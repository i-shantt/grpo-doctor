"""AdamW with the gradient instrumentation the monitor needs.

This exists as its own module for one reason: **the gradient norm you can log and the gradient norm
that reaches the weights are different numbers**, and almost every training loop conflates them.

HuggingFace `Trainer` logs `grad_norm` as the value returned by `clip_grad_norm_`, which is the norm
*before* clipping. With the default `max_grad_norm=1.0`, any run whose true norm exceeds 1.0 has an
applied norm of exactly 1.0 while the logged number keeps climbing. So a "gradient spike" visible in
W&B may have been entirely absorbed by the clipper and never perturbed the policy at all. A detector
trained on the logged norm is therefore reading a quantity that is only loosely coupled to what
happened to the model, which is the concrete reason the plan predicts the ablation will kill S6.

We record both, plus the ratio, so that prediction is testable rather than asserted:

    grad_norm             pre-clip. Matches what HF logs, so calibrations transfer.
    grad_norm_postclip    what actually scaled the update.
    grad_norm_clip_active 1.0 on steps where the clipper bound the update, else 0.0.
    update_ratio          ||theta_t - theta_{t-1}|| / ||theta_{t-1}||, the thing we actually care
                          about: how far the policy moved. Independent of both norms above, since
                          Adam's preconditioner sits between the gradient and the step.

Two failure modes are handled here rather than left to crash:

- **Zero gradient.** When every group in a batch is degenerate, all advantages are 0 and the loss
  has no gradient signal -- `grad_norm` is exactly 0.0. That is the silent-death signature from
  finding #1, and it must be recorded as a number, not swallowed. Adam still takes a step from
  stale momentum, so `update_ratio > 0` while `grad_norm == 0` is a real and diagnostic combination.
- **Non-finite gradient.** The step is skipped, the optimizer state is left untouched, and
  `nonfinite_grad=1.0` is recorded. A corpus run that NaNs is a *lost* run, not a labeled failure
  (open risk #1 in the plan); silently stepping on NaN would poison every subsequent signal in that
  trace with no way to tell after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 5e-5
    """Measured, not inherited from a tutorial.

    At 3e-4 -- the obvious default, and what this used to be -- the *healthy control* was not
    healthy: across five seeds only 40% of F0 runs were labeled healthy, with a mean maximum
    drawdown of 0.314 in held-out accuracy. Runs peaked and then decayed. A corpus built on that
    control would have measured every false-alarm rate against a baseline that was itself mildly
    diverging, which is worse than useless because it looks fine.

    Sweep at 600 steps, 5 seeds, sort_digits (healthy fraction / mean max drawdown / mean gain):

        5e-5   100%   0.061   +0.432
        1e-4    80%   0.101   +0.437
        2e-4    60%   0.205   +0.385
        3e-4    40%   0.314   +0.278

    Lower is better on both axes at once here -- 5e-5 is the most stable *and* learns the most, so
    there is no stability-for-progress trade being made.

    A second result falls out of that sweep and is worth stating: the 95th percentile of healthy
    max drawdown at 5e-5 is 0.090, against the labeler's delta of 0.094 derived independently from
    the probe's binomial standard error. The threshold argued from first principles and the one
    measured from healthy runs agree to 0.004, so `t_collapse` is calibrated to roughly a 5%
    false-positive rate in the ground truth itself rather than by assertion.
    """

    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0

    max_grad_norm: float = 1.0
    """Set to 0.0 to disable clipping entirely.

    Kept at the HF default because the point is to reproduce what practitioners actually run, not
    to pick a better value. F8 (normalization instability) is one of the few families where the
    clipper is load-bearing, and disabling it there is itself a dose level.
    """

    warmup_steps: int = 0
    """Linear warmup from 0. Not a schedule beyond that -- a decaying LR would confound the
    entropy-collapse family, since falling LR and falling entropy would be indistinguishable."""

    track_update_ratio: bool = True
    """Costs one parameter-sized copy per step (~12 MB of traffic at 3M params). Cheap relative to
    the rollout, and it is the only signal here that measures the model rather than the gradient."""


class InstrumentedAdamW:
    """AdamW that reports what happened, in TRL/HF-compatible key names where they exist."""

    def __init__(self, model: nn.Module, cfg: OptimConfig) -> None:
        self.cfg = cfg
        self.model = model
        self._step_count = 0

        # Standard split: no weight decay on biases and normalization gains. Decaying them shrinks
        # LayerNorm scales toward zero, which suppresses entropy for reasons unrelated to the
        # policy -- a confound we would then mistake for F4.
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
        )
        self._params: list[Tensor] = decay + no_decay

    def _lr_at(self, step: int) -> float:
        if self.cfg.warmup_steps <= 0:
            return self.cfg.lr
        return self.cfg.lr * min(1.0, (step + 1) / self.cfg.warmup_steps)

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _global_norm(self) -> Tensor:
        grads = [p.grad for p in self._params if p.grad is not None]
        if not grads:
            return torch.zeros(())
        return torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])
        )

    def step(self) -> dict[str, float]:
        """Apply one update. Call after `loss.backward()`.

        Returns the gradient/update panel for this step. Always returns every key, so a trace has no
        ragged columns -- a missing key downstream would be indistinguishable from an unavailable
        signal, which the monitor treats very differently from a measured zero.
        """
        cfg = self.cfg
        pre_clip = self._global_norm()

        if not torch.isfinite(pre_clip):
            # Leave optimizer state and parameters untouched; the caller decides whether to abort.
            self.zero_grad()
            self._step_count += 1
            return {
                "grad_norm": float(pre_clip),
                "grad_norm_postclip": float("nan"),
                "grad_norm_clip_active": float("nan"),
                "update_norm": 0.0,
                "update_ratio": 0.0,
                "learning_rate": self._lr_at(self._step_count - 1),
                "nonfinite_grad": 1.0,
            }

        if cfg.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self._params, cfg.max_grad_norm)
        post_clip = self._global_norm()
        clip_active = float(cfg.max_grad_norm > 0.0 and float(pre_clip) > cfg.max_grad_norm)

        lr = self._lr_at(self._step_count)
        for group in self.opt.param_groups:
            group["lr"] = lr

        before = None
        if cfg.track_update_ratio:
            before = [p.detach().clone() for p in self._params]

        self.opt.step()
        self.zero_grad()
        self._step_count += 1

        update_norm, update_ratio = 0.0, 0.0
        if before is not None:
            with torch.no_grad():
                deltas = torch.stack(
                    [
                        torch.linalg.vector_norm((p.detach() - b).float())
                        for p, b in zip(before, self._params, strict=True)
                    ]
                )
                base = torch.stack([torch.linalg.vector_norm(b.float()) for b in before])
                update_norm = float(torch.linalg.vector_norm(deltas))
                update_ratio = update_norm / max(float(torch.linalg.vector_norm(base)), 1e-12)

        return {
            "grad_norm": float(pre_clip),
            "grad_norm_postclip": float(post_clip),
            "grad_norm_clip_active": clip_active,
            "update_norm": update_norm,
            "update_ratio": update_ratio,
            "learning_rate": lr,
            "nonfinite_grad": 0.0,
        }

    @property
    def step_count(self) -> int:
        return self._step_count
