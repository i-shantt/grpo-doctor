"""One GRPO run, start to finish, emitting one record per optimizer step.

This is the thing the corpus is made of. Three decisions here are load-bearing for whether the
corpus means anything.

**Oracle quarantine is mechanical, not procedural.** Every quantity the monitor must never see is
emitted under an `oracle/` key prefix. The labeler reads those keys; the monitor is handed the
record with them stripped, by a function rather than by anyone remembering to. That is what makes
the blindness property checkable instead of aspirational -- if a signal ever starts depending on
held-out accuracy, the prefix convention is what catches it.

**The importance ratio is only meaningful because `old_logprobs` come from the sampler.** They are
carried on the `Rollout` from generation and reused unchanged across all mu inner iterations, which
is exactly what TRL does with `_old_per_token_logps`. Recomputing them from the current policy would
force the ratio to 1 at every inner step and silently delete the entire clipping family.

**Per-step metrics are averaged across inner iterations, matching TRL.** This matters more than it
looks: at inner iteration 0 the policy has not moved since the rollout, so the ratio is exactly 1
and the clip fractions are exactly 0 *by construction*. Only iterations >= 1 can clip. So the
logged clip fraction at mu=2 is already halved relative to the clipping that actually occurred, and
a threshold calibrated without knowing that is calibrated against an artifact of mu.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from testbed.core.grpo import GRPOConfig, compute_advantages, grpo_loss
from testbed.core.model import TinyGPT
from testbed.core.optim import InstrumentedAdamW, OptimConfig
from testbed.core.rollout import generate, score
from testbed.core.warmstart import DEFAULT_CACHE_DIR, load_or_train
from testbed.tasks.base import Batch, Task, VerifierConfig, decode

ORACLE_PREFIX = "oracle/"

PROBE_SEED = 20260801
"""Fixed, so the probe set and its sampling noise are identical at every step and across every run.

Common random numbers: the accuracy *trajectory* within a run is then far less noisy than fresh
sampling would make it, which sharpens `t_collapse` since the label is defined on a drawdown. The
cost is that the binomial standard error used to set delta slightly overstates the noise, making
the collapse threshold conservative -- it will miss marginal collapses before it invents any.
"""


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    steps: int = 300
    n_prompts: int = 8
    """Prompts per optimizer step. The rollout batch is n_prompts * group_size sequences."""

    difficulty: int = 6
    """The difficulty the **probe** always uses. Training difficulty comes from
    `difficulty_range`."""

    difficulty_range: tuple[int, int] | None = None
    """Inclusive range of training difficulties, drawn afresh each step.

    Training on a single difficulty makes `difficulty` unusable as a failure knob, and the bug it
    caused is worth recording. With a policy warm-started only on 6-digit problems, injecting
    difficulty 2 -- *easier* -- drove reward to exactly 0.000 and the degenerate-group fraction to
    1.00, identical to injecting difficulty 8. Both directions broke the policy equally, because
    what changed was not the difficulty but the shape of the input: a different number of prompt
    tokens, a different pad pattern, and a different answer length, none of which it had ever seen.
    F1 (starvation) and F2 (saturation) were producing byte-identical traces as a result.

    Training over a range makes difficulty a real axis. Narrowing the range then genuinely means
    harder or easier problems rather than an out-of-distribution prompt format.
    """

    temperature: float = 1.0
    sampler_noise: float = 0.0

    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4

    warm_start_steps: int = 150
    """Supervised steps before RL begins -- a *maximum* when `warm_start_target` is set.

    Not a detail: measured, 300 SFT steps left the policy at reward 0.969 with entropy 0.064, i.e.
    nothing left to collapse, and 150 gave 0.547. A run that starts saturated cannot exhibit any
    failure mode and is a wasted corpus slot.
    """
    warm_start_lr: float = 3e-3

    warm_start_target: tuple[float, float] | None = None
    """Held-out accuracy band to warm-start *into*, instead of a fixed step count.

    Below the band there is nothing to fall from; above it there is no headroom. Corpus runs set
    this so every seed in a cell starts from a comparable policy -- see `warm_start` for the
    measurement that motivates it.
    """
    warm_start_probe_every: int = 100

    probe_every: int = 10
    probe_n: int = 256

    onset_step: int | None = None
    """Step at which `onset_overrides` is applied.

    Randomized per run over [50, 250] by the corpus builder rather than fixed, because a fixed
    onset makes the corpus a step-counter benchmark that the step-index-only control would win --
    which would invalidate every number downstream.
    """
    onset_overrides: dict[str, Any] = field(default_factory=dict)
    """Dotted paths into this config, e.g. {"grpo.num_iterations": 8, "verifier.flip_p": 0.2}."""


def apply_overrides(cfg: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Return a copy with dotted-path fields replaced. Unknown paths raise rather than no-op."""
    out = cfg
    for path, value in overrides.items():
        head, _, tail = path.partition(".")
        if not hasattr(out, head):
            raise KeyError(f"RunConfig has no field {head!r} (from {path!r})")
        if tail:
            sub = getattr(out, head)
            if not hasattr(sub, tail):
                raise KeyError(f"{type(sub).__name__} has no field {tail!r} (from {path!r})")
            out = replace(out, **{head: replace(sub, **{tail: value})})
        else:
            out = replace(out, **{head: value})
    return out


def draw_difficulty(cfg: RunConfig, rng: np.random.Generator) -> int:
    """Training difficulty for one step. Fixed at `cfg.difficulty` when no range is configured."""
    if cfg.difficulty_range is None:
        return cfg.difficulty
    lo, hi = cfg.difficulty_range
    return int(rng.integers(lo, hi + 1))


def strip_oracle(record: dict[str, float]) -> dict[str, float]:
    """What the monitor is allowed to see. The blindness guarantee, as one function."""
    return {k: v for k, v in record.items() if not k.startswith(ORACLE_PREFIX)}


def _completion_targets(task: Task, batch: Batch, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth completions (answer + EOS), right-padded to `width`."""
    n = len(batch.problems)
    tgt = np.full((n, width), task.pad_id, dtype=np.int64)
    mask = np.zeros((n, width), dtype=np.float32)
    for i, p in enumerate(batch.problems):
        seq = [*p.answer, task.eos_id]
        if len(seq) > width:
            raise ValueError(f"answer of length {len(seq)} exceeds completion width {width}")
        tgt[i, : len(seq)] = seq
        mask[i, : len(seq)] = 1.0
    return tgt, mask


def warm_start(model: TinyGPT, task: Task, cfg: RunConfig) -> dict[str, float]:
    """Supervised fine-tuning to put the initial pass rate in a regime that can actually collapse.

    Two modes, and the second is the one corpus runs use.

    **Fixed budget** (`warm_start_target=None`): train for exactly `warm_start_steps`.

    **Targeted** (the default): train until held-out accuracy first lands inside
    `warm_start_target`, checking every `warm_start_probe_every` steps, and stop there. This exists
    because a fixed budget does not produce a fixed starting point. Measured across five seeds at
    the same 1000 supervised steps, `ca_rule` started at 0.277, 0.281, 0.199, 0.031 and 0.152 --
    one seed had learned essentially nothing while another was three-quarters of the way to the
    band. Seeds inside a cell would then differ in their initial accuracy as much as the failure
    knob differs from the control, and initialization variance would be attributed to the
    injection. Targeting the accuracy instead of the step count removes that confound, and it costs
    almost nothing: a probe is 0.07s against a warm start measured in minutes.

    Stopping on a *sampled* probe means stopping on a noisy estimate, so the run records the
    accuracy it actually stopped at rather than assuming the target was hit exactly.

    Trained on the `train` split only, so the probe stays untouched even during warm start -- a run
    that memorized probe answers before RL began would show an inflated, unfalsifiable baseline.
    """
    max_steps = cfg.warm_start_steps
    if max_steps <= 0:
        return {
            "warm_start/final_loss": float("nan"),
            "warm_start/steps_used": 0.0,
            "warm_start/accuracy": float("nan"),
            "warm_start/hit_target": float("nan"),
        }

    rng = np.random.default_rng(cfg.seed + 1)
    opt = InstrumentedAdamW(model, OptimConfig(lr=cfg.warm_start_lr, max_grad_norm=1.0))
    model.train()
    width = task.max_completion_len
    loss_val = float("nan")
    target = cfg.warm_start_target
    used = 0

    for step in range(max_steps):
        # Same difficulty distribution the RL phase will see, so the policy is never surprised by
        # a prompt shape at onset that it was not trained on.
        batch = task.sample(
            cfg.n_prompts * cfg.grpo.group_size, draw_difficulty(cfg, rng), rng, "train"
        )
        prompts = torch.from_numpy(batch.prompts)
        tgt_np, mask_np = _completion_targets(task, batch, width)
        tgt = torch.from_numpy(tgt_np)
        mask = torch.from_numpy(mask_np)

        full = torch.cat([prompts, tgt], dim=1)
        logits, _ = model(full[:, :-1])
        p = prompts.shape[1]
        logits = logits[:, p - 1 :, :]

        lp = F.log_softmax(logits.float(), dim=-1)
        token_ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = -(token_ll * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
        used = step + 1

        if target is not None and used % cfg.warm_start_probe_every == 0:
            acc = evaluate_probe(model, task, cfg)
            model.train()
            if acc >= target[0]:
                # Stop on the first crossing rather than the first value *inside* the band: past
                # the upper edge the policy is saturated, and overshooting is not recoverable by
                # training further.
                return {
                    "warm_start/final_loss": loss_val,
                    "warm_start/steps_used": float(used),
                    "warm_start/accuracy": float(acc),
                    "warm_start/hit_target": float(target[0] <= acc <= target[1]),
                }

    acc = evaluate_probe(model, task, cfg) if target is not None else float("nan")
    model.train()
    return {
        "warm_start/final_loss": loss_val,
        "warm_start/steps_used": float(used),
        "warm_start/accuracy": float(acc),
        # Exhausted the budget without reaching the band. Not an error -- the run is still valid --
        # but it starts below the collapsible regime and will almost certainly be a STALL.
        "warm_start/hit_target": 0.0 if target is not None else float("nan"),
    }


@torch.no_grad()
def evaluate_probe(model: TinyGPT, task: Task, cfg: RunConfig) -> float:
    """Held-out accuracy under `verify_true`. The labeling oracle, never a reward, never a feature.

    **Always evaluated at the run's base difficulty, never the post-onset one.** This is called with
    the original `cfg` even after an injection has changed `active.difficulty`, and that is
    deliberate: if the probe followed the injection, an F1 difficulty-starvation run would show an
    accuracy drop simply because the evaluation got harder, and `t_collapse` would be labeling a
    change in the ruler rather than a change in the policy. The measuring stick has to stay fixed
    for a drawdown to mean anything.
    """
    rng = np.random.default_rng(PROBE_SEED + cfg.difficulty)
    batch = task.sample(cfg.probe_n, cfg.difficulty, rng, "probe")
    gen = torch.Generator().manual_seed(PROBE_SEED)
    roll = generate(
        model,
        torch.from_numpy(batch.prompts),
        max_new_tokens=task.max_completion_len,
        eos_id=task.eos_id,
        pad_id=task.pad_id,
        temperature=1.0,
        generator=gen,
    )
    ids = roll.completion_ids.numpy()
    correct = sum(
        task.verify_true(decode(ids[i], task.eos_id), p) for i, p in enumerate(batch.problems)
    )
    return correct / len(batch.problems)


def _aggregate(per_iter: list[dict[str, float]]) -> dict[str, float]:
    """Mean across inner iterations, matching TRL's logging, except that maxima take the max.

    `nanmean` over a column that is NaN at every iteration yields NaN, which is what we want: the
    mu=1 case must stay explicitly unmeasurable rather than becoming a zero.
    """
    keys = per_iter[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([d[k] for d in per_iter], dtype=float)
        if np.all(np.isnan(vals)):
            out[k] = float("nan")
        elif k.endswith("/max"):
            out[k] = float(np.nanmax(vals))
        else:
            out[k] = float(np.nanmean(vals))
    return out


def run(
    task: Task,
    cfg: RunConfig,
    *,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> Iterator[dict[str, float]]:
    """Execute one run, yielding a metrics record per optimizer step.

    Yields eagerly so a caller can stream to disk and so an auto-halt demo can stop mid-run.

    The warm start is cached by default. Caching cannot change the run: model init is the only
    consumer of the global torch RNG, sampling draws from an explicitly seeded generator, and the
    supervised phase is deterministic given the key -- so a cached run and a freshly warm-started
    one produce byte-identical records. `test_cached_and_fresh_runs_are_identical` holds that.
    """
    model, warm = load_or_train(task, cfg, cache_dir=cache_dir)

    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = InstrumentedAdamW(model, cfg.optim)
    active = cfg
    g = cfg.grpo.group_size
    last_probe = evaluate_probe(model, task, cfg)

    for step in range(cfg.steps):
        if cfg.onset_step is not None and step == cfg.onset_step:
            active = apply_overrides(cfg, cfg.onset_overrides)
            # The optimizer is rebuilt only if its config changed, so momentum is preserved
            # otherwise; a silent Adam-state reset would look like a shock the knob did not cause.
            if active.optim != cfg.optim:
                opt = InstrumentedAdamW(model, active.optim)

        batch = task.sample(cfg.n_prompts, draw_difficulty(active, rng), rng, "train")
        prompts = torch.from_numpy(batch.prompts).repeat_interleave(g, dim=0)

        model.train()
        roll = generate(
            model,
            prompts,
            max_new_tokens=task.max_completion_len,
            eos_id=task.eos_id,
            pad_id=task.pad_id,
            temperature=active.temperature,
            sampler_noise=active.sampler_noise,
            generator=gen,
        )

        ids = roll.completion_ids.numpy()
        rewards, true_hits = [], 0
        for i in range(ids.shape[0]):
            completion = decode(ids[i], task.eos_id)
            problem = batch.problems[i // g]
            rewards.append(task.verify_train(completion, problem, active.verifier, rng))
            true_hits += bool(task.verify_true(completion, problem))

        reward_t = torch.tensor(rewards, dtype=torch.float32).view(cfg.n_prompts, g)
        advantages, adv_metrics = compute_advantages(reward_t, active.grpo)

        # None at mu=1 so the ratio is 1 by construction and the clip metrics report NaN rather
        # than a fake zero -- the same distinction TRL's `old_per_token_logps is None` branch makes.
        old_lp = roll.logprobs if active.grpo.num_iterations >= 2 else None

        per_iter = []
        for _ in range(active.grpo.num_iterations):
            logits = score(model, roll, temperature=active.temperature)
            loss, loss_metrics = grpo_loss(
                logits,
                roll.completion_ids,
                roll.completion_mask,
                advantages,
                old_lp,
                active.grpo,
            )
            loss.backward()
            per_iter.append({**loss_metrics, **opt.step()})

        record: dict[str, float] = {
            "step": float(step),
            **adv_metrics,
            **_aggregate(per_iter),
            f"{ORACLE_PREFIX}train_true_accuracy": true_hits / ids.shape[0],
        }

        if step % cfg.probe_every == 0:
            last_probe = evaluate_probe(model, task, cfg)
            record[f"{ORACLE_PREFIX}heldout_accuracy"] = last_probe
            record[f"{ORACLE_PREFIX}probe_fresh"] = 1.0
        else:
            # Carried forward so the column is dense, flagged so the labeler can tell the K-step
            # quantization apart from a genuine plateau.
            record[f"{ORACLE_PREFIX}heldout_accuracy"] = last_probe
            record[f"{ORACLE_PREFIX}probe_fresh"] = 0.0

        if step == 0:
            # Quarantined wholesale, including `steps_used` and `final_loss`. It is tempting to
            # expose the step count as harmless configuration, but with a targeted warm start the
            # *stopping rule reads the probe*, so how long the warm start ran is a function of
            # oracle data: "this run needed 4200 supervised steps to reach 0.25" tells a monitor
            # how hard the task turned out to be, which is exactly the sort of side channel the
            # blindness guarantee exists to close.
            record.update({f"{ORACLE_PREFIX}{k.replace('/', '_')}": v for k, v in warm.items()})
        yield record
