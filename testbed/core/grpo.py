"""Group-Relative Policy Optimization: advantages, clipped surrogate loss, and instrumentation.

Written from scratch rather than wrapping TRL, for three reasons that are specific rather than
not-invented-here:

1. `scale_rewards` and `loss_type` are first-class ablation axes here, not configuration. The whole
   project is about what the *estimator* does to the signals, so the advantage math has to be
   visible and directly testable against hand-computed values.
2. The testbed policy is a from-scratch `nn.Module` with a ~20-token vocabulary and no tokenizer.
   TRL's abstractions are all model-object-centric and cannot run it.
3. Several quantities we need are not logged by TRL at all: the full per-token importance-ratio
   distribution (TRL logs only mean clip fractions), per-group advantage vectors, and the fraction
   of sequences contributing exactly zero gradient.

TRL is instead used as the **correctness oracle**: `tests/test_grpo_math.py` asserts these losses
match TRL's documented closed forms on hand-computed batches for all three `loss_type` values
crossed with all three `scale_rewards` values, and the metric keys emitted here deliberately mirror
TRL's own key names so the monitor is drop-in against a real TRL log.

Reference closed forms (TRL `grpo_trainer.py`), with l_{i,t} the per-token clipped surrogate:

    grpo     loss = -(1/N) * sum_i  (1/|o_i|) * sum_t l_{i,t}
    dr_grpo  loss = -(1/(N*L)) *    sum_i sum_t l_{i,t}          L = max_completion_length
    dapo     loss = -(1/sum_i|o_i|) sum_i sum_t l_{i,t}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

LossType = Literal["grpo", "dr_grpo", "dapo"]
ScaleRewards = Literal["group", "batch", "none"]


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    """G: completions sampled per prompt. Bounds the advantage magnitude -- see advantage_bound()."""

    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    loss_type: LossType = "dapo"
    scale_rewards: ScaleRewards = "group"

    num_iterations: int = 2
    """mu: gradient updates per rollout batch.

    MUST be >= 2 for any run in the corpus. At mu=1 the policy that generated the rollouts is the
    policy being updated, so the importance ratio is identically 1 and both clip metrics are exactly
    0.0 for the entire run. That is not a quiet signal, it is a structurally absent one, and a
    monitor that reads it as "healthy" is reading a constant column. Measured, not assumed:
    mu=1 -> clip fraction 0.000 across 80 steps; mu=8 -> ratio_max 39.6.
    """

    max_completion_length: int = 32
    advantage_eps: float = 1e-4
    """Matches TRL's `+ 1e-4` in the advantage denominator.

    Note this epsilon only ever *shrinks* |A| for degenerate groups -- at reward spread 1e-5 it
    takes |A|max from 2.646 down to 0.085. It cannot cause the "advantage explosion" that zero-
    variance groups are often blamed for; see advantage_bound().
    """

    zero_std_eps: float = 1e-3
    """Threshold for counting a group as degenerate.

    Deliberately NOT TRL's test. TRL uses `torch.isclose(std, 0)` (default atol 1e-8), so a group
    with std=1e-3 is not counted -- yet such a group produces advantages that are pure amplified
    noise. We report both: `frac_reward_zero_std` (TRL-compatible) and `acr` (this threshold).
    """

    entropy_coef: float = 0.0


def advantage_bound(group_size: int, unbiased: bool = True) -> float:
    """The exact supremum of |A| under `scale_rewards="group"`.

    Group standardization is scale-invariant. For a reward vector r of length G that is not
    constant, A = (r - mean)/std satisfies sum(A) = 0, and the sum of squares is fixed by the std
    convention:

        unbiased (n-1, Bessel):  sum(A^2) = G - 1   ->  sup |A| = (G - 1) / sqrt(G)
        biased   (n, population): sum(A^2) = G      ->  sup |A| = sqrt(G - 1)

    In both cases the maximum is attained when one component differs and the other G-1 are equal.

    **TRL uses the unbiased convention**, so `(G - 1) / sqrt(G)` is the bound that applies to real
    runs: `nanstd` in `trl/trainer/utils.py` computes `torch.nanmean((x - mean)**2)` and then
    multiplies by `count / (count - 1)`. For G=8 that is 2.4749, not the 2.6458 you get from a
    population std. This distinction is easy to get wrong -- we did, initially, and only caught it
    by reading TRL's source rather than reasoning from a textbook definition.

    What matters for the project is unaffected by which convention is in play: the bound is
    *finite and independent of the reward scale, the spread, and how degenerate the group is*. So
    the widely repeated claim that zero-variance groups cause advantage explosion is false. As the
    spread goes to zero the numerator vanishes at the same rate as the denominator, and TRL's
    `+ 1e-4` pushes the advantage toward *zero*. Degenerate groups cause silent death (A = 0, no
    gradient), not blow-up -- a failure no advantage-magnitude threshold can detect.

    Explosion is real only under `scale_rewards` in {"batch", "none"}, where the per-group
    normalization is gone and this bound does not hold. That is why it is its own failure family.
    """
    g = float(group_size)
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    return (g - 1.0) / g**0.5 if unbiased else (g - 1.0) ** 0.5


def compute_advantages(
    rewards: Tensor,
    cfg: GRPOConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Group-relative advantages plus the reward-side diagnostics.

    Args:
        rewards: (n_groups, G) or (n_groups * G,). Completions of one prompt must share a group.

    Returns:
        advantages flattened to (n_groups * G,), and a metrics dict using TRL-compatible key names.
    """
    g = cfg.group_size
    if rewards.dim() == 1:
        if rewards.numel() % g != 0:
            raise ValueError(f"reward count {rewards.numel()} is not a multiple of group_size={g}")
        grouped = rewards.view(-1, g)
    elif rewards.dim() == 2:
        if rewards.shape[1] != g:
            raise ValueError(f"rewards have group dim {rewards.shape[1]}, expected {g}")
        grouped = rewards
    else:
        raise ValueError(f"rewards must be 1-D or 2-D, got shape {tuple(rewards.shape)}")

    grouped = grouped.float()
    group_mean = grouped.mean(dim=1, keepdim=True)
    # Unbiased (n-1) std, matching torch.std default and TRL.
    group_std = grouped.std(dim=1, keepdim=True)

    centered = grouped - group_mean
    if cfg.scale_rewards == "group":
        adv = centered / (group_std + cfg.advantage_eps)
    elif cfg.scale_rewards == "batch":
        # Lite-PPO: group-level mean, batch-level std.
        adv = centered / (grouped.std() + cfg.advantage_eps)
    elif cfg.scale_rewards == "none":
        # Dr. GRPO: mean-centred only. No sqrt(G-1) bound applies here.
        adv = centered
    else:
        raise ValueError(f"unknown scale_rewards: {cfg.scale_rewards!r}")

    # TRL-compatible degeneracy count, and our own with a usable epsilon.
    trl_zero_std = torch.isclose(group_std.squeeze(1), torch.zeros(1)).float().mean()
    acr = (group_std.squeeze(1) < cfg.zero_std_eps).float().mean()

    # Fraction of sequences whose advantage is numerically zero, i.e. that contribute no policy
    # gradient at all. This is the quantity that actually matters for starvation, and it is the one
    # that is invisible if you only watch advantage magnitude.
    dead = (adv.abs() < 1e-8).float().mean()

    metrics = {
        "reward": float(grouped.mean()),
        "reward_std": float(grouped.std()),
        "frac_reward_zero_std": float(trl_zero_std),
        "acr": float(acr),
        "frac_zero_advantage": float(dead),
        "advantage_abs_max": float(adv.abs().max()),
        "group_reward_std_mean": float(group_std.mean()),
    }
    return adv.reshape(-1), metrics


def grpo_loss(
    logits: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
    advantages: Tensor,
    old_logprobs: Tensor | None,
    cfg: GRPOConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Clipped surrogate loss over completion tokens, plus per-token instrumentation.

    Args:
        logits: (B, T, V) logits for the completion positions under the *current* policy.
        completion_ids: (B, T) sampled token ids.
        completion_mask: (B, T) 1.0 for real completion tokens, 0.0 for padding past EOS.
        advantages: (B,) one scalar per sequence, from compute_advantages().
        old_logprobs: (B, T) log-probs under the rollout policy, or None for a strictly on-policy
            step. When None the ratio is 1 by construction and the clip metrics are reported as
            NaN rather than 0.0 -- "not measurable" is different from "no clipping happened", and
            conflating them is precisely the bug this project exists to catch.
    """
    logprobs_all = F.log_softmax(logits.float(), dim=-1)
    logprobs = logprobs_all.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)

    mask = completion_mask.float()
    n_tokens = mask.sum().clamp(min=1.0)

    # Per-token entropy of the full distribution, averaged over real tokens only.
    with torch.no_grad():
        token_entropy = -(logprobs_all.exp() * logprobs_all).sum(-1)  # (B, T)
        mean_entropy = float((token_entropy * mask).sum() / n_tokens)

    if old_logprobs is None:
        ratio = torch.ones_like(logprobs)
        log_ratio = torch.zeros_like(logprobs)
        measurable = False
    else:
        log_ratio = logprobs - old_logprobs
        ratio = log_ratio.exp()
        measurable = True

    adv = advantages.unsqueeze(1)  # (B, 1) broadcast over tokens
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.epsilon_low, 1.0 + cfg.epsilon_high) * adv
    per_token_loss = -torch.min(unclipped, clipped)

    if cfg.loss_type == "grpo":
        # Sequence-mean then batch-mean: short completions get more per-token weight.
        seq_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        loss = seq_loss.mean()
    elif cfg.loss_type == "dr_grpo":
        loss = (per_token_loss * mask).sum() / (
            per_token_loss.size(0) * cfg.max_completion_length
        )
    elif cfg.loss_type == "dapo":
        loss = (per_token_loss * mask).sum() / n_tokens
    else:
        raise ValueError(f"unknown loss_type: {cfg.loss_type!r}")

    if cfg.entropy_coef != 0.0:
        loss = loss - cfg.entropy_coef * (
            -(logprobs_all.exp() * logprobs_all).sum(-1) * mask
        ).sum() / n_tokens

    with torch.no_grad():
        metrics: dict[str, float] = {
            "entropy": mean_entropy,
            "completions/mean_length": float(mask.sum(-1).mean()),
            # A completion that never emitted EOS is truncated; near-1 is a repetition loop.
            "completions/clipped_ratio": float((mask.sum(-1) >= mask.shape[1]).float().mean()),
            "policy_loss": float(loss),
        }
        if measurable:
            low = ((ratio < 1.0 - cfg.epsilon_low) & (adv < 0)).float()
            high = ((ratio > 1.0 + cfg.epsilon_high) & (adv > 0)).float()
            metrics.update(
                {
                    "clip_ratio/low_mean": float((low * mask).sum() / n_tokens),
                    "clip_ratio/high_mean": float((high * mask).sum() / n_tokens),
                    "clip_ratio/region_mean": float(((low + high) * mask).sum() / n_tokens),
                    "importance_ratio/max": float(ratio[mask > 0].max()),
                    "importance_ratio/mean": float((ratio * mask).sum() / n_tokens),
                    # Heavy-tail probe: log-ratio dispersion catches drift before the clip
                    # fractions move, because clipping saturates.
                    "importance_ratio/log_std": float(log_ratio[mask > 0].std()),
                }
            )
        else:
            nan = float("nan")
            metrics.update(
                {
                    "clip_ratio/low_mean": nan,
                    "clip_ratio/high_mean": nan,
                    "clip_ratio/region_mean": nan,
                    "importance_ratio/max": nan,
                    "importance_ratio/mean": nan,
                    "importance_ratio/log_std": nan,
                }
            )

    return loss, metrics
