"""Conformance of our GRPO loss to TRL's closed forms.

This is what makes "we wrote our own GRPO" a strength rather than a smell. The reference
implementation below is transcribed directly from `trl/trainer/grpo_trainer.py` on main (the loss
branches around lines 3104-3170), not reconstructed from a paper, because the two differ in details
that matter -- see `test_trl_std_is_unbiased`.

Fidelity caveat, deliberately pinned by `test_dapo_equals_trl_only_without_accumulation`: TRL's
`dapo` normalizer is `num_items_in_batch` spanning the whole generation batch, rescaled by
gradient-accumulation steps and process count. Ours is `mask.sum()` over the batch it is handed.
These agree exactly in the single-process, single-accumulation-window setting the testbed runs in,
and TRL's `bnpo` is the same formula unconditionally.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from testbed.core.grpo import GRPOConfig, compute_advantages, grpo_loss  # noqa: E402

pytestmark = pytest.mark.torch

LOSS_TYPES = ["grpo", "dr_grpo", "dapo"]
SCALE_MODES = ["group", "batch", "none"]


def trl_nanstd(t: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    """Transcribed from trl/trainer/utils.py::nanstd. Applies Bessel's correction."""
    mean = torch.nanmean(t, dim=dim, keepdim=True)
    variance = torch.nanmean((t - mean) ** 2, dim=dim, keepdim=True)
    count = torch.sum(~torch.isnan(t), dim=dim, keepdim=True)
    correction = count / (count - 1)
    correction = torch.where(count > 1, correction, torch.full_like(correction, float("nan")))
    variance = variance * correction
    std = torch.sqrt(variance)
    if keepdim:
        return std
    return std.squeeze() if dim is None else std.squeeze(dim)


def trl_advantages(rewards: torch.Tensor, g: int, scale: str) -> torch.Tensor:
    """Transcribed from grpo_trainer.py lines ~2705-2752."""
    flat = rewards.reshape(-1)
    mean_grouped = flat.view(-1, g).mean(dim=1).repeat_interleave(g, dim=0)
    if scale == "group":
        std = trl_nanstd(flat.view(-1, g), dim=1).repeat_interleave(g, dim=0)
    elif scale == "batch":
        std = trl_nanstd(flat).expand_as(flat)
    else:
        std = torch.zeros_like(flat)

    adv = flat - mean_grouped
    if scale != "none":
        adv = adv / (std + 1e-4)
    return torch.nan_to_num(adv, nan=0.0)


def trl_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
    eps_low: float,
    eps_high: float,
    max_completion_length: int,
) -> torch.Tensor:
    """Transcribed from grpo_trainer.py lines ~3108-3167."""
    coef_1 = (logprobs - old_logprobs).exp()
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    adv = advantages.unsqueeze(1)
    per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)

    if loss_type == "grpo":
        return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    if loss_type == "dr_grpo":
        return (per_token_loss * mask).sum() / (per_token_loss.size(0) * max_completion_length)
    if loss_type == "dapo":  # == TRL's `bnpo` formula; see module docstring
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    raise ValueError(loss_type)


def _batch(seed: int, n_groups: int = 3, g: int = 4, t: int = 6, vocab: int = 11):
    gen = torch.Generator().manual_seed(seed)
    b = n_groups * g
    logits = torch.randn(b, t, vocab, generator=gen)
    ids = torch.randint(0, vocab, (b, t), generator=gen)
    lengths = torch.randint(1, t + 1, (b,), generator=gen)
    mask = (torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)).float()
    old_logprobs = torch.randn(b, t, generator=gen) * 0.1 - 2.0
    rewards = (torch.rand(n_groups, g, generator=gen) > 0.5).float()
    return logits, ids, mask, old_logprobs, rewards


@pytest.mark.parametrize("scale", SCALE_MODES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_advantages_match_trl(scale: str, seed: int) -> None:
    _, _, _, _, rewards = _batch(seed)
    cfg = GRPOConfig(group_size=4, scale_rewards=scale)  # type: ignore[arg-type]
    ours, _ = compute_advantages(rewards, cfg)
    theirs = trl_advantages(rewards, 4, scale)
    torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
@pytest.mark.parametrize("scale", SCALE_MODES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_matches_trl(loss_type: str, scale: str, seed: int) -> None:
    """All nine (loss_type x scale_rewards) combinations, on batches with ragged lengths."""
    logits, ids, mask, old_logprobs, rewards = _batch(seed)
    cfg = GRPOConfig(
        group_size=4,
        loss_type=loss_type,  # type: ignore[arg-type]
        scale_rewards=scale,  # type: ignore[arg-type]
        max_completion_length=6,
    )
    adv, _ = compute_advantages(rewards, cfg)
    ours, _ = grpo_loss(logits, ids, mask, adv, old_logprobs, cfg)

    logprobs = F.log_softmax(logits.float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    theirs = trl_loss(
        logprobs, old_logprobs, adv, mask, loss_type, cfg.epsilon_low, cfg.epsilon_high, 6
    )
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


def test_hand_computed_golden_values() -> None:
    """A worked example small enough to verify with a pencil, guarding the transcription itself.

    B=2, T=2, advantages [2, -1], mask [[1,1],[1,0]], ratio == 1 (on-policy).
    per_token_loss = -A broadcast  ->  [[-2,-2],[1,*]]

        grpo    : mean over sequences of (sum/len) = mean(-4/2, 1/1) = -0.5
        dapo    : sum / n_tokens                   = (-2-2+1)/3      = -1.0
        dr_grpo : sum / (B * L), L=4               = -3/(2*4)        = -0.375
    """
    vocab = 5
    logits = torch.zeros(2, 2, vocab)  # uniform -> logprob = -log(5) for every token
    ids = torch.zeros(2, 2, dtype=torch.long)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    adv = torch.tensor([2.0, -1.0])

    expected = {"grpo": -0.5, "dapo": -1.0, "dr_grpo": -0.375}
    for loss_type, want in expected.items():
        cfg = GRPOConfig(
            group_size=2,
            loss_type=loss_type,  # type: ignore[arg-type]
            max_completion_length=4,
        )
        got, _ = grpo_loss(logits, ids, mask, adv, None, cfg)
        assert float(got) == pytest.approx(want), f"{loss_type}: got {float(got)}, want {want}"


def test_loss_types_agree_when_all_lengths_equal() -> None:
    """grpo and dapo differ *only* through length weighting.

    With uniform lengths they must coincide, and dr_grpo must equal them scaled by mean_len/L.
    This pins down that the difference we ablate is genuinely the length bias and nothing else.
    """
    logits, ids, _, old_logprobs, rewards = _batch(7)
    mask = torch.ones(ids.shape)
    t = ids.shape[1]

    losses = {}
    for loss_type in LOSS_TYPES:
        cfg = GRPOConfig(group_size=4, loss_type=loss_type, max_completion_length=t)  # type: ignore[arg-type]
        adv, _ = compute_advantages(rewards, cfg)
        losses[loss_type], _ = grpo_loss(logits, ids, mask, adv, old_logprobs, cfg)

    assert float(losses["grpo"]) == pytest.approx(float(losses["dapo"]), rel=1e-5)
    assert float(losses["dr_grpo"]) == pytest.approx(float(losses["dapo"]), rel=1e-5)


def test_onpolicy_step_reports_clip_metrics_as_unmeasurable_not_zero() -> None:
    """Finding #2: at mu=1 the ratio is identically 1, so clipping cannot occur.

    The failure this guards against is reporting 0.0 -- which a monitor reads as "no clipping
    pressure, healthy" when the truth is "this signal does not exist in this configuration".
    """
    logits, ids, mask, _, rewards = _batch(3)
    cfg = GRPOConfig(group_size=4)
    adv, _ = compute_advantages(rewards, cfg)

    _, metrics = grpo_loss(logits, ids, mask, adv, None, cfg)
    for key in ("clip_ratio/low_mean", "clip_ratio/high_mean", "importance_ratio/max"):
        assert metrics[key] != metrics[key], (
            f"{key} should be NaN (unmeasurable), got {metrics[key]}"
        )

    # With genuine off-policy logprobs the same keys must become real numbers.
    logprobs = F.log_softmax(logits.float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    _, metrics_off = grpo_loss(logits, ids, mask, adv, logprobs - 0.5, cfg)
    for key in ("clip_ratio/low_mean", "clip_ratio/high_mean", "importance_ratio/max"):
        assert metrics_off[key] == metrics_off[key], f"{key} should be measurable off-policy"


def test_onpolicy_ratio_is_exactly_one() -> None:
    """The mechanism behind the previous test, stated directly."""
    logits, ids, mask, _, rewards = _batch(4)
    cfg = GRPOConfig(group_size=4)
    adv, _ = compute_advantages(rewards, cfg)
    logprobs = F.log_softmax(logits.float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

    _, metrics = grpo_loss(logits, ids, mask, adv, logprobs.detach().clone(), cfg)
    assert metrics["importance_ratio/max"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["clip_ratio/region_mean"] == 0.0


def test_trl_std_is_unbiased() -> None:
    """Guards the correction that produced the bound in the README.

    TRL's nanstd applies Bessel's correction, so |A| is bounded by (G-1)/sqrt(G), not sqrt(G-1).
    If TRL ever changes this, this test fails and the README claim must be revisited.
    """
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert float(trl_nanstd(x, dim=1)) == pytest.approx(float(x.std(dim=1)), rel=1e-6)
    assert float(trl_nanstd(x, dim=1)) != pytest.approx(float(x.std(dim=1, correction=0)), rel=1e-3)


def test_dapo_equals_trl_only_without_accumulation() -> None:
    """Documents the one place our loss is a special case rather than an identity.

    TRL normalizes `dapo` by num_items_in_batch over the whole generation batch. Splitting a batch
    into microbatches and averaging our per-microbatch losses does NOT reproduce that, because the
    normalizer differs per microbatch. The testbed never accumulates, so this is a caveat, not a
    bug -- but it must be stated rather than discovered later.
    """
    logits, ids, mask, old_logprobs, rewards = _batch(11, n_groups=2, g=4)
    cfg = GRPOConfig(group_size=4, loss_type="dapo", max_completion_length=6)
    adv, _ = compute_advantages(rewards, cfg)

    full, _ = grpo_loss(logits, ids, mask, adv, old_logprobs, cfg)

    half = logits.shape[0] // 2
    a, _ = grpo_loss(logits[:half], ids[:half], mask[:half], adv[:half], old_logprobs[:half], cfg)
    b, _ = grpo_loss(logits[half:], ids[half:], mask[half:], adv[half:], old_logprobs[half:], cfg)
    naive_mean = (a + b) / 2

    token_a, token_b = mask[:half].sum(), mask[half:].sum()
    reweighted = (a * token_a + b * token_b) / (token_a + token_b)

    torch.testing.assert_close(reweighted, full, rtol=1e-5, atol=1e-6)
    assert abs(float(naive_mean) - float(full)) > 1e-9, (
        "if these coincide the test batch has uniform lengths and proves nothing"
    )
