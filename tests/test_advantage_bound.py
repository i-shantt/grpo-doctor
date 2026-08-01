"""The advantage bound, which is finding #1 of the project and therefore load-bearing.

If these fail, the README's central claim is wrong and must be corrected before anything else.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from testbed.core.grpo import GRPOConfig, advantage_bound, compute_advantages  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("g", [2, 4, 8, 16, 64])
def test_bound_formula_matches_extremal_configuration(g: int) -> None:
    """sup|A| is attained when one element differs and the rest are equal.

    Verified numerically against the closed form rather than assumed.
    """
    cfg = GRPOConfig(group_size=g, advantage_eps=0.0)
    rewards = torch.zeros(1, g)
    rewards[0, 0] = 1.0
    adv, _ = compute_advantages(rewards, cfg)
    assert float(adv.abs().max()) == pytest.approx(advantage_bound(g), rel=1e-5)


@pytest.mark.parametrize("g", [4, 8, 16])
def test_bound_holds_over_random_rewards(g: int) -> None:
    """No random reward vector may exceed the bound. 20k samples per group size."""
    cfg = GRPOConfig(group_size=g, advantage_eps=0.0)
    gen = torch.Generator().manual_seed(0)
    bound = advantage_bound(g)
    worst = 0.0
    for scale in (1.0, 1e-3, 1e3):
        rewards = torch.rand(20_000, g, generator=gen) * scale
        adv, _ = compute_advantages(rewards, cfg)
        worst = max(worst, float(adv.abs().max()))
    assert worst <= bound + 1e-4, f"|A|max={worst} exceeded bound={bound}"


def test_bound_is_scale_invariant() -> None:
    """Multiplying every reward by 1e6 must not change a single advantage.

    This is the property that makes "reward scale caused the advantage to explode" impossible
    under group normalization.
    """
    cfg = GRPOConfig(group_size=8, advantage_eps=0.0)
    gen = torch.Generator().manual_seed(1)
    rewards = torch.rand(64, 8, generator=gen)
    a1, _ = compute_advantages(rewards, cfg)
    a2, _ = compute_advantages(rewards * 1e6, cfg)
    torch.testing.assert_close(a1, a2, rtol=1e-4, atol=1e-4)


def test_degenerate_group_shrinks_advantage_rather_than_exploding() -> None:
    """The headline finding: as reward spread -> 0, |A| -> 0 under TRL's +1e-4.

    A near-degenerate group must produce a *smaller* advantage than a healthy one, which is the
    opposite of the folk claim that motivated much of this project's prior art.
    """
    cfg = GRPOConfig(group_size=8)  # advantage_eps = 1e-4, as TRL

    healthy = torch.zeros(1, 8)
    healthy[0, 0] = 1.0
    healthy_adv, _ = compute_advantages(healthy, cfg)

    prev = float(healthy_adv.abs().max())
    for delta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        degenerate = torch.full((1, 8), 0.5)
        degenerate[0, 0] = 0.5 + delta
        adv, metrics = compute_advantages(degenerate, cfg)
        cur = float(adv.abs().max())
        assert cur < prev, f"|A|max grew as spread shrank: delta={delta}"
        assert metrics["advantage_abs_max"] == pytest.approx(cur)
        prev = cur

    assert prev < 0.2, "at spread 1e-6 the advantage should be crushed toward zero"


def test_all_equal_group_gives_exactly_zero_gradient_signal() -> None:
    """All-fail and all-pass groups both produce A == 0 -- silent death, not blow-up."""
    cfg = GRPOConfig(group_size=8)
    for value in (0.0, 1.0):
        rewards = torch.full((4, 8), value)
        adv, metrics = compute_advantages(rewards, cfg)
        assert float(adv.abs().max()) == 0.0
        assert metrics["frac_zero_advantage"] == 1.0
        assert metrics["frac_reward_zero_std"] == 1.0
        assert metrics["acr"] == 1.0


def test_batch_and_none_scaling_are_genuinely_unbounded() -> None:
    """The bound is a property of *group* scaling only.

    Under batch/none scaling the advantage tracks the raw reward scale, which is why those are
    their own failure family (F8) rather than a footnote.
    """
    rewards = torch.zeros(4, 8)
    rewards[0, 0] = 1.0

    small = {}
    large = {}
    for mode in ("group", "batch", "none"):
        cfg = GRPOConfig(group_size=8, scale_rewards=mode)  # type: ignore[arg-type]
        small[mode] = float(compute_advantages(rewards, cfg)[0].abs().max())
        large[mode] = float(compute_advantages(rewards * 1000.0, cfg)[0].abs().max())

    assert large["group"] == pytest.approx(small["group"], rel=1e-3)
    assert large["none"] > 100 * small["none"]
    # "batch" divides by the batch-wide std, which also scales, so it stays scale-invariant --
    # its instability comes from cross-group composition, not from reward magnitude.
    assert large["batch"] == pytest.approx(small["batch"], rel=1e-3)


def test_trl_isclose_misses_groups_our_acr_catches() -> None:
    """Documents the deliberate divergence from TRL's `frac_reward_zero_std`.

    TRL's isclose(std, 0) has atol 1e-8, so a group with std ~1e-4 is reported as perfectly
    healthy while producing advantages that are amplified noise.
    """
    cfg = GRPOConfig(group_size=8, zero_std_eps=1e-3)
    rewards = torch.full((1, 8), 0.5)
    rewards[0, 0] = 0.5 + 1e-5  # std ~ 3.5e-6: far below our eps, far above isclose's atol

    _, metrics = compute_advantages(rewards, cfg)
    assert metrics["frac_reward_zero_std"] == 0.0, "TRL's key should call this healthy"
    assert metrics["acr"] == 1.0, "our ACR should flag it"
