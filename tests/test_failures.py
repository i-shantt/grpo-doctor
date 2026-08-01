"""The failure grid must be well-formed before 450 runs are spent on it.

The expensive mistakes this catches are silent ones: an override path that no longer exists (the
run would execute happily under the *default* setting and be filed under a family it never
exhibited), a family whose null dose is not actually null, or an onset distribution narrow enough
to hand the step-index control a free win.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from testbed.core.grpo import GRPOConfig  # noqa: E402
from testbed.core.train import RunConfig, apply_overrides  # noqa: E402
from testbed.inject.failures import (  # noqa: E402
    ALL_SPECS,
    FAILURES,
    HARD_NEGATIVES,
    HEALTHY,
    ONSET_RANGE,
    families,
    sample_onset,
    spec_by_cell,
)


def _cfg(**kw) -> RunConfig:
    base = dict(seed=0, steps=10, n_prompts=2, grpo=GRPOConfig(group_size=4, num_iterations=2))
    base.update(kw)
    return RunConfig(**base)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.cell for s in ALL_SPECS])
def test_every_override_path_exists(spec) -> None:
    """The expensive silent failure: a renamed field means the knob is never applied, the run
    executes under the default, and the trace is filed under a family it never exhibited."""
    out = apply_overrides(_cfg(), spec.overrides)
    assert out is not None


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.cell for s in ALL_SPECS])
def test_every_override_actually_changes_something(spec) -> None:
    base = _cfg()
    out = apply_overrides(base, spec.overrides)
    if spec is HEALTHY:
        assert out == base
    else:
        assert out != base, f"{spec.cell} left the config untouched"


def test_healthy_is_the_null_dose() -> None:
    assert HEALTHY.overrides == {}
    assert HEALTHY.needs_onset is False


def test_cells_are_unique() -> None:
    cells = [s.cell for s in ALL_SPECS]
    assert len(cells) == len(set(cells))


def test_every_family_has_more_than_one_dose_or_is_the_control() -> None:
    """Graded doses, not switches: a binary knob cannot answer how large a perturbation must be
    before it becomes detectable, nor reveal that a knob does nothing."""
    counts: dict[str, int] = {}
    for s in ALL_SPECS:
        counts[s.family] = counts.get(s.family, 0) + 1
    for family, n in counts.items():
        if family in {"F0", "H2", "H3", "H4", "H5"}:
            continue
        assert n >= 2, f"family {family} has only one dose"


def test_lookup_by_cell_round_trips() -> None:
    for s in ALL_SPECS:
        assert spec_by_cell(s.cell) is s
    with pytest.raises(KeyError):
        spec_by_cell("F99/nonexistent")


# --- onset ----------------------------------------------------------------------------------------


def test_onset_is_spread_across_the_whole_range() -> None:
    """A narrow or centered onset distribution hands the step-index-only control a free win, and
    if that control wins the entire study is an artifact."""
    rng = np.random.default_rng(0)
    draws = np.array([sample_onset(rng) for _ in range(4000)])
    lo, hi = ONSET_RANGE
    assert draws.min() <= lo + 5 and draws.max() >= hi - 5
    assert abs(draws.mean() - (lo + hi) / 2) < 5
    # Roughly uniform: no decile should hold more than 1.5x its share.
    hist, _ = np.histogram(draws, bins=10, range=(lo, hi + 1))
    assert hist.max() < 1.5 * len(draws) / 10


def test_onset_is_reproducible_from_the_seed() -> None:
    a = [sample_onset(np.random.default_rng(7)) for _ in range(3)]
    b = [sample_onset(np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_normalization_family_is_set_from_step_zero() -> None:
    """Switching scale_rewards mid-run injects a discontinuity in the loss scale that has nothing
    to do with the instability being studied."""
    for s in ALL_SPECS:
        if s.family == "F8":
            assert not s.needs_onset


def test_simulated_families_are_flagged() -> None:
    """F9 perturbs rollout logits to imitate a train/inference gap that is exactly zero in this
    testbed. The flag has to travel with the run or the caveat gets lost in the results."""
    simulated = {s.family for s in ALL_SPECS if s.simulated}
    assert simulated == {"F9"}


def test_hard_negatives_are_separate_from_failures() -> None:
    """They are negatives. If one ended up in FAILURES it would be labeled positive by family and
    the false-alarm rate would be computed against the wrong denominator."""
    assert not ({s.family for s in HARD_NEGATIVES} & {s.family for s in FAILURES})
    assert all(s.family.startswith("H") for s in HARD_NEGATIVES)
    assert all(s.family.startswith("F") for s in FAILURES)


def test_a_plateau_hard_negative_exists() -> None:
    """The one that matters. Flat reward with a high zero-variance fraction is indistinguishable
    from starvation collapse until it resolves, and it is where the false-alarm budget goes."""
    assert any(s.family == "H2" for s in HARD_NEGATIVES)


def test_families_are_stable_and_ordered() -> None:
    fams = families()
    assert fams[0] == "F0"
    assert len(fams) == len(set(fams))
    assert "F5" in fams and "H2" in fams
