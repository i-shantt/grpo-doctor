"""Split hygiene. Every test here is guarding against a form of leakage.

None of these can fail loudly in a report -- a leaked split produces *better* numbers, which is
exactly why they need to be asserted rather than eyeballed.
"""

from __future__ import annotations

import pytest

from grpo_doctor.eval.splits import (
    NEGATIVE_FAMILIES,
    RunRef,
    Split,
    loco_splits,
    lomo_splits,
)

FAILURE_FAMILIES = ("F1", "F5", "F9")


def _corpus(seeds: int = 6) -> list[RunRef]:
    runs: list[RunRef] = []
    for task in ("sort_digits", "countdown_lite"):
        for fam in (*FAILURE_FAMILIES, *sorted(NEGATIVE_FAMILIES)):
            for seed in range(seeds):
                runs.append(RunRef(f"{task}_{fam}_s{seed}", task, fam, seed))
    return runs


def test_the_held_out_family_never_appears_in_fit() -> None:
    """The whole point of LOMO. Includes the family's *negative* runs, which still carry its dose."""
    runs = _corpus()
    by_id = {r.run_id: r for r in runs}
    for split in lomo_splits(runs):
        assert not any(by_id[r].family == split.held_out for r in split.fit)


def test_every_run_of_the_held_out_family_is_evaluated() -> None:
    """Dropping the survivors would evaluate only on runs that collapsed, inflating detection."""
    runs = _corpus()
    by_id = {r.run_id: r for r in runs}
    for split in lomo_splits(runs):
        held = {r.run_id for r in runs if r.family == split.held_out}
        assert held <= set(split.eval)
        assert all(
            by_id[r].family == split.held_out or by_id[r].is_negative_family for r in split.eval
        )


def test_calibration_and_evaluation_negatives_are_disjoint() -> None:
    """A threshold solved for 5% FAR on the same runs that report FAR would report 5% by fiat."""
    for split in lomo_splits(_corpus()):
        assert set(split.calib) & set(split.eval) == set()
        assert len(split.calib) > 0 and len(split.eval) > 0


def test_no_negative_run_is_ever_fit_on() -> None:
    """Controls are spent entirely on calibration and FAR; fit negatives come from dosed runs."""
    runs = _corpus()
    by_id = {r.run_id: r for r in runs}
    for split in lomo_splits(runs):
        assert not any(by_id[r].is_negative_family for r in split.fit)


def test_negatives_are_split_evenly_within_every_cell() -> None:
    """Seed parity, not a shuffle: a random half could empty a hard-negative type from the FAR
    breakdown without anything failing."""
    runs = _corpus(seeds=6)
    split = lomo_splits(runs)[0]
    by_id = {r.run_id: r for r in runs}
    for fam in NEGATIVE_FAMILIES:
        for task in ("sort_digits", "countdown_lite"):
            c = sum(1 for r in split.calib if by_id[r].family == fam and by_id[r].task == task)
            e = sum(1 for r in split.eval if by_id[r].family == fam and by_id[r].task == task)
            assert c == e == 3


def test_one_fold_per_failure_family_and_none_per_negative_family() -> None:
    """A held-out F0 fold would report a lead time over an empty positive set."""
    held = {s.held_out for s in lomo_splits(_corpus())}
    assert held == set(FAILURE_FAMILIES)


def test_loco_holds_out_a_whole_task() -> None:
    """LOMO and LOCO answer different questions; a monitor can pass one and fail the other."""
    runs = _corpus()
    by_id = {r.run_id: r for r in runs}
    splits = loco_splits(runs)
    assert {s.held_out for s in splits} == {"sort_digits", "countdown_lite"}
    for split in splits:
        assert not any(by_id[r].task == split.held_out for r in split.fit)
        assert all(by_id[r].task == split.held_out for r in split.eval)
        assert all(by_id[r].task == split.held_out for r in split.calib)


def test_overlapping_sets_are_rejected_at_construction() -> None:
    """The invariant is enforced by the type, so a hand-built split cannot leak quietly."""
    with pytest.raises(ValueError, match="overlaps"):
        Split(name="x", held_out="F1", fit=("a",), calib=("a",), eval=("b",))
    with pytest.raises(ValueError, match="overlap"):
        Split(name="x", held_out="F1", fit=("a",), calib=("b",), eval=("b",))


def test_an_unsupported_loco_key_is_refused_rather_than_ignored() -> None:
    """Silently falling back to task would report a split that is not the one that was asked for."""
    with pytest.raises(ValueError, match="unsupported"):
        loco_splits(_corpus(), key="lr_band")
