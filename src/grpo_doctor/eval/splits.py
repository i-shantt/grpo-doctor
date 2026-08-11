"""Leave-one-mode-out and leave-one-config-out splits, with the calibration set kept separate.

The headline protocol is LOMO: fit on every failure family except `F`, evaluate on held-out `F`.
That answers "does a detector trained on other pathologies transfer to one it has never seen",
which is the only question worth asking of a monitor somebody would actually leave running -- the
next pathology is by definition not in the corpus.

**Three sets, not two, and the third is the one that is easy to get wrong.** A threshold fixed so
that 5% of healthy runs alarm is itself fit to those runs. If the same healthy runs then supply the
reported false-alarm rate, that rate is 5% by construction and carries no information at all. So
every split here partitions the negatives into a `calib` half, which the threshold is solved
against, and an `eval` half, which the reported FAR is measured on. Neither is ever used to fit the
model. The cost is that each number rests on half the negatives; the benefit is that the FAR is a
measurement rather than a restatement of the target.

**Negatives are partitioned by seed, not at random.** Seeds are the corpus's own independence axis
-- every seed is a different initialization and a different randomized onset -- and splitting on
them keeps the two halves exchangeable without needing a shuffle to be reproducible. Runs from the
same cell and seed always travel together, so a detector cannot be calibrated on one dose of a hard
negative and evaluated on a sibling that shares its initialization.

**Positives from the held-out family are never in `fit`, and neither is anything else from it.**
That includes its *negative* runs -- cells where the knob was set and the run stayed healthy. Those
are the most informative negatives in the corpus and it is tempting to keep them in the fit set,
but a run that was dosed with `F` and survived still carries `F`'s signature in its training
signals. Leaving it in would leak the held-out mode through the back door.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

NEGATIVE_FAMILIES: frozenset[str] = frozenset({"F0", "H2", "H3", "H4", "H5"})
"""Families that exist to *not* collapse: the control and the hard negatives.

Membership here is a statement about the design of the cell, never about the label a run received.
A dosed run that stayed healthy is still a failure-family run; it is a negative *example* but it is
not a member of a negative family, and the distinction is what keeps `held_out` honest.
"""


@dataclass(frozen=True)
class RunRef:
    """The minimum a split needs to know about a run.

    Deliberately not `RunOutcome`: splits are decided before any detector runs, and depending on an
    outcome type would make it possible to write a split that peeks at whether a run collapsed.
    """

    run_id: str
    task: str
    family: str
    seed: int

    @property
    def is_negative_family(self) -> bool:
        return self.family in NEGATIVE_FAMILIES


@dataclass(frozen=True)
class Split:
    """One fold: what to fit on, what to calibrate the threshold on, what to report."""

    name: str
    held_out: str
    """The family (or config value) withheld. `""` for a pooled split."""

    fit: tuple[str, ...]
    calib: tuple[str, ...]
    """Negative runs only. The threshold is solved here so the FAR can be measured elsewhere."""

    eval: tuple[str, ...]
    """Held-out positives plus the negative half `calib` did not take."""

    def __post_init__(self) -> None:
        overlap = (set(self.fit) & set(self.eval)) | (set(self.fit) & set(self.calib))
        if overlap:
            raise ValueError(f"{self.name}: fit overlaps another set on {sorted(overlap)[:3]}")
        if set(self.calib) & set(self.eval):
            raise ValueError(f"{self.name}: calib and eval overlap")

    @property
    def sizes(self) -> dict[str, int]:
        return {"fit": len(self.fit), "calib": len(self.calib), "eval": len(self.eval)}


def _partition_negatives(negatives: Sequence[RunRef]) -> tuple[list[str], list[str]]:
    """Split negatives into (calib, eval) halves on seed parity.

    Parity rather than a shuffle so the partition is reproducible without carrying a seed around,
    and balanced within every cell rather than only in aggregate: an odd/even split of a cell with
    10 seeds gives 5 and 5, where a random half of the whole pool could empty a cell entirely and
    silently drop a hard-negative type out of the FAR breakdown.
    """
    calib = [r.run_id for r in negatives if r.seed % 2 == 0]
    ev = [r.run_id for r in negatives if r.seed % 2 == 1]
    return calib, ev


def lomo_splits(runs: Iterable[RunRef]) -> list[Split]:
    """One split per failure family: fit on the rest, evaluate on it.

    Negative families never form a fold of their own -- there is nothing to detect in them, so a
    "held-out F0" fold would report a lead time over an empty set. They are instead partitioned
    into calib and eval and appear in every fold, which is what makes the per-fold false-alarm
    rates comparable to each other.

    Note what that leaves in `fit`: only failure-family runs, most of which did *not* collapse.
    That is deliberate and it is the better training set of the two available. The model learns to
    tell collapse apart from "a knob was applied and the run survived it", which is the hard and
    useful discrimination, while the pure controls and hard negatives are spent entirely on
    calibrating and then measuring the false-alarm rate -- the numbers they are uniquely able to
    support and the ones the study is judged on.
    """
    all_runs = list(runs)
    negatives = [r for r in all_runs if r.is_negative_family]
    calib_ids, eval_neg_ids = _partition_negatives(negatives)

    families = sorted({r.family for r in all_runs if not r.is_negative_family})
    out: list[Split] = []
    for fam in families:
        held = [r.run_id for r in all_runs if r.family == fam]
        fit = [r.run_id for r in all_runs if r.family != fam and not r.is_negative_family]
        out.append(
            Split(
                name=f"lomo:{fam}",
                held_out=fam,
                fit=tuple(fit),
                calib=tuple(calib_ids),
                eval=tuple(held + eval_neg_ids),
            )
        )
    return out


def loco_splits(runs: Iterable[RunRef], key: str = "task") -> list[Split]:
    """Leave-one-config-out: hold out a whole *setup* rather than a pathology.

    Nested with LOMO in the report because they answer different questions. LOMO asks whether a
    detector transfers to an unseen pathology; LOCO asks whether it transfers to an unseen task at
    pathologies it has already seen. A monitor can pass one and fail the other, and reporting only
    the flattering one would be the easiest unearned claim in the study.
    """
    all_runs = list(runs)
    if key != "task":
        raise ValueError(f"unsupported LOCO key: {key!r}")

    out: list[Split] = []
    for value in sorted({r.task for r in all_runs}):
        held_runs = [r for r in all_runs if r.task == value]
        fit_runs = [r for r in all_runs if r.task != value]
        calib_ids, eval_neg_ids = _partition_negatives(
            [r for r in held_runs if r.is_negative_family]
        )
        out.append(
            Split(
                name=f"loco:{key}={value}",
                held_out=value,
                fit=tuple(r.run_id for r in fit_runs if not r.is_negative_family),
                calib=tuple(calib_ids),
                eval=tuple(
                    [r.run_id for r in held_runs if not r.is_negative_family] + eval_neg_ids
                ),
            )
        )
    return out


__all__ = [
    "NEGATIVE_FAMILIES",
    "RunRef",
    "Split",
    "loco_splits",
    "lomo_splits",
]
