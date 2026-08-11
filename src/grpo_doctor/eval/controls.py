"""The four negative controls, in the headline table rather than an appendix.

A monitor's own numbers mean nothing on their own. These four say what the same numbers look like
when the detector is replaced by something that cannot possibly work, and each one falsifies a
different way the study could be fooling itself.

**Step-index-only** is the control a skeptical reviewer asks for first. Its only feature is how far
into the run the step is, so if it matches the real monitor the corpus is time-confounded and every
result in the study is an artifact of collapses happening at predictable steps. It is the reason
onsets are randomized in [50, 250]. If it fires, that is the finding, and it gets published as one.

**Shuffled-label** gives the null at this sample size. Without it a detection rate of 0.7 is
uninterpretable, because nobody knows what chance looks like over 472 runs with this family
imbalance. The permutation is *across* runs and never within: permuting within a run would leave
every run's collapse time in place and leak the thing being tested.

**Reward-only** is the minimum bar. It is what a practitioner already watches in W&B, so a monitor
that does not beat it is a worse version of scrolling a dashboard. It is also the control the
project's headline case is designed to defeat: under reward hacking the training reward *rises*
through the collapse, so a drawdown on reward cannot fire at all. That failure is the point, and
reporting it next to the monitor is what makes the hacking case legible.

**Constant-alarm** anchors the far end. It fires on everything, so its detection rate is 1.0 at a
false-alarm rate of 1.0, which is what keeps the FAR axis honest -- a detection rate means nothing
until you can see what it costs.

Every control here produces a per-step score, exactly like a real detector, so the identical alarm
logic runs over all of them and nothing in the evaluation path knows which arm it is scoring. That
is deliberate: a control evaluated by a different code path is not a control, it is a second
experiment. The single exception is `constant_alarm`, which is defined by its threshold rather than
by its score and therefore cannot be calibrated -- see `Control.calibrated`, where the exception is
carried in the type rather than left as a special case for a caller to remember.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from grpo_doctor.eval.metrics import RunOutcome, calibrate_threshold
from grpo_doctor.record import StepRecord


def _series(records: Sequence[StepRecord], field: str) -> np.ndarray:
    """A float array with missing entries carried forward, then filled with the first real value.

    Forward-fill rather than zero-fill: a missing metric means "not logged this step", and zero is
    a legitimate value for most of these. Substituting it would manufacture a drawdown.
    """
    out = np.full(len(records), np.nan, dtype=float)
    for i, rec in enumerate(records):
        v = getattr(rec, field, None)
        if v is not None:
            out[i] = float(v)
    if np.all(np.isnan(out)):
        return np.zeros(len(records), dtype=float)
    idx = np.where(~np.isnan(out), np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    first = out[~np.isnan(out)]
    filled: np.ndarray = np.nan_to_num(out, nan=float(first[0]) if first.size else 0.0)
    return filled


def step_index_only(records: Sequence[StepRecord]) -> np.ndarray:
    """Score = fraction of the run elapsed. Monotone, so a threshold is "alarm after step k".

    Uses `t/T` rather than `t` so runs of different lengths are on one scale; with every corpus run
    at 600 steps the two are equivalent here, but a monitor calibrated on this corpus and run
    against a longer job would otherwise inherit an absolute step count as if it were a rate.
    """
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=float)
    return np.arange(n, dtype=float) / max(n - 1, 1)


def reward_only(records: Sequence[StepRecord]) -> np.ndarray:
    """Score = drawdown of training reward below its running peak.

    The practitioner's heuristic, stated precisely enough to be evaluated. It is *structurally*
    incapable of the case the project exists for: when the policy games the verifier, training
    reward rises monotonically and this score stays pinned at zero through the entire collapse.
    """
    r = _series(records, "reward_mean")
    if r.size == 0:
        return r
    drawdown: np.ndarray = np.maximum.accumulate(r) - r
    return drawdown


def constant_alarm(records: Sequence[StepRecord]) -> np.ndarray:
    """Fires at step 0 of every run. Detection 1.0 at a false-alarm rate of 1.0."""
    return np.full(len(records), np.inf, dtype=float)


@dataclass(frozen=True)
class Control:
    """A control arm and whether its threshold is solved for like a detector's.

    The flag exists because `constant_alarm` cannot be calibrated, and pretending otherwise would
    quietly turn it into the opposite of what it is for. `calibrate_threshold` finds the smallest
    threshold at which at most 5% of healthy runs cross. Against a score that is identical at every
    step of every run there is no such threshold: any cut either lets all runs through or none, and
    the routine correctly picks none -- so an arm defined as "always alarms" would be evaluated as
    one that never does, and the far end of the false-alarm axis would silently go missing from the
    table it exists to anchor.
    """

    name: str
    score: Callable[[Sequence[StepRecord]], np.ndarray]
    calibrated: bool = True
    """False means the threshold is fixed at -inf rather than solved for a target FAR."""

    def threshold_for(self, healthy: Sequence[np.ndarray], target_far: float = 0.05) -> float:
        if not self.calibrated:
            return float("-inf")
        return calibrate_threshold(healthy, target_far)


CONTROLS: dict[str, Control] = {
    c.name: c
    for c in (
        Control("step_index_only", step_index_only),
        Control("reward_only", reward_only),
        Control("constant_alarm", constant_alarm, calibrated=False),
    )
}
"""The three *scoring* controls. `shuffled_label` is a label permutation and cannot join them."""


@dataclass(frozen=True)
class ShuffledLabels:
    """The null: outcomes with their labels permuted across runs.

    Not a scorer. It leaves the detector and its alarms exactly as they are and destroys only the
    correspondence between a run's alarm and that run's ground truth, which is what isolates "the
    detector found something" from "something in this corpus makes any detector look good".
    """

    seed: int = 0

    def apply(self, outcomes: Sequence[RunOutcome]) -> list[RunOutcome]:
        """Permute `collapsed` and `t_collapse` across runs, keeping alarms in place.

        The two travel together. Permuting `collapsed` while leaving `t_collapse` attached to its
        original run would produce runs marked collapsed with no collapse step and vice versa, and
        the null would be measuring an incoherent label rather than a shuffled one.
        """
        runs = list(outcomes)
        if not runs:
            return []
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(runs))
        return [
            replace(
                run,
                collapsed=runs[j].collapsed,
                t_collapse=runs[j].t_collapse,
                censored=runs[j].censored,
            )
            for run, j in zip(runs, perm, strict=True)
        ]


def alarm_step(scores: np.ndarray, threshold: float) -> int | None:
    """First step whose score crosses the threshold, or None.

    Strictly greater than, so a threshold placed at a score a run actually attains does not count
    as a crossing for that run. With `calibrate_threshold` solving for a target false-alarm rate on
    healthy runs, `>=` would make the realized rate overshoot the target by however many runs sit
    exactly on the boundary -- which for `constant_alarm`, whose scores are all identical, is all
    of them.
    """
    hits = np.flatnonzero(np.asarray(scores, dtype=float) > threshold)
    return int(hits[0]) if hits.size else None


__all__ = [
    "CONTROLS",
    "Control",
    "ShuffledLabels",
    "alarm_step",
    "constant_alarm",
    "reward_only",
    "step_index_only",
]
