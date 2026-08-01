"""S5 -- completion length and truncation.

This is the cheapest signal on the panel and, going in, the one most likely to beat the fuser on
its own. That makes it the **baseline to beat**, and the project reports it that way: if a
nine-signal model cannot separate itself from `truncation` alone under leave-one-mode-out, the
honest conclusion is that the information was concentrated in one free number all along.

Two readings, and the distinction matters more than it looks:

`truncation` is `completions/clipped_ratio` -- the fraction of completions that hit the token
budget without emitting EOS. It is highly specific: a run whose completions stop terminating is
almost always in a repetition loop or a degenerate long-output mode.

`len_drift` is the standardized change in mean completion length. It is much *less* specific,
because long-CoT training grows completion length legitimately, all the way to the budget. Growth
alone is not pathology, which is why the two are separate readings and why the drift signal is
reported as a z-score against the run's own history rather than as a level.

Measured caveat carried from the testbed: under the `sort_digits` and `ca_rule` verifier leaks the
policy hacks the reward with **no change in length at all**. The exploit is not shorter, merely
wrong. So this signal is structurally blind to two of the four reward-hacking families here, and
any claim that length detects reward hacking in general is contradicted by that.
"""

from __future__ import annotations

import math
from typing import Any

from grpo_doctor.record import StepRecord
from grpo_doctor.signals.base import EWMA, RunningMoments, missing
from grpo_doctor.snapshot import NOT_LOGGED, WARMING_UP, SignalReading


class Truncation:
    name = "truncation"

    MIN_STD = 0.002
    """A truncation rate flat to within 0.2% of completions carries no information."""

    MIN_COUNT = 20

    def __init__(self) -> None:
        self._moments = RunningMoments()

    def update(self, rec: StepRecord) -> SignalReading:
        v = rec.completion_clipped_ratio
        if v is None or not math.isfinite(v):
            return missing(self.name, NOT_LOGGED)
        v = float(v)
        z = self._moments.z(v, floor=self.MIN_STD, min_count=self.MIN_COUNT)
        self._moments.update(v)
        return SignalReading(name=self.name, available=True, value=v, z=z)

    def state_dict(self) -> dict[str, Any]:
        return {"moments": self._moments.state_dict()}


class LengthDrift:
    """Standardized change in smoothed mean completion length.

    The `burn_in` is not a tuning knob, it is a correctness requirement. This signal z-scores the
    *step-to-step change* in an EWMA, and while that EWMA is still converging toward the series its
    increments are large and systematically shrinking -- a non-stationary stretch whose statistics
    describe the filter warming up rather than the run. Accumulating those into the baseline
    produced a measured z of -6.64 at step 35 on a healthy stretch of a real trace, an alarm 133
    steps before anything happened, which then got credited as lead time. Two halflives of burn-in
    removes it.
    """

    name = "len_drift"

    MIN_STD = 0.01
    """Tokens per step. Below this the mean length is constant and the deltas are numerical noise."""

    MIN_COUNT = 20

    def __init__(self, halflife: float = 10.0, burn_in: int | None = None) -> None:
        self._smooth = EWMA(halflife)
        self._moments = RunningMoments()
        self._prev: float | None = None
        self._seen = 0
        self._burn_in = int(2 * halflife) if burn_in is None else burn_in

    def update(self, rec: StepRecord) -> SignalReading:
        v = rec.completion_len_mean
        if v is None or not math.isfinite(v):
            return missing(self.name, NOT_LOGGED)
        v = float(v)
        self._seen += 1
        smoothed = self._smooth.update(v)
        if self._prev is None or smoothed is None:
            self._prev = smoothed
            return SignalReading(name=self.name, available=True, value=0.0, z=None)

        delta = smoothed - self._prev
        self._prev = smoothed
        if self._seen <= self._burn_in:
            # Observed but deliberately not accumulated: the filter is still converging.
            return SignalReading(
                name=self.name, available=True, value=delta, z=None, reason=WARMING_UP
            )

        z = self._moments.z(delta, floor=self.MIN_STD, min_count=self.MIN_COUNT)
        self._moments.update(delta)
        return SignalReading(name=self.name, available=True, value=delta, z=z)

    def state_dict(self) -> dict[str, Any]:
        return {
            "smooth": self._smooth.state_dict(),
            "moments": self._moments.state_dict(),
            "prev": self._prev,
            "seen": self._seen,
        }


__all__ = ["LengthDrift", "Truncation"]
