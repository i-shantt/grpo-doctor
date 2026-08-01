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
from grpo_doctor.snapshot import NOT_LOGGED, SignalReading


class Truncation:
    name = "truncation"

    def __init__(self) -> None:
        self._moments = RunningMoments()

    def update(self, rec: StepRecord) -> SignalReading:
        v = rec.completion_clipped_ratio
        if v is None or not math.isfinite(v):
            return missing(self.name, NOT_LOGGED)
        v = float(v)
        z = self._moments.z(v)
        self._moments.update(v)
        return SignalReading(name=self.name, available=True, value=v, z=z)

    def state_dict(self) -> dict[str, Any]:
        return {"moments": self._moments.state_dict()}


class LengthDrift:
    name = "len_drift"

    def __init__(self, halflife: float = 10.0) -> None:
        self._smooth = EWMA(halflife)
        self._moments = RunningMoments()
        self._prev: float | None = None

    def update(self, rec: StepRecord) -> SignalReading:
        v = rec.completion_len_mean
        if v is None or not math.isfinite(v):
            return missing(self.name, NOT_LOGGED)
        v = float(v)
        smoothed = self._smooth.update(v)
        if self._prev is None or smoothed is None:
            self._prev = smoothed
            return SignalReading(name=self.name, available=True, value=0.0, z=None)

        delta = smoothed - self._prev
        self._prev = smoothed
        z = self._moments.z(delta)
        self._moments.update(delta)
        return SignalReading(name=self.name, available=True, value=delta, z=z)

    def state_dict(self) -> dict[str, Any]:
        return {
            "smooth": self._smooth.state_dict(),
            "moments": self._moments.state_dict(),
            "prev": self._prev,
        }


__all__ = ["LengthDrift", "Truncation"]
