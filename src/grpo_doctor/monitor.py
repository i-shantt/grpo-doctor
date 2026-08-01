"""The streaming monitor. One record in, one snapshot out, forever, in constant memory.

Three properties are guaranteed here and each is enforced by a test rather than by care:

**Causality.** `update` is a pure function of the records seen so far, in order. Feeding a fresh
Monitor the prefix [0..t] must produce a snapshot identical to the one the full run produced at t.
This is what makes a reported lead time mean anything; without it a monitor can peek at the future
through any run-level statistic and quietly inflate every number.

**Blindness.** No signal may read a field in `ORACLE_FIELDS`. `update` strips them on the way in,
so a signal cannot read held-out accuracy even if it tries. Enforced by replaying a run twice with
that field replaced by noise and asserting bit-identical output.

**Constant memory.** No history is retained. Verified by pickling `state_dict()` after 10 steps and
after 10,000 and asserting equal size.

The fusion here is deliberately the simplest thing that works -- the maximum standardized deviation
across available signals, thresholded. That is rung R0/R1 of the ladder. The calibrated logistic
and the CUSUM on its log-odds (R2, R3) are fitted from the corpus and loaded as weights; shipping
a hand-tuned fuser before the corpus exists would be picking the answer first.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any

from grpo_doctor.record import StepRecord
from grpo_doctor.signals.base import Signal
from grpo_doctor.signals.length import LengthDrift, Truncation
from grpo_doctor.signals.starvation import Starvation
from grpo_doctor.snapshot import WARMING_UP, Alert, Level, SignalReading, VitalsSnapshot

DEFAULT_WARMUP = 30
"""Steps before any alarm may fire.

Not cosmetic: every signal standardizes against the run's own past, so before there is a past there
is no z-score. Firing during warmup would mean firing on an estimate built from three samples.
"""

THRESHOLDS: dict[Level, float] = {Level.WATCH: 2.0, Level.WARN: 3.0, Level.ALARM: 4.0}


def default_signals() -> list[Signal]:
    return [Starvation(), Truncation(), LengthDrift()]


class Monitor:
    """Watch a GRPO run. Call `update` once per optimizer step, in order."""

    def __init__(
        self,
        signals: Sequence[Signal] | None = None,
        *,
        warmup: int = DEFAULT_WARMUP,
    ) -> None:
        self.signals: list[Signal] = list(signals) if signals is not None else default_signals()
        self.warmup = warmup
        self._seen = 0
        self._last: VitalsSnapshot | None = None

    def update(self, rec: StepRecord) -> VitalsSnapshot:
        # Blindness, enforced at the boundary. A signal never receives the oracle fields at all,
        # so it cannot depend on them by accident or by edit.
        rec = rec.visible()
        self._seen += 1
        warming = self._seen <= self.warmup

        readings: list[SignalReading] = []
        for sig in self.signals:
            reading = sig.update(rec)
            if warming and reading.available:
                # The value is real, the z-score is not yet trustworthy. Report the value and
                # suppress the score rather than dropping the reading, so coverage stays honest.
                reading = SignalReading(
                    name=reading.name,
                    available=True,
                    value=reading.value,
                    z=None,
                    reason=WARMING_UP,
                )
            readings.append(reading)

        scores = [abs(r.z) for r in readings if r.available and r.z is not None]
        score = max(scores) if scores else 0.0
        level = Level.OK
        if not warming:
            for lvl in (Level.ALARM, Level.WARN, Level.WATCH):
                if score >= THRESHOLDS[lvl]:
                    level = lvl
                    break

        alerts: tuple[Alert, ...] = ()
        if level >= Level.WARN:
            worst = max(
                (r for r in readings if r.available and r.z is not None),
                key=lambda r: abs(r.z),  # type: ignore[arg-type]
            )
            alerts = (
                Alert(
                    step=rec.step,
                    level=level,
                    message=f"{worst.name} at z={worst.z:+.2f} (value {worst.value:.4g})",
                ),
            )

        snap = VitalsSnapshot(
            step=rec.step,
            level=level,
            score=score,
            signals=tuple(readings),
            warming_up=warming,
            alerts=alerts,
        )
        self._last = snap
        return snap

    @property
    def last(self) -> VitalsSnapshot | None:
        return self._last

    def state_dict(self) -> dict[str, Any]:
        """Everything the monitor remembers. Must not grow with the number of steps seen."""
        return {
            "seen": self._seen,
            "warmup": self.warmup,
            "signals": {s.name: s.state_dict() for s in self.signals},
        }

    def state_bytes(self) -> int:
        return len(pickle.dumps(self.state_dict()))


__all__ = ["DEFAULT_WARMUP", "THRESHOLDS", "Monitor", "default_signals"]
