"""What the monitor hands back after every step.

`coverage` is deliberately prominent. A monitor reporting OK on two of nine signals is saying
something much weaker than one reporting OK on all nine, and a panel that hides the difference
invites exactly the misreading the project is about -- treating an absent signal as a healthy one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Level(IntEnum):
    OK = 0
    WATCH = 1
    WARN = 2
    ALARM = 3


class Unavailable(str):
    """Reasons a signal can be missing, as constants so they are greppable and testable."""


NOT_LOGGED = "not_logged"
"""The trainer never reported the underlying quantity."""

NOT_MEASURABLE = "not_measurable"
"""Reported, but structurally undefined in this configuration -- e.g. clip ratios at
num_iterations=1, where the ratio is 1 by construction and 0.0 means "cannot happen", not "did not
happen"."""

WARMING_UP = "warming_up"
"""Fewer than `warmup` steps seen, so there is no in-run baseline to standardize against yet."""


@dataclass(frozen=True)
class SignalReading:
    name: str
    available: bool
    value: float | None = None
    z: float | None = None
    """Standardized against **this run's own past only**. There is no corpus-wide scaler anywhere
    in this codebase; one would leak future information into every step and silently break the
    causality guarantee."""
    reason: str = ""


@dataclass(frozen=True)
class Alert:
    step: int
    level: Level
    message: str


@dataclass(frozen=True)
class VitalsSnapshot:
    step: int
    level: Level
    score: float
    signals: tuple[SignalReading, ...]
    warming_up: bool
    alerts: tuple[Alert, ...] = ()

    @property
    def coverage(self) -> float:
        """Fraction of the panel that was live this step."""
        if not self.signals:
            return 0.0
        return sum(s.available for s in self.signals) / len(self.signals)

    def by_name(self, name: str) -> SignalReading | None:
        return next((s for s in self.signals if s.name == name), None)


__all__ = [
    "NOT_LOGGED",
    "NOT_MEASURABLE",
    "WARMING_UP",
    "Alert",
    "Level",
    "SignalReading",
    "Unavailable",
    "VitalsSnapshot",
]
