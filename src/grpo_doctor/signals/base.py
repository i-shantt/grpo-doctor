"""Signal protocol and the O(1) accumulators every signal is built from.

The constraint that shapes this file: **a signal may only standardize against the run's own past.**

That follows directly from the causality guarantee. If a signal were z-scored against corpus-wide
statistics, its value at step 40 would depend on runs that had not happened yet, and the
prefix-replay test would fail -- or worse, would pass while the reported lead times quietly
included information from the future. So there is no corpus-wide scaler anywhere in this codebase,
and every accumulator here is both streaming and bounded in memory.

Bounded memory is not a nicety either. A monitor installed in someone's training loop for 100k
steps must not grow, and `test_state_is_constant_memory` pins that by comparing the pickled state
after 10 steps against after 10,000.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from grpo_doctor.record import StepRecord
from grpo_doctor.snapshot import SignalReading


class RunningMoments:
    """Welford's algorithm. Streaming mean and unbiased variance in constant memory."""

    __slots__ = ("_count", "_m2", "_mean")

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (x - self._mean)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._count - 1))

    def z(self, x: float, floor: float = 1e-8) -> float | None:
        """Standardize against everything seen so far.

        Returns None rather than a large number when the history is degenerate. A constant signal
        that moves once would otherwise produce an unbounded z and a certain false alarm -- which
        is precisely the hard-negative case (a plateau) the false-alarm budget is spent on.
        """
        if self._count < 2:
            return None
        s = self.std
        if s < floor:
            return None
        return (x - self._mean) / s

    def state_dict(self) -> dict[str, float]:
        return {"count": self._count, "mean": self._mean, "m2": self._m2}


class EWMA:
    """Exponentially weighted mean. `halflife` steps to forget half the weight."""

    __slots__ = ("_alpha", "_value")

    def __init__(self, halflife: float = 10.0) -> None:
        if halflife <= 0:
            raise ValueError(f"halflife must be positive, got {halflife}")
        self._alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
        self._value: float | None = None

    def update(self, x: float) -> float | None:
        if not math.isfinite(x):
            return self._value
        self._value = x if self._value is None else self._value + self._alpha * (x - self._value)
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    def state_dict(self) -> dict[str, Any]:
        return {"alpha": self._alpha, "value": self._value}


class RunningMax:
    """Running maximum and drawdown from it.

    Drawdown rather than an absolute level because runs start at different places: a fall from 0.8
    to 0.5 and a fall from 0.5 to 0.2 are the same event, and only one of them crosses any fixed
    threshold you could pick.
    """

    __slots__ = ("_max",)

    def __init__(self) -> None:
        self._max: float | None = None

    def update(self, x: float) -> None:
        if math.isfinite(x) and (self._max is None or x > self._max):
            self._max = x

    def drawdown(self, x: float) -> float | None:
        if self._max is None or not math.isfinite(x):
            return None
        return self._max - x

    @property
    def value(self) -> float | None:
        return self._max

    def state_dict(self) -> dict[str, Any]:
        return {"max": self._max}


@runtime_checkable
class Signal(Protocol):
    """One panel entry.

    `update` is called once per step, in order, exactly once. It must be deterministic given the
    sequence of records seen so far and must not retain per-step history.
    """

    name: str

    def update(self, rec: StepRecord) -> SignalReading: ...

    def state_dict(self) -> dict[str, Any]: ...


def missing(name: str, reason: str) -> SignalReading:
    return SignalReading(name=name, available=False, reason=reason)


__all__ = ["EWMA", "RunningMax", "RunningMoments", "Signal", "missing"]
