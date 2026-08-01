"""S1 -- advantage starvation, disambiguated from mastery.

The reason this signal is a *pair* rather than a number: the fraction of groups with zero reward
variance is bimodally confounded. It is high at the start of training, when every completion fails,
and high again at successful convergence, when every completion passes. A threshold on the raw
fraction fires on both, so as a level it carries almost no information.

What disambiguates them is where the reward sits at the same moment:

    starvation = zero_std_frac * (1 - reward_norm)     all groups agree, and they agree on failure
    mastery    = zero_std_frac * reward_norm           all groups agree, and they agree on success

Only `starvation` is pathological. `mastery` is what a finished run looks like, and it is exposed
as its own reading precisely so the ablation can show it carries no predictive weight -- a clean,
checkable negative rather than a signal quietly dropped.

Both are computed from `frac_reward_zero_std`, TRL's own key, whose test is
`torch.isclose(std, 0)`. That default `atol=1e-8` means a group with std 1e-3 is not counted even
though its advantages are pure amplified noise. Where the richer `group_rewards` array is available
we recompute with a usable epsilon and report the gap, since a monitor that inherits a threshold
this tight would systematically under-report starvation on any real run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from grpo_doctor.record import StepRecord
from grpo_doctor.signals.base import RunningMoments, missing
from grpo_doctor.snapshot import NOT_LOGGED, SignalReading

ZERO_STD_EPS = 1e-3


class Starvation:
    name = "starvation"

    def __init__(self, zero_std_eps: float = ZERO_STD_EPS) -> None:
        self.zero_std_eps = zero_std_eps
        self._moments = RunningMoments()
        self._lo = math.inf
        self._hi = -math.inf
        self.last_mastery: float | None = None
        self.last_acr_gap: float | None = None

    def _zero_std_frac(self, rec: StepRecord) -> float | None:
        """Prefer the per-group array with our epsilon; fall back to TRL's stricter scalar."""
        if rec.group_rewards is not None and rec.group_rewards.size:
            g = np.asarray(rec.group_rewards, dtype=float)
            if g.ndim == 1:
                g = g.reshape(1, -1)
            if g.shape[1] < 2:
                return None
            stds = g.std(axis=1, ddof=1)
            ours = float((stds < self.zero_std_eps).mean())
            if rec.frac_reward_zero_std is not None:
                self.last_acr_gap = ours - float(rec.frac_reward_zero_std)
            return ours
        if rec.frac_reward_zero_std is not None:
            return float(rec.frac_reward_zero_std)
        return None

    def _reward_norm(self, rec: StepRecord) -> float | None:
        """Reward mapped to [0, 1] using the run's own observed range.

        Binary-reward RLVR already lives in [0, 1] and this is the identity. Shaped rewards do not,
        and a fixed assumption would make the starvation/mastery split meaningless for them.
        """
        if rec.reward_mean is None or not math.isfinite(rec.reward_mean):
            return None
        r = float(rec.reward_mean)
        self._lo = min(self._lo, r)
        self._hi = max(self._hi, r)
        span = self._hi - self._lo
        if span < 1e-9:
            # Nothing has moved yet; assume the RLVR-standard [0, 1] rather than inventing a scale.
            return min(max(r, 0.0), 1.0)
        return (r - self._lo) / span

    def update(self, rec: StepRecord) -> SignalReading:
        acr = self._zero_std_frac(rec)
        rnorm = self._reward_norm(rec)
        if acr is None or rnorm is None:
            return missing(self.name, NOT_LOGGED)

        starvation = acr * (1.0 - rnorm)
        self.last_mastery = acr * rnorm
        z = self._moments.z(starvation)
        self._moments.update(starvation)
        return SignalReading(name=self.name, available=True, value=starvation, z=z)

    def state_dict(self) -> dict[str, Any]:
        return {
            "moments": self._moments.state_dict(),
            "lo": self._lo,
            "hi": self._hi,
            "mastery": self.last_mastery,
            "acr_gap": self.last_acr_gap,
        }


__all__ = ["ZERO_STD_EPS", "Starvation"]
