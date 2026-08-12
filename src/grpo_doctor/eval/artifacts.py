"""Is a positive label a real collapse, or the probe measuring the wrong thing?

`t_collapse` fires when held-out accuracy falls below its own running peak. That is the right
definition, but it can be satisfied by a policy that did not get worse: if the probe is too small,
or if it is pointed somewhere the policy stopped visiting, ordinary drift on a handful of instances
registers as a drawdown. The `ca_rule` exclusion came out of exactly this -- 29 of its 39 positives
were policies that had genuinely *improved*, on a probe drawing 256 samples from two distinct
problems.

The check that caught it: **a hacking or degrading policy fails `verify_true` on training problems
too.** `verify_true` is the same strict verifier the probe uses, and it is evaluated every step on
the training batch, so a run whose training-distribution true accuracy is *rising* across its own
collapse point has not lost competence. Something moved, but it was the ruler.

**The comparison is only meaningful when the two measurements look at the same thing**, and this is
the part that is easy to get wrong. `evaluate_probe` is deliberately pinned to the run's *base*
configuration -- base difficulty range, temperature fixed at 1.0 -- so that an F1 run is not scored
against a harder exam simply because the injection made training harder. Any knob that moves the
training distribution or the training decode away from those therefore breaks the comparison rather
than revealing anything:

- **`difficulty_range`** -- F2 trains at difficulty (3,3) or (3,4) while the probe stays at 4-6.
  True accuracy on the easier training mix rises while the fixed probe correctly reports lost
  competence at the difficulty that was never abandoned. That divergence *is* F2's mechanism; it is
  what saturation means. Reading it as an artifact would throw away the family's every positive.
- **`temperature`** -- the probe always decodes at 1.0. A run trained at 0.3 or 0.5 measures its
  training accuracy under a sharper sampler, which raises it on any partly-learned task without any
  change in the policy. This one is structural rather than observed: no positive in the corpus so
  far overrides temperature without also overriding `difficulty_range`, so there is no run that
  demonstrates it in isolation. It is excluded on the argument, not on evidence.

So the check returns three outcomes, not two, and `NOT_APPLICABLE` is a real answer rather than a
failure. Collapsing it into "clean" would silently claim the run passed an audit that never ran.

Measured on the 193-run `sort_digits` corpus: of 29 positives, five show rising `train_true` and
all five override `difficulty_range`. **No confound-free positive rises at all** -- the largest is
-0.016, against a margin of 0.187. The result does not depend on where the margin is put.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

CONFOUNDING_OVERRIDES: tuple[str, ...] = ("difficulty_range", "temperature")
"""Injected knobs that move training away from what the probe measures.

Matched on the final dotted segment of an override key, so a future `sampler.temperature` is caught
the same way a bare `temperature` is. Deliberately not a list of *families*: families are named by
convention and can acquire a dose that changes what they touch, whereas the override keys are the
thing that actually breaks the comparison.
"""


class ArtifactVerdict(str, Enum):
    CLEAN = "clean"
    """The check ran and the positive survived it: training-distribution true accuracy did not
    rise across the collapse."""

    ARTIFACT = "artifact"
    """The check ran and the positive failed it: the policy's true accuracy on training problems
    rose while the probe fell, so the probe -- not the policy -- is what moved."""

    NOT_APPLICABLE = "not_applicable"
    """The run's injection moved training off the probe's distribution or decode, so the two
    numbers are not comparable. Not a pass, and not a failure."""


@dataclass(frozen=True)
class ArtifactCheck:
    verdict: ArtifactVerdict
    train_true_rise: float | None
    """Median training-distribution true accuracy after the collapse minus before the peak.
    `None` when the check did not run."""

    margin: float | None
    """How large a rise has to be to count. `None` when the check did not run."""

    confounds: tuple[str, ...]
    """The override keys that made the run unusable, empty when there were none."""

    reason: str

    @property
    def is_artifact(self) -> bool:
        """True only on a positive finding. A run the check could not evaluate is not an artifact."""
        return self.verdict is ArtifactVerdict.ARTIFACT


def confounding_overrides(overrides: Mapping[str, Any]) -> tuple[str, ...]:
    """The subset of an injection's overrides that breaks the train-vs-probe comparison."""
    return tuple(k for k in overrides if k.rsplit(".", 1)[-1] in CONFOUNDING_OVERRIDES)


def rise_margin(p: float, completions_per_step: int, sigmas: float = 3.0) -> float:
    """How big a rise in `train_true` has to be before it is more than sampling noise.

    Derived the same way `LabelConfig.delta` is, from the binomial noise floor of the measurement
    itself, at `n = completions_per_step` rather than at the number of completions in the whole
    comparison window. That is deliberately conservative: consecutive steps share a policy, so the
    completions in a window are nowhere near independent and dividing by all of them would claim a
    precision the measurement does not have. Erring wide here costs at most a missed artifact,
    while erring narrow would discard real positives -- and only one of those two mistakes is
    recoverable by looking at the run.
    """
    p = min(max(p, 0.0), 1.0)
    se = math.sqrt(max(p * (1.0 - p), 1e-12) / max(completions_per_step, 1))
    return sigmas * se


def check_train_true_artifact(
    steps: np.ndarray,
    train_true: np.ndarray,
    peak_step: int,
    t_collapse: int,
    overrides: Mapping[str, Any],
    completions_per_step: int,
    window: int = 20,
    sigmas: float = 3.0,
) -> ArtifactCheck:
    """Audit one positive label.

    Args:
        steps: (T,) optimizer step index.
        train_true: (T,) `verify_true` accuracy on the training batch, every step.
        peak_step: the labeler's peak step -- where held-out accuracy was highest.
        t_collapse: the labeler's collapse step.
        overrides: the run spec's injected overrides, as stored in the trace header.
        completions_per_step: `n_prompts * group_size`, the sample size behind one `train_true`
            reading.
        window: half-open comparison window in steps, on both sides. Medians rather than point
            values, for the same reason `_hack_or_degrade` uses them: `train_true` at 64 completions
            a step is noisy enough that a single-step comparison would be close to a coin flip.
        sigmas: width of the margin, in standard errors.

    Only call this on a run the labeler called positive; a run with no `t_collapse` has no collapse
    to audit.
    """
    confounds = confounding_overrides(overrides)
    if confounds:
        return ArtifactCheck(
            verdict=ArtifactVerdict.NOT_APPLICABLE,
            train_true_rise=None,
            margin=None,
            confounds=confounds,
            reason=(
                f"injection overrides {', '.join(confounds)}, so training is not measured on what "
                "the probe measures and the comparison would report the knob rather than the policy"
            ),
        )

    steps = np.asarray(steps)
    train_true = np.asarray(train_true, dtype=float)
    before = train_true[(steps >= peak_step - window) & (steps <= peak_step)]
    after = train_true[(steps >= t_collapse) & (steps <= t_collapse + window)]
    if not len(before) or not len(after):
        return ArtifactCheck(
            verdict=ArtifactVerdict.NOT_APPLICABLE,
            train_true_rise=None,
            margin=None,
            confounds=(),
            reason=(
                f"no train_true samples within {window} steps of "
                f"{'the peak' if not len(before) else 'the collapse'}"
            ),
        )

    at_peak = float(np.median(before))
    rise = float(np.median(after) - at_peak)
    margin = rise_margin(at_peak, completions_per_step, sigmas)
    if rise > margin:
        return ArtifactCheck(
            verdict=ArtifactVerdict.ARTIFACT,
            train_true_rise=rise,
            margin=margin,
            confounds=(),
            reason=(
                f"true accuracy on training problems rose {rise:+.3f} across the collapse "
                f"(margin {margin:.3f}); a policy that had lost competence would have failed "
                "verify_true on training problems too"
            ),
        )
    return ArtifactCheck(
        verdict=ArtifactVerdict.CLEAN,
        train_true_rise=rise,
        margin=margin,
        confounds=(),
        reason=(
            f"true accuracy on training problems moved {rise:+.3f} across the collapse, "
            f"not a rise beyond the {margin:.3f} margin"
        ),
    )


__all__ = [
    "CONFOUNDING_OVERRIDES",
    "ArtifactCheck",
    "ArtifactVerdict",
    "check_train_true_artifact",
    "confounding_overrides",
    "rise_margin",
]
