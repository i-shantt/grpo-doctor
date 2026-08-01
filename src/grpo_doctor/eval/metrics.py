"""Lead time, false-alarm rate, and the statistics that make them defensible.

Four decisions here are the difference between a number and a claim.

**The unit of analysis is the run, not the step.** A corpus of 660 runs at 600 steps has 396,000
step-level rows and about 660 independent observations. Bootstrapping over steps would produce
confidence intervals roughly sqrt(600) ~ 24x too narrow, and every comparison would come out
significant. Every interval here resamples *whole runs* with replacement.

**Lead time is reported as a median with an explicit fired-in-time rate, never as a mean.** Runs
that never alarm are right-censored: their lead time is not zero and it is not missing, it is
"greater than the observation window". A mean over the runs that did fire silently conditions on
success and flatters the monitor exactly where it is weakest. `median | fired` and `P(fired)`
together are the honest pair, and neither alone means anything.

**The operating point is a fixed false-alarm rate, not an AUC.** ROC-AUC rewards a monitor that
fires at step 1 on everything, because it is computed over all thresholds including useless ones.
The threshold is instead set so 5% of held-out *healthy* runs produce at least one alarm, and lead
time is reported at that threshold.

**Comparison is against the strongest control, never the average.** Beating the mean of
{reward-only, truncation-alone, step-index-only} is easy and meaningless; beating whichever of them
happens to be best is the actual bar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunOutcome:
    """One run's contribution to every metric here."""

    run_id: str
    family: str
    collapsed: bool
    """From `label_run`, never from the manifest."""

    t_collapse: int | None
    t_alarm: int | None
    """First step at which the detector's score crossed the threshold, or None."""

    n_steps: int
    censored: bool = False
    """The label was censored: a drawdown was seen but the persistence window ran past the trace."""

    @property
    def fired(self) -> bool:
        return self.t_alarm is not None

    @property
    def fired_in_time(self) -> bool:
        """Alarmed strictly before the collapse. An alarm *after* t_collapse is not a warning."""
        if self.t_alarm is None or self.t_collapse is None:
            return False
        return self.t_alarm < self.t_collapse

    @property
    def lead_time(self) -> int | None:
        if not self.fired_in_time:
            return None
        assert self.t_collapse is not None and self.t_alarm is not None
        return self.t_collapse - self.t_alarm


def false_alarm_rate(negatives: Sequence[RunOutcome]) -> float:
    """Fraction of non-collapsing runs that raised at least one alarm.

    Per run, not per step: a monitor that alarms on 300 consecutive steps of one healthy run has
    made one mistake, not 300, and a per-step rate would make that run look like a catastrophe
    while a single spurious alarm in each of 300 runs looked identical.
    """
    if not negatives:
        return float("nan")
    return sum(r.fired for r in negatives) / len(negatives)


def far_by_family(negatives: Sequence[RunOutcome]) -> dict[str, float]:
    """FAR broken out by hard-negative type.

    An aggregate that hides 40% false alarms on plateaus while clean runs stay silent is worse than
    no number at all, since plateaus are where the budget is actually spent.
    """
    groups: dict[str, list[RunOutcome]] = {}
    for r in negatives:
        groups.setdefault(r.family, []).append(r)
    return {k: false_alarm_rate(v) for k, v in sorted(groups.items())}


def calibrate_threshold(healthy_scores: Sequence[np.ndarray], target_far: float = 0.05) -> float:
    """Smallest threshold at which at most `target_far` of healthy runs ever cross.

    Args:
        healthy_scores: one array of per-step scores per held-out healthy run.

    Calibrated on *held-out* healthy runs. Fitting the threshold on the same runs used to report
    the false-alarm rate would make that rate exactly the target by construction.
    """
    if not len(healthy_scores):
        raise ValueError("need at least one healthy run to calibrate against")
    peaks = np.array([np.nanmax(s) if len(s) else -np.inf for s in healthy_scores])
    # The threshold must exceed the (1 - target) quantile of per-run maxima. Using the peak per run
    # is what makes this a per-run rate rather than a per-step one.
    k = int(np.ceil((1.0 - target_far) * len(peaks)))
    ordered = np.sort(peaks)
    idx = min(k, len(ordered)) - 1
    return float(np.nextafter(ordered[idx], np.inf))


@dataclass(frozen=True)
class LeadTimeReport:
    n_positive: int
    n_fired_in_time: int
    p_fired_in_time: float
    median_lead: float
    iqr_lead: tuple[float, float]
    median_lead_fraction: float
    """Lead time as a fraction of the run's healthy phase, so fast and slow collapses compare."""

    n_censored: int

    def __str__(self) -> str:
        lo, hi = self.iqr_lead
        return (
            f"fired in time on {self.n_fired_in_time}/{self.n_positive} "
            f"({self.p_fired_in_time:.0%}); median lead {self.median_lead:.0f} steps "
            f"[IQR {lo:.0f}-{hi:.0f}], {self.median_lead_fraction:.0%} of the healthy phase"
        )


def lead_time_report(positives: Sequence[RunOutcome]) -> LeadTimeReport:
    """Both halves of the honest answer: how often it fires, and how early when it does."""
    leads = [r.lead_time for r in positives if r.lead_time is not None]
    fractions = [
        r.lead_time / r.t_collapse for r in positives if r.lead_time is not None and r.t_collapse
    ]
    if leads:
        q1, med, q3 = np.percentile(leads, [25, 50, 75])
    else:
        q1 = med = q3 = float("nan")
    return LeadTimeReport(
        n_positive=len(positives),
        n_fired_in_time=len(leads),
        p_fired_in_time=len(leads) / len(positives) if positives else float("nan"),
        median_lead=float(med),
        iqr_lead=(float(q1), float(q3)),
        median_lead_fraction=float(np.median(fractions)) if fractions else float("nan"),
        n_censored=sum(r.censored for r in positives),
    )


def cluster_bootstrap_ci(
    runs: Sequence[RunOutcome],
    statistic: Callable[[Sequence[RunOutcome]], float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample whole runs with replacement. Returns (point estimate, lo, hi).

    The cluster is the run because steps within a run are massively correlated -- consecutive
    signal values differ by an EWMA increment. Resampling steps would treat 600 nearly identical
    observations as 600 independent ones.
    """
    rng = np.random.default_rng(seed)
    point = statistic(runs)
    if not runs:
        return point, float("nan"), float("nan")
    idx = np.arange(len(runs))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        draws[b] = statistic([runs[i] for i in pick])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_cluster_bootstrap(
    a: Sequence[RunOutcome],
    b: Sequence[RunOutcome],
    statistic: Callable[[Sequence[RunOutcome]], float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Difference in a statistic between two detectors on the *same* runs.

    Paired: each bootstrap draw resamples run indices once and applies them to both arms, so the
    interval is on the difference rather than on two independent estimates. Unpaired intervals
    would be far wider and would hide exactly the moderate effects worth arguing about.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    rng = np.random.default_rng(seed)
    point = statistic(a) - statistic(b)
    idx = np.arange(len(a))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        draws[i] = statistic([a[j] for j in pick]) - statistic([b[j] for j in pick])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mcnemar_p(a_hit: Sequence[bool], b_hit: Sequence[bool]) -> float:
    """Exact two-sided McNemar test on paired detections.

    The right test for "did A catch runs B missed": only the discordant pairs carry information,
    and runs both detectors agree on say nothing about which is better. Exact binomial rather than
    the chi-square approximation because the discordant count is usually small -- with 28 positives
    a handful of disagreements is typical, and the approximation is unreliable there.
    """
    a = np.asarray(a_hit, dtype=bool)
    b = np.asarray(b_hit, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return 1.0

    # Exact binomial tail, two-sided, under p=0.5.
    from math import comb

    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def best_control(
    controls: dict[str, Sequence[RunOutcome]], statistic: Callable[[Sequence[RunOutcome]], float]
) -> tuple[str, float]:
    """The strongest control, which is the bar. Never the average of them."""
    scored = {name: statistic(runs) for name, runs in controls.items()}
    name = max(scored, key=lambda k: scored[k])
    return name, scored[name]


def detection_rate(runs: Sequence[RunOutcome]) -> float:
    """Fraction of collapsing runs alarmed before collapse. The statistic most reports use."""
    if not runs:
        return float("nan")
    return sum(r.fired_in_time for r in runs) / len(runs)


def median_lead(runs: Sequence[RunOutcome]) -> float:
    leads = [r.lead_time for r in runs if r.lead_time is not None]
    return float(np.median(leads)) if leads else float("nan")


__all__ = [
    "LeadTimeReport",
    "RunOutcome",
    "best_control",
    "calibrate_threshold",
    "cluster_bootstrap_ci",
    "detection_rate",
    "false_alarm_rate",
    "far_by_family",
    "lead_time_report",
    "mcnemar_p",
    "median_lead",
    "paired_cluster_bootstrap",
]
