"""The statistics that turn measurements into claims.

Two tests here are the ones that would catch a wrong headline rather than a wrong number.
`test_bootstrapping_steps_would_have_been_far_too_narrow` demonstrates the interval inflation that
resampling steps instead of runs produces, and `test_mean_lead_time_would_flatter_a_worse_monitor`
demonstrates why the mean is never reported.
"""

from __future__ import annotations

import numpy as np
import pytest

from grpo_doctor.eval.metrics import (
    RunOutcome,
    best_control,
    calibrate_threshold,
    cluster_bootstrap_ci,
    detection_rate,
    false_alarm_rate,
    far_by_family,
    lead_time_report,
    mcnemar_p,
    median_lead,
    paired_cluster_bootstrap,
)


def pos(run_id: str, t_collapse: int, t_alarm: int | None, family: str = "F5", **kw) -> RunOutcome:
    return RunOutcome(
        run_id=run_id,
        family=family,
        collapsed=True,
        t_collapse=t_collapse,
        t_alarm=t_alarm,
        n_steps=600,
        **kw,
    )


def neg(run_id: str, t_alarm: int | None, family: str = "F0") -> RunOutcome:
    return RunOutcome(
        run_id=run_id,
        family=family,
        collapsed=False,
        t_collapse=None,
        t_alarm=t_alarm,
        n_steps=600,
    )


# --- false alarms ---------------------------------------------------------------------------


def test_far_counts_runs_not_steps() -> None:
    """A monitor that alarms on 300 consecutive steps of one healthy run made one mistake."""
    runs = [neg("a", 100), neg("b", None), neg("c", None), neg("d", None)]
    assert false_alarm_rate(runs) == pytest.approx(0.25)


def test_far_is_broken_out_by_hard_negative_type() -> None:
    """An aggregate hiding 40% false alarms on plateaus while clean runs stay silent is worse than
    no number, because plateaus are where the budget is actually spent."""
    runs = [neg(f"h{i}", 50, "H2") for i in range(4)] + [neg(f"c{i}", None, "F0") for i in range(6)]
    assert false_alarm_rate(runs) == pytest.approx(0.4)
    by = far_by_family(runs)
    assert by["H2"] == pytest.approx(1.0)
    assert by["F0"] == pytest.approx(0.0)


def test_calibration_hits_the_target_rate() -> None:
    rng = np.random.default_rng(0)
    healthy = [rng.normal(0, 1, size=200) for _ in range(100)]
    thr = calibrate_threshold(healthy, target_far=0.05)
    fired = sum(np.nanmax(s) > thr for s in healthy)
    assert fired / len(healthy) <= 0.05


def test_a_higher_target_admits_more_alarms() -> None:
    rng = np.random.default_rng(1)
    healthy = [rng.normal(0, 1, size=200) for _ in range(200)]
    assert calibrate_threshold(healthy, 0.20) < calibrate_threshold(healthy, 0.01)


# --- lead time ------------------------------------------------------------------------------


def test_an_alarm_after_the_collapse_is_not_a_warning() -> None:
    r = pos("late", t_collapse=200, t_alarm=250)
    assert r.fired is True
    assert r.fired_in_time is False
    assert r.lead_time is None


def test_report_gives_both_halves_of_the_answer() -> None:
    """median | fired and P(fired) together. Either alone is misleading."""
    runs = [pos("a", 300, 200), pos("b", 300, 250), pos("c", 300, None), pos("d", 300, None)]
    rep = lead_time_report(runs)
    assert rep.n_positive == 4
    assert rep.n_fired_in_time == 2
    assert rep.p_fired_in_time == pytest.approx(0.5)
    assert rep.median_lead == pytest.approx(75.0)


def test_mean_lead_time_would_flatter_a_worse_monitor() -> None:
    """Why the mean is never reported.

    A cautious monitor catches every collapse with a modest lead. A reckless one fires enormously
    early on the single easiest run and misses the rest. Their *mean lead among runs that fired*
    ranks the reckless one far higher; the pair (P(fired), median) ranks them correctly.
    """
    cautious = [pos(f"c{i}", 300, 250) for i in range(10)]
    reckless = [pos("r0", 300, 10)] + [pos(f"r{i}", 300, None) for i in range(1, 10)]

    mean_of_fired = lambda rs: float(  # noqa: E731
        np.mean([r.lead_time for r in rs if r.lead_time is not None])
    )
    assert mean_of_fired(reckless) > mean_of_fired(cautious)  # the misleading ranking

    assert lead_time_report(cautious).p_fired_in_time == 1.0
    assert lead_time_report(reckless).p_fired_in_time == pytest.approx(0.1)
    assert detection_rate(cautious) > detection_rate(reckless)


def test_lead_is_also_reported_as_a_fraction_of_the_healthy_phase() -> None:
    """50 steps before a collapse at step 100 is a very different warning from 50 steps before one
    at step 500."""
    fast = lead_time_report([pos("f", 100, 50)])
    slow = lead_time_report([pos("s", 500, 450)])
    assert fast.median_lead == slow.median_lead == 50
    assert fast.median_lead_fraction > slow.median_lead_fraction


def test_censored_runs_are_counted_and_reported() -> None:
    runs = [pos("a", 300, 200), pos("b", 560, 500, censored=True)]
    assert lead_time_report(runs).n_censored == 1


# --- the statistics -------------------------------------------------------------------------


def test_bootstrapping_steps_would_have_been_far_too_narrow() -> None:
    """The trap, made concrete.

    Treating each step as an observation inflates the sample size by the run length, and the
    interval shrinks by roughly its square root. At 600 steps that is a factor of ~24 -- enough to
    make almost any comparison look significant.
    """
    runs = [pos(f"r{i}", 300, 300 - (i % 7) * 10) for i in range(40)]
    _, lo, hi = cluster_bootstrap_ci(runs, median_lead, n_boot=800, seed=0)
    run_width = hi - lo

    # The same statistic bootstrapped as if every step were independent.
    leads = np.array([r.lead_time for r in runs if r.lead_time is not None], dtype=float)
    inflated = np.repeat(leads, 600)
    rng = np.random.default_rng(0)
    draws = [np.median(rng.choice(inflated, size=len(inflated), replace=True)) for _ in range(200)]
    step_width = float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5))

    assert step_width < run_width, (
        f"step-level interval {step_width:.1f} should be far narrower than the honest "
        f"run-level {run_width:.1f}"
    )


def test_cluster_bootstrap_brackets_the_point_estimate() -> None:
    runs = [pos(f"r{i}", 300, 300 - (i % 5) * 20) for i in range(30)]
    point, lo, hi = cluster_bootstrap_ci(runs, median_lead, n_boot=500, seed=1)
    assert lo <= point <= hi


def test_paired_bootstrap_is_tighter_than_two_independent_ones() -> None:
    """Paired resampling is what makes a moderate improvement arguable at this sample size."""
    a = [pos(f"r{i}", 300, 200) for i in range(30)]
    b = [pos(f"r{i}", 300, 210) for i in range(30)]
    point, lo, hi = paired_cluster_bootstrap(a, b, median_lead, n_boot=500, seed=2)
    assert point == pytest.approx(10.0)
    assert hi - lo < 20.0


def test_paired_bootstrap_rejects_mismatched_arms() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        paired_cluster_bootstrap([pos("a", 300, 200)], [], median_lead)


def test_mcnemar_uses_only_discordant_pairs() -> None:
    """Runs both detectors agree on carry no information about which is better."""
    a = [True] * 20 + [True, True, False]
    b = [True] * 20 + [False, False, True]
    agree_only = mcnemar_p([True] * 20, [True] * 20)
    assert agree_only == 1.0
    assert mcnemar_p(a, b) == mcnemar_p(a[20:], b[20:])


def test_mcnemar_is_two_sided_and_symmetric() -> None:
    a = [True] * 8 + [False] * 2
    b = [False] * 8 + [True] * 2
    assert mcnemar_p(a, b) == pytest.approx(mcnemar_p(b, a))
    assert mcnemar_p(a, b) < 0.15


def test_mcnemar_on_a_lopsided_split_is_significant() -> None:
    a = [True] * 12 + [False] * 0
    b = [False] * 12
    assert mcnemar_p(a, b) < 0.001


def test_the_bar_is_the_strongest_control_not_their_average() -> None:
    """Beating the mean of the controls is easy and meaningless."""
    controls = {
        "reward_only": [pos("a", 300, None), pos("b", 300, None)],
        "truncation": [pos("a", 300, 100), pos("b", 300, 100)],
        "step_index": [pos("a", 300, None), pos("b", 300, 200)],
    }
    name, score = best_control(controls, detection_rate)
    assert name == "truncation"
    assert score == pytest.approx(1.0)
    average = float(np.mean([detection_rate(v) for v in controls.values()]))
    assert score > average
