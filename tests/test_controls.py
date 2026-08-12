"""What each control must do to be worth putting in the headline table.

A control that is quietly broken is worse than no control, because it makes the monitor look good
against nothing and the failure is invisible in the report.
"""

from __future__ import annotations

import numpy as np
import pytest

from grpo_doctor.eval.controls import (
    CONTROLS,
    ShuffledLabels,
    alarm_step,
    constant_alarm,
    is_degenerate,
    operating_points,
    reward_only,
    step_index_only,
)
from grpo_doctor.eval.metrics import RunOutcome, calibrate_threshold, false_alarm_rate
from grpo_doctor.record import StepRecord


def _run(rewards: list[float]) -> list[StepRecord]:
    return [StepRecord(step=i, reward_mean=r) for i, r in enumerate(rewards)]


def test_reward_only_cannot_see_a_hacking_collapse() -> None:
    """The headline case, stated as a property of the control rather than a hope.

    Under reward hacking the training reward rises monotonically through the collapse. A drawdown
    on reward is then exactly zero at every step, so this control cannot fire at any threshold --
    which is why beating it on hacking runs is the project's minimum claim.
    """
    rising = _run([0.5 + 0.01 * i for i in range(50)])
    assert float(np.max(reward_only(rising))) == 0.0
    assert alarm_step(reward_only(rising), threshold=0.0) is None


def test_reward_only_does_fire_on_an_ordinary_degradation() -> None:
    """Otherwise the control would be trivially beatable and the comparison would prove nothing."""
    falling = _run([0.5] * 20 + [0.5 - 0.02 * i for i in range(30)])
    assert float(np.max(reward_only(falling))) > 0.1
    assert alarm_step(reward_only(falling), threshold=0.1) is not None


def test_step_index_only_is_monotone_and_length_normalized() -> None:
    """Its threshold has to mean 'alarm after this fraction of the run', not 'after step k', or a
    monitor calibrated here would inherit an absolute step count when run on a longer job."""
    short, long = step_index_only(_run([0.0] * 50)), step_index_only(_run([0.0] * 500))
    assert np.all(np.diff(short) > 0)
    assert short[0] == long[0] == 0.0
    assert short[-1] == long[-1] == 1.0


def test_constant_alarm_fires_on_everything_at_its_fixed_threshold() -> None:
    """It anchors the far end of the axis, so a detection rate can be read against what it costs."""
    ctl = CONTROLS["constant_alarm"]
    assert ctl.calibrated is False
    thr = ctl.threshold_for([constant_alarm(_run([0.0] * 10))])
    assert alarm_step(constant_alarm(_run([0.0] * 10)), thr) == 0


def test_calibrating_a_constant_score_would_have_silenced_it() -> None:
    """The reason `calibrated` exists. Solving for 5% FAR against a score that never varies picks
    the threshold that lets nothing through, turning 'always alarms' into 'never alarms'."""
    healthy = [constant_alarm(_run([0.0] * 10)) for _ in range(20)]
    naive = calibrate_threshold(healthy, target_far=0.05)
    assert alarm_step(constant_alarm(_run([0.0] * 10)), naive) is None


def test_step_index_only_is_degenerate_on_equal_length_runs() -> None:
    """The reason its 0.000 detection rate is not evidence against a time confound.

    Every corpus run is 600 steps, so `t/T` is the same array for all of them, every per-run maximum
    is 1.0, and a threshold can only fire on all runs or none. Reporting that as a pass would be
    reading a vacuous number as a result.
    """
    runs = [step_index_only(_run([0.0] * 600)) for _ in range(20)]
    assert operating_points(runs) == 1
    assert is_degenerate(runs)
    thr = calibrate_threshold(runs, target_far=0.05)
    assert all(alarm_step(s, thr) is None for s in runs)


def test_ragged_run_lengths_would_give_step_index_only_something_to_separate() -> None:
    """And it is not degenerate then, so the check tracks the corpus rather than the control."""
    runs = [step_index_only(_run([0.0] * n)) for n in (1, 2, 3, 600)]
    assert operating_points(runs) > 1
    assert not is_degenerate(runs)


def test_reward_only_is_not_degenerate_on_the_kind_of_runs_it_sees() -> None:
    rising = _run([0.5 + 0.01 * i for i in range(50)])
    falling = _run([0.5] * 20 + [0.5 - 0.02 * i for i in range(30)])
    assert not is_degenerate([reward_only(rising), reward_only(falling)])


def test_calibration_holds_the_false_alarm_rate_at_or_under_target() -> None:
    """The operating point the whole study is reported at."""
    rng = np.random.default_rng(0)
    healthy = [np.abs(rng.normal(size=100)) for _ in range(100)]
    thr = calibrate_threshold(healthy, target_far=0.05)
    fired = sum(alarm_step(s, thr) is not None for s in healthy)
    assert fired <= 5


def test_a_run_sitting_exactly_on_the_threshold_does_not_count_as_a_crossing() -> None:
    """`>` not `>=`: with ties the realized false-alarm rate would overshoot its target."""
    assert alarm_step(np.array([1.0, 1.0]), threshold=1.0) is None
    assert alarm_step(np.array([1.0, 1.5]), threshold=1.0) == 1


def test_shuffling_labels_moves_collapse_and_its_step_together() -> None:
    """A permuted `collapsed` with an unpermuted `t_collapse` would be an incoherent label, not a
    null -- runs marked collapsed with no collapse step, and the reverse."""
    runs = [
        RunOutcome(
            f"r{i}",
            "F5",
            collapsed=i < 5,
            t_collapse=100 + i if i < 5 else None,
            t_alarm=50,
            n_steps=600,
        )
        for i in range(20)
    ]
    out = ShuffledLabels(seed=3).apply(runs)
    assert sum(r.collapsed for r in out) == sum(r.collapsed for r in runs)
    for r in out:
        assert (r.t_collapse is not None) == r.collapsed
    assert [r.t_alarm for r in out] == [r.t_alarm for r in runs]


def test_shuffling_is_reproducible_and_actually_permutes() -> None:
    runs = [
        RunOutcome(
            f"r{i}",
            "F5",
            collapsed=i < 5,
            t_collapse=100 + i if i < 5 else None,
            t_alarm=50,
            n_steps=600,
        )
        for i in range(20)
    ]
    a = ShuffledLabels(seed=1).apply(runs)
    assert [r.collapsed for r in a] == [r.collapsed for r in ShuffledLabels(seed=1).apply(runs)]
    assert [r.collapsed for r in a] != [r.collapsed for r in runs]


def test_missing_metrics_are_carried_forward_not_zero_filled() -> None:
    """Zero is a legitimate reward, so filling gaps with it would manufacture a drawdown."""
    recs = [
        StepRecord(step=0, reward_mean=0.8),
        StepRecord(step=1, reward_mean=None),
        StepRecord(step=2, reward_mean=0.8),
    ]
    assert float(np.max(reward_only(recs))) == pytest.approx(0.0)


def test_every_control_handles_an_empty_trace() -> None:
    """The corpus should never contain one, but a control that raises here would take down the
    whole report rather than the one run."""
    for ctl in CONTROLS.values():
        assert ctl.score([]).shape == (0,)
    assert false_alarm_rate([]) != false_alarm_rate([])  # nan
