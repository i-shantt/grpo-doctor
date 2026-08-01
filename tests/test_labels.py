"""The labeler decides what "collapsed" means, so its failure modes are the project's failure modes.

Two tests here matter more than the rest. `test_a_knob_that_produced_no_collapse_is_labeled_healthy`
is the corpus's central honesty property: injecting a pathological setting is not the same as the
run going wrong, and a dataset that conflates them is testing whether the detector can read the
manifest. `test_hacking_is_not_mistaken_for_health` covers the case a reward-based label gets
exactly backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from grpo_doctor.eval.labels import (
    LabelConfig,
    RunLabel,
    centered_median,
    label_run,
)

K = 10


def _trace(acc_fn, reward_fn, n: int = 300, zero_std: float = 0.2, seed: int = 0):
    """Build (steps, accuracy, probe_fresh, reward, frac_zero_std) with probes every K steps."""
    rng = np.random.default_rng(seed)
    steps = np.arange(n)
    fresh = (steps % K == 0).astype(float)
    acc, reward, last = [], [], None
    for t in steps:
        if fresh[t]:
            last = float(np.clip(acc_fn(t) + rng.normal(0, 0.01), 0.0, 1.0))
        acc.append(last)
        reward.append(float(np.clip(reward_fn(t) + rng.normal(0, 0.01), 0.0, 1.5)))
    return (
        steps,
        np.array(acc),
        fresh,
        np.array(reward),
        np.full(n, zero_std),
    )


# --- the central honesty property ----------------------------------------------------------------


def test_a_knob_that_produced_no_collapse_is_labeled_healthy() -> None:
    """A run where a pathological setting was injected but accuracy never fell is a NEGATIVE.

    This is the property that separates this corpus from one that merely re-derives its own
    manifest. The label is a function of what happened, never of what was configured.
    """
    tr = _trace(lambda t: 0.40 + 0.0008 * t, lambda t: 0.45 + 0.0008 * t)
    res = label_run(*tr)
    assert res.label is RunLabel.HEALTHY
    assert res.t_collapse is None


def test_hacking_is_not_mistaken_for_health() -> None:
    """Reward rising, accuracy falling. A reward-based label would call this a triumph."""
    tr = _trace(
        acc_fn=lambda t: 0.60 if t < 120 else max(0.0, 0.60 - 0.006 * (t - 120)),
        reward_fn=lambda t: min(1.0, 0.55 + 0.004 * t),
    )
    res = label_run(*tr)
    assert res.label is RunLabel.HACK
    assert res.t_collapse is not None and 120 <= res.t_collapse <= 150


def test_degradation_is_distinguished_from_hacking() -> None:
    """Same accuracy trajectory, reward falling with it. Different prediction problem, different
    label -- one is visible in W&B and the other is not."""
    tr = _trace(
        acc_fn=lambda t: 0.60 if t < 120 else max(0.0, 0.60 - 0.006 * (t - 120)),
        reward_fn=lambda t: 0.65 if t < 120 else max(0.0, 0.65 - 0.006 * (t - 120)),
    )
    res = label_run(*tr)
    assert res.label is RunLabel.DEGRADE
    assert res.t_collapse is not None


def test_a_run_that_never_learned_is_a_stall_not_a_collapse() -> None:
    """No peak means nothing to fall from, so a drawdown test on it is meaningless."""
    tr = _trace(lambda t: 0.02, lambda t: 0.03, zero_std=0.97)
    res = label_run(*tr)
    assert res.label is RunLabel.STALL
    assert res.t_collapse is None


def test_a_flat_run_without_degenerate_groups_is_not_a_stall() -> None:
    """STALL requires both conditions. A run that is merely slow is healthy, and mislabeling it
    would put a hard negative in the positive class."""
    tr = _trace(lambda t: 0.30, lambda t: 0.35, zero_std=0.10)
    assert label_run(*tr).label is RunLabel.HEALTHY


# --- the persistence window ----------------------------------------------------------------------


def test_a_transient_dip_is_not_a_collapse() -> None:
    """Measured from a real healthy run: accuracy fell 0.641 -> 0.367 and recovered within 60 steps.

    Without the persistence requirement that dip is a false positive in the ground truth itself,
    which would poison both the labels and the false-alarm rate computed against them.
    """

    def acc(t: float) -> float:
        return 0.37 if 230 <= t <= 270 else 0.63

    tr = _trace(acc, lambda t: 0.70, n=400)
    res = label_run(*tr)
    assert res.label is RunLabel.HEALTHY, f"labeled {res.label} because {res.reason}"


def test_a_dip_that_never_recovers_is_a_collapse() -> None:
    def acc(t: float) -> float:
        return 0.37 if t >= 230 else 0.63

    tr = _trace(acc, lambda t: 0.70, n=400)
    res = label_run(*tr)
    assert res.label is not RunLabel.HEALTHY
    assert res.t_collapse is not None and 225 <= res.t_collapse <= 245


def test_persistence_window_is_respected_exactly() -> None:
    """A drop holding for H-K steps must not count; holding for H must."""
    cfg = LabelConfig(persistence=50)

    def make(hold: int):
        return _trace(lambda t: 0.37 if 200 <= t < 200 + hold else 0.63, lambda t: 0.7, n=400)

    assert label_run(*make(30), cfg=cfg).label is RunLabel.HEALTHY
    assert label_run(*make(200), cfg=cfg).t_collapse is not None


# --- right censoring ------------------------------------------------------------------------------


def test_a_collapse_at_the_very_end_is_flagged_censored() -> None:
    """The persistence window extends past the run, so persistence cannot be confirmed.

    Reported rather than silently resolved either way: dropping these biases the corpus toward fast
    collapses, and counting them inflates the positive rate.
    """
    tr = _trace(lambda t: 0.60 if t < 270 else 0.20, lambda t: 0.70, n=300)
    res = label_run(*tr)
    assert res.t_collapse is not None
    assert res.censored is True


def test_a_collapse_with_room_to_spare_is_not_censored() -> None:
    tr = _trace(lambda t: 0.60 if t < 120 else 0.20, lambda t: 0.70, n=300)
    res = label_run(*tr)
    assert res.t_collapse is not None and res.censored is False


# --- the threshold --------------------------------------------------------------------------------


def test_delta_is_three_standard_errors_of_the_probe() -> None:
    cfg = LabelConfig(probe_n=256, delta_sigmas=3.0)
    assert cfg.delta(0.5) == pytest.approx(3 * (0.25 / 256) ** 0.5, rel=1e-9)
    assert cfg.delta(0.5) == pytest.approx(0.09375, abs=1e-5)


def test_delta_shrinks_as_the_probe_grows() -> None:
    """The threshold is a property of the measurement, so a bigger probe must sharpen it."""
    small = LabelConfig(probe_n=64).delta(0.5)
    large = LabelConfig(probe_n=1024).delta(0.5)
    assert large == pytest.approx(small / 4, rel=1e-9)


def test_a_drop_smaller_than_the_noise_floor_is_not_a_collapse() -> None:
    """0.05 at N=256 is inside the evaluation's own error bar."""
    tr = _trace(lambda t: 0.60 if t < 150 else 0.56, lambda t: 0.70, n=400)
    assert label_run(*tr).label is RunLabel.HEALTHY


def test_drawdown_is_measured_from_the_running_peak_not_an_absolute_level() -> None:
    """Two runs, same 0.3 fall, different starting points. Both must be collapses."""
    high = _trace(lambda t: 0.80 if t < 150 else 0.50, lambda t: 0.7, n=400)
    low = _trace(lambda t: 0.45 if t < 150 else 0.15, lambda t: 0.7, n=400)
    assert label_run(*high).t_collapse is not None
    assert label_run(*low).t_collapse is not None


# --- smoothing ------------------------------------------------------------------------------------


def test_centered_median_shrinks_at_the_edges_rather_than_padding() -> None:
    """Padding would invent data exactly where the peak and the censoring decision are made."""
    v = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    out = centered_median(v, 3)
    assert out[0] == pytest.approx(np.median(v[:2]))
    assert out[-1] == pytest.approx(np.median(v[-2:]))
    assert out[2] == pytest.approx(np.median(v[1:4]))
    assert len(out) == len(v)


def test_width_one_is_the_identity() -> None:
    v = np.array([1.0, 5.0, 2.0])
    assert np.array_equal(centered_median(v, 1), v)


def test_single_probe_outlier_does_not_move_the_label() -> None:
    """One anomalous probe should not create a collapse, which is what the median buys."""
    acc = np.full(300, 0.6)
    steps = np.arange(300)
    fresh = (steps % K == 0).astype(float)
    acc[150] = 0.05  # a single catastrophic probe
    res = label_run(steps, acc, fresh, np.full(300, 0.7))
    assert res.label is RunLabel.HEALTHY


# --- input handling ---------------------------------------------------------------------------------


def test_carried_forward_probes_are_ignored() -> None:
    """Only freshly measured probes count. Carried-forward values repeat, and a repeated value
    inside the persistence window would look like confirmed persistence that was never measured."""
    steps = np.arange(300)
    fresh = (steps % K == 0).astype(float)
    acc = np.where(steps < 150, 0.6, 0.2).astype(float)
    with_carry = label_run(steps, acc, fresh, np.full(300, 0.7))
    only_fresh = label_run(steps[::K], acc[::K], np.ones(30), np.full(30, 0.7))
    assert with_carry.t_collapse == only_fresh.t_collapse


def test_too_few_probes_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="two fresh probes"):
        label_run(np.arange(5), np.full(5, 0.5), np.array([1, 0, 0, 0, 0]), np.full(5, 0.5))


def test_probe_interval_travels_with_the_result() -> None:
    """t_collapse is only resolved to K, so K must be reportable next to every lead time."""
    tr = _trace(lambda t: 0.6, lambda t: 0.7)
    assert label_run(*tr, cfg=LabelConfig(probe_every=25)).probe_interval == 25
