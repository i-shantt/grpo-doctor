"""The three properties that make a reported lead time believable.

These are the tests the plan marks as never-cut. They do not check that the monitor is *good*; they
check that it is not cheating. A monitor that peeks at the future, or that reads the ground truth
it is supposed to predict, will produce excellent numbers and mean nothing at all.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from grpo_doctor.monitor import DEFAULT_WARMUP, Monitor
from grpo_doctor.record import ORACLE_FIELDS, StepRecord
from grpo_doctor.snapshot import Level


def synthetic_run(n: int = 200, seed: int = 0, collapse_at: int | None = 120) -> list[StepRecord]:
    """A trace with a plausible shape. Deterministic, and not from the simulator module.

    After `collapse_at` the reward keeps rising while groups go degenerate and completions start
    truncating -- the reward-hacking geometry, since that is the case where a monitor watching the
    reward alone is structurally blind.
    """
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        hacking = collapse_at is not None and t >= collapse_at
        base = min(0.35 + 0.002 * t, 0.75)
        reward = min(base + (0.01 * (t - collapse_at) if hacking else 0.0), 1.0)
        zero_std = 0.1 + (0.7 * min(1.0, (t - collapse_at) / 40) if hacking else 0.0)
        trunc = 0.02 + (0.4 * min(1.0, (t - collapse_at) / 40) if hacking else 0.0)
        g = rng.random((4, 8)) < reward
        out.append(
            StepRecord(
                step=t,
                reward_mean=float(reward + rng.normal(0, 0.01)),
                reward_std=float(abs(rng.normal(0.3, 0.02))),
                frac_reward_zero_std=float(zero_std),
                entropy=float(max(0.05, 1.2 - 0.004 * t)),
                grad_norm=float(abs(rng.normal(1.0, 0.1))),
                completion_len_mean=float(8.0 + rng.normal(0, 0.05)),
                completion_clipped_ratio=float(trunc),
                group_rewards=g.astype(float),
                heldout_accuracy=float(
                    max(0.0, 0.6 - (0.02 * (t - collapse_at) if hacking else 0.0))
                ),
                source="testbed",
            )
        )
    return out


# --- 1. causality -------------------------------------------------------------------------------


def test_prefix_replay_is_identical_to_streaming() -> None:
    """For every t, a fresh Monitor fed only [0..t] must produce the same snapshot as the streaming
    Monitor produced at t.

    This is what makes forward-peeking mechanically impossible, including the subtle version where
    a signal standardizes against statistics of the whole run.
    """
    records = synthetic_run(n=90)

    m = Monitor()
    streamed = [m.update(r) for r in records]

    for t in (0, 1, DEFAULT_WARMUP, DEFAULT_WARMUP + 1, 60, len(records) - 1):
        fresh = Monitor()
        replayed = None
        for r in records[: t + 1]:
            replayed = fresh.update(r)
        assert replayed == streamed[t], f"prefix replay diverged at step {t}"


def test_two_monitors_on_the_same_stream_agree_everywhere() -> None:
    records = synthetic_run(n=120, seed=3)
    a, b = Monitor(), Monitor()
    for r in records:
        assert a.update(r) == b.update(r)


# --- 2. blindness -------------------------------------------------------------------------------


def test_oracle_field_cannot_influence_the_monitor() -> None:
    """Replay a run twice, once with held-out accuracy intact and once replaced with noise.

    If these ever differ, some signal is reading the ground truth it is meant to predict, and every
    lead time the project reports is circular.
    """
    records = synthetic_run(n=150, seed=1)
    rng = np.random.default_rng(99)
    poisoned = [
        StepRecord(
            **{
                **{f: getattr(r, f) for f in r.__dataclass_fields__},
                "heldout_accuracy": float(rng.random()),
            }
        )
        for r in records
    ]

    clean_m, dirty_m = Monitor(), Monitor()
    for clean, dirty in zip(records, poisoned, strict=True):
        assert clean_m.update(clean) == dirty_m.update(dirty)


def test_visible_clears_every_oracle_field() -> None:
    rec = synthetic_run(n=1)[0]
    assert rec.heldout_accuracy is not None
    vis = rec.visible()
    for field in ORACLE_FIELDS:
        assert getattr(vis, field) is None
    # ...and changes nothing else.
    for field in rec.__dataclass_fields__:
        if field not in ORACLE_FIELDS:
            a, b = getattr(rec, field), getattr(vis, field)
            assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_oracle_fields_is_not_silently_empty() -> None:
    """A refactor that emptied this set would make the blindness test pass vacuously."""
    assert "heldout_accuracy" in ORACLE_FIELDS


# --- 3. constant memory --------------------------------------------------------------------------


def _flat(step: int) -> StepRecord:
    return StepRecord(
        step=step,
        reward_mean=0.5,
        frac_reward_zero_std=0.1,
        completion_len_mean=8.0,
        completion_clipped_ratio=0.02,
    )


def test_state_is_constant_memory() -> None:
    """State size must not grow with the number of steps seen.

    Measured between two large step counts rather than against a small one, because pickle encodes
    a step counter of 10 in one byte and 10,010 in four. That 4-byte difference is integer width,
    not retained history, and asserting exact equality across it would be testing the pickle
    protocol. `test_no_container_in_the_state_grows` covers the case that actually matters.
    """
    m = Monitor()
    for i in range(10_000):
        m.update(_flat(i))
    at_10k = len(pickle.dumps(m.state_dict()))

    for i in range(10_000, 20_000):
        m.update(_flat(i))
    assert len(pickle.dumps(m.state_dict())) == at_10k


def test_no_container_in_the_state_grows() -> None:
    """The property the byte count is a proxy for: nothing in the state accumulates entries."""

    def shape(state: object) -> object:
        if isinstance(state, dict):
            return {k: shape(v) for k, v in state.items()}
        if isinstance(state, (list, tuple, set)):
            return (type(state).__name__, len(state))
        return type(state).__name__

    m = Monitor()
    for i in range(20):
        m.update(_flat(i))
    early = shape(m.state_dict())
    for i in range(20, 5_000):
        m.update(_flat(i))
    assert shape(m.state_dict()) == early


def test_monitor_holds_no_record_history() -> None:
    m = Monitor()
    for r in synthetic_run(n=50):
        m.update(r)
    blob = pickle.dumps(m.state_dict())
    assert len(blob) < 2000, f"state is {len(blob)} bytes; something is accumulating"


# --- behavior that the structural tests do not cover ---------------------------------------------


def test_missing_fields_never_raise() -> None:
    """The premise of the whole package: you get what your trainer happens to log."""
    m = Monitor()
    snap = m.update(StepRecord(step=0))
    assert snap.coverage == 0.0
    assert snap.level is Level.OK
    assert all(not s.available for s in snap.signals)


def test_coverage_reports_what_was_live() -> None:
    m = Monitor()
    snap = m.update(StepRecord(step=0, completion_clipped_ratio=0.1))
    assert snap.coverage == pytest.approx(1 / 3)
    assert snap.by_name("truncation").available
    assert not snap.by_name("starvation").available


def test_no_alarm_during_warmup() -> None:
    """Before there is a past to standardize against, a z-score is an estimate from three samples."""
    m = Monitor(warmup=30)
    for i, r in enumerate(synthetic_run(n=30, collapse_at=5)):
        snap = m.update(r)
        assert snap.warming_up
        assert snap.level is Level.OK, f"alarmed at step {i} during warmup"


def test_a_collapse_eventually_raises_the_level() -> None:
    """Not a performance claim -- just that the plumbing carries a signal end to end."""
    m = Monitor()
    levels = [m.update(r).level for r in synthetic_run(n=200, collapse_at=120)]
    assert max(levels[:120]) < Level.WARN, "fired before anything happened"
    assert max(levels[120:]) >= Level.WARN, "never fired at all"


def test_alerts_name_the_signal_responsible() -> None:
    m = Monitor()
    alerts = [a for r in synthetic_run(n=200) for a in m.update(r).alerts]
    assert alerts
    assert any(a.message.startswith(("starvation", "truncation", "len_drift")) for a in alerts)


# --- flat is not extreme ---------------------------------------------------------------------


def test_a_constant_signal_reports_no_z_rather_than_a_huge_one() -> None:
    """Measured on a real trace: completion length was effectively constant at 7.00 tokens, the
    step-to-step change had a standard deviation around 1e-3, and a numerical-epsilon floor duly
    turned a 0.0019-token wiggle into z=+7.42 and an ALARM at step 34 of a healthy stretch.

    The signal was not anomalous, it was flat. Flat must read as uninformative, never as extreme.
    """
    m = Monitor()
    levels = []
    for i in range(400):
        levels.append(
            m.update(
                StepRecord(
                    step=i,
                    reward_mean=0.5,
                    frac_reward_zero_std=0.2,
                    # Constant to within floating-point noise, as a converged length series is.
                    completion_len_mean=7.0 + 1e-9 * ((i * 37) % 11),
                    completion_clipped_ratio=0.02 + 1e-9 * ((i * 17) % 7),
                )
            ).level
        )
    assert max(levels) is Level.OK, "a flat run must never alarm"


def test_a_real_change_still_fires_once_the_floor_is_cleared() -> None:
    """The floor must not deafen the signal -- a length change large enough to matter still fires."""
    m = Monitor()
    fired = False
    for i in range(400):
        length = 7.0 if i < 200 else 7.0 + 0.25 * (i - 200)
        snap = m.update(
            StepRecord(
                step=i,
                reward_mean=0.5,
                frac_reward_zero_std=0.2,
                completion_len_mean=length,
                completion_clipped_ratio=0.02,
            )
        )
        if i > 210 and snap.level >= Level.WARN:
            fired = True
    assert fired, "a sustained real drift must still be detected"
