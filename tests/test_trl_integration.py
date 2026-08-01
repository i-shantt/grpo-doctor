"""The TRL bridge, tested without TRL installed.

`check_config` and the key mapping are the parts that break silently, and neither needs a trainer
to exercise. The two configuration errors it guards against both produce *plausible* monitoring
rather than an obvious failure, which is exactly why they are raised at `on_train_begin` instead of
being discovered in a results table.

The live 3-step run against a real `GRPOTrainer` belongs in CI's integration job, not here: the
unit suite installs numpy only, on purpose.
"""

from __future__ import annotations

import warnings

import pytest

from grpo_doctor.integrations.trl import check_config
from grpo_doctor.record import TRL_KEY_MAP, StepRecord


class FakeArgs:
    """Stands in for a TRL GRPOConfig. Only the attributes check_config reads."""

    def __init__(self, **kw):
        self.logging_steps = kw.get("logging_steps", 1)
        self.num_iterations = kw.get("num_iterations", 2)
        self.log_completions = kw.get("log_completions", True)


def test_a_good_config_passes_quietly() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_config(FakeArgs())


def test_logging_steps_other_than_one_is_an_error() -> None:
    """TRL averages metrics over the logging window and clears them, so at the default of 500 the
    monitor silently receives 500-step means and still produces numbers."""
    with pytest.raises(ValueError, match="logging_steps=1"):
        check_config(FakeArgs(logging_steps=500))
    with pytest.raises(ValueError, match="logging_steps=1"):
        check_config(FakeArgs(logging_steps=10))


def test_on_policy_is_a_warning_not_an_error() -> None:
    """mu=1 is a legitimate way to train. It just makes the clipping signals structurally absent,
    and the user has to be told that rather than shown zeros."""
    with pytest.warns(UserWarning, match="structurally absent"):
        check_config(FakeArgs(num_iterations=1))


def test_missing_completion_logs_is_a_warning() -> None:
    with pytest.warns(UserWarning, match="frac_reward_zero_std"):
        check_config(FakeArgs(log_completions=False))


def test_missing_attributes_are_tolerated() -> None:
    """TRL renames things. A monitor that dies on an unfamiliar config is worse than one that
    checks what it can find -- and it must not warn about an attribute that is merely absent, or
    users learn to ignore the warning that matters.
    """

    class Bare:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_config(Bare())


# --- the key mapping -------------------------------------------------------------------------


def test_a_realistic_trl_payload_maps_across() -> None:
    logs = {
        "loss": 0.031,
        "grad_norm": 1.42,
        "learning_rate": 5e-06,
        "reward": 0.6875,
        "reward_std": 0.2951,
        "frac_reward_zero_std": 0.125,
        "entropy": 0.4271,
        "kl": 0.0,
        "clip_ratio/low_mean": 0.0012,
        "clip_ratio/high_mean": 0.0031,
        "clip_ratio/region_mean": 0.0043,
        "completions/mean_length": 214.5,
        "completions/clipped_ratio": 0.0625,
        "epoch": 0.4,
    }
    rec = StepRecord.from_trl_log(logs, step=42)
    assert rec.step == 42 and rec.source == "trl"
    assert rec.reward_mean == pytest.approx(0.6875)
    assert rec.frac_reward_zero_std == pytest.approx(0.125)
    assert rec.clip_region == pytest.approx(0.0043)
    assert rec.completion_clipped_ratio == pytest.approx(0.0625)


def test_unknown_keys_are_ignored_not_rejected() -> None:
    """TRL shipped nine releases in two months; dying on a new log key is the wrong failure mode."""
    rec = StepRecord.from_trl_log({"reward": 0.5, "some_new_2027_metric": 1.0}, step=1)
    assert rec.reward_mean == 0.5


def test_absent_metrics_stay_none_rather_than_zero() -> None:
    """The distinction the whole package rests on: not logged is not the same as measured zero."""
    rec = StepRecord.from_trl_log({"reward": 0.5}, step=1)
    assert rec.clip_low is None and rec.entropy is None
    assert rec.reward_mean == 0.5


def test_step_falls_back_to_the_payload() -> None:
    assert StepRecord.from_trl_log({"reward": 0.1, "global_step": 7}).step == 7


def test_every_mapped_key_names_a_real_field() -> None:
    fields = set(StepRecord.__dataclass_fields__)
    unknown = set(TRL_KEY_MAP.values()) - fields
    assert not unknown, f"TRL_KEY_MAP points at nonexistent fields: {unknown}"


def test_the_map_covers_the_signals_the_panel_needs() -> None:
    """A signal with no route from a TRL log is unreachable in the setting the package targets."""
    for key in (
        "reward",
        "frac_reward_zero_std",
        "entropy",
        "completions/mean_length",
        "completions/clipped_ratio",
        "clip_ratio/region_mean",
        "grad_norm",
    ):
        assert key in TRL_KEY_MAP
