"""The artifact check decides which positives survive, so its scoping is the thing to pin.

The check exists because a probe that is too small turns ordinary drift into a labeled collapse --
that is what disqualified `ca_rule`. But applied without scoping it deletes F2 wholesale, because
F2 trains below the probed difficulty *on purpose* and the divergence it flags is the family's
mechanism. `test_a_difficulty_override_makes_the_check_inapplicable` and
`test_a_confounded_run_is_never_called_an_artifact` are the two that keep both of those true at
once.
"""

from __future__ import annotations

import numpy as np
import pytest

from grpo_doctor.eval.artifacts import (
    ArtifactVerdict,
    check_train_true_artifact,
    confounding_overrides,
    rise_margin,
)

G = 64  # completions per step: n_prompts 8 * group_size 8


def _run(train_true_fn, n: int = 300):
    steps = np.arange(n)
    return steps, np.array([train_true_fn(t) for t in steps], dtype=float)


def _check(train_true_fn, overrides=None, peak_step=100, t_collapse=200, **kw):
    steps, tt = _run(train_true_fn)
    return check_train_true_artifact(
        steps, tt, peak_step, t_collapse, overrides or {}, completions_per_step=G, **kw
    )


def test_rising_train_true_on_an_unconfounded_run_is_an_artifact():
    # The ca_rule signature: the probe fell, but the policy got better at the very problems it was
    # training on, which a hacking or degrading policy cannot do.
    res = _check(lambda t: 0.30 + 0.002 * t, overrides={"verifier.leak_level": "structure"})
    assert res.verdict is ArtifactVerdict.ARTIFACT
    assert res.is_artifact
    assert res.train_true_rise is not None and res.train_true_rise > 0


def test_falling_train_true_is_a_real_collapse():
    res = _check(lambda t: 0.60 - 0.002 * t, overrides={"verifier.leak_level": "structure"})
    assert res.verdict is ArtifactVerdict.CLEAN
    assert not res.is_artifact


def test_flat_train_true_is_a_real_collapse():
    res = _check(lambda t: 0.42, overrides={"grpo.num_iterations": 8})
    assert res.verdict is ArtifactVerdict.CLEAN


def test_a_rise_inside_the_noise_margin_does_not_condemn_a_positive():
    # Half a margin of drift is what 64 completions a step produces on its own.
    half = rise_margin(0.5, G) / 2
    res = _check(lambda t: 0.5 + (half if t >= 150 else 0.0))
    assert res.verdict is ArtifactVerdict.CLEAN
    assert res.train_true_rise is not None
    assert 0.0 < res.train_true_rise <= res.margin  # type: ignore[operator]


@pytest.mark.parametrize("key", ["difficulty_range", "temperature"])
def test_a_difficulty_or_temperature_override_makes_the_check_inapplicable(key):
    # The probe is pinned to base difficulty and decodes at 1.0. A run that moved either is not
    # measuring its training accuracy on the same thing, so the comparison reports the knob.
    res = _check(lambda t: 0.30 + 0.002 * t, overrides={key: (3, 3)})
    assert res.verdict is ArtifactVerdict.NOT_APPLICABLE
    assert res.confounds == (key,)
    assert res.train_true_rise is None and res.margin is None


def test_a_confounded_run_is_never_called_an_artifact():
    # F2's actual shape: training true accuracy climbs steeply on the easier mix while the fixed
    # probe correctly reports lost competence at the difficulty training abandoned. All six F2
    # positives in the sort_digits corpus land here, and none of them is a defect.
    res = _check(lambda t: 0.20 + 0.004 * t, overrides={"difficulty_range": (3, 3)})
    assert res.verdict is not ArtifactVerdict.ARTIFACT
    assert not res.is_artifact


def test_not_applicable_is_distinguishable_from_clean():
    # Folding the two together would claim a run passed an audit that never ran on it.
    confounded = _check(lambda t: 0.42, overrides={"temperature": 0.5})
    clean = _check(lambda t: 0.42, overrides={"temperature_schedule": 0.5})
    assert confounded.verdict is ArtifactVerdict.NOT_APPLICABLE
    assert clean.verdict is ArtifactVerdict.CLEAN


def test_confounds_are_matched_on_the_final_dotted_segment():
    assert confounding_overrides({"difficulty_range": (3, 3)}) == ("difficulty_range",)
    assert confounding_overrides({"sampler.temperature": 0.5}) == ("sampler.temperature",)
    assert confounding_overrides({"grpo.num_iterations": 8, "verifier.leak_p": 0.7}) == ()
    # A key that merely contains the name is not a confound.
    assert confounding_overrides({"difficulty_range_note": "x", "max_temperature_cap": 1}) == ()


def test_margin_follows_the_binomial_noise_floor_of_the_measurement():
    # Same derivation as LabelConfig.delta, so it moves correctly if the batch size changes rather
    # than being a number someone liked.
    assert rise_margin(0.5, 64) == pytest.approx(3.0 * (0.25 / 64) ** 0.5)
    assert rise_margin(0.5, 256) < rise_margin(0.5, 64)
    assert rise_margin(0.02, 64) < rise_margin(0.5, 64)
    assert rise_margin(0.0, 64) > 0.0  # never zero, or any drift at all would condemn a run


def test_an_empty_comparison_window_is_not_applicable_rather_than_clean():
    steps = np.arange(50)
    tt = np.full(50, 0.4)
    res = check_train_true_artifact(steps, tt, 10, 900, {}, completions_per_step=G)
    assert res.verdict is ArtifactVerdict.NOT_APPLICABLE
    assert res.confounds == ()
