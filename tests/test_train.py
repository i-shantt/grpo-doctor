"""Run-loop invariants: reproducibility, oracle quarantine, and the probe split.

`test_probe_prompts_are_never_sampled_for_training` and `test_visible_record_contains_no_oracle`
are the two that the labeling story rests on. If the probe leaks into training, `t_collapse`
understates every collapse; if an oracle key reaches the monitor, every lead time is circular.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from testbed.core.grpo import GRPOConfig  # noqa: E402
from testbed.core.optim import OptimConfig  # noqa: E402
from testbed.core.train import (  # noqa: E402
    ORACLE_PREFIX,
    RunConfig,
    apply_overrides,
    run,
    strip_oracle,
)
from testbed.tasks.base import LeakLevel, is_probe_prompt  # noqa: E402
from testbed.tasks.sort_digits import SortDigits  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Warm starts are cached to `warmstarts/` relative to the cwd.

    Every test in this module runs in its own temporary directory so the suite neither pollutes the
    repo nor picks up a checkpoint another test wrote -- a cache hit across tests would make a
    reproducibility failure invisible.
    """
    monkeypatch.chdir(tmp_path)


def _cfg(**kw) -> RunConfig:
    """Deliberately tiny: these test control flow and bookkeeping, not learning."""
    base = dict(
        seed=0,
        steps=6,
        n_prompts=2,
        difficulty=4,
        warm_start_steps=2,
        probe_every=3,
        probe_n=32,
        d_model=32,
        n_layer=1,
        n_head=4,
        grpo=GRPOConfig(group_size=4, num_iterations=2),
    )
    base.update(kw)
    return RunConfig(**base)


# --- probe split -------------------------------------------------------------------------------


def test_probe_prompts_are_never_sampled_for_training() -> None:
    task, rng = SortDigits(), np.random.default_rng(0)
    for _ in range(40):
        for p in task.sample(64, 5, rng, "train").problems:
            assert not is_probe_prompt(p.prompt)


def test_probe_split_yields_only_probe_prompts() -> None:
    task, rng = SortDigits(), np.random.default_rng(0)
    for p in task.sample(256, 5, rng, "probe").problems:
        assert is_probe_prompt(p.prompt)


def test_the_split_is_stable_across_processes() -> None:
    """crc32, not `hash()`. Python salts `hash()` per process, which would put the same prompt on
    both sides of the split in different corpus workers."""
    assert is_probe_prompt((1, 2, 3)) == is_probe_prompt((1, 2, 3))
    known = [is_probe_prompt(tuple(range(i, i + 4))) for i in range(8)]
    assert any(known) and not all(known), "split should be non-degenerate on this sample"


def test_split_leaves_the_task_distribution_alone() -> None:
    """Rejection sampling must not distort digit frequencies, or train and probe would differ in
    difficulty and the accuracy gap would be an artifact of the split."""
    task, rng = SortDigits(), np.random.default_rng(1)
    tr = np.concatenate([p.answer for p in task.sample(3000, 4, rng, "train").problems])
    pr = np.concatenate([p.answer for p in task.sample(3000, 4, rng, "probe").problems])
    assert abs(tr.mean() - pr.mean()) < 0.25


# --- oracle quarantine -------------------------------------------------------------------------


def test_visible_record_contains_no_oracle() -> None:
    rec = next(iter(run(SortDigits(), _cfg())))
    assert any(k.startswith(ORACLE_PREFIX) for k in rec), "the run must emit oracle keys at all"
    visible = strip_oracle(rec)
    assert not any(k.startswith(ORACLE_PREFIX) for k in visible)
    assert not any("accuracy" in k for k in visible)


def test_heldout_accuracy_is_present_and_dense() -> None:
    """Dense so traces have no ragged columns, but flagged so the labeler can tell a carried-forward
    value from a freshly measured one."""
    recs = list(run(SortDigits(), _cfg()))
    key = f"{ORACLE_PREFIX}heldout_accuracy"
    assert all(key in r for r in recs)
    fresh = [r[f"{ORACLE_PREFIX}probe_fresh"] for r in recs]
    assert fresh[0] == 1.0 and sum(fresh) == len(recs[::3])
    assert all(0.0 <= r[key] <= 1.0 for r in recs)


# --- reproducibility ---------------------------------------------------------------------------


def test_run_is_bitwise_reproducible() -> None:
    """Without this no corpus run can be re-derived and no trace can be audited after the fact."""
    a = list(run(SortDigits(), _cfg()))
    b = list(run(SortDigits(), _cfg()))
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert ra.keys() == rb.keys()
        for k in ra:
            if k == "warm_start/cached":
                # Cache-hit bookkeeping, not run behavior: the second run legitimately reports 1.0.
                # test_cached_and_fresh_runs_are_identical covers that the weights match regardless.
                continue
            assert (ra[k] == rb[k]) or (np.isnan(ra[k]) and np.isnan(rb[k])), f"{k} diverged"


def test_cached_and_fresh_runs_are_identical() -> None:
    """Caching the warm start must be an optimization, not a change in behavior.

    It holds only because model initialization is the sole consumer of the global torch RNG and
    sampling draws from an explicitly seeded generator. If either of those ever stops being true,
    every cached corpus run would silently diverge from its own manifest and this is the test that
    catches it.
    """
    cfg = _cfg(warm_start_steps=3)
    fresh = list(run(SortDigits(), cfg, cache_dir=None))
    written = list(run(SortDigits(), cfg))  # populates the cache
    reused = list(run(SortDigits(), cfg))  # hits it

    assert written[0]["warm_start/cached"] == 0.0
    assert reused[0]["warm_start/cached"] == 1.0
    for a, b in zip(fresh, reused, strict=True):
        for k in a:
            if k.startswith("warm_start/"):
                continue
            assert (a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])), f"{k} diverged"


def test_cache_key_covers_every_input_that_changes_the_weights() -> None:
    """A key that missed a field would serve one task's checkpoint to another's run."""
    from testbed.core.warmstart import key_for

    base = _cfg()
    digest = key_for(SortDigits(), base).digest()
    for change in (
        {"difficulty": 5},
        {"warm_start_steps": 3},
        {"warm_start_lr": 1e-2},
        {"d_model": 64},
        {"n_layer": 2},
        {"seed": 7},
        {"n_prompts": 4},
    ):
        assert key_for(SortDigits(), _cfg(**change)).digest() != digest, (
            f"{change} did not change the key"
        )


def test_different_seeds_give_different_runs() -> None:
    a = list(run(SortDigits(), _cfg(seed=0)))
    b = list(run(SortDigits(), _cfg(seed=1)))
    assert a[-1]["reward"] != b[-1]["reward"] or a[-1]["entropy"] != b[-1]["entropy"]


def test_every_step_emits_the_same_keys() -> None:
    recs = list(run(SortDigits(), _cfg()))
    # Step 0 additionally carries the warm-start summary; the rest must be identical.
    assert {k for r in recs[1:] for k in r} == set(recs[1].keys())


# --- onset ------------------------------------------------------------------------------------


def test_onset_leaves_the_run_untouched_before_it_fires() -> None:
    """A knob applied at step k must not perturb steps < k, or the corpus confounds the injection
    with everything that preceded it."""
    plain = list(run(SortDigits(), _cfg(steps=6)))
    injected = list(
        run(
            SortDigits(),
            _cfg(steps=6, onset_step=3, onset_overrides={"verifier.leak_level": LeakLevel.FORMAT}),
        )
    )
    for i in range(3):
        assert plain[i]["reward"] == injected[i]["reward"]
        assert plain[i]["entropy"] == injected[i]["entropy"]


def test_onset_actually_changes_the_reward() -> None:
    injected = list(
        run(
            SortDigits(),
            _cfg(steps=6, onset_step=3, onset_overrides={"verifier.leak_level": LeakLevel.FORMAT}),
        )
    )
    # FORMAT accepts anything well-formed, so the proxy reward should jump once it fires.
    assert injected[-1]["reward"] > injected[2]["reward"]


def test_apply_overrides_reaches_nested_fields() -> None:
    cfg = _cfg()
    out = apply_overrides(cfg, {"grpo.num_iterations": 8, "temperature": 0.5})
    assert out.grpo.num_iterations == 8 and out.temperature == 0.5
    assert cfg.grpo.num_iterations == 2, "the original config must not be mutated"


@pytest.mark.parametrize("bad", ["nonexistent", "grpo.nonexistent"])
def test_apply_overrides_rejects_unknown_paths(bad: str) -> None:
    """A silently ignored override would produce a run labeled as a failure family that was never
    actually injected -- a mislabeled corpus row that nothing downstream could detect."""
    with pytest.raises(KeyError):
        apply_overrides(_cfg(), {bad: 1})


# --- mu and clip availability -------------------------------------------------------------------


def test_mu1_reports_clip_metrics_as_unmeasurable() -> None:
    """Finding #2 end to end: at mu=1 the clip columns are NaN, never a healthy-looking 0.0."""
    rec = next(iter(run(SortDigits(), _cfg(grpo=GRPOConfig(group_size=4, num_iterations=1)))))
    for k in ("clip_ratio/region_mean", "importance_ratio/max", "importance_ratio/log_std"):
        assert np.isnan(rec[k]), f"{k} should be NaN at mu=1, got {rec[k]}"


def test_mu2_makes_clipping_measurable() -> None:
    rec = next(iter(run(SortDigits(), _cfg(grpo=GRPOConfig(group_size=4, num_iterations=2)))))
    assert not np.isnan(rec["clip_ratio/region_mean"])
    assert rec["importance_ratio/max"] >= 1.0


def test_gradient_panel_is_present() -> None:
    rec = next(iter(run(SortDigits(), _cfg())))
    for k in ("grad_norm", "grad_norm_postclip", "update_ratio", "learning_rate"):
        assert k in rec
    assert rec["nonfinite_grad"] == 0.0


def test_warm_start_can_be_skipped() -> None:
    recs = list(run(SortDigits(), _cfg(warm_start_steps=0)))
    assert np.isnan(recs[0]["warm_start/final_loss"])


def test_optimizer_is_rebuilt_only_when_its_config_changes() -> None:
    """Silently resetting Adam state at onset would look like a shock the injected knob did not
    cause, and would show up as a spike in update_ratio attributable to nothing."""
    recs = list(
        run(
            SortDigits(),
            _cfg(steps=6, onset_step=3, onset_overrides={"optim.lr": 1e-2}),
        )
    )
    assert recs[4]["learning_rate"] == pytest.approx(1e-2)
    assert recs[0]["learning_rate"] == pytest.approx(OptimConfig().lr)
