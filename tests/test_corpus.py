"""The grid and the runner, checked before an overnight is spent on them.

The failures worth catching here are the ones that produce a *plausible* corpus: every run
executes, every trace parses, and the numbers are quietly meaningless. A cell whose override never
applied, seeds inside a cell sharing an onset step, or healthy runs shorter than failing ones are
all invisible in the output and fatal to the conclusions.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from testbed.corpus.manifest import (  # noqa: E402
    PROFILE_BY_TASK,
    PROFILES,
    TASKS,
    RunSpec,
    build_config,
    build_task,
    make_grid,
    read_manifest,
    write_manifest,
)
from testbed.corpus.runner import (  # noqa: E402
    execute,
    read_trace,
    run_grid,
    summarize,
    trace_path,
)
from testbed.inject.failures import ALL_SPECS  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _tiny(**kw) -> RunSpec:
    base = dict(
        run_id="t0",
        task="sort_digits",
        family="F0",
        dose="none",
        seed=0,
        steps=4,
        difficulty=4,
        warm_start_steps=2,
        onset_step=None,
        overrides={},
        n_prompts=2,
        group_size=4,
        probe_every=2,
        probe_n=16,
    )
    base.update(kw)
    return RunSpec(**base)


# --- the grid ------------------------------------------------------------------------------------


def test_every_profile_names_a_real_task() -> None:
    for p in PROFILES:
        assert p.task in TASKS
    assert set(PROFILE_BY_TASK) == set(TASKS), "every task needs a measured warm-start budget"


def test_measured_accuracy_is_inside_the_collapsible_band() -> None:
    """Below the band every run stalls; above it there is no headroom to fall from.

    These are recorded measurements, so a change that moves one out of band shows up as a diff
    rather than as a corpus full of unusable runs.
    """
    for p in PROFILES:
        assert 0.20 <= p.measured_accuracy <= 0.60, f"{p.task} starts at {p.measured_accuracy}"


def test_run_ids_are_unique() -> None:
    ids = [s.run_id for s in make_grid()]
    assert len(ids) == len(set(ids))


def test_every_cell_is_present_for_every_task() -> None:
    grid = make_grid()
    cells = {(s.task, s.family, s.dose) for s in grid}
    assert len(cells) == len(TASKS) * len(ALL_SPECS)


def test_seeds_within_a_cell_do_not_share_an_onset() -> None:
    """Otherwise every seed in a cell collapses at the same step, and the effective sample size for
    anything time-related is the number of cells rather than the number of runs."""
    grid = [s for s in make_grid() if s.onset_step is not None]
    by_cell: dict[tuple[str, str, str], list[int]] = {}
    for s in grid:
        by_cell.setdefault((s.task, s.family, s.dose), []).append(s.onset_step)
    shared = [k for k, v in by_cell.items() if len(set(v)) == 1 and len(v) > 1]
    assert not shared, f"cells with a single shared onset: {shared[:3]}"


def test_negative_classes_are_densely_sampled() -> None:
    """FAR is reported broken out by hard-negative type, and each breakdown needs its own sample.

    Stated per *cell* rather than per family: F1 has 60 runs only because it has three doses, which
    is three separate conditions of 5 seeds each. What matters is how many seeds sit behind a
    single false-alarm-rate number, and at 5 seeds a per-type FAR moves in 20-point steps per task
    -- uselessly coarse against a 5% operating point.
    """
    grid = make_grid()
    per_cell = Counter((s.task, s.family, s.dose) for s in grid)
    healthy_seeds = min(v for (_, f, _), v in per_cell.items() if f == "F0")
    failure_seeds = max(v for (_, f, _), v in per_cell.items() if f.startswith("F") and f != "F0")
    assert healthy_seeds >= 3 * failure_seeds

    for h in ("H2", "H3", "H4", "H5"):
        assert min(v for (_, f, _), v in per_cell.items() if f == h) >= 10

    negatives = sum(1 for s in grid if s.family in {"F0", "H2", "H3", "H4", "H5"})
    assert negatives / len(grid) >= 0.30, "too few guaranteed negatives to calibrate a 5% FAR"


def test_all_runs_are_the_same_length() -> None:
    """Shorter healthy runs would have fewer chances to raise a false alarm, which makes the
    false-alarm rate optimistic for free."""
    assert len({s.steps for s in make_grid()}) == 1


def test_grid_is_reproducible_from_the_onset_seed() -> None:
    a, b = make_grid(), make_grid()
    assert [s.to_json() for s in a] == [s.to_json() for s in b]
    c = make_grid(onset_seed=999)
    assert [s.onset_step for s in a] != [s.onset_step for s in c]


def test_manifest_round_trips(tmp_path) -> None:
    grid = make_grid(tasks=("sort_digits",))
    path = tmp_path / "m.jsonl"
    write_manifest(str(path), grid)
    assert [s.to_json() for s in read_manifest(str(path))] == [s.to_json() for s in grid]


def test_the_manifest_records_no_label() -> None:
    """An entry says what was configured, never what happened. If the label lived here the corpus
    would be testing whether a detector can read its own configuration file."""
    fields = set(json.loads(make_grid()[0].to_json()))
    assert not (fields & {"label", "t_collapse", "collapsed", "outcome"})


# --- config reconstruction -------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s.cell for s in ALL_SPECS])
def test_every_cell_builds_a_config(spec) -> None:
    grid = make_grid(tasks=("sort_digits",), specs=(spec,), seeds=1)
    cfg = build_config(grid[0])
    assert cfg.steps == grid[0].steps


def test_step_zero_overrides_are_folded_in_not_scheduled() -> None:
    """Applying them at onset as well would be a no-op, but leaving onset_step set would make the
    trace claim an injection point that never existed."""
    spec = _tiny(onset_step=None, overrides={"grpo.scale_rewards": "none"})
    cfg = build_config(spec)
    assert cfg.grpo.scale_rewards == "none"
    assert cfg.onset_step is None and cfg.onset_overrides == {}


def test_onset_overrides_are_scheduled_not_folded_in() -> None:
    spec = _tiny(onset_step=3, overrides={"difficulty": 8})
    cfg = build_config(spec)
    assert cfg.difficulty == 4, "the pre-onset difficulty must be untouched"
    assert cfg.onset_step == 3 and cfg.onset_overrides == {"difficulty": 8}


def test_build_task_returns_the_named_task() -> None:
    for name in TASKS:
        assert build_task(_tiny(task=name)).name == name


# --- the runner ------------------------------------------------------------------------------------


def test_execute_writes_a_readable_trace(tmp_path) -> None:
    out = execute(_tiny(), tmp_path)
    assert out.status == "ok" and out.steps_written == 4
    spec, recs = read_trace(trace_path(tmp_path, "t0"))
    assert spec["run_id"] == "t0"
    assert len(recs) == 4
    assert all(r["source"] == "testbed" for r in recs)


def test_simulated_runs_are_tagged_at_the_record_level(tmp_path) -> None:
    """The report generator refuses `sim` traces, so the tag has to be on the data itself rather
    than only in a manifest someone has to remember to consult."""
    execute(_tiny(run_id="s1", simulated=True, overrides={"sampler_noise": 0.5}), tmp_path)
    _, recs = read_trace(trace_path(tmp_path, "s1"))
    assert all(r["source"] == "sim" for r in recs)


def test_a_crash_is_recorded_rather_than_lost(tmp_path) -> None:
    """A run that vanishes silently biases the corpus toward whatever succeeds, and the families
    most likely to crash are the ones whose absence would matter most."""
    out = execute(_tiny(run_id="bad", task="nonexistent_task"), tmp_path)
    assert out.status == "crashed"
    assert out.error
    assert not trace_path(tmp_path, "bad").exists(), "no partial trace may survive"


def test_no_partial_trace_is_left_behind(tmp_path) -> None:
    execute(_tiny(run_id="bad2", task="nope"), tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_traces_are_skipped_so_a_grid_resumes(tmp_path) -> None:
    assert execute(_tiny(), tmp_path).status == "ok"
    assert execute(_tiny(), tmp_path).status == "skipped"
    assert execute(_tiny(), tmp_path, overwrite=True).status == "ok"


def test_the_trace_is_self_describing(tmp_path) -> None:
    """A trace must be re-derivable without the manifest it came from."""
    execute(_tiny(run_id="sd", seed=3), tmp_path)
    with gzip.open(trace_path(tmp_path, "sd"), "rt") as fh:
        header = json.loads(fh.readline())
    assert header["_spec"]["seed"] == 3
    assert build_config(RunSpec(**header["_spec"])).seed == 3


def test_traces_carry_the_oracle_for_the_labeler(tmp_path) -> None:
    execute(_tiny(), tmp_path)
    _, recs = read_trace(trace_path(tmp_path, "t0"))
    assert all("oracle/heldout_accuracy" in r for r in recs)
    assert any(r["oracle/probe_fresh"] == 1.0 for r in recs)


def test_run_grid_serially_covers_every_spec(tmp_path) -> None:
    specs = [_tiny(run_id=f"g{i}", seed=i) for i in range(3)]
    outcomes = run_grid(specs, tmp_path, workers=1, progress=False)
    assert {o.run_id for o in outcomes} == {"g0", "g1", "g2"}
    assert all(o.status == "ok" for o in outcomes)
    assert summarize(outcomes)["by_status"]["ok"] == 3


def test_summarize_names_the_crashed_runs(tmp_path) -> None:
    outcomes = run_grid(
        [_tiny(run_id="ok1"), _tiny(run_id="bad1", task="nope")],
        tmp_path,
        workers=1,
        progress=False,
    )
    s = summarize(outcomes)
    assert s["crashed_ids"] == ["bad1"]
    assert s["total"] == 2


def test_a_run_is_reproducible_from_its_trace_header(tmp_path) -> None:
    """The strongest property the corpus can offer: the trace alone regenerates itself."""
    execute(_tiny(run_id="rep"), tmp_path)
    header, original = read_trace(trace_path(tmp_path, "rep"))
    execute(RunSpec(**header), tmp_path, overwrite=True)
    _, again = read_trace(trace_path(tmp_path, "rep"))
    for a, b in zip(original, again, strict=True):
        for k in a:
            if k in ("source", "warm_start/cached"):
                continue
            assert (a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])), f"{k} diverged"
