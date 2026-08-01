"""The corpus grid: which runs exist, what each one is, and how to reproduce it exactly.

A manifest entry is the *complete* specification of a run. Given one, `build_config` reconstructs
the RunConfig bit for bit, so any trace in the corpus can be re-derived from a single JSON line
without consulting the code that generated it. That is the difference between a released dataset
and a directory of numbers.

Two things are deliberately kept out of the manifest.

**The label.** An entry records what was *configured*, never what happened. `t_collapse` is decided
afterwards from held-out accuracy by `label_run`, and cells where a knob was set but the run stayed
healthy are negatives. If the manifest carried the label, the corpus would be testing whether a
detector can read its own configuration file.

**Per-task warm-start budgets are data, not defaults.** They are recorded here explicitly because
they were measured, not chosen, and because they differ by more than an order of magnitude between
tasks. `modarith` needs ~5000 supervised steps where `sort_digits` needs 150 -- and modarith
reaches that band during a grokking transition, going 0.105 -> 1.000 between 2500 and 8000 steps,
so its budget is a measurement with a tolerance rather than a round number someone liked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from testbed.core.grpo import GRPOConfig
from testbed.core.train import RunConfig, apply_overrides
from testbed.inject.failures import ALL_SPECS, FailureSpec, sample_onset
from testbed.tasks.base import Task
from testbed.tasks.ca_rule import CARule
from testbed.tasks.countdown_lite import CountdownLite
from testbed.tasks.modarith import ModArith
from testbed.tasks.sort_digits import SortDigits

TASKS: dict[str, type] = {
    "sort_digits": SortDigits,
    "modarith": ModArith,
    "countdown_lite": CountdownLite,
    "ca_rule": CARule,
}


TARGET_BAND = (0.25, 0.55)
"""Held-out accuracy every run warm-starts *into*.

Below it there is nothing to fall from and every run is a STALL; above it the policy is saturated
with no headroom, and a run that starts at 0.97 cannot exhibit a failure mode either. Both ends
were measured rather than assumed: 300 supervised steps on sort_digits gave reward 0.969 at entropy
0.064 -- nothing left to collapse.
"""


@dataclass(frozen=True)
class TaskProfile:
    """A task plus the supervision budget it needs to reach `TARGET_BAND`.

    `warm_start_steps` is a *ceiling*, not a schedule. Training stops when the probe first crosses
    into the band, because a fixed budget does not produce a fixed starting point: measured across
    five seeds at the same 1000 supervised steps, ca_rule started at 0.277, 0.281, 0.199, 0.031 and
    0.152. One seed had learned essentially nothing while another was nearly in band. Seeds inside
    a cell would then differ in initial accuracy by as much as the failure knob differs from the
    control, and initialization variance would be read as an effect of the injection.
    """

    task: str
    difficulty: int
    """Fixed probe difficulty -- the measuring stick, held constant for the whole run."""

    difficulty_range: tuple[int, int]
    """Training difficulties, drawn per step. Must contain `difficulty`, or the probe would measure
    something the policy is never trained on."""

    warm_start_steps: int
    """Ceiling. Generous, since overshooting costs only time and stopping is governed by the probe."""

    measured_accuracy: float
    """Held-out accuracy at seed 0 under the budget originally measured. Kept as a record of the
    calibration sweep so a change that moves a task out of band is visible in a diff."""


PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile("sort_digits", 6, (3, 8), 2500, 0.344),
    TaskProfile("countdown_lite", 3, (2, 5), 4000, 0.445),
    # Steep between 1000 and 2500 steps (0.277 -> 0.906), which is exactly why the stopping rule
    # is a probe crossing rather than a step count.
    TaskProfile("ca_rule", 6, (4, 8), 6000, 0.277),
    # Groks: 0.105 at 2500 steps, 1.000 at 7000, and non-monotone inside the usable window
    # (0.406 at 5000, 0.309 at 6000). The ceiling sits past the transition so every seed reaches
    # the band wherever its own transition happens to fall.
    TaskProfile("modarith", 3, (2, 5), 12000, 0.406),
)

PROFILE_BY_TASK = {p.task: p for p in PROFILES}


@dataclass(frozen=True)
class RunSpec:
    """One corpus run, fully specified."""

    run_id: str
    task: str
    family: str
    dose: str
    seed: int
    steps: int
    difficulty: int
    warm_start_steps: int
    warm_start_target: tuple[float, float] | None
    difficulty_range: tuple[int, int] | None
    onset_step: int | None
    overrides: dict[str, Any]
    simulated: bool = False
    group_size: int = 8
    num_iterations: int = 2
    n_prompts: int = 8
    probe_every: int = 10
    probe_n: int = 256
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cell(self) -> str:
        return f"{self.task}/{self.family}/{self.dose}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> RunSpec:
        return cls(**json.loads(line))


def build_task(spec: RunSpec) -> Task:
    return TASKS[spec.task]()  # type: ignore[no-any-return]


def build_config(spec: RunSpec) -> RunConfig:
    """Reconstruct the exact RunConfig. The only path from a manifest entry to a run."""
    base = RunConfig(
        seed=spec.seed,
        steps=spec.steps,
        n_prompts=spec.n_prompts,
        difficulty=spec.difficulty,
        difficulty_range=(
            tuple(spec.difficulty_range) if spec.difficulty_range else None  # type: ignore[arg-type]
        ),
        warm_start_steps=spec.warm_start_steps,
        warm_start_target=(
            tuple(spec.warm_start_target) if spec.warm_start_target else None  # type: ignore[arg-type]
        ),
        probe_every=spec.probe_every,
        probe_n=spec.probe_n,
        grpo=GRPOConfig(group_size=spec.group_size, num_iterations=spec.num_iterations),
        onset_step=spec.onset_step,
        onset_overrides=dict(spec.overrides),
    )
    # Overrides that must hold from step 0 are folded into the base config instead of being
    # scheduled. Applying them here AND at onset would be a no-op, but leaving onset_step set
    # would make the trace claim an injection point that never existed.
    if spec.onset_step is None and spec.overrides:
        base = replace(apply_overrides(base, spec.overrides), onset_overrides={})
    return base


SEEDS_BY_FAMILY: dict[str, int] = {
    # The negative classes get more seeds than the failure families, because the false-alarm rate
    # is reported *broken out by hard-negative type* and each of those breakdowns needs its own
    # sample. At five seeds a per-type FAR moves in steps of 5 percentage points per task, which is
    # far coarser than the 5% operating point the whole study is calibrated to.
    "F0": 15,
    "H2": 10,  # the plateau, where the false-alarm budget is actually spent
    "H3": 10,
    "H4": 10,
    "H5": 10,
}
DEFAULT_SEEDS = 5


def make_grid(
    *,
    tasks: tuple[str, ...] = tuple(TASKS),
    specs: tuple[FailureSpec, ...] = ALL_SPECS,
    seeds: int = DEFAULT_SEEDS,
    seeds_by_family: dict[str, int] | None = None,
    steps: int = 600,
    seed0: int = 0,
    onset_seed: int = 20260801,
) -> list[RunSpec]:
    """The full corpus grid: tasks x cells x seeds.

    Onset steps are drawn from a single seeded generator so the whole grid is reproducible from
    `onset_seed` alone, and drawn *per run* rather than per cell so that seeds within a cell do not
    share an injection point -- otherwise every seed in a cell would collapse at the same step and
    the effective sample size for anything time-related would be the number of cells, not runs.

    Every run is the same length. Shorter healthy runs would give them fewer opportunities to raise
    a false alarm, which makes the false-alarm rate optimistic for free.
    """
    rng = np.random.default_rng(onset_seed)
    by_family = SEEDS_BY_FAMILY if seeds_by_family is None else seeds_by_family
    out: list[RunSpec] = []
    for task in tasks:
        profile = PROFILE_BY_TASK[task]
        for spec in specs:
            for s in range(by_family.get(spec.family, seeds)):
                seed = seed0 + s
                onset = sample_onset(rng) if spec.needs_onset else None
                out.append(
                    RunSpec(
                        run_id=f"{task}_{spec.family}_{spec.dose}_s{seed}",
                        task=task,
                        family=spec.family,
                        dose=spec.dose,
                        seed=seed,
                        steps=steps,
                        difficulty=profile.difficulty,
                        warm_start_steps=profile.warm_start_steps,
                        warm_start_target=TARGET_BAND,
                        difficulty_range=profile.difficulty_range,
                        onset_step=onset,
                        overrides=dict(spec.overrides),
                        simulated=spec.simulated,
                    )
                )
    return out


def write_manifest(path: str, specs: list[RunSpec]) -> None:
    with open(path, "w") as fh:
        for s in specs:
            fh.write(s.to_json() + "\n")


def read_manifest(path: str) -> list[RunSpec]:
    with open(path) as fh:
        return [RunSpec.from_json(line) for line in fh if line.strip()]


__all__ = [
    "PROFILES",
    "PROFILE_BY_TASK",
    "TASKS",
    "RunSpec",
    "TaskProfile",
    "build_config",
    "build_task",
    "make_grid",
    "read_manifest",
    "write_manifest",
]
