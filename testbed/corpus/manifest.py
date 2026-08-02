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
tasks. `modarith` needs ~10000 supervised steps where `sort_digits` needs a few hundred -- and it
reaches the band during a grokking transition, going 0.105 -> 1.000 between 2500 and 8000 steps, so
its budget is a measurement with a tolerance rather than a round number someone liked.

**Which tasks are in the corpus is also data.** `TaskRole` records it per task, with the measurement
that decided it. A task whose control misbehaves, or that cannot be made to fail at any dose, is
worse than no task at all: the first poisons the negatives the false-alarm rate is calibrated on,
and the second inflates the denominator of every rate with runs that could never have been
positives. Both are recorded rather than quietly deleted, because "we tried four tasks and two did
not work" is a result and a repo that shows only the two survivors is hiding it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
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


class TaskRole(str, Enum):
    """What a task contributes to the corpus.

    Not every implemented task earns a place in the grid, and the reason a task is out belongs next
    to the task rather than in a commit message.
    """

    FULL = "full"
    """Every cell: failure families and negatives alike.

    Including for a task that has never once collapsed. Dropping a task's failure cells *because*
    they came out negative would select the corpus on its outcomes, and it would delete exactly the
    runs that are hardest to get right -- a knob was set, the training signal changed visibly, and
    held-out accuracy survived. Those are the false alarms worth measuring.
    """

    EXCLUDED = "excluded"
    """Not generated at all. The task stays implemented and tested; it is out of the corpus.

    Reserved for a task whose *control* misbehaves. That is the one defect no downstream analysis
    can absorb: with a broken F0 there is no baseline to measure a failure against and no trustworthy
    negatives to calibrate a false-alarm rate on. A task that simply refuses to collapse is not
    excluded -- it is `FULL` and contributes negatives.
    """


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
    task_kwargs: dict[str, Any]
    """Constructor arguments, chiefly the maximum answer length.

    Short answers are not a convenience. Sequence accuracy is per-token accuracy raised to the
    answer length, so every extra token compounds the error and starves the reward signal.
    Measured on sort_digits at 4 seeds: cutting the maximum from 8 digits to 6 raised mean RL gain
    from +0.202 to +0.229 with the worst seed going +0.043 -> +0.160, and cut the supervised warm
    start from 3400-6000 steps to 300-1100. Adding capacity instead (192x4 -> 256x6) made it
    *worse*, +0.077, because the bottleneck was compounding per-token error rather than anything
    the model could not represent.
    """

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

    role: TaskRole = TaskRole.FULL
    """What this task contributes. See `TaskRole`; the evidence is in `role_reason`."""

    role_reason: str = ""
    """The measurement behind `role`, in one line. Required for anything but a full task, so a
    demotion cannot be made silently."""

    excluded_seeds: tuple[int, ...] = ()
    """Seeds whose warm start never reached `TARGET_BAND`, skipped by `seeds_for`.

    Measured, not chosen, and written down per seed by `scripts/build_warmstarts.py`. ca_rule misses
    the band on roughly a quarter of seeds: s0 finished at 0.055, s8 at 0.043, s11 at 0.090, s12 at
    0.066 and s13 at 0.047, each after spending the entire 12000-step ceiling.

    Excluding them is selection on *initialization*, decided before any failure knob is applied, and
    it is the confound `TARGET_BAND` exists to remove -- not selection on outcome. A run that starts
    at 0.055 has no headroom to fall through: every ca_rule seed-0 run in the F5 probe peaked
    between 0.09 and 0.24, so the labeler could never have seen a drawdown, and the cell would have
    contributed a negative decided by its random initialization rather than by the condition under
    study. That is the same defect that disqualified modarith, one notch less severe.

    The grid substitutes the next seed rather than shrinking the cell, so every cell keeps the same
    sample size and the manifest still reproduces from `onset_seed` alone.
    """

    seed_scale: int = 1
    """Multiplier on every per-cell seed count for this task.

    The grid is deliberately unbalanced across tasks, because collapse-proneness turned out to be
    task-specific and seeds spent on a task that does not collapse buy no positives. Scaling here
    rather than editing `SEEDS_BY_FAMILY` keeps the *shape* of the negative-to-positive sampling --
    the ratio the false-alarm calibration depends on -- identical across tasks while their absolute
    sizes differ.
    """


# Ceilings are deliberately generous. Warm start stops the moment the probe crosses into the band,
# so a high ceiling costs nothing for a task that gets there early and is the difference between a
# usable run and a dead one for a task that does not. Measured: at a 2500-step ceiling with a
# difficulty range, most sort_digits seeds never reached the band at all, RL had no reward signal
# to work with, and F0 peaked at 0.22 instead of 0.85. At 8000, 4/4 seeds reached it using a mean
# of 3200 steps -- the ceiling was the whole problem, not the range.
PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile(
        "ca_rule",
        {"max_width": 6},
        5,
        (3, 6),
        12000,
        0.277,
        seed_scale=2,
        # Measured by scripts/build_warmstarts.py: 15 of the first 20 seeds reached the band. These
        # five spent the whole 12000-step ceiling and finished at 0.055, 0.043, 0.090, 0.066 and
        # 0.047 -- far under the 0.25 floor, with no headroom for anything to collapse through.
        excluded_seeds=(0, 8, 11, 12, 13),
        role_reason=(
            "Double weight because positives are the scarce resource in this corpus and ca_rule "
            "is where they are. Per-task smoke over 26 cells x 2 seeds: ca_rule collapsed 6 cells "
            "(F3/mu8_hot 2/2, 0.537 -> 0.000) against a control healthy 2/2, where sort_digits "
            "collapsed 2 and countdown_lite 0. Lead time is undefined on a run that never "
            "collapses, so seeds spent here buy headline sample size and seeds spent elsewhere "
            "buy false-alarm sample size."
        ),
    ),
    TaskProfile("sort_digits", {"max_digits": 6}, 4, (2, 6), 12000, 0.348),
    # Kept at full breadth despite contributing no positives, and that is the point. 0 collapses in
    # 29 cells x 2 seeds, but re-running the three most violent knobs showed it is resilient rather
    # than floored: F8/none_unclipped fell 0.551 -> 0.254 and came back, F3/mu8_hot 0.547 -> 0.352
    # and came back, both inside the H=50 persistence window. A drawdown of 0.30 that recovers is
    # the hardest negative in the corpus -- three times the labeler's 0.094 noise floor, and every
    # cheap signal will be screaming through it. Its failure cells are worth more as negatives than
    # its hard-negative cells are.
    TaskProfile("countdown_lite", {"max_numbers": 5}, 3, (2, 5), 12000, 0.445),
    # Groks: 0.105 at 2500 supervised steps and 1.000 at 7000 on a single difficulty. Over a range
    # it needs more still, so this ceiling sits far past the transition.
    TaskProfile(
        "modarith",
        {"max_terms": 5},
        3,
        (2, 5),
        24000,
        0.406,
        role=TaskRole.EXCLUDED,
        role_reason=(
            "The F0 control STALLs on 1 of 2 seeds, which disqualifies it outright: a task whose "
            "*healthy* condition dies cannot supply negatives, and its positives cannot be told "
            "from its baseline. The cause is the grokking transition -- warm start lands both "
            "seeds at the very bottom of the band (0.258, 0.262 against a 0.25 floor) after "
            "10-11k supervised steps, because the accuracy curve crosses the whole band almost "
            "vertically and the stopping rule catches it on the way through. What is lost with it "
            "is stated in docs/STATE.md rather than waved away: modarith was carried for the "
            "length-collapsing reward hack, and the length baseline now has a thinner set of "
            "positives to be right about."
        ),
    ),
)

PROFILE_BY_TASK = {p.task: p for p in PROFILES}

CORPUS_TASKS: tuple[str, ...] = tuple(p.task for p in PROFILES if p.role is not TaskRole.EXCLUDED)
"""Tasks the corpus is actually generated from, in profile order."""


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
    task_kwargs: dict[str, Any]
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
    return TASKS[spec.task](**spec.task_kwargs)  # type: ignore[no-any-return]


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
    # 18 rather than 15 because the F5 shaped-leak doses added three failure cells per task, which
    # pushed the expected-negative share of the grid to 0.289 and tripped the 0.30 floor. The floor
    # is the thing being protected -- a 5% operating point needs negatives to be calibrated on --
    # so the control grew to meet it rather than the guard being lowered to meet the control.
    "F0": 18,
    "H2": 10,  # the plateau, where the false-alarm budget is actually spent
    "H3": 10,
    "H4": 10,
    "H5": 10,
}
DEFAULT_SEEDS = 5


def cells_for(profile: TaskProfile, specs: tuple[FailureSpec, ...]) -> tuple[FailureSpec, ...]:
    """The cells a task actually contributes, after its role."""
    return () if profile.role is TaskRole.EXCLUDED else specs


def seeds_for(profile: TaskProfile, n: int, seed0: int = 0) -> list[int]:
    """The first `n` usable seeds for a task, skipping the ones that never warm-started into band.

    Substituting rather than shrinking keeps every cell at the same sample size, which matters
    because per-cell counts are what the false-alarm calibration and the seeds-share-no-onset
    guarantee are both stated over. The cost is that a task's seeds are no longer contiguous, so
    the exclusions have to be readable from the profile -- which is why they live there with their
    measurements attached rather than in a filter someone applies later.
    """
    out: list[int] = []
    seed = seed0
    while len(out) < n:
        if seed not in profile.excluded_seeds:
            out.append(seed)
        seed += 1
    return out


def make_grid(
    *,
    tasks: tuple[str, ...] = tuple(TASKS),
    specs: tuple[FailureSpec, ...] = ALL_SPECS,
    seeds: int = DEFAULT_SEEDS,
    seeds_by_family: dict[str, int] | None = None,
    steps: int = 600,
    seed0: int = 0,
    onset_seed: int = 20260801,
    roles: bool = True,
) -> list[RunSpec]:
    """The full corpus grid: tasks x cells x seeds.

    Onset steps are drawn from a single seeded generator so the whole grid is reproducible from
    `onset_seed` alone, and drawn *per run* rather than per cell so that seeds within a cell do not
    share an injection point -- otherwise every seed in a cell would collapse at the same step and
    the effective sample size for anything time-related would be the number of cells, not runs.

    Every run is the same length. Shorter healthy runs would give them fewer opportunities to raise
    a false alarm, which makes the false-alarm rate optimistic for free.

    `roles=False` ignores `TaskProfile.role`, `seed_scale` and `excluded_seeds`, and gives the
    uniform tasks x cells product over contiguous seeds. That is the diagnostic view -- it is how a
    demotion gets re-examined, and how `--smoke` asks "does this knob do anything on this task"
    about a task the grid no longer carries. It is never the corpus.
    """
    rng = np.random.default_rng(onset_seed)
    by_family = SEEDS_BY_FAMILY if seeds_by_family is None else seeds_by_family
    out: list[RunSpec] = []
    for task in tasks:
        profile = PROFILE_BY_TASK[task]
        task_specs = cells_for(profile, specs) if roles else specs
        scale = profile.seed_scale if roles else 1
        # Under `roles=False` the excluded seeds come back too: that path exists to re-examine a
        # judgement, and a seed cannot be re-examined if the grid refuses to emit it.
        source = profile if roles else replace(profile, excluded_seeds=())
        for spec in task_specs:
            for seed in seeds_for(source, by_family.get(spec.family, seeds) * scale, seed0):
                onset = sample_onset(rng) if spec.needs_onset else None
                out.append(
                    RunSpec(
                        run_id=f"{task}_{spec.family}_{spec.dose}_s{seed}",
                        task=task,
                        task_kwargs=dict(profile.task_kwargs),
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
    "CORPUS_TASKS",
    "PROFILES",
    "PROFILE_BY_TASK",
    "TASKS",
    "RunSpec",
    "TaskProfile",
    "TaskRole",
    "build_config",
    "build_task",
    "cells_for",
    "make_grid",
    "read_manifest",
    "seeds_for",
    "write_manifest",
]
