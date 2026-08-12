"""Loading the corpus into the shapes the evaluation actually consumes.

Two things live here, and both exist because getting them wrong is silent.

**The label is read from `labels.json`, never from the manifest.** A run's family says which knob was
turned; it does not say whether the run collapsed, and most dosed runs did not. Anything that treats
"F5" as a positive is measuring the grid instead of the model, so `collapsed` comes from the
labeler's verdict and from nowhere else.

**Alarm positions are converted from record index to optimizer step.** A scorer returns an array
parallel to the record list, so its crossing is an *index*; `t_collapse` is a *step*. In this corpus
the two happen to coincide -- every trace logs all 600 steps -- which is exactly the condition under
which mixing them up would go unnoticed here and produce nonsense on a trace that logged every tenth
step. `outcome_for` does the conversion through `records[i].step` so the coincidence is never relied
on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grpo_doctor.eval.controls import alarm_step
from grpo_doctor.eval.metrics import RunOutcome
from grpo_doctor.eval.splits import RunRef
from grpo_doctor.record import StepRecord
from grpo_doctor.trace import load_trace, to_records

HEALTHY_LABEL = "healthy"


@dataclass(frozen=True)
class LoadedRun:
    """One corpus run: what it is, what happened to it, and the signals it produced."""

    ref: RunRef
    records: list[StepRecord]
    collapsed: bool
    t_collapse: int | None
    censored: bool
    simulated: bool
    """The pathology was simulated rather than reproduced (F9). Carried, never silently dropped."""

    @property
    def run_id(self) -> str:
        return self.ref.run_id

    @property
    def n_steps(self) -> int:
        return len(self.records)


def load_labels(path: Path) -> list[dict[str, Any]]:
    with open(path) as fh:
        rows: list[dict[str, Any]] = json.load(fh)
    return rows


def load_corpus(corpus_dir: Path, *, traces: str = "traces") -> list[LoadedRun]:
    """Every labeled run, with its trace, in `labels.json` order.

    Raises rather than skipping when a labeled run has no trace. A missing trace is a corrupted
    corpus, and quietly evaluating on the runs that happened to load would report a detection rate
    over a denominator nobody chose.
    """
    labels = load_labels(corpus_dir / "labels.json")
    out: list[LoadedRun] = []
    for row in labels:
        path = corpus_dir / traces / f"{row['run_id']}.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"labeled run {row['run_id']} has no trace at {path}")
        _, rows = load_trace(path)
        out.append(
            LoadedRun(
                ref=RunRef(
                    run_id=row["run_id"],
                    task=row["task"],
                    family=row["family"],
                    seed=int(row["seed"]),
                ),
                records=to_records(rows),
                collapsed=row["label"] != HEALTHY_LABEL,
                t_collapse=row["t_collapse"],
                censored=bool(row["censored"]),
                simulated=bool(row["simulated"]),
            )
        )
    return out


def outcome_for(run: LoadedRun, scores: np.ndarray, threshold: float) -> RunOutcome:
    """Score array plus threshold -> the run's contribution to every metric.

    The index-to-step conversion is the whole reason this is a function rather than a comprehension
    at each call site.
    """
    i = alarm_step(scores, threshold)
    return RunOutcome(
        run_id=run.run_id,
        family=run.ref.family,
        collapsed=run.collapsed,
        t_collapse=run.t_collapse,
        t_alarm=run.records[i].step if i is not None else None,
        n_steps=run.n_steps,
        censored=run.censored,
    )


def by_id(runs: Sequence[LoadedRun]) -> dict[str, LoadedRun]:
    return {r.run_id: r for r in runs}


__all__ = [
    "HEALTHY_LABEL",
    "LoadedRun",
    "by_id",
    "load_corpus",
    "load_labels",
    "outcome_for",
]
