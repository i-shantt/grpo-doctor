"""Execute manifest entries into trace files, one process per run.

Multiprocessing with `torch.set_num_threads(1)` per worker, because it was measured to be about 5x
faster than a single MPS process at this model size (8 CPU workers ~16.7 steps/s aggregate against
MPS ~3.1 steps/s). Intra-op threading fights itself on a 3M-parameter model; the parallelism that
pays is across runs.

Three behaviors here exist because of how corpus generation actually fails.

**A crashed run is recorded, not lost.** NaN gradients, an unexpected exception, anything -- the
outcome is written to the manifest result with the traceback. A run that vanishes silently biases
the corpus toward whatever succeeds, and the families most likely to crash (F8 unclipped, F3 at
high mu) are exactly the ones whose absence would matter most.

**Traces are written atomically and runs are resumable.** An overnight grid that dies at 80% must
resume rather than restart, and a half-written trace must never be readable as a complete one.

**`source` is stamped on every record.** Traces from the simulator are tagged `sim` and the report
generator refuses them, so a simulated trace cannot end up inside a published number.
"""

from __future__ import annotations

import gzip
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testbed.corpus.manifest import RunSpec, build_config, build_task


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: str
    """"ok", "crashed", or "skipped" (already present)."""

    steps_written: int
    seconds: float
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))


def trace_path(out_dir: Path | str, run_id: str) -> Path:
    return Path(out_dir) / f"{run_id}.jsonl.gz"


def execute(spec: RunSpec, out_dir: Path | str, *, overwrite: bool = False) -> RunOutcome:
    """Run one manifest entry to completion and write its trace.

    Imports torch lazily and pins threads here rather than at module scope so this is safe to call
    as a process-pool worker, where the module is re-imported in each child.
    """
    import time

    import torch

    torch.set_num_threads(1)
    from testbed.core.train import run

    path = trace_path(out_dir, spec.run_id)
    if path.exists() and not overwrite:
        return RunOutcome(spec.run_id, "skipped", 0, 0.0)

    started = time.time()
    tmp = path.with_suffix(".gz.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        task = build_task(spec)
        cfg = build_config(spec)
        source = "sim" if spec.simulated else "testbed"
        with gzip.open(tmp, "wt") as fh:
            # The spec is the first line of its own trace, so a trace file is self-describing and
            # can be re-derived without the manifest it came from.
            fh.write(json.dumps({"_spec": json.loads(spec.to_json()), "source": source}) + "\n")
            for rec in run(task, cfg):
                fh.write(json.dumps({**rec, "source": source}, separators=(",", ":")) + "\n")
                n += 1
        tmp.replace(path)
        return RunOutcome(spec.run_id, "ok", n, time.time() - started)
    except Exception:
        tmp.unlink(missing_ok=True)
        return RunOutcome(
            spec.run_id, "crashed", n, time.time() - started, error=traceback.format_exc()[-2000:]
        )


def read_trace(path: Path | str) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Return (spec dict, records). The inverse of `execute`'s writer."""
    with gzip.open(path, "rt") as fh:
        header = json.loads(fh.readline())
        return header.get("_spec", {}), [json.loads(line) for line in fh if line.strip()]


def run_grid(
    specs: list[RunSpec],
    out_dir: Path | str,
    *,
    workers: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> list[RunOutcome]:
    """Execute a grid across processes, returning one outcome per spec.

    Defaults to `cpu_count - 2`: leaving headroom keeps the machine usable and avoids the
    memory-bandwidth contention that makes the last couple of workers cost more than they add.
    """
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[RunOutcome] = []

    if workers == 1:
        for i, spec in enumerate(specs):
            outcomes.append(execute(spec, out_dir, overwrite=overwrite))
            if progress:
                _report(outcomes[-1], i + 1, len(specs))
        return outcomes

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, s, out_dir, overwrite=overwrite): s for s in specs}
        for i, fut in enumerate(as_completed(futures)):
            outcomes.append(fut.result())
            if progress:
                _report(outcomes[-1], i + 1, len(specs))
    return outcomes


def _report(outcome: RunOutcome, done: int, total: int) -> None:
    mark = {"ok": "  ", "skipped": "= ", "crashed": "!!"}.get(outcome.status, "??")
    print(
        f"{mark} [{done:4d}/{total}] {outcome.run_id:44s} "
        f"{outcome.status:8s} {outcome.steps_written:4d} steps {outcome.seconds:6.1f}s",
        flush=True,
    )


def write_outcomes(path: Path | str, outcomes: list[RunOutcome]) -> None:
    with open(path, "w") as fh:
        for o in outcomes:
            fh.write(o.to_json() + "\n")


def summarize(outcomes: list[RunOutcome]) -> dict[str, Any]:
    """Crash rate is a headline number, not a footnote: it says which families are under-sampled."""
    by_status: dict[str, int] = {}
    for o in outcomes:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    total_time = sum(o.seconds for o in outcomes)
    return {
        "total": len(outcomes),
        "by_status": by_status,
        "crashed_ids": [o.run_id for o in outcomes if o.status == "crashed"],
        "wall_seconds": total_time,
        "mean_seconds": total_time / max(1, len(outcomes)),
    }


__all__ = [
    "RunOutcome",
    "execute",
    "read_trace",
    "run_grid",
    "summarize",
    "trace_path",
    "write_outcomes",
]
