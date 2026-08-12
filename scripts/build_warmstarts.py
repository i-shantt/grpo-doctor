#!/usr/bin/env python3
"""Generate every warm start the grid needs, in parallel, before the grid runs.

    python scripts/build_warmstarts.py --report          # what exists, what is missing
    python scripts/build_warmstarts.py                   # build the missing ones
    python scripts/build_warmstarts.py --task ca_rule --seeds 8

Two reasons this is a separate pass rather than something the corpus runner does on demand.

**It is shared work, and on-demand it is duplicated work.** A warm start is keyed by
(task, difficulty, seed) and every one of a task's 29 cells at that seed loads the same checkpoint.
Under six corpus workers the uncached case is not rare, it is the opening minute of the run: six
processes pick up six cells of the same seed and all six train the same supervised model, then five
of them throw it away. Building them first turns the most expensive shared step into a cache hit.

**It produces a measurement the corpus depends on and cannot report itself.** Warm start stops when
the held-out probe first enters `TARGET_BAND`, and it does not always get there -- measured, ca_rule
seed 0 spent the entire 12000-step ceiling and finished at 0.055 against a 0.25 floor, while seed 1
crossed at 5200 steps. A run starting that far under the band has nothing to collapse from, so its
cell is a negative decided by initialization rather than by the knob. The hit rate is therefore a
property of the corpus worth knowing *before* generating it, and `--report` prints it per task.

Nothing here decides anything. It writes checkpoints and a table; whether a miss rate is acceptable
is a judgement made against the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from testbed.core.warmstart import DEFAULT_CACHE_DIR, key_for  # noqa: E402
from testbed.corpus.manifest import (  # noqa: E402
    CORPUS_TASKS,
    RunSpec,
    build_config,
    build_task,
    make_grid,
)


def needed(specs: list[RunSpec]) -> dict[tuple[str, int, int], RunSpec]:
    """One spec per distinct warm start, keyed by (task, difficulty, seed).

    The grid has 700 entries and roughly 60 distinct warm starts behind them; deduplicating here is
    the whole point of the pass.
    """
    out: dict[tuple[str, int, int], RunSpec] = {}
    for s in specs:
        out.setdefault((s.task, s.difficulty, s.seed), s)
    return out


def _build_one(spec: RunSpec, cache_dir: str) -> dict[str, Any]:
    """Warm start one (task, difficulty, seed). A process-pool worker, so it pins threads itself."""
    import torch

    torch.set_num_threads(1)
    from testbed.core.warmstart import load_or_train

    started = time.time()
    task = build_task(spec)
    cfg = build_config(spec)
    _, info = load_or_train(task, cfg, cache_dir=cache_dir)
    return {
        "task": spec.task,
        "difficulty": spec.difficulty,
        "seed": spec.seed,
        "cached": bool(info.get("warm_start/cached", 0.0)),
        "accuracy": float(info.get("warm_start/accuracy", float("nan"))),
        "steps_used": int(info.get("warm_start/steps_used", 0)),
        "hit_target": bool(info.get("warm_start/hit_target", 0.0)),
        "seconds": time.time() - started,
    }


def existing(spec: RunSpec, cache_dir: Path) -> Path | None:
    ckpt = cache_dir / f"{key_for(build_task(spec), build_config(spec)).filename()}.pt"
    return ckpt if ckpt.exists() else None


def report(rows: list[dict[str, Any]]) -> None:
    """Per-task hit rate and cost. The number that matters is `hit`: a task whose seeds routinely
    finish under the band is a task whose cells are decided by initialization."""
    print(
        f"\n{'task':16s} {'starts':>7s} {'hit band':>9s} {'median acc':>11s} {'median steps':>13s}"
    )
    print("-" * 60)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    for task in sorted(by_task):
        rs = by_task[task]
        hits = sum(r["hit_target"] for r in rs)
        accs = sorted(r["accuracy"] for r in rs)
        steps = sorted(r["steps_used"] for r in rs)
        print(
            f"{task:16s} {len(rs):7d} {f'{hits}/{len(rs)}':>9s} "
            f"{accs[len(accs) // 2]:11.3f} {steps[len(steps) // 2]:13d}"
        )

    missed = [r for r in rows if not r["hit_target"]]
    if missed:
        print(f"\n{len(missed)} warm start(s) finished outside the band:")
        for r in sorted(missed, key=lambda r: r["accuracy"]):
            print(
                f"  {r['task']}/s{r['seed']:<3d} accuracy {r['accuracy']:.3f} "
                f"after {r['steps_used']} steps"
            )
        print(
            "  These runs start with less headroom than the labeler's threshold assumes. Their\n"
            "  cells become negatives for a reason that has nothing to do with the knob."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", action="append", help="restrict to these tasks (repeatable)")
    p.add_argument("--seeds", type=int, default=None, help="cap the seed count per task")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--report", action="store_true", help="report what exists; build nothing")
    p.add_argument("--out", default="results/warmstarts.json")
    args = p.parse_args()

    tasks = tuple(args.task) if args.task else CORPUS_TASKS
    grid = make_grid(tasks=tasks, roles=bool(not args.task))
    if args.seeds is not None:
        grid = [s for s in grid if s.seed < args.seeds]
    work = needed(grid)
    cache_dir = Path(args.cache_dir)

    have = {k: v for k, v in work.items() if existing(v, cache_dir)}
    print(f"{len(work)} distinct warm starts for {len(grid)} runs; {len(have)} already cached")
    print(dict(Counter(t for t, _, _ in work)))

    if args.report:
        if not have:
            return 0
        rows = [_build_one(v, str(cache_dir)) for v in have.values()]
        report(rows)
        return 0

    started = time.time()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_one, s, str(cache_dir)): k for k, s in work.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            flag = "cached" if row["cached"] else f"{row['seconds']:.0f}s"
            band = "in band" if row["hit_target"] else "MISSED BAND"
            print(
                f"  [{i:3d}/{len(work)}] {row['task']}/s{row['seed']:<3d} "
                f"acc {row['accuracy']:.3f} after {row['steps_used']:5d} steps  {band:11s} {flag}",
                flush=True,
            )

    report(rows)
    print(f"\nwall {(time.time() - started) / 60:.1f} min on {args.workers} workers")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(rows, key=lambda r: (r["task"], r["seed"])), indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
