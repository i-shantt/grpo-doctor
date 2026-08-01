#!/usr/bin/env python3
"""Generate the corpus, or smoke-test the grid before spending an overnight on it.

    python scripts/build_corpus.py --smoke              # one run per family, ~10 min
    python scripts/build_corpus.py --manifest-only      # write the grid, run nothing
    python scripts/build_corpus.py --full               # the whole thing

`--smoke` is the gate. It runs a single seed of every family and reports the label each one
actually produced, so the question "does this knob do anything?" is answered for 660 runs' worth of
compute by about ten minutes of it. The expected failure it catches is a family whose dose is too
weak to collapse anything -- which is not a bug, but is something to know *before* committing the
grid rather than after.

Nothing here decides a label from the manifest. Labels come from `label_run` reading held-out
accuracy, and a family that produces HEALTHY is reported as HEALTHY.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from grpo_doctor.eval.labels import LabelConfig, label_run  # noqa: E402
from testbed.corpus.manifest import RunSpec, make_grid, write_manifest  # noqa: E402
from testbed.corpus.runner import (  # noqa: E402
    read_trace,
    run_grid,
    summarize,
    trace_path,
    write_outcomes,
)
from testbed.inject.failures import ALL_SPECS  # noqa: E402


def label_trace(path: Path, probe_every: int) -> tuple[str, int | None, float, float]:
    """Label one written trace. Returns (label, t_collapse, peak accuracy, final accuracy)."""
    _, recs = read_trace(path)
    steps = np.array([r["step"] for r in recs])
    acc = np.array([r["oracle/heldout_accuracy"] for r in recs])
    fresh = np.array([r["oracle/probe_fresh"] for r in recs])
    reward = np.array([r["reward"] for r in recs])
    zero_std = np.array([r.get("acr", 0.0) for r in recs])
    res = label_run(steps, acc, fresh, reward, zero_std, LabelConfig(probe_every=probe_every))
    return res.label.value, res.t_collapse, res.peak_accuracy, res.final_accuracy


def smoke(args: argparse.Namespace) -> int:
    """One seed of every family on one task, then report what each actually produced."""
    specs: list[RunSpec] = []
    for spec in ALL_SPECS:
        specs += make_grid(tasks=(args.task,), specs=(spec,), seeds=1, steps=args.steps)
    out_dir = Path(args.out) / "smoke"

    print(f"smoke: {len(specs)} runs x {args.steps} steps on {args.task}\n")
    started = time.time()
    outcomes = run_grid(specs, out_dir, workers=args.workers, overwrite=args.overwrite)
    summary = summarize(outcomes)

    print(f"\n{'cell':22s} {'label':9s} {'t_collapse':>10s} {'peak':>6s} {'final':>6s}")
    print("-" * 60)
    labels: dict[str, str] = {}
    for spec in sorted(specs, key=lambda s: (s.family, s.dose)):
        path = trace_path(out_dir, spec.run_id)
        if not path.exists():
            print(f"{spec.family + '/' + spec.dose:22s} {'CRASHED':9s}")
            continue
        label, t, peak, final = label_trace(path, spec.probe_every)
        labels[f"{spec.family}/{spec.dose}"] = label
        print(
            f"{spec.family + '/' + spec.dose:22s} {label:9s} "
            f"{('-' if t is None else t):>10} {peak:6.3f} {final:6.3f}"
        )

    print(f"\n{json.dumps(summary['by_status'])}  wall {time.time() - started:.0f}s")

    # The gate. Not "did every knob collapse" -- a dose that does nothing is a legitimate negative --
    # but the two conditions that would make the grid worthless.
    problems = []
    healthy_control = labels.get("F0/none")
    if healthy_control != "healthy":
        problems.append(f"F0 control was labeled {healthy_control!r}, not healthy")
    if not any(v != "healthy" for k, v in labels.items() if k.startswith("F")):
        problems.append("no failure family collapsed at all; every dose is too weak")
    if summary["by_status"].get("crashed"):
        problems.append(f"crashed runs: {summary['crashed_ids']}")

    print()
    for p in problems:
        print(f"PROBLEM: {p}")
    if not problems:
        collapsed = sum(1 for k, v in labels.items() if k.startswith("F") and v != "healthy")
        print(f"OK: control is healthy, {collapsed} failure cells collapsed")
    return 1 if problems else 0


def full(args: argparse.Namespace) -> int:
    specs = make_grid(steps=args.steps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(str(out_dir / "manifest.jsonl"), specs)
    print(f"{len(specs)} runs -> {out_dir}")
    if args.manifest_only:
        print(dict(Counter(s.family for s in specs)))
        return 0

    started = time.time()
    outcomes = run_grid(specs, out_dir / "traces", workers=args.workers, overwrite=args.overwrite)
    write_outcomes(str(out_dir / "outcomes.jsonl"), outcomes)
    summary = summarize(outcomes)
    summary["wall_seconds"] = time.time() - started
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{json.dumps(summary['by_status'])}  wall {summary['wall_seconds'] / 60:.1f} min")
    return 1 if summary["by_status"].get("crashed") else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--task", default="sort_digits")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="corpus")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.smoke:
        return smoke(args)
    if args.full or args.manifest_only:
        return full(args)
    p.error("choose one of --smoke, --full, --manifest-only")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
