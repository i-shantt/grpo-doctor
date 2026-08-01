"""`grpo-doctor` command line.

    grpo-doctor replay corpus/traces/sort_digits_F5_structure_s0.jsonl.gz
    grpo-doctor replay <trace> --panel        # per-step signal readings
    grpo-doctor label  <trace>                # what the labeler makes of it

`replay` is the ten-second reproduction: point it at any released trace, on any machine, with no
GPU and no torch, and it re-derives the alarm and the lead time from the raw record stream. Nothing
is cached and no result is read from a file -- the monitor is actually run.

The output deliberately shows training reward and held-out accuracy side by side, because the point
of the whole project is the gap between them. On a reward-hacking trace the reward column climbs to
1.000 while the accuracy column falls to 0.000, and the alarm fires somewhere in between.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import IO, Any

import numpy as np

from grpo_doctor.monitor import Monitor
from grpo_doctor.record import StepRecord
from grpo_doctor.snapshot import Level

LEVEL_MARK = {Level.OK: ".", Level.WATCH: "-", Level.WARN: "!", Level.ALARM: "#"}


def _open(path: Path) -> IO[str]:
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a testbed trace or a plain JSONL record stream.

    Tolerates both because a corpus trace carries a `_spec` header line and a trace someone
    captured from their own trainer will not.
    """
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with _open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "_spec" in obj:
                header = obj["_spec"]
                continue
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path} contained no records")
    return header, rows


def to_records(rows: list[dict[str, Any]]) -> list[StepRecord]:
    """Map testbed metric dicts onto StepRecord.

    The oracle columns are carried through so `label` can use them; `Monitor.update` strips them at
    its own boundary, which is where the guarantee belongs.
    """
    out = []
    for r in rows:
        out.append(
            StepRecord(
                step=int(r.get("step", 0)),
                reward_mean=r.get("reward"),
                reward_std=r.get("reward_std"),
                frac_reward_zero_std=r.get("frac_reward_zero_std"),
                entropy=r.get("entropy"),
                grad_norm=r.get("grad_norm"),
                learning_rate=r.get("learning_rate"),
                clip_low=r.get("clip_ratio/low_mean"),
                clip_high=r.get("clip_ratio/high_mean"),
                clip_region=r.get("clip_ratio/region_mean"),
                completion_len_mean=r.get("completions/mean_length"),
                completion_clipped_ratio=r.get("completions/clipped_ratio"),
                importance_ratio_max=r.get("importance_ratio/max"),
                importance_ratio_log_std=r.get("importance_ratio/log_std"),
                heldout_accuracy=r.get("oracle/heldout_accuracy"),
                source=r.get("source", "unknown"),
            )
        )
    return out


def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.trace)
    header, rows = load_trace(path)
    records = to_records(rows)

    monitor = Monitor()
    snaps = [monitor.update(r) for r in records]
    first_alarm = next((s for s in snaps if s.level >= Level.WARN), None)

    if header:
        print(f"run    {header.get('run_id', '?')}")
        print(
            f"cell   {header.get('task', '?')} / {header.get('family', '?')}/{header.get('dose', '?')}"
            f"   onset={header.get('onset_step')}  seed={header.get('seed')}"
        )
    sources = {r.source for r in records}
    print(
        f"source {'+'.join(sorted(sources))}   steps={len(records)}   coverage={snaps[-1].coverage:.0%}"
    )
    if "sim" in sources:
        print("NOTE   this trace is SIMULATED; it is excluded from every reported number")
    print()

    every = max(1, len(records) // args.rows)
    print(f"{'step':>5} {'lvl':>3} {'score':>6} {'reward':>7} {'heldout':>8}  signals")
    for i in range(0, len(records), every):
        s, r = snaps[i], records[i]
        vals = " ".join(
            f"{sig.name}={sig.z:+.1f}" if sig.available and sig.z is not None else f"{sig.name}=--"
            for sig in s.signals
        )
        acc = "   n/a" if r.heldout_accuracy is None else f"{r.heldout_accuracy:8.3f}"
        rew = "    n/a" if r.reward_mean is None else f"{r.reward_mean:7.3f}"
        print(f"{s.step:5d} {LEVEL_MARK[s.level]:>3} {s.score:6.2f} {rew} {acc}  {vals}")

    print()
    if first_alarm is None:
        print("no alarm raised")
    else:
        print(f"first alarm at step {first_alarm.step} ({first_alarm.level.name})")
        for a in first_alarm.alerts:
            print(f"  {a.message}")

    lead = _lead_time(records, first_alarm)
    if lead is not None:
        print(f"lead time: {lead} steps before held-out accuracy collapsed")
    return 0


def _lead_time(records: list[StepRecord], alarm: Any) -> int | None:
    """Steps between the alarm and `t_collapse`, when the trace carries the oracle."""
    from grpo_doctor.eval.labels import LabelConfig, label_run

    acc = [r.heldout_accuracy for r in records]
    if any(a is None for a in acc) or alarm is None:
        return None
    steps = np.array([r.step for r in records])
    reward = np.array([r.reward_mean if r.reward_mean is not None else np.nan for r in records])
    # Every step is treated as a fresh probe here: a trace from someone else's trainer has no
    # probe_fresh column, and over-counting probes only makes the label more conservative.
    res = label_run(steps, np.array(acc, dtype=float), np.ones(len(acc)), reward, cfg=LabelConfig())
    if res.t_collapse is None:
        return None
    return int(res.t_collapse) - int(alarm.step)


def cmd_label(args: argparse.Namespace) -> int:
    from grpo_doctor.eval.labels import LabelConfig, label_run

    _, rows = load_trace(Path(args.trace))
    if "oracle/heldout_accuracy" not in rows[0]:
        print("this trace carries no held-out accuracy, so it cannot be labeled", file=sys.stderr)
        return 2
    steps = np.array([r["step"] for r in rows])
    acc = np.array([r["oracle/heldout_accuracy"] for r in rows], dtype=float)
    fresh = np.array([r.get("oracle/probe_fresh", 1.0) for r in rows], dtype=float)
    reward = np.array([r.get("reward", np.nan) for r in rows], dtype=float)
    zero = np.array([r.get("acr", np.nan) for r in rows], dtype=float)

    res = label_run(steps, acc, fresh, reward, zero, LabelConfig(probe_every=args.probe_every))
    print(f"label        {res.label.value}")
    print(f"t_collapse   {res.t_collapse}  (+/- {res.probe_interval}, the probe interval)")
    print(f"peak         {res.peak_accuracy:.3f} at step {res.peak_step}")
    print(f"final        {res.final_accuracy:.3f}")
    print(f"delta        {res.delta:.4f}   (3 SE of the probe at the peak)")
    print(f"censored     {res.censored}")
    print(f"reason       {res.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grpo-doctor", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("replay", help="stream a trace through the monitor")
    rp.add_argument("trace")
    rp.add_argument("--rows", type=int, default=30, help="approximate number of rows to print")
    rp.add_argument("--panel", action="store_true", help="(reserved) full per-signal panel")
    rp.set_defaults(fn=cmd_replay)

    lp = sub.add_parser("label", help="apply the collapse labeler to a trace")
    lp.add_argument("trace")
    lp.add_argument("--probe-every", type=int, default=10)
    lp.set_defaults(fn=cmd_label)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
