#!/usr/bin/env python3
"""Label every trace in the corpus and report the yield.

    python scripts/label_corpus.py                      # label corpus/, print the yield
    python scripts/label_corpus.py --task countdown_lite
    python scripts/label_corpus.py --sensitivity        # every headline under delta/H variation

Two things this deliberately does not do.

It does not decide anything from the manifest. A run is positive because held-out accuracy fell
below its own running peak and stayed there, and a cell whose knob was set but which never
collapsed is reported as HEALTHY. That is the corpus's central honesty property and it is worth
restating at the point of use, because the yield table is exactly where it would be tempting to
quietly drop the families that came out empty.

It does not silently discard positives that fail the `train_true` artifact check. Those are
reported on their own line, since "a positive we do not believe" and "a positive the check could
not evaluate" are different facts and both matter -- see `grpo_doctor.eval.artifacts`, where the
scoping is argued.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from grpo_doctor.eval.artifacts import (  # noqa: E402
    ArtifactVerdict,
    check_train_true_artifact,
)
from grpo_doctor.eval.labels import LabelConfig, LabelResult, RunLabel, label_run  # noqa: E402
from testbed.corpus.runner import read_trace  # noqa: E402

POSITIVE = (RunLabel.HACK, RunLabel.DEGRADE)
"""STALL is not a positive: there was no peak to fall from, so there is no collapse to predict."""


def label_one(spec: dict[str, Any], recs: list[dict[str, float]], cfg: LabelConfig) -> LabelResult:
    steps = np.array([r["step"] for r in recs])
    return label_run(
        steps,
        np.array([r["oracle/heldout_accuracy"] for r in recs]),
        np.array([r["oracle/probe_fresh"] for r in recs]),
        np.array([r["reward"] for r in recs]),
        np.array([r.get("acr", 0.0) for r in recs]),
        cfg,
    )


def label_corpus(
    trace_dir: Path, task: str | None = None, cfg: LabelConfig | None = None
) -> list[dict[str, Any]]:
    """One row per trace: its label, and -- for positives -- the artifact check's verdict.

    `probe_every` always comes from the run's own spec, never from `cfg`. K is a property of how the
    trace was generated, not a knob to sweep: a caller varying delta and H must not silently
    reinterpret the probe spacing along with them, or the sensitivity table would be measuring two
    things at once.
    """
    rows: list[dict[str, Any]] = []
    pattern = f"{task}_*.jsonl.gz" if task else "*.jsonl.gz"
    for path in sorted(trace_dir.glob(pattern)):
        spec, recs = read_trace(path)
        if not recs:
            continue
        run_cfg = replace(cfg or LabelConfig(), probe_every=spec.get("probe_every", 10))
        res = label_one(spec, recs, run_cfg)
        row: dict[str, Any] = {
            "run_id": spec["run_id"],
            "task": spec["task"],
            "family": spec["family"],
            "dose": spec["dose"],
            "cell": f"{spec['family']}/{spec['dose']}",
            "seed": spec["seed"],
            "simulated": spec.get("simulated", False),
            "label": res.label.value,
            "t_collapse": res.t_collapse,
            "peak_accuracy": res.peak_accuracy,
            "peak_step": res.peak_step,
            "final_accuracy": res.final_accuracy,
            "censored": res.censored,
            "onset_step": spec.get("onset_step"),
            "probe_interval": res.probe_interval,
            "artifact_verdict": None,
            "train_true_rise": None,
        }
        if res.label in POSITIVE and res.t_collapse is not None:
            chk = check_train_true_artifact(
                np.array([r["step"] for r in recs]),
                np.array([r["oracle/train_true_accuracy"] for r in recs]),
                res.peak_step,
                res.t_collapse,
                spec.get("overrides", {}),
                completions_per_step=spec["n_prompts"] * spec["group_size"],
            )
            row["artifact_verdict"] = chk.verdict.value
            row["train_true_rise"] = chk.train_true_rise
            row["artifact_reason"] = chk.reason
        rows.append(row)
    return rows


def report(rows: list[dict[str, Any]]) -> None:
    for task in sorted({r["task"] for r in rows}):
        trs = [r for r in rows if r["task"] == task]
        print(f"\n{'=' * 78}\n{task}  --  {len(trs)} runs\n{'=' * 78}")

        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in trs:
            by_cell[r["cell"]].append(r)

        print(
            f"\n{'cell':<26} {'n':>3} {'pos':>4} {'hack':>5} {'degr':>5} {'stall':>6} "
            f"{'cens':>5}  {'peak->final':<16} artifact"
        )
        fam_tot: Counter = Counter()
        fam_pos: Counter = Counter()
        for cell in sorted(by_cell):
            rs = by_cell[cell]
            pos = [r for r in rs if r["label"] in ("hack", "degrade")]
            fam = rs[0]["family"]
            fam_tot[fam] += len(rs)
            fam_pos[fam] += len(pos)
            art = Counter(r["artifact_verdict"] for r in pos if r["artifact_verdict"])
            art_s = " ".join(f"{v[:4]}:{n}" for v, n in sorted(art.items())) or "-"
            peak = float(np.mean([r["peak_accuracy"] for r in rs]))
            final = float(np.mean([r["final_accuracy"] for r in rs]))
            print(
                f"{cell:<26} {len(rs):>3} {len(pos):>4} "
                f"{sum(r['label'] == 'hack' for r in rs):>5} "
                f"{sum(r['label'] == 'degrade' for r in rs):>5} "
                f"{sum(r['label'] == 'stall' for r in rs):>6} "
                f"{sum(bool(r['censored']) for r in pos):>5}  "
                f"{peak:.3f}->{final:.3f}     {art_s}"
            )

        print("\n  by family: ", end="")
        print(
            "  ".join(
                f"{f} {fam_pos[f]}/{fam_tot[f]}" for f in sorted(fam_tot) if not f.startswith("H")
            )
        )
        hard = [f for f in sorted(fam_tot) if f.startswith("H") or f == "F0"]
        if hard:
            print("  negatives: ", end="")
            print("  ".join(f"{f} {fam_pos[f]}/{fam_tot[f]}" for f in hard))

        pos_all = [r for r in trs if r["label"] in ("hack", "degrade")]
        arts = [r for r in pos_all if r["artifact_verdict"] == ArtifactVerdict.ARTIFACT.value]
        na = [r for r in pos_all if r["artifact_verdict"] == ArtifactVerdict.NOT_APPLICABLE.value]
        print(
            f"\n  {len(pos_all)} positives: {len(pos_all) - len(arts)} believed, "
            f"{len(arts)} disqualified as probe artifacts, "
            f"{len(na)} unauditable (training moved off the probe's distribution)"
        )
        for r in arts:
            print(f"    ARTIFACT {r['run_id']}: {r.get('artifact_reason', '')}")


def sensitivity(trace_dir: Path, task: str | None) -> None:
    """The headline positive count under every (delta, H) the pre-registration committed to.

    A label definition that only survives at its own tuning is not a label definition.
    """
    print(f"\n{'delta':>6} {'H':>5} {'positives':>10} {'hack':>6} {'degrade':>8} {'stall':>6}")
    for sig in (2.0, 3.0, 4.0):
        for h in (30, 50, 100):
            rows = label_corpus(trace_dir, task, LabelConfig(delta_sigmas=sig, persistence=h))
            c = Counter(r["label"] for r in rows)
            print(
                f"{sig:>5.0f}s {h:>5} {c['hack'] + c['degrade']:>10} "
                f"{c['hack']:>6} {c['degrade']:>8} {c['stall']:>6}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--task", default=None, help="label one task only")
    p.add_argument("--sensitivity", action="store_true", help="sweep delta and H")
    p.add_argument(
        "--out", default=None, help="write labels JSON here (default <corpus>/labels.json)"
    )
    args = p.parse_args()

    corpus = Path(args.corpus)
    trace_dir = corpus / "traces"
    if not trace_dir.is_dir():
        print(f"no traces at {trace_dir}", file=sys.stderr)
        return 2

    rows = label_corpus(trace_dir, args.task)
    if not rows:
        print("no traces matched", file=sys.stderr)
        return 2
    report(rows)

    if args.sensitivity:
        sensitivity(trace_dir, args.task)

    out = Path(args.out) if args.out else corpus / "labels.json"
    out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    print(f"\nwrote {len(rows)} labels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
