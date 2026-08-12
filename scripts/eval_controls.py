"""The four negative controls over the finished corpus, before any detector is fit.

This runs first on purpose. Three of the four controls need no model at all, so their numbers are
available now, and one of them decides whether the rest of the study is worth running:

    if step-index-only detects collapses, the corpus is time-confounded

-- meaning collapses happen at predictable points in a run and *any* score that rises with time will
appear to work. Onsets are randomized in [50, 250] specifically to prevent that, but the check is
cheap and the failure is invisible downstream, so it is made before a detector exists to be flattered
by it.

Thresholds are solved on the even-seeded negatives and every rate is reported on the odd-seeded
ones, so the false-alarm column is a measurement rather than a restatement of the 5% target. See
`eval/splits.py` for why that partition is on seed parity.

    python3 scripts/eval_controls.py --corpus corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grpo_doctor.eval.controls import CONTROLS, ShuffledLabels, is_degenerate
from grpo_doctor.eval.corpus import LoadedRun, load_corpus, outcome_for
from grpo_doctor.eval.metrics import (
    RunOutcome,
    detection_rate,
    false_alarm_rate,
    far_by_family,
    lead_time_report,
)
from grpo_doctor.eval.splits import _partition_negatives

TARGET_FAR = 0.05


def evaluate(
    name: str,
    runs: list[LoadedRun],
    calib_ids: set[str],
    eval_neg_ids: set[str],
) -> tuple[str, float, list[RunOutcome], list[RunOutcome], bool]:
    """Calibrate on the calib negatives, then score everything at that one threshold."""
    ctl = CONTROLS[name]
    scores = {r.run_id: ctl.score(r.records) for r in runs}
    thr = ctl.threshold_for([scores[i] for i in sorted(calib_ids)], TARGET_FAR)
    outcomes = {r.run_id: outcome_for(r, scores[r.run_id], thr) for r in runs}
    positives = [outcomes[r.run_id] for r in runs if r.collapsed]
    negatives = [outcomes[i] for i in sorted(eval_neg_ids)]
    degenerate = ctl.calibrated and is_degenerate([scores[i] for i in sorted(calib_ids)])
    return name, thr, positives, negatives, degenerate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus")
    args = ap.parse_args()

    runs = load_corpus(Path(args.corpus))
    negatives = [r for r in runs if r.ref.is_negative_family]
    calib_ids, eval_neg_ids = _partition_negatives([r.ref for r in negatives])
    calib, eval_neg = set(calib_ids), set(eval_neg_ids)
    n_pos = sum(r.collapsed for r in runs)

    print(f"{len(runs)} runs: {n_pos} positive, {len(negatives)} in negative families")
    print(f"  calibrate on {len(calib)} (even seeds), report FAR on {len(eval_neg)} (odd seeds)")
    print(f"  {sum(r.simulated for r in runs)} simulated runs (F9), all labeled healthy\n")

    hdr = f"{'control':<18} {'threshold':>10} {'FAR':>7} {'detect':>7} {'in time':>8} {'lead':>7}"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    degenerate = []
    for name in ("step_index_only", "reward_only", "constant_alarm"):
        name, thr, pos, neg, is_vacuous = evaluate(name, runs, calib, eval_neg)
        results[name] = (pos, neg)
        rep = lead_time_report(pos)
        lead = "n/a" if rep.n_fired_in_time == 0 else f"{rep.median_lead:.0f}"
        t = "-inf" if thr == float("-inf") else f"{thr:10.4f}"
        mark = "  <- vacuous, see below" if is_vacuous else ""
        if is_vacuous:
            degenerate.append(name)
        print(
            f"{name:<18} {t:>10} {false_alarm_rate(neg):7.3f} {detection_rate(pos):7.3f} "
            f"{rep.p_fired_in_time:8.3f} {lead:>7}{mark}"
        )

    # The null. Not a scorer: it takes the best real arm's alarms and destroys only the
    # correspondence between a run's alarm and that run's ground truth.
    best = "reward_only"
    pos, neg = results[best]
    shuffled = ShuffledLabels(seed=0).apply(pos + neg)
    s_pos = [o for o in shuffled if o.collapsed]
    rep = lead_time_report(s_pos)
    lead = "n/a" if rep.n_fired_in_time == 0 else f"{rep.median_lead:.0f}"
    print(
        f"{'shuffled_label':<18} {'(' + best + ')':>10} {'-':>7} {detection_rate(s_pos):7.3f} "
        f"{rep.p_fired_in_time:8.3f} {lead:>7}"
    )

    print("\nfalse alarms by negative family, at each control's own threshold")
    for name, (_, neg) in results.items():
        by_fam = far_by_family(neg)
        print(f"  {name:<18} " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(by_fam.items())))

    if degenerate:
        print(
            f"\nvacuous: {', '.join(degenerate)} -- every run's score has the same maximum, so no\n"
            "threshold separates them and only false-alarm rates of 0.0 and 1.0 are reachable.\n"
            "Their rates above are what calibration does with an unusable score, not measurements."
        )

    # The time-confound question, asked in the way this corpus can actually answer it. Every run is
    # the same length by design, so a time-only score is degenerate and cannot be compared at a
    # matched false-alarm rate. What remains is whether collapse times are predictable at all.
    tc = sorted(o.t_collapse for o in results["reward_only"][0] if o.t_collapse is not None)
    q1, med, q3 = (tc[len(tc) // 4], tc[len(tc) // 2], tc[3 * len(tc) // 4])
    print(
        f"\ntime confound: t_collapse spans {tc[0]}-{tc[-1]}, median {med} [IQR {q1}-{q3}] over "
        f"{len(tc)} positives.\nEvery run is 600 steps, so a fixed-step alarm fires on 100% of "
        "negatives at any threshold that\ndetects anything -- a time-only rule buys detection only "
        "at FAR 1.0, and cannot be confounded\nwith a real detector at the 5% operating point."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
