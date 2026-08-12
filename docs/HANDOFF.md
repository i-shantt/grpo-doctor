# Handoff — paste this into a fresh session

Continuing work on grpo-doctor (`~/rlProject`, github.com/i-shantt/grpo-doctor).
Read `docs/STATE.md` and the plan at `~/.claude/plans/i-have-a-bunch-toasty-crane.md` first.
`docs/NEGATIVE_RESULTS.md` is a pre-registration written before any of this ran — do not retrofit it.

## Status

Phase 2. **380 tests, `mypy --strict` clean.** Branch `feat/rebalance-grid`, 11 commits,
**4 unpushed**. PR #2 is open but **its description is stale** — it describes `ca_rule` at double
weight as the corpus's main source of positives, and `ca_rule` has since been excluded entirely.

**Corpus is 217 of 386 runs complete**, paused mid-generation. All traces verified intact.
`sort_digits` (193) is finished and labeled; `countdown_lite` has 24 of 193.

Resume with:

```
nice -n 15 python3 scripts/build_corpus.py --full --workers 5 --out corpus
```

~1.5h left. Finished runs skip safely — `runner.is_stale` compares each existing trace's stored spec
against the requested one, so a config change forces regeneration rather than silent reuse.

## Results already in hand (sort_digits, 193 runs)

- **F0 control: 0 of 18 positive** (was 3 of 54 before the probe fix). The false-alarm ground truth
  is coherent now.
- **29 positives across 8 cells and 4 families**: F5 18/40, F2 6/10, F3 4/15, F1 1/15. None are
  artifacts.
- Silent: F4, F6, F7, F8, F9, H2–H5.

## Do these next, in order

1. **Finish the corpus** (command above), then label `countdown_lite` and report the yield.
2. **Scope the `train_true` artifact check** to families that do not override `difficulty_range`
   *before* it goes into analysis code. It currently false-alarms on all 6 F2 positives: F2 sets
   training difficulty to (3,3) or (3,4), below the probed 4–6, so `train_true` rises with the easier
   training mix while the fixed probe correctly reports lost competence. For F1/F2 that divergence is
   the mechanism, not a defect.
3. **Push and rewrite the PR #2 body** around the probe finding — ask first, it is outward-facing.
4. **F4/F7/F9 knob tuning.** Two derivations are already done (see below); do not repeat them.
5. **Phase 3**: fit the detector ladder R0→R3, evaluate under leave-one-mode-out against the four
   negative controls, report lead time at a fixed 5% false-alarm rate with run-level cluster
   bootstrap. `eval/metrics.py` is written and tested.

## Hard-won context — do not rediscover these

**The probe was measuring two problems.** `t_collapse` sets `δ = 3·SE` assuming N=256 *independent*
held-out problems, but the split is a hash bucket (1 in 16) and nobody checked what that leaves.
`ca_rule` has a binary alphabet, so width *w* gives only 2^w rows — its probe drew 256 samples from
**1–4 distinct problems**. Consequences: 3 of 54 F0 controls labeled positive with no knob applied,
and 29 of 39 of its positives were policies that had genuinely *improved*. `ca_rule` is excluded.
Three guards now exist (probe pool ≥ samples drawn; training range **equals** probed range; probe
budget spent in full). `modarith` is excluded separately — its F0 control STALLs.

**Only two tasks remain: `sort_digits` and `countdown_lite`.** Both train on the narrowed ranges
(4–6 and 4–5) that they are actually probed on. Narrowing *helped* — sort_digits warm-starts 18/18
into band at a median 400 steps versus 600 before.

**Four failure families had doses that could never have worked**, all found by algebra rather than
sweeps:
- **F5** — leaks accept a *superset* of correct answers, so the exploit is weakly dominated at every
  `leak_p` and ties at 1 (the zero-variance freeze). **Fixed** with shaped doses: a negative length
  bonus makes the one-token exploit strictly outscore a correct answer. `structure_full_terse` is
  HACK 2/2 with held-out accuracy → **0.000** while training reward *rises* 0.512 → 0.923. The
  unshaped doses are kept as honest negatives.
- **F6 and H5** — group standardization is invariant to positive affine transforms, and under a
  strict verifier every correct completion in a group has the same length, so the bonus cancels
  exactly. H5 is a duplicate of F0, which matters because FAR-by-hard-negative-type is a headline
  commitment and it has three real types, not four.
- **F8** — for binary rewards, max |A| is 0.875 under `scale_rewards="none"` against 2.474 under
  `"group"`. Removing normalization makes updates *gentler*.
- **F7** — symmetric label noise scales the signal by exactly `(1−2p)`; measured correlation matches
  to three decimals. Doses of 0.10/0.25 retain 80%/50% of signal — a slowdown, not a pathology. A
  dose that could bite needs p ≈ 0.40–0.45.
- **F4** is configured to do the opposite of its name: narrowing the clip epsilon shrinks the trust
  region, which *preserves* entropy by slowing the policy's sharpening, and with `num_iterations=2`
  inner iteration 0 has `ratio ≡ 1` so only half the iterations clip at all. The knob that actually
  targets entropy is `entropy_coef`, entering as `loss − coef·H` — a **negative** coefficient
  minimizes entropy directly.

**GPU will not help.** Measured: 8 CPU workers ≈ 16.7 steps/s aggregate against MPS single-process
3.1. The model is 3M params with ~832 tokens per step — launch-overhead bound, not FLOP bound.

**A knob that fires is not a collapse.** Runs where a knob was set and nothing happened are
negatives, and `countdown_lite` is carried at full breadth for exactly that reason. Dropping cells
because they came out negative would select the corpus on its own outcomes.

## Working preferences

- **Ask before Kaggle or any GPU compute** — another session shares the quota.
- **Say explicitly what needs the user and what does not.**
- **Always `--smoke` before a full grid.** It has caught nine real defects.
- Run long jobs niced (`nice -n 15 --workers 5`); they are resumable, so they can be killed freely.
- Commit in small atomic commits with real reasoning in the message; merge commits, never squash.
- `gh` 2.97 is installed at `~/.local/bin/gh` and authenticated as `i-shantt`.
- The project memory directory now has entries; there were none before 2026-08-08.
