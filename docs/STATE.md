# Current state

Living document. What is built, what is measured, and what the open problem is. Updated as the work
moves; it is not a changelog.

## Built and green

- **Testbed** (`testbed/`) — TinyGPT with KV cache, from-scratch GRPO verified against TRL's closed
  forms for all three `loss_type` values crossed with all three `scale_rewards` values, AdamW that
  separates the logged gradient norm from the applied one, four verifiable tasks with a leak ladder,
  the F0–F9 failure grid, a resumable multiprocess corpus runner.
- **Package** (`src/grpo_doctor/`) — `StepRecord`, `Monitor`, signal panel, the `t_collapse` labeler,
  evaluation metrics with run-level cluster bootstrap and exact McNemar, a TRL `TrainerCallback`,
  and a `replay`/`label` CLI. **numpy-only**; CI asserts `torch` never enters `sys.modules`.
- **377 tests**, `mypy --strict` clean, CI green on Python 3.10–3.13 and macOS.

## The probe was measuring two problems

The largest finding of the corpus work, and it invalidated a task that had been reported as the best
one. `t_collapse` sets its threshold at `δ = 3·SE` with `SE = √(p̂(1−p̂)/N)`, `N = 256` — which
assumes **256 independent held-out problems**. The held-out split is a hash bucket, 1 prompt in 16.
Nobody had checked how many prompts that leaves.

| task | difficulty | problem space | **distinct probe problems** |
|---|---|---|---|
| `ca_rule` | 3 / 4 / 5 / 6 | 8 / 16 / 32 / 64 | **1 / 1 / 2 / 4** |
| `sort_digits` | 4 | 10,000 | 625 |
| `countdown_lite` | 4 | 6,561 | 410 |

`ca_rule` has a binary alphabet, so width *w* admits only 2^w rows. Its probe drew 256 samples from
**two** distinct problems. Forgetting one moves measured accuracy by 0.5 — five times the collapse
threshold — so ordinary drift on two instances registered as a collapse.

Measured consequences, all from the finished 579-run corpus:

- **3 of 54 F0 controls labeled positive**, with no leak and no knob applied. A 5% false-alarm
  operating point is incoherent when the ground truth fires at 5.6%.
- **29 of 39 `ca_rule` positives were artifacts** — strict-verifier accuracy on the *training*
  distribution was rising while the probe fell, meaning the policy had genuinely improved. A hacking
  policy would fail `verify_true` on training problems too, so rising `train_true` rules hacking out.
- A per-difficulty profile of 0.000 / 0.010 / 0.245 / 0.021 — a spike at exactly the probed width,
  not a difficulty gradient. The warm start halts on the first crossing of the probe it is later
  labeled by, so it selects checkpoints caught at a favourable fluctuation on the labeling oracle
  itself. RL then regresses to the mean, and the probe calls it collapse.

Pinning `ca_rule` to a single width does not help — two problems is still two problems, and controls
still collapsed 2 of 4. **The task is excluded.** It remains implemented and is kept as the
regression case for the guard below.

### Three guards, each of which would have caught it

1. Every probed difficulty must supply at least as many distinct held-out problems as samples drawn
   from it. `ca_rule` fails by construction and `probe_difficulties` raises rather than degrading.
2. **Training range must equal probed range exactly.** The old test only required the probe point to
   be *inside* the training range, which is what allowed a policy to specialise where nobody was
   measuring.
3. The probe budget is spent in full, so narrowing never silently shrinks `N` and the noise floor
   behind `δ` stays honest.

Guard 2 is why `sort_digits` now trains on 4–6 rather than 2–6 and `countdown_lite` on 4–5:
difficulties 2 and 3 are trainable but not *labelable*, holding 6 and 62 distinct held-out prompts.

## Families that could not have fired, for reasons in the algebra

Six families had doses that were provably inert. Each would otherwise have been written up as
*"GRPO resists this pathology"* — a claim about the algorithm — when the truth was a claim about our
knob. Only F5's has been fixed so far; F4, F7 and F9 are derived and not yet re-dosed.

**F5 (verifier leakage).** Every leak accepts a *superset* of the correct answers, so a correct
completion still scores 1.0 on a leaked problem: `E[always correct] = 1.0` against
`E[always exploit] = leak_p`. The exploit is dominated at every `leak_p < 1` and ties at 1, which is
the zero-variance freeze. Measured: `structure_p40`, `structure_p70`, `format_p40` all healthy 2/2
while reward climbed 0.51 → 0.91.

*Fixed.* A negative length bonus breaks the tie, since the exploit is one token and the answer is
not, and it restores reward variance at `leak_p = 1`. `structure_full_terse` is now HACK 2/2 on
sort_digits: held-out accuracy 0.41 → **0.000**, mean length 4.46 → 2.00, training reward **rising**
0.512 → 0.923. The unshaped doses are kept as honest negatives.

**F6 (length hacking) and H5.** Group standardization is invariant to positive affine transforms of
the reward: `(c·x − mean)/std = (x − mean)/std`. Under a strict verifier a correct completion *is*
the answer, so every correct completion in a group has the same length and the same bonus — the term
cancels exactly. Verified: advantages agree to five decimals across bonuses of ±0.05 and +0.5. **H5
is a duplicate of F0**, which matters because FAR broken out by hard-negative type is a headline
commitment and it has three real types, not four.

**F8 (normalization).** For *binary* rewards, max |A| is 0.875 under `scale_rewards="none"` against
2.474 under `"group"` — removing per-group normalization makes updates **gentler**. The
"|A| is unbounded" claim holds only for unbounded rewards, which binary verification never supplies.

Both are pinned by property tests in `tests/test_advantage_bound.py`.

**F7 (reward noise).** Symmetric label noise is affine in expectation, `E[r'] = r(1−2p) + p`, so it
scales the learning signal by exactly `(1−2p)`; measured correlation between noisy and clean
advantages matches to three decimals. The grid's doses of 0.10 and 0.25 retain 80% and 50% of the
signal — a slowdown, not a pathology. A dose that could bite needs p ≈ 0.40–0.45.

**F4 (entropy collapse).** Configured to do the opposite of its name. The clipped surrogate caps
policy change in *both* directions, so narrowing epsilon shrinks the trust region and **preserves**
entropy by slowing the policy's sharpening. With `num_iterations=2`, inner iteration 0 has
`ratio ≡ 1`, so only half the iterations can clip at all. The knob that actually targets entropy is
`entropy_coef`, entering as `loss − coef·H`: a **negative** coefficient minimizes entropy directly.

**F9 (sampler/trainer mismatch).** The testbed supplies exactly the correction the real pathology
omits. `generate()` scores each sampled token under the *noised* distribution it was actually drawn
from (`rollout.py:92`, and the comment there says so deliberately), so `old_logprobs = log π_b`, the
true behavior policy. The GRPO ratio is then `π_θ/π_b` — a correctly importance-weighted off-policy
estimator, unbiased before clipping. That is a legitimate algorithm, so it degrades gracefully.

The real train/inference gap is the *absence* of that weight: TRL computes `old_per_token_logps`
from the trainer's own forward pass, so `old_logps = log π_θ`, the ratio is 1 at inner iteration 0,
and the vLLM↔HF discrepancy is never corrected. arXiv 2602.01103's mechanism is the missing
correction.

Measured, and it matches: the log-ratio bias is `E_{a∼π_b}[log π_θ(a) − log π_b(a)] = −KL(π_b‖π_θ)`,
strictly negative, so the ratio sits below 1 and clipping is one-sided. On sort_digits, `ratio_mean`
0.99992 (F0) → 0.99975 (σ=0.25) → 0.99878 (σ=0.75), `ratio_max` 1.8 → 2.9 → 13–33, and
`clip_low` 0.000 → 0.005 → 0.023 against `clip_high` roughly a third of that. So F9 produces a loud,
correctly-shaped *signal* and no collapse at all: reward still rises 0.285 → 0.456 against F0's
0.285 → 0.439, and held-out accuracy does not move.

*The fix, not yet applied:* score the sampled tokens under the **clean** logits while sampling from
the noised ones. Then `ratio ≡ 1` at iteration 0 exactly as in TRL, the correction is omitted, and
the gradient carries the bias. That version is also the harder detection problem — the clip metrics
look normal while the estimator is quietly wrong — which is what makes the real thing dangerous. The
existing noise doses stay as honest negatives, as F5's unshaped leaks did.

## The corpus

386 runs over two tasks. A previous 579-run corpus completed cleanly (zero crashes, 266 min) and was
discarded when the probe defect was found — its `sort_digits` half is sound but the labels move under
the corrected probe, so mixing definitions was not an option.

| task | role | why |
|---|---|---|
| `sort_digits` | full | 625 distinct probe problems at d=4; the only source of positives |
| `countdown_lite` | full | 2 collapses in 193 runs, both F5. Resilient rather than immune, and its 38 other failure cells are the hardest negatives available. |
| `ca_rule` | **excluded** | probe holds 1–4 distinct problems |
| `modarith` | **excluded** | F0 control STALLs on 1 of 2 seeds |

`countdown_lite`'s failure cells are kept deliberately. Dropping cells *because* they came out
negative would select the grid on its own outcomes, which `docs/NEGATIVE_RESULTS.md` pre-registered
against.

## The yield, now that the grid is finished

386 runs, 169 generated in the second sitting and 217 resumed, zero crashes, 81 minutes at 5 niced
workers. `scripts/label_corpus.py` produces the table below and writes `corpus/labels.json`.

| task | positives | by family |
|---|---|---|
| `sort_digits` | **29 / 193** | F5 18/40, F2 6/10, F3 4/15, F1 1/15 |
| `countdown_lite` | **2 / 193** | F5 2/40 |
| both | **0 / 36** F0 controls | every hard-negative type 0/20 |

Silent on both tasks: F4, F6, F7, F8, F9, and H2–H5.

None of the 31 positives is a probe artifact. Seven are unauditable rather than clean — all F1/F2,
which move training off the probe's distribution by construction, so the check reports
`NOT_APPLICABLE` rather than pretending to have cleared them (`grpo_doctor.eval.artifacts`).

**The label definition does not depend on its own tuning.** Positives across the pre-registered
grid: 45 / 45 / 43 at δ=2·SE, 31 / 31 / 31 at 3·SE, 26 / 26 / 26 at 4·SE, for H = 30 / 50 / 100.
Monotone and gentle in δ, and almost exactly flat in H — these collapses persist, so widening the
confirmation window from 30 steps to 100 changes essentially nothing.

## What that implies

31 positives over 386 runs, 29 of them from one task and 20 of those from one family. That is thin
for leave-one-mode-out and it is exactly Outcome 2 in `docs/NEGATIVE_RESULTS.md`, which was written
before any of this ran.

The headline is still the negative result, but one clause of it has to change. `countdown_lite` does
**not** resist every knob in the taxonomy: `F5/prefix` and `F5/format_p70_terse` each collapsed one
seed. Two of 40 is resilience, not immunity, and the shaped-leak cells that did it were added after
the four measurements the earlier claim rested on. What survives intact is the stronger half — five
families' silence was traced to arithmetic rather than left as a shrug, and a task that dips 0.30 and
recovers inside the window supplies the hardest negatives in the corpus.

The README must say plainly that two of four tasks were disqualified, and why — a probe too small to
measure anything is a mistake worth publishing, since it is invisible in every downstream number and
would have produced a confident, wrong result.

## Next

1. **Re-dose F4, F7 and F9** from the derivations above — `entropy_coef` negative rather than a
   narrower epsilon, `flip_p` around 0.40–0.45, and an uncorrected sampler gap. Keep the current
   doses as negatives. Smoke gate first; each is a grid change.
2. **Phase 3.** Fit the detector ladder R0→R3, evaluate under leave-one-mode-out against the four
   negative controls, report lead time at a fixed 5% false-alarm rate with run-level cluster
   bootstrap. `eval/metrics.py` is written and tested; `eval/splits.py`, `controls.py` and
   `ablation.py` are not written yet.
3. **Look at the controls before anything else.** If step-index-only matches the real monitor the
   corpus is time-confounded and Phase 4 is a rebuild, not polish.
4. Rewrite the README around the probe finding and the two disqualified tasks.
