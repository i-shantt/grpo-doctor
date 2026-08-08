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

Three families had doses that were provably inert. Each would otherwise have been written up as
*"GRPO resists this pathology"* — a claim about the algorithm — when the truth was a claim about our
knob.

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

## The corpus

386 runs over two tasks. A previous 579-run corpus completed cleanly (zero crashes, 266 min) and was
discarded when the probe defect was found — its `sort_digits` half is sound but the labels move under
the corrected probe, so mixing definitions was not an option.

| task | role | why |
|---|---|---|
| `sort_digits` | full | 625 distinct probe problems at d=4; the only source of positives |
| `countdown_lite` | full, `expects_collapse=False` | 0 collapses in 29 cells, but resilient rather than floored: 0.551 → 0.254 and back inside the H=50 window. Its failure cells are the hardest negatives available. |
| `ca_rule` | **excluded** | probe holds 1–4 distinct problems |
| `modarith` | **excluded** | F0 control STALLs on 1 of 2 seeds |

`countdown_lite`'s failure cells are kept deliberately. Dropping cells *because* they came out
negative would select the grid on its own outcomes, which `docs/NEGATIVE_RESULTS.md` pre-registered
against.

## What that implies

Genuine positives are roughly 23, all from `sort_digits`, concentrated in F5 and F3. That is thin for
leave-one-mode-out and it is exactly Outcome 2 in `docs/NEGATIVE_RESULTS.md`, which was written
before any of this ran. The honest headline remains the negative result, now better supported:
`countdown_lite` resists every knob in the taxonomy while dipping 0.30 and recovering, and three
families' silence was traced to arithmetic rather than left as a shrug.

The README must say plainly that two of four tasks were disqualified, and why — a probe too small to
measure anything is a mistake worth publishing, since it is invisible in every downstream number and
would have produced a confident, wrong result.

## Next

1. Rebuild warm starts under the corrected probe; record the real `measured_accuracy` per task.
2. Smoke the two-task grid, then regenerate (386 runs, ~2h).
3. Diagnose the remaining silent families (F4, F7, F9) the way F5/F6/F8 were — derive the effect on
   the advantage first, spend compute only where the arithmetic says a dose can work.
4. Fit and evaluate the detector ladder R0→R3 against the four negative controls under
   leave-one-mode-out, reporting lead time at a fixed 5% false-alarm rate.
