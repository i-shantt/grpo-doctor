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
- **360 tests**, `mypy --strict` clean, CI green on Python 3.10–3.13 and macOS.
- Probe false-accept audit passes: every leak fully exploitable, and `ca_rule`'s 5.6% residual is
  pinned to an exact identity (rotation is the identity precisely on constant rows) rather than to a
  tolerance someone chose.

## The open problem

Collapse-proneness turned out to be **task-specific**, and two of the four tasks are not currently
usable. Per-task smoke, 26 cells x 2 seeds:

| task | collapsing cells | control | verdict |
|---|---|---|---|
| `ca_rule` | **6** (F3/mu8_hot 2/2, 0.537 → 0.000) | healthy 2/2 | best vehicle |
| `sort_digits` | 2 reliable (3 seeds) | healthy 3/3 | usable |
| `countdown_lite` | **0 of 29 cells** | healthy | too weak to induce failure |
| `modarith` | — | **STALLs on 1 of 2 seeds** | unusable until fixed |

`countdown_lite`'s weakness is consistent with the probe audit: its leak only partially breaks the
task (30% of exploit outputs are valid solutions by coincidence), so there is less to lose.
`modarith` groks — it is non-monotone through the trainable band and its RL phase is seed-fragile.

Separately, on `sort_digits` only two of 26 conditions collapse GRPO at all, and that survived a
direct test: at 1500 steps rather than 600, six silent families stayed silent, so the
right-censoring explanation is falsified rather than merely unlikely.

## What that implies

If the pattern holds, the honest headline is the negative result — a testbed with 26 induced
pathologies and a labeler whose threshold is *derived* rather than chosen, finding that GRPO at a
sane learning rate resists most of them. That shape was pre-registered in `docs/NEGATIVE_RESULTS.md`
**before any of it ran**, which is the only reason it counts as a finding rather than a
rationalisation. Leave-one-mode-out would then be limited by having few modes, and the README has to
say so plainly.

## Next

1. Decide `modarith` and `countdown_lite`: fix, demote to negatives-only, or drop. Dropping is
   preferable to keeping a task whose control does not behave.
2. Rebalance the grid toward `ca_rule` and `sort_digits`.
3. Generate the corpus, then fit and evaluate the detector ladder R0→R3 against the four negative
   controls, reporting lead time at a fixed 5% false-alarm rate.
