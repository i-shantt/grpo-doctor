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
- **371 tests**, `mypy --strict` clean, CI green on Python 3.10–3.13 and macOS.
- Probe false-accept audit passes: every leak fully exploitable, and `ca_rule`'s 5.6% residual is
  pinned to an exact identity (rotation is the identity precisely on constant rows) rather than to a
  tolerance someone chose.

## The corpus grid, and why it has the shape it has

772 runs over three tasks. Every decision below is a measurement, and each one is recorded next to
the thing it decided (`TaskProfile.role_reason`, the F5 comment block in `inject/failures.py`) so it
can be re-examined rather than taken on faith.

| task | role | runs | why |
|---|---|---|---|
| `ca_rule` | full, **×2 seeds** | 386 | 6 collapsing cells of 26 against a healthy control. The only reliable source of positives, and lead time is undefined without them. |
| `sort_digits` | full | 193 | 2 reliably collapsing cells, healthy control 3/3. |
| `countdown_lite` | full | 193 | **0 collapses in 29 cells** — carried anyway, see below. |
| `modarith` | **excluded** | 0 | Its F0 control STALLs on 1 of 2 seeds. |

**`modarith` is out.** A broken *control* is the one defect no downstream analysis absorbs: there is
no baseline to measure a failure against and no trustworthy negatives to calibrate a false-alarm
rate on. The cause is grokking — warm start lands both seeds at the very bottom of the band (0.258,
0.262 against a 0.25 floor) after 10–11k supervised steps, because the accuracy curve crosses the
whole band almost vertically and the stopping rule catches it in transit. The task stays implemented
and tested; `make_grid(roles=False)` still runs it, so the judgement is re-checkable.

What that costs is stated rather than waved away: modarith was carried for a reward hack that
collapses completion length, so the length baseline now has a thinner set of positives to be right
about. A claim in `modarith.py`'s docstring — that sort_digits' leak leaves length untouched while
modarith's collapses it — turned out to be **false for the doses actually in the grid**. The
`6.95 → 2.00` length collapse cited in `tasks/base.py` was measured under the *full* leak, which now
freezes training instead of hacking it.

**`countdown_lite` stays at full breadth despite contributing no positives**, and that is the
interesting decision. Re-running the three most violent knobs in the taxonomy showed it is
*resilient*, not floored: `F8/none_unclipped` fell 0.551 → 0.254 and recovered, `F3/mu8_hot` fell
0.547 → 0.352 and recovered, both inside the H=50 persistence window. A 0.30 drawdown that recovers
is three times the labeler's 0.094 noise floor with every cheap signal screaming through it — the
hardest negative in the corpus. Dropping its failure cells *because* they came out negative would
select the grid on its own outcomes and delete exactly the runs that are hardest to get right.

## F5 did not Goodhart, and it was arithmetic rather than a weak dose

The reward-hacking family is the project's headline — the mode a reward-watching practitioner is
structurally blind to. Measured across both live tasks at two seeds, three of its four doses never
fired: `structure_p40`, `structure_p70` and `format_p40` were **healthy 2/2 on sort_digits** while
training reward climbed 0.51 → 0.91, held-out accuracy stayed flat, and mean length did not move.

That is not tuning slack. Every leak in the ladder accepts a **superset** of the correct answers, so
a correct completion still scores 1.0 on a leaked problem and abandoning correctness can only lose:

```
E[always correct] = 1.0        E[always exploit] = leak_p
```

The exploit is dominated at every `leak_p < 1`, and at `leak_p = 1` the two tie — which is the
zero-variance freeze `VerifierConfig.leak_p` already documents. **No dose of a bare superset leak
produces Goodhart**, so pushing `leak_p` harder was never going to find one.

A negative length bonus breaks the tie, because the exploit is one token and the answer is not:

```
correct ~ 1 + bonus·len(answer)      exploit ~ 1 + bonus·1
```

The shaped doses are now in the grid (`structure_p70_terse`, `structure_full_terse`,
`format_p70_terse`), and they produce the signature the whole project exists to catch. On
sort_digits, `structure_full_terse` is HACK 2/2: held-out accuracy 0.41 → **0.000**, mean length
4.46 → 2.00, and training reward **rising** 0.512 → 0.923. On ca_rule the strongest cell is
`format_p70_terse` (3 of 4 seeds to 0.000–0.010), because ca_rule's STRUCTURE leak grades a
population count and so cannot be exploited by a short answer — only FORMAT frees the length.

The bonus also restores reward variance at `leak_p = 1`, so the full-leak case now hacks instead of
freezing. The four unshaped doses are **kept**: they are honest negatives with a mechanism behind
them, and they are the corpus's cleanest example of a knob that fires and changes nothing.

One mechanism note worth keeping, because it is not obvious: `leaks_for` hashes the prompt, so the
policy cannot tell a leaked problem from a strictly graded one. An exploit that pays on 70% of
problems is therefore generalized to all of them. **A partial leak does not produce partial
hacking.**

## The open problem

**Warm starts do not reliably reach the trainable band, and `ca_rule` is the worst offender.**
Measured so far: sort_digits 15/15 in band at a median 500 supervised steps, countdown_lite 2/2 at
500, `ca_rule` **1 of 2** on the first pair — seed 0 spent its entire 12000-step ceiling and finished
at **0.055** against a 0.25 floor.

That is not a cosmetic miss. Every ca_rule seed-0 run in the F5 probe peaked at 0.09–0.24, so the
labeler could never have seen a meaningful drawdown: the cell becomes a negative for a reason that
has nothing to do with the knob, which is precisely the initialization confound `TARGET_BAND` exists
to remove. It is also the same failure that disqualified modarith, one notch less severe, on the
task carrying half the corpus.

`scripts/build_warmstarts.py` builds all 72 of them in parallel and reports the per-task hit rate
before the grid runs — both because it is shared work (a warm start is keyed by task/difficulty/seed
and all 32 of a seed's cells load the same checkpoint, so on demand six workers train it six times
and discard five) and because the hit rate is a property of the corpus worth knowing in advance.

## What that implies

Unchanged, and now better supported: if the pattern holds, the honest headline is the negative
result — a testbed with 29 induced pathologies and a labeler whose threshold is *derived* rather
than chosen, finding that GRPO at a sane learning rate resists most of them. That shape was
pre-registered in `docs/NEGATIVE_RESULTS.md` **before any of it ran**. Two things sharpen it now:
`countdown_lite` resists every knob in the taxonomy while dipping 0.30 and recovering, and F5's
failure to fire was traced to an argument rather than left as a shrug.

Separately, on `sort_digits` only two of 26 conditions collapsed GRPO at all, and that survived a
direct test: at 1500 steps rather than 600, six silent families stayed silent, so the
right-censoring explanation is falsified rather than merely unlikely.

## Next

1. Finish the warm-start pass; record which seeds land in band and exclude the ones that do not.
2. Smoke the rebalanced grid, then generate it. Measured cost: 93 s/run at 600 steps and ~130 s for
   the heavier cells, so 772 runs on 6 workers is **3–4h**, not the 2h previously assumed.
3. Fit and evaluate the detector ladder R0→R3 against the four negative controls under
   leave-one-mode-out, reporting lead time at a fixed 5% false-alarm rate.
