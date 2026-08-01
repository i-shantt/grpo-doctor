# Pre-registration: what a negative result looks like

**Written and committed before any corpus was generated or any detector was fit.**
Check `git log` on this file against the corpus and results commits — that ordering is the point.
If this document were written afterwards it would be worthless, because every one of the outcomes
below can be made to sound like a finding in hindsight.

The purpose is to make it impossible to retrofit a story. Each plausible outcome below has its
reporting decided **now**, while the answer is still unknown.

---

## The headline claim being tested

> A streaming monitor, using only signals available during training and never seeing held-out
> accuracy, raises an alarm a useful number of steps before held-out accuracy collapses — at a
> false-alarm rate low enough to leave on.

The two ways this fails are *no lead time* and *too many false alarms*. Both are reported.

## Success criterion, fixed in advance

Fix the alarm threshold so that **5% of held-out healthy runs** produce at least one alarm. At that
operating point, the fused detector must beat the **strongest** trivial control — not the average of
them — on detections-before-collapse, with a cluster-bootstrap CI on the paired difference that
excludes zero.

Strongest control is `max(reward_only, best_single_signal, step_index_only)`, computed per split.
Taking the max rather than the mean is deliberate: averaging controls would let the panel look good
merely by beating the weakest one.

If that CI includes zero, **the project reports no improvement from fusion.** It does not go looking
for a fifth model class until one clears the bar.

---

## Outcome 1 — Fusion does not beat the best single signal

**Prior: this is the most likely outcome.** `completions/clipped_ratio` is specific, free, and
famous; much of the collapse signal may simply live there.

*Reported as the primary finding, in RESULTS.md, in this shape:*

> Under leave-one-mode-out, a nine-signal fuser did not significantly outperform
> `completions/clipped_ratio` alone (Δ detections = +N/M, McNemar p = P, Δ median lead = L steps,
> 95% CI [a, b], cluster-bootstrapped over runs). The information is concentrated in one signal
> that TRL already logs.

This is a useful result and the library still has a reason to exist: it packages the signal that
works, with calibration, availability detection, and a drop-in callback. What it must **not** become
is a search for a signal set that wins. Reporting rung R0 through R3 under identical evaluation is
required precisely so that "the simple thing won" stays visible.

## Outcome 2 — Detection does not transfer across failure modes

Leave-one-mode-out means training on eight pathologies and testing on the ninth. With ~9 families
there may not be enough diversity to generalize.

*Reported as:* the full LOMO matrix, one row per held-out family, and the plain statement that the
monitor is a family-specific detector rather than a general one, plus what N families would likely
be needed. A per-family profile is more useful to a practitioner than a single averaged number,
which is why the matrix is the headline format whether or not it generalizes.

## Outcome 3 — The step-index-only control matches the monitor

A detector whose only features are `[t, t/T_run]` should be near-useless. If it is competitive, the
corpus is **time-confounded**: pathologies were injected on a predictable schedule and the "monitor"
learned a clock.

*Reported as:* a methodological finding, prominently, because it would call into question any
result in this area built on a synthetic collapse corpus without this control. Then the corpus is
regenerated with randomized onset (the injection step is already drawn uniformly from [50, 250] for
this reason) and everything is re-run. **Publishing this rather than quietly fixing it is the
commitment.**

## Outcome 4 — Lead times are real but too small to act on

A median lead of 1–3 steps is statistically detectable and operationally worthless.

*Reported as:* the median and IQR in steps **and** in wall-clock seconds at a realistic per-step
cost, with the explicit sentence that this is not actionable. No framing of a 2-step lead as an
early warning.

## Outcome 5 — Detectors do not transfer from the testbed to real TRL runs

The central external-validity claim is that thresholds calibrated on a 3M-parameter from-scratch
model carry over, without recalibration, to Qwen2.5-0.5B under real TRL.

*Reported as:* frozen-detector performance on the LLM runs next to testbed performance, with no
recalibration attempted first. If it fails, the failure is the result, and the honest conclusion is
that the cheap testbed is a development environment rather than a calibration source. Recalibrating
on the LLM runs and reporting *that* number as the headline would be circular given how few LLM runs
the compute budget allows.

## Outcome 6 — The held-out probe is itself gameable

If `verify_true` accepts the hacked outputs that the reward-hacking family produces, then
`t_collapse` is not measuring what it claims and **every downstream number is void**.

*Reported as:* the probe false-accept audit runs **before** corpus generation and its table is
published regardless of outcome. If the false-accept rate is materially above zero on constructed
hacked-format cases, corpus generation does not proceed until the probe is fixed. This gates the
project rather than qualifying it.

---

## Standing methodological commitments

- **The unit of analysis is the run, never the step.** Steps within a run are heavily
  autocorrelated; bootstrapping over step-rows would produce confidence intervals roughly √(steps
  per run) times too narrow. All intervals are cluster-bootstrapped over runs, and over cells for
  the grid.
- **Runs where a failure knob was set but no collapse occurred are labeled negative.** Injecting a
  pathological hyperparameter is not evidence that the run collapsed. Labeling by intent rather than
  by measurement is the most common way corpora like this cheat, and it inflates every metric.
- **Right-censoring is reported.** Lead time is reported as `P(fired before collapse)` alongside
  `median lead | fired`. A mean over only the runs that fired is survivorship bias.
- **False-alarm rate is broken out by hard-negative type** (plateau, noisy-but-recovering,
  slow-but-healthy, legitimate length growth). An aggregate FAR that hides most of its false alarms
  on plateaus is not informative, and plateaus are the case a starvation detector is most likely to
  get wrong.
- **Simulator traces never appear in a reported number.** They exist for CI and API development.
  They carry `source="sim"` and the report generator refuses to include them.
- **Held-out accuracy is never an input to the monitor.** Enforced by a test that replays each run
  with the oracle field replaced by noise and asserts bit-identical output, not by discipline.
- **`RESULTS.md` is generated from run artifacts and never hand-edited.** CI asserts byte-equality,
  so a number in the repo cannot drift from the artifact that produced it.
