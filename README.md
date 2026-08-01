# grpo-doctor

**Streaming early-warning monitor for GRPO/RLVR training collapse.**

Watches a reinforcement-learning-from-verifiable-rewards run and raises an alarm *before*
held-out accuracy falls — including the case that matters most, where the training reward is
still going **up** because the policy has learned to game the verifier.

```python
from trl import GRPOTrainer, GRPOConfig
from grpo_doctor.integrations.trl import VitalsCallback

cfg = GRPOConfig(logging_steps=1, log_completions=True, num_iterations=2, output_dir="out")
trainer = GRPOTrainer(..., args=cfg, callbacks=[VitalsCallback(on_alarm="warn")])
```

The monitor is `numpy`-only and O(1) per step, so it installs into a training image without
pulling a single extra tensor library.

> **Status: in development.** Numbers are not in this README yet because the experiments that
> produce them have not been run. Everything reported here will be generated from committed run
> artifacts by `scripts/make_report.py`, and CI asserts the generated file matches byte-for-byte.

---

## Why this exists

The algorithms for *fixing* GRPO pathologies are well covered — Dr. GRPO, DAPO, GSPO, Lite-PPO —
and the knobs already ship in TRL (`scale_rewards`, `loss_type`, `top_entropy_quantile`). What is
missing is **instrumentation**: the 2026 papers proposing collapse metrics either released no code
([arXiv 2602.01103](https://arxiv.org/abs/2602.01103)) or a project page with no implementation
([AVSPO, arXiv 2605.21125](https://arxiv.org/abs/2605.21125)). TRL logs the right raw signals and
does nothing with them — no fusion, no thresholding, no calibration, no forecast.

### Three things we measured that make the naive monitor wrong

These are the reason this is not a five-line threshold script. Each shows an obvious approach
failing for a demonstrable reason.

**1. Degenerate groups cause silent death, not advantage explosion.**
The widely repeated story is that zero-variance groups blow up the normalized advantage. They do
not. Under `scale_rewards="group"`, `A = (r − mean)/std` is scale-invariant with

```
sup |A| = sqrt(G − 1)      exactly, independent of reward scale or spread
```

Measured: a group with `std = 3.3e-4` gave `|A|max = 2.03` — *smaller* than a healthy group's
`2.65 = sqrt(7)`. All-fail groups give `A ≡ 0` and `grad_norm = 0.00`. The pathology is a policy
that quietly stops learning, which no advantage-magnitude threshold will ever catch.

**2. On-policy GRPO cannot exhibit clipping pathology at all.**
With TRL's default `num_iterations=1`, `old_per_token_logps` is `None`, so the importance ratio is
identically 1 and `clip_ratio/low_mean == clip_ratio/high_mean == 0.0` for the entire run. A monitor
reading those keys is reading a constant column and calling it healthy. `grpo-doctor` detects this
and marks the signal *unavailable* rather than *nominal*.

**3. Entropy moves in opposite directions for different collapse types.**
In a catastrophic off-policy run entropy **rose** (0.427 → 0.446) as reward died; in a
reward-hacking run entropy also **rose** (0.355 → 0.532) while reward hit 1.000. Meanwhile entropy
falling steadily is what a *healthy* run looks like. Any single-sided entropy threshold is wrong
about half the time.

## How it is validated

- A corpus of labeled runs with **deliberately induced** pathologies, released alongside the code.
- Ground truth is a persistent drop on a **held-out verifier that the training reward cannot
  touch** — never the training reward itself, which rises during reward hacking.
- **Leave-one-failure-mode-out**: fit on every pathology except one, evaluate on the one held out.
  A detector that only recognizes pathologies it was trained on is not an early-warning system.
- **Four negative controls in the headline table**, including a step-index-only detector. If a
  model that sees nothing but the step counter matches the real monitor, the corpus is
  time-confounded and the result is an artifact — we would rather find that ourselves.
- Detectors are calibrated on the cheap testbed, **frozen**, and then applied to real TRL runs. The
  transfer number is reported whether or not it is flattering.

Predictions about what the negative outcomes would look like were written down **before** the
experiments were run: [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md).

## Prior art

[arXiv 2606.03238](https://arxiv.org/html/2606.03238) trains early-warning classifiers for reward
hacking (ROC-AUC 0.821) and ships code. It studies **PPO/DPO/RLHF with learned reward models**,
where there is no group — so group reward variance, advantage starvation, and pass-rate structure,
the signals this project is built on, do not exist in that setting. It also classifies 31 checkpoint
transitions offline; this is per-optimizer-step, streaming, and causal.

## License

MIT
