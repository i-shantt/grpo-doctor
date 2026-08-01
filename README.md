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
not. Under `scale_rewards="group"`, `A = (r − mean)/std` is scale-invariant and bounded:

```
sup |A| = (G − 1) / sqrt(G)     exactly, independent of reward scale or spread
```

That constant is TRL's, and getting it right required reading TRL's source rather than a textbook:
`nanstd` applies Bessel's correction, so the std is unbiased and the bound is `(G−1)/√G = 2.4749`
at `G=8` — not the `√(G−1) = 2.6458` you get from a population std.

The consequence is what matters. A group with `std = 3.3e-4` measures `|A|max = 2.03` — *smaller*
than a healthy group's. All-fail groups give `A ≡ 0` and `grad_norm = 0.00`, measured here for 60
consecutive steps. The pathology is a policy that quietly stops learning, and no
advantage-magnitude threshold will ever catch it.

**2. On-policy GRPO cannot exhibit clipping pathology at all.**
With TRL's default `num_iterations=1`, `old_per_token_logps` is `None`, so the importance ratio is
identically 1 and `clip_ratio/low_mean == clip_ratio/high_mean == 0.0` for the entire run. A monitor
reading those keys is reading a constant column and calling it healthy. `grpo-doctor` detects this
and marks the signal *unavailable* rather than *nominal*.

**3. Falling entropy is what health looks like. Collapse comes with entropy going *up*.**
The intuition to watch for "entropy collapse" points the wrong way. Across 77 runs on the testbed,
comparing mean entropy just before the injection against the last 50 steps:

| | runs | mean Δ entropy | mean Δ held-out accuracy |
|---|---|---|---|
| healthy controls | 15 | **−0.149** | +0.382 |
| runs that collapsed | 3 | **+0.108** | −0.371 |

Every healthy run lost entropy, and every collapsed run gained it. So a monitor thresholding on
"entropy is dropping" would flag its healthiest runs and miss every real failure — it is not merely
unreliable, it is anti-correlated. This is why the entropy signal here is `Δreward` *per unit of
entropy spent* rather than a level or a slope: the pathology is paying entropy and getting nothing
back, not the paying itself.

Three collapsed runs is a thin sample and this is a Phase-1 measurement, not the headline; it gets
re-derived on the full corpus with cluster-bootstrap intervals. It already contradicts what this
README previously claimed on the basis of two ad-hoc runs, which is the reason the number is
reported with its `n` attached.

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
