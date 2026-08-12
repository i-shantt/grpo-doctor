"""The failure taxonomy F0-F9, as configuration deltas applied at a randomized onset step.

Three design rules here are what keep the corpus from being a benchmark for reading its own
manifest.

**Doses, not switches.** Every family is graded, with a null level that must behave exactly like
F0. A binary on/off knob turns the corpus into a two-class problem with no way to ask how large a
perturbation has to be before it becomes detectable, and no way to notice that a knob does nothing.

**Randomized onset.** The pathological setting is introduced at a uniformly random step in
[50, 250], never at a fixed one. With a fixed onset the corpus becomes a step-counter benchmark
that the step-index-only control wins outright -- and if that control ever does win, every number
in the study is an artifact. Randomizing removes the confound rather than hoping it is absent.

**A knob that fires is not a collapse.** Nothing here labels anything. These functions produce run
configurations; whether the run actually collapsed is decided afterwards by `label_run` from
held-out accuracy alone. Runs where a knob was set and nothing happened are negatives, and they are
expected to be a substantial fraction of the grid.

The hard-negative families H2-H5 live here too, and they matter more than the failure families for
the headline number. A false-alarm rate measured only against clean, successful runs is measured
against the easy case; the budget is really spent on plateaus, which look exactly like starvation
until they resolve.

They are *intended* negatives, not guaranteed ones. H3 at its chosen dose leaves about 60% of runs
healthy and genuinely degrades the rest, and that ambiguity is the point -- a hard negative that
never gets close to collapsing is not hard. As everywhere else here, the label comes from held-out
accuracy afterwards and an H-family run that really did collapse is counted as a positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from testbed.tasks.base import LeakLevel

ONSET_RANGE = (50, 250)


@dataclass(frozen=True)
class FailureSpec:
    """One cell of the grid: a family, a dose, and the overrides that produce it."""

    family: str
    dose: str
    overrides: dict[str, Any]
    needs_onset: bool = True
    """False for families that must be set from step 0.

    F8 is the clear case: `scale_rewards` changes the meaning of the advantage, and switching it
    mid-run would inject a discontinuity in the loss scale that has nothing to do with the
    normalization instability being studied.
    """

    simulated: bool = False
    """F9 only. The train/inference logprob gap is *simulated* by perturbing rollout logits, because
    in this testbed the same code samples and trains so the genuine discrepancy is exactly zero.
    The flag travels with the run into the results, and no lead-time number is claimed for it.
    """

    @property
    def cell(self) -> str:
        return f"{self.family}/{self.dose}"


# F0 -- healthy. The control, and the null dose every family's ladder starts from.
HEALTHY = FailureSpec("F0", "none", {}, needs_onset=False)

FAILURES: tuple[FailureSpec, ...] = (
    HEALTHY,
    # F1 difficulty starvation: nothing is solvable, every group agrees on failure, and the
    # gradient goes to exactly zero. Silent death, not explosion.
    #
    # These narrow the *training* range rather than setting a single difficulty. Setting one used
    # to send reward to exactly 0.000 in both directions -- difficulty 2 broke the policy as
    # thoroughly as difficulty 8 -- because the change was in prompt shape, not in difficulty.
    FailureSpec("F1", "mild", {"difficulty_range": (7, 8)}),
    FailureSpec("F1", "strong", {"difficulty_range": (8, 8), "temperature": 0.5}),
    FailureSpec("F1", "cold", {"temperature": 0.3}),
    # F2 saturation: zero variance from the *success* side. Included because it produces the same
    # zero-variance reading as F1 while being the opposite situation, which is exactly why raw ACR
    # is worthless as a level and has to be split into starvation and mastery.
    FailureSpec("F2", "easy", {"difficulty_range": (3, 3)}),
    FailureSpec("F2", "hot_easy", {"difficulty_range": (3, 4), "temperature": 0.5}),
    # F3 off-policy ratio blowup. mu>=2 is required for any clipping signal to exist at all, so
    # the null dose is mu=2 rather than mu=1.
    FailureSpec("F3", "mu4", {"grpo.num_iterations": 4}),
    FailureSpec("F3", "mu8", {"grpo.num_iterations": 8}),
    FailureSpec("F3", "mu8_hot", {"grpo.num_iterations": 8, "optim.lr": 5e-4}),
    # F4 entropy collapse. Both doses are negatives, and the family has no dose that is not.
    #
    # `narrow_clip` does the OPPOSITE of what the family is named for. The clipped surrogate caps
    # policy change in *both* directions, so narrowing epsilon shrinks the trust region and thereby
    # *preserves* entropy by slowing the policy's sharpening. On top of that, `num_iterations=2`
    # means inner iteration 0 has ratio == 1 by construction, so only half the iterations can clip
    # at all. Measured: healthy 5/5 on both tasks, entropy essentially flat.
    #
    # The knob that actually targets entropy is `entropy_coef`, entering the loss as
    #   loss - coef * H
    # so a NEGATIVE coefficient minimizes entropy directly. It was tried at -0.05 and -0.25 and it
    # is deliberately NOT in the grid, because it works on entropy and does nothing to accuracy:
    # final entropy 0.090 (control) against 0.062 at -0.25, held-out accuracy untouched, healthy at
    # both doses. The control also dips to 0.027 at its lowest, below anything the dosed run ever
    # reaches -- low entropy is a thing that happens to healthy runs here, not the pathology.
    #
    # That is the whole finding, and it is a fact about the regime rather than about the dose:
    # healthy runs in this corpus LOSE entropy (mean delta -0.149) and collapsed runs GAIN it
    # (+0.108). Driving entropy down pushes the policy along the healthy direction, so no
    # coefficient can collapse a run by this route. Premature entropy collapse needs a policy that
    # has not learned the task yet, and the warm start puts every run in TARGET_BAND by design.
    # Adding a stronger dose would not fix that; it would need a different testbed.
    FailureSpec("F4", "narrow_clip", {"grpo.epsilon_low": 0.05, "grpo.epsilon_high": 0.05}),
    FailureSpec("F4", "cold_narrow", {"temperature": 0.5, "grpo.epsilon_high": 0.05}),
    # F5 verifier leakage. The Goodhart families: proxy reward rises while true accuracy falls.
    #
    # Leaks are PARTIAL by default, and that is a measured requirement rather than a stylistic
    # choice. A verifier that leaks on every problem sends reward to 1.000 everywhere, which makes
    # every group degenerate, every advantage exactly zero and grad_norm exactly 0.0000 -- training
    # stops dead at the injection step and the policy is frozen, not corrupted. Held-out accuracy
    # therefore never moves and the run is correctly labeled healthy. Keeping some problems strictly
    # graded preserves reward variance, so the gradient survives and the policy actually drifts
    # toward exploiting the leak.
    FailureSpec("F5", "prefix", {"verifier.leak_level": LeakLevel.PREFIX}),
    FailureSpec(
        "F5", "structure_p40", {"verifier.leak_level": LeakLevel.STRUCTURE, "verifier.leak_p": 0.4}
    ),
    FailureSpec(
        "F5", "structure_p70", {"verifier.leak_level": LeakLevel.STRUCTURE, "verifier.leak_p": 0.7}
    ),
    FailureSpec(
        "F5", "format_p40", {"verifier.leak_level": LeakLevel.FORMAT, "verifier.leak_p": 0.4}
    ),
    # Retained at full strength as the *freeze* case: reward pinned at 1.000, gradient at exactly
    # zero, nothing learned again. It belongs in the corpus precisely because a reward-watching
    # practitioner reads it as total success.
    FailureSpec("F5", "structure_full", {"verifier.leak_level": LeakLevel.STRUCTURE}),
    #
    # SHAPED LEAKS. The four doses above are the leak alone, and measured across both live tasks at
    # two seeds they almost never Goodhart: structure_p40, structure_p70 and format_p40 were healthy
    # 2/2 on sort_digits while reward climbed 0.51 -> 0.91, held-out accuracy stayed flat, and mean
    # length did not move.
    #
    # That is not a weak dose, it is the arithmetic. Every leak here accepts a *superset* of the
    # correct answers, so a correct completion still scores 1.0 on a leaked problem and abandoning
    # correctness can only lose. Always-correct earns 1.0; always-exploit earns leak_p. The exploit
    # is dominated at every leak_p < 1, and at leak_p = 1 the two tie, which is the zero-variance
    # freeze `VerifierConfig.leak_p` documents. There is no dose of a bare superset leak that
    # produces Goodhart, so tuning leak_p harder was never going to find one.
    #
    # A negative length bonus breaks the tie, because the exploit is short and the answer is not:
    #   correct  ~ 1 + bonus * len(answer)      exploit ~ 1 + bonus * 1
    # The one-token exploit now strictly outscores a correct answer wherever the leak applies, and
    # the bonus also restores reward variance at leak_p = 1, so the full-leak case hacks instead of
    # freezing. Measured on sort_digits: HACK 2/2 with held-out accuracy 0.41 -> 0.000, mean length
    # 4.46 -> 2.00, and reward *rising* 0.512 -> 0.923. That is the signature the whole project is
    # built to catch, and without these cells the corpus did not contain it.
    #
    # Note what this says about the mechanism: the policy cannot tell a leaked problem from a
    # strictly graded one (`leaks_for` hashes the prompt), so an exploit that pays on 70% of
    # problems is generalized to all of them. A partial leak does not produce partial hacking.
    FailureSpec(
        "F5",
        "structure_p70_terse",
        {
            "verifier.leak_level": LeakLevel.STRUCTURE,
            "verifier.leak_p": 0.7,
            "verifier.length_bonus": -0.05,
        },
    ),
    FailureSpec(
        "F5",
        "structure_full_terse",
        {"verifier.leak_level": LeakLevel.STRUCTURE, "verifier.length_bonus": -0.05},
    ),
    # The strongest cell measured on ca_rule (3 of 4 seeds, held-out accuracy to 0.000-0.010),
    # because ca_rule's STRUCTURE leak checks a population count and so cannot be exploited by a
    # short answer -- only FORMAT frees the length. One dose per task shape, not one per taxonomy
    # slot.
    FailureSpec(
        "F5",
        "format_p70_terse",
        {
            "verifier.leak_level": LeakLevel.FORMAT,
            "verifier.leak_p": 0.7,
            "verifier.length_bonus": -0.05,
        },
    ),
    # F6 length hacking: reward decoupled from correctness by a per-token bonus.
    FailureSpec("F6", "verbose", {"verifier.length_bonus": 0.05}),
    FailureSpec("F6", "terse", {"verifier.length_bonus": -0.05}),
    # F7 flaky grader. Both doses are negatives, and NO value of flip_p below 0.5 can be anything
    # else -- which is a sharper statement than "these two doses were too weak", so it is worth the
    # lines.
    #
    # Symmetric label noise is affine in expectation: E[r'] = r(1-2p) + p. Group standardization
    # removes the offset, so the whole effect is a scale factor of exactly (1-2p) on the learning
    # signal -- confirmed by the correlation between noisy and clean advantages matching (1-2p) to
    # three decimals. For every p < 0.5 that factor is POSITIVE: the signal is attenuated but its
    # sign is intact, so the policy stops improving rather than getting worse. A stall leaves no
    # drawdown, and a drawdown is what `t_collapse` labels on. p=0.10 and p=0.25 retain 80% and 50%
    # of the signal and are healthy 5/5 on both tasks.
    #
    # Pushing toward the vanishing point does not change the sign, only the rate, and the measured
    # answer agrees: at p=0.40 held-out accuracy fell 0.480 -> 0.457, a drawdown of 0.023 against a
    # threshold near 0.094, while the training reward ROSE to 0.516 (above the control's 0.344) on
    # the strength of the flipped labels alone. Attenuation plus reward inflation, still healthy.
    #
    # The ordering inverts only at p > 0.5, where E[r|wrong] = p exceeds E[r|correct] = 1 - p and
    # the grader is training the policy toward incorrect completions. A probe at 0.55 and 0.65
    # collapses decisively (drawdown 0.246 and 0.453; at 0.65 held-out accuracy ends at 0.004 while
    # reward ends at 0.625). That is a real pathology but it is not a flaky grader -- it is an
    # INVERTED one, and it belongs to the taxonomy under that name or not at all. It is left out
    # rather than smuggled in under F7's label; see docs/STATE.md.
    FailureSpec("F7", "p10", {"verifier.flip_p": 0.10}),
    FailureSpec("F7", "p25", {"verifier.flip_p": 0.25}),
    # F8 normalization instability -- the only family where |A| is genuinely unbounded, since the
    # per-group standardization that gives the (G-1)/sqrt(G) bound is gone. Set from step 0.
    FailureSpec("F8", "batch", {"grpo.scale_rewards": "batch"}, needs_onset=False),
    FailureSpec("F8", "none", {"grpo.scale_rewards": "none"}, needs_onset=False),
    FailureSpec(
        "F8",
        "none_unclipped",
        {"grpo.scale_rewards": "none", "optim.max_grad_norm": 0.0},
        needs_onset=False,
    ),
    # F9 sampler/trainer mismatch. Simulated, and labeled as such everywhere it appears.
    #
    # Both doses supply the importance correction the real pathology omits, so what they simulate
    # is the cure rather than the disease -- and naming that is the point of keeping them.
    # `generate` scores each sampled token under the noised distribution it came from, making
    # old_logprobs = log pi_behavior and the GRPO ratio pi_theta/pi_behavior: a correct importance
    # weight, i.e. ordinary off-policy PPO, unbiased before clipping. Measured on sort_digits, a
    # loud and correctly-shaped signal with no collapse. The log-ratio bias is
    # -KL(pi_b || pi_theta) < 0, so clipping is one-sided (clip_low 0.023 against clip_high 0.007
    # at sigma=0.75) while reward rises as fast as the F0 control.
    #
    # TRL computes old_per_token_logps from the *trainer's* own forward pass, so its ratio is 1 at
    # inner iteration 0 and the vLLM/HF gap is never corrected. The missing weight IS the mechanism
    # (arXiv 2602.01103), and `correct_sampler_gap=False` reproduces it: sample from the noised
    # distribution, score under the clean one.
    #
    # No cell uses that setting, on purpose. Measured at sigma=0.75 the uncorrected arm is
    # signal-IDENTICAL to the F0 control -- clip_low 0.000, clip_high 0.000, ratio_max 1.442, the
    # control's exact values -- and it does not collapse either. That invisibility is the finding
    # and it is what the flag exists to state; a corpus cell would only add runs that are
    # indistinguishable from the control by construction. Raising sigma further would not rescue
    # it, because a sampler that far from the trainer is a broken sampler rather than a
    # train/inference gap, which is a different pathology wearing F9's name.
    FailureSpec("F9", "noise_lo", {"sampler_noise": 0.25}, simulated=True),
    FailureSpec("F9", "noise_hi", {"sampler_noise": 0.75}, simulated=True),
)

HARD_NEGATIVES: tuple[FailureSpec, ...] = (
    # H2 plateau. The important one. Learning simply stops: flat reward, high zero-variance
    # fraction, nothing improving -- indistinguishable from starvation collapse except that nothing
    # is actually lost. This is where the false-alarm budget is really spent, and an aggregate FAR
    # that hides 40% false alarms here would be meaningless, so FAR is broken out by type.
    FailureSpec("H2", "plateau", {"optim.lr": 1e-6}),
    # H3 noisy but recovering. 2e-4 measured a mean max drawdown of 0.205 with 60% of runs still
    # ending healthy -- a genuinely ambiguous condition rather than a disguised catastrophe. The
    # previous value here was 3e-3, which did not produce a dip at all: every run went to 0.000
    # final accuracy, making it a failure family wearing a hard negative's name.
    FailureSpec("H3", "dip", {"optim.lr": 2e-4}),
    # H4 slow but healthy: improving the whole way, just far below the median rate. Below the new
    # 5e-5 default, since that default is now itself the stable setting.
    FailureSpec("H4", "slow", {"optim.lr": 1e-5}, needs_onset=False),
    # H5 legitimate length growth. Rewards longer *correct* answers, so length rises with accuracy
    # rather than against it -- the case that must not fire the length baseline.
    FailureSpec("H5", "longer", {"verifier.length_bonus": 0.02}),
)

ALL_SPECS: tuple[FailureSpec, ...] = FAILURES + HARD_NEGATIVES

NEGATIVE_FAMILIES: frozenset[str] = frozenset({"F0"}) | {s.family for s in HARD_NEGATIVES}
"""Families a run is *expected* to survive: the control plus the hard negatives.

"Expected" is doing all the work here, and this set is deliberately not the definition of a
negative. Nothing is negative by construction except F0. An H3 run that genuinely collapses is a
positive, and -- measured -- whole failure families come out negative on some tasks. So the
false-alarm rate is computed from `label_run`'s verdict, never from membership here; this exists to
report FAR *broken out by intended type*, which is a different question from what a run turned out
to be.
"""


def sample_onset(
    rng: np.random.Generator, low: int = ONSET_RANGE[0], high: int = ONSET_RANGE[1]
) -> int:
    """Uniform onset step. Uniform, not centered, so the step-index control gets no free signal."""
    return int(rng.integers(low, high + 1))


def spec_by_cell(cell: str) -> FailureSpec:
    for spec in ALL_SPECS:
        if spec.cell == cell:
            return spec
    raise KeyError(f"unknown cell {cell!r}")


def families() -> tuple[str, ...]:
    """Distinct family names, in declaration order. The unit of leave-one-mode-out."""
    seen: list[str] = []
    for spec in ALL_SPECS:
        if spec.family not in seen:
            seen.append(spec.family)
    return tuple(seen)


__all__ = [
    "ALL_SPECS",
    "FAILURES",
    "HARD_NEGATIVES",
    "HEALTHY",
    "NEGATIVE_FAMILIES",
    "ONSET_RANGE",
    "FailureSpec",
    "families",
    "sample_onset",
    "spec_by_cell",
]
