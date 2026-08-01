"""Sampling correctness.

The load-bearing test here is `test_scoring_reproduces_sampling_logprobs`. If the KV-cache path and
the teacher-forced path disagree -- by an off-by-one in the position offset, a cache misalignment,
or a temperature mismatch -- then the importance ratio is wrong at every step. It would not crash;
it would produce a plausible-looking clipping signal out of nothing, and every downstream number
about off-policy drift would be measuring a bug.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from testbed.core.model import ModelConfig, TinyGPT  # noqa: E402
from testbed.core.rollout import generate, score  # noqa: E402

pytestmark = pytest.mark.torch

VOCAB, EOS, PAD = 13, 11, 12


def _model(seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(ModelConfig(vocab_size=VOCAB, d_model=64, n_layer=2, n_head=4, max_seq_len=48))


def _prompts(b: int = 6, p: int = 5) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    return torch.randint(0, 10, (b, p), generator=gen)


def test_scoring_reproduces_sampling_logprobs() -> None:
    """An unchanged policy must score its own samples exactly as it sampled them.

    This is what makes the importance ratio equal 1 at mu=1 as a *consequence* rather than an
    assumption, and it is the single test that guards cache and offset correctness.
    """
    model = _model()
    gen = torch.Generator().manual_seed(3)
    roll = generate(
        model, _prompts(), max_new_tokens=10, eos_id=EOS, pad_id=PAD, generator=gen
    )

    logits = score(model, roll)
    lp = F.log_softmax(logits.float(), -1)
    recomputed = lp.gather(-1, roll.completion_ids.unsqueeze(-1)).squeeze(-1)

    m = roll.completion_mask > 0
    assert m.any(), "test batch produced no live tokens"
    torch.testing.assert_close(recomputed[m], roll.logprobs[m], rtol=1e-4, atol=1e-5)


def test_ratio_is_one_when_policy_unchanged() -> None:
    """The finding-#2 mechanism, stated end to end on a real model."""
    model = _model()
    gen = torch.Generator().manual_seed(4)
    roll = generate(model, _prompts(), max_new_tokens=8, eos_id=EOS, pad_id=PAD, generator=gen)

    lp = F.log_softmax(score(model, roll).float(), -1)
    recomputed = lp.gather(-1, roll.completion_ids.unsqueeze(-1)).squeeze(-1)
    ratio = (recomputed - roll.logprobs).exp()

    m = roll.completion_mask > 0
    torch.testing.assert_close(ratio[m], torch.ones_like(ratio[m]), rtol=1e-4, atol=1e-4)


def test_ratio_departs_from_one_after_an_update() -> None:
    """Guards against the ratio being trivially 1 for the wrong reason (e.g. scoring the stale
    policy). After perturbing the weights, the ratio must move."""
    model = _model()
    gen = torch.Generator().manual_seed(5)
    roll = generate(model, _prompts(), max_new_tokens=8, eos_id=EOS, pad_id=PAD, generator=gen)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    lp = F.log_softmax(score(model, roll).float(), -1)
    recomputed = lp.gather(-1, roll.completion_ids.unsqueeze(-1)).squeeze(-1)
    ratio = (recomputed - roll.logprobs).exp().detach()
    m = roll.completion_mask > 0
    assert float((ratio[m] - 1.0).abs().max()) > 0.01


def test_mask_stops_at_eos_and_includes_it() -> None:
    """EOS is a real, scored token; everything after it is not."""
    model = _model()
    gen = torch.Generator().manual_seed(6)
    roll = generate(model, _prompts(b=16), max_new_tokens=12, eos_id=EOS, pad_id=PAD, generator=gen)

    for i in range(roll.completion_ids.shape[0]):
        row_mask = roll.completion_mask[i]
        # The mask must be a prefix: no gaps.
        n = int(row_mask.sum())
        assert torch.all(row_mask[:n] == 1.0)
        assert torch.all(row_mask[n:] == 0.0)
        if n < roll.completion_ids.shape[1]:
            # Terminated early => the last live token is EOS and padding follows.
            assert int(roll.completion_ids[i, n - 1]) == EOS
            assert torch.all(roll.completion_ids[i, n:] == PAD)


def test_generation_is_deterministic_given_a_seed() -> None:
    """Bitwise reproducibility, without which no corpus run can be re-derived."""
    model = _model()
    outs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(11)
        outs.append(generate(model, _prompts(), max_new_tokens=8, eos_id=EOS, pad_id=PAD, generator=gen))
    assert torch.equal(outs[0].completion_ids, outs[1].completion_ids)
    torch.testing.assert_close(outs[0].logprobs, outs[1].logprobs, rtol=0, atol=0)


def test_sampler_noise_creates_a_behavior_policy_gap() -> None:
    """F9's mechanism: with noise the sampled logprobs no longer match a clean rescoring.

    Reported honestly as a simulation of train/inference mismatch, not a reproduction -- the
    testbed samples and trains with the same code, so the genuine gap is zero.
    """
    model = _model()
    gen = torch.Generator().manual_seed(9)
    roll = generate(
        model, _prompts(), max_new_tokens=8, eos_id=EOS, pad_id=PAD,
        sampler_noise=0.5, generator=gen,
    )
    lp = F.log_softmax(score(model, roll).float(), -1)
    recomputed = lp.gather(-1, roll.completion_ids.unsqueeze(-1)).squeeze(-1).detach()
    m = roll.completion_mask > 0
    assert float((recomputed[m] - roll.logprobs[m]).abs().mean()) > 0.01


def test_temperature_mismatch_would_bias_the_ratio() -> None:
    """Documents why `score` takes a temperature at all.

    Scoring at a different temperature than sampling manufactures a systematic ratio bias that is
    indistinguishable from real off-policy drift. Pinned so nobody 'simplifies' it away.
    """
    model = _model()
    gen = torch.Generator().manual_seed(12)
    roll = generate(
        model, _prompts(), max_new_tokens=8, eos_id=EOS, pad_id=PAD,
        temperature=1.3, generator=gen,
    )
    m = roll.completion_mask > 0

    lp_ok = F.log_softmax(score(model, roll, temperature=1.3).float(), -1)
    ok = lp_ok.gather(-1, roll.completion_ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(ok[m], roll.logprobs[m], rtol=1e-4, atol=1e-5)

    lp_bad = F.log_softmax(score(model, roll, temperature=1.0).float(), -1)
    bad = lp_bad.gather(-1, roll.completion_ids.unsqueeze(-1)).squeeze(-1).detach()
    assert float((bad[m] - roll.logprobs[m]).abs().mean()) > 0.01
