"""Batched sampling with a KV cache, plus the teacher-forced scoring pass used for updates.

Generation dominates the cost of a GRPO step, so this is the hot path. Two things matter for
correctness of the *signals* rather than of the model:

- `old_logprobs` returned here are the log-probs under the policy that actually produced the
  tokens. They are what makes the importance ratio meaningful at mu >= 2. Recomputing them from the
  updated policy would silently force the ratio to 1 and quietly delete the clipping signals.
- `sampler_noise` perturbs the rollout logits *only*, creating a deliberate mismatch between the
  behavior policy and the policy being scored. This simulates the train/inference discrepancy that
  arises in real setups when vLLM and the trainer disagree numerically. It is a **simulation of the
  mechanism, not a reproduction of it** -- in this testbed the same code samples and trains, so the
  genuine discrepancy is exactly zero. Runs using this knob are labeled as simulated and that
  caveat travels with them into the results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from testbed.core.model import TinyGPT


@dataclass
class Rollout:
    """One batch of sampled completions."""

    prompt_ids: Tensor
    """(B, P) padded prompt tokens, equal length by construction for these synthetic tasks."""

    completion_ids: Tensor
    """(B, T) sampled tokens, padded past EOS."""

    completion_mask: Tensor
    """(B, T) 1.0 for real tokens including the EOS itself, 0.0 after."""

    logprobs: Tensor
    """(B, T) log-probs under the *sampling* policy. This is `old_logprobs` for the GRPO update."""

    def lengths(self) -> Tensor:
        return self.completion_mask.sum(-1)


@torch.no_grad()
def generate(
    model: TinyGPT,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    temperature: float = 1.0,
    sampler_noise: float = 0.0,
    generator: torch.Generator | None = None,
) -> Rollout:
    """Sample `max_new_tokens` per sequence, stopping each at EOS.

    Every sequence is decoded to the same tensor width; `completion_mask` carries which positions
    are real. We do not early-exit when all sequences finish, because in the failure modes we care
    about (repetition loops) they usually do not.
    """
    was_training = model.training
    model.eval()

    b = prompt_ids.shape[0]
    device = prompt_ids.device

    logits, caches = model(prompt_ids)
    next_logits = logits[:, -1, :]

    tokens = torch.full((b, max_new_tokens), pad_id, dtype=torch.long, device=device)
    logprobs = torch.zeros(b, max_new_tokens, device=device)
    mask = torch.zeros(b, max_new_tokens, device=device)
    alive = torch.ones(b, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        step_logits = next_logits.float()
        if sampler_noise > 0.0:
            step_logits = step_logits + sampler_noise * torch.randn(
                step_logits.shape, generator=generator, device=device
            )
        scaled = step_logits / max(temperature, 1e-6)
        probs = F.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

        # Score the sampled token under the distribution it was actually drawn from, so that the
        # importance ratio at mu>=2 reflects the true behavior policy including any injected noise.
        step_logprobs = F.log_softmax(scaled, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        tokens[:, t] = torch.where(alive, sampled, torch.full_like(sampled, pad_id))
        logprobs[:, t] = torch.where(alive, step_logprobs, torch.zeros_like(step_logprobs))
        mask[:, t] = alive.float()  # EOS itself is a real, scored token
        alive = alive & (sampled != eos_id)

        if t + 1 < max_new_tokens:
            step_in = tokens[:, t : t + 1]
            pos = prompt_ids.shape[1] + t
            next_logits_full, caches = model(step_in, caches=caches, pos_offset=pos)
            next_logits = next_logits_full[:, -1, :]

    if was_training:
        model.train()

    return Rollout(
        prompt_ids=prompt_ids,
        completion_ids=tokens,
        completion_mask=mask,
        logprobs=logprobs,
    )


def score(
    model: TinyGPT,
    rollout: Rollout,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Teacher-forced logits over the completion positions under the *current* policy.

    Returns (B, T, V). The GRPO loss gathers from these, so gradients flow here and only here.

    Temperature must match the value used at sampling time. If it does not, the importance ratio
    picks up a constant multiplicative bias that looks exactly like off-policy drift -- a subtle way
    to fabricate a clipping signal that isn't real.
    """
    full = torch.cat([rollout.prompt_ids, rollout.completion_ids], dim=1)
    logits, _ = model(full[:, :-1])
    p = rollout.prompt_ids.shape[1]
    # Position i predicts token i+1, so completion token j is predicted from index p+j-1.
    completion_logits = logits[:, p - 1 :, :]
    return completion_logits / max(temperature, 1e-6)
