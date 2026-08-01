"""A small decoder-only transformer, sized so that a full GRPO run fits in minutes on one CPU core.

This is deliberately *not* an LLM. It exists so that the corpus of labeled collapse runs can be
generated hundreds of runs at a time on a laptop, which is what makes leave-one-failure-mode-out
evaluation statistically possible at all. See docs/NEGATIVE_RESULTS.md for the transfer claim being
tested and README.md for what this tier can and cannot reproduce.

Sizing note: at d_model=192 / n_layer=4 the parameter count is ~1.8M and a forward+backward at
batch 64 costs ~0.2s on a single CPU thread, so a 600-step run lands around 4 minutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4
    max_seq_len: int = 96
    dropout: float = 0.0  # kept at 0: dropout would inject noise into the entropy signals we study

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with an optional incremental KV cache.

    The cache matters more than it looks: rollout generation is the dominant cost of a GRPO step,
    and recomputing the full prefix at every decode position would roughly triple run time.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, n_head, T, d_head)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # is_causal must be False when decoding with a cache: the query is a single new position
        # attending to the whole cached prefix, which a causal mask over (t, t_total) would break.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None and t > 1)
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.d_head)
        return self.proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, new_cache = self.attn(self.ln1(x), cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyGPT(nn.Module):
    """Pre-LN decoder-only transformer with tied embeddings and learned positions."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied

        self.apply(self._init_weights)
        # Scale residual-path projections by 1/sqrt(2*n_layer) (GPT-2 init). Without this the
        # residual stream variance grows with depth and early training is needlessly unstable --
        # which would show up as spurious grad-norm spikes in the very signals we are studying.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # Tied head shares storage with tok_emb, so count unique tensors only.
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(
        self,
        idx: Tensor,
        caches: list[tuple[Tensor, Tensor]] | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Return logits (B, T, V) and the updated per-layer KV caches.

        `pos_offset` is the absolute position of `idx[:, 0]`, needed when decoding incrementally.
        """
        _, t = idx.shape
        if pos_offset + t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {pos_offset + t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        pos = torch.arange(pos_offset, pos_offset + t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, caches[i] if caches is not None else None)
            new_caches.append(cache)

        return self.head(self.ln_f(x)), new_caches
