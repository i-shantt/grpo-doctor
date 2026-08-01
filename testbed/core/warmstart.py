"""Cached supervised warm starts.

Every corpus run begins from a supervised checkpoint, because a policy that starts at 0% or 97%
cannot exhibit any interesting failure -- there is nothing to collapse from. Warm starting is
therefore mandatory, and it is also the one part of a run whose cost does not depend on the failure
knob being studied: a healthy run and a hacked run at the same (task, difficulty, seed) start from
*identical* weights.

So it is computed once and reused. That matters for two reasons beyond speed.

**It makes expensive warm starts affordable.** `modarith` needs an order of magnitude more
supervision than `sort_digits` to reach a trainable pass rate. Paying that per run would dominate
the corpus budget; paying it once per (task, difficulty, seed) is negligible.

**It removes a confound.** With a shared checkpoint, two runs in the same cell differ only in the
knob under study. Without it they would also differ in their initial policy, and part of any
measured effect would be initialization noise attributed to the knob.

The cache key covers every input that can change the resulting weights, and the parameters are
written alongside the checkpoint as JSON so a stale cache is auditable rather than silent. Nothing
here is committed: `warmstarts/` is gitignored, and any run can regenerate it from the key.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from testbed.core.model import ModelConfig, TinyGPT

if TYPE_CHECKING:  # pragma: no cover
    from testbed.core.train import RunConfig
    from testbed.tasks.base import Task

DEFAULT_CACHE_DIR = Path("warmstarts")


@dataclass(frozen=True)
class WarmStartKey:
    """Everything that determines the resulting weights. Anything omitted here is a cache bug."""

    task: str
    difficulty: int
    steps: int
    lr: float
    batch_size: int
    d_model: int
    n_layer: int
    n_head: int
    max_seq_len: int
    vocab_size: int
    seed: int
    target: tuple[float, float] | None = None
    """The accuracy band, when warm-starting to a target rather than a step count.

    Part of the key because it changes where training stops: two runs with the same maximum budget
    but different bands end at different checkpoints, and omitting this would serve one the other's
    weights.
    """
    target_probe_every: int = 50
    """Also part of the key: it sets the granularity of the stopping rule, so a run that probes
    every 50 steps stops at a different point than one probing every 200."""

    difficulty_range: tuple[int, int] | None = None
    """The warm start trains over this range, so it determines the weights directly."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "difficulty": self.difficulty,
            "steps": self.steps,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "seed": self.seed,
            "target": list(self.target) if self.target else None,
            "target_probe_every": self.target_probe_every,
            "difficulty_range": list(self.difficulty_range) if self.difficulty_range else None,
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def filename(self) -> str:
        return f"{self.task}_d{self.difficulty}_s{self.seed}_{self.digest()}"


def key_for(task: Task, cfg: RunConfig) -> WarmStartKey:
    return WarmStartKey(
        task=task.name,
        difficulty=cfg.difficulty,
        steps=cfg.warm_start_steps,
        lr=cfg.warm_start_lr,
        batch_size=cfg.n_prompts * cfg.grpo.group_size,
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        max_seq_len=task.prompt_len + task.max_completion_len + 1,
        vocab_size=task.vocab_size,
        seed=cfg.seed,
        target=cfg.warm_start_target,
        target_probe_every=cfg.warm_start_probe_every,
        difficulty_range=cfg.difficulty_range,
    )


def build_model(task: Task, cfg: RunConfig) -> TinyGPT:
    """Fresh model at the configured shape. Seeds the global RNG, as model init consumes it."""
    torch.manual_seed(cfg.seed)
    return TinyGPT(
        ModelConfig(
            vocab_size=task.vocab_size,
            d_model=cfg.d_model,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            max_seq_len=task.prompt_len + task.max_completion_len + 1,
        )
    )


def load_or_train(
    task: Task,
    cfg: RunConfig,
    *,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> tuple[TinyGPT, dict[str, float]]:
    """Return a warm-started model, from cache when possible.

    `cache_dir=None` disables the cache entirely, which is what the tests use to prove that a
    cached run and a freshly trained one produce the same thing.
    """
    from testbed.core.train import warm_start  # circular at module scope; deliberate

    model = build_model(task, cfg)
    if cfg.warm_start_steps <= 0 or cache_dir is None:
        return model, {**warm_start(model, task, cfg), "warm_start/cached": 0.0}

    key = key_for(task, cfg)
    directory = Path(cache_dir)
    ckpt = directory / f"{key.filename()}.pt"
    meta = directory / f"{key.filename()}.json"

    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        # The whole info dict is stored, not just the loss, so a cached run reports the same
        # columns as a fresh one. Otherwise `warm_start/steps_used` and `warm_start/accuracy`
        # would be present on first generation and missing on every rerun -- a ragged trace, and
        # the calibration record would survive only until someone regenerated the corpus.
        info = {k: float(v) for k, v in state.get("info", {}).items()}
        return model, {**info, "warm_start/cached": 1.0}

    info = warm_start(model, task, cfg)
    directory.mkdir(parents=True, exist_ok=True)
    # Write through a temporary name so a killed process cannot leave a truncated checkpoint that
    # every later run would silently load.
    #
    # The pid is in the name because the corpus runs across processes and two workers hitting the
    # same uncached key concurrently is the common case, not the rare one -- every failure family
    # at a given seed shares one warm start. With a shared temporary name both wrote to the same
    # file and the second `replace` died with FileNotFoundError, losing the run. Observed twice in
    # a 38-run sweep at six workers.
    tmp = ckpt.with_suffix(f".pt.tmp{os.getpid()}")
    try:
        torch.save({"model": model.state_dict(), "info": info}, tmp)
        tmp.replace(ckpt)
    finally:
        tmp.unlink(missing_ok=True)
    meta.write_text(json.dumps({**key.to_dict(), **info}, indent=1, sort_keys=True))
    return model, {**info, "warm_start/cached": 0.0}


__all__ = ["DEFAULT_CACHE_DIR", "WarmStartKey", "build_model", "key_for", "load_or_train"]
