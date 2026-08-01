"""The one input type. Everything the monitor knows about a training step arrives as a StepRecord.

Two constraints shape this module and neither is negotiable.

**Every field is optional.** The whole premise is that you get different signals depending on what
your trainer happens to log, so a record with only `reward_mean` must work. The monitor never
raises on a missing field; it marks the dependent signal unavailable and reports reduced coverage.
Optional is also why `None` and `nan` mean different things here: `None` is "this trainer does not
log it", `nan` is "logged, but not measurable in this configuration" -- which is what TRL's clip
ratios are at `num_iterations=1`. Collapsing either one into `0.0` is the specific bug this project
exists to catch, so neither is ever defaulted.

**`heldout_accuracy` is an oracle and is quarantined.** It rides along in the record because the
labeler needs it and because a trace should be self-contained, but no signal may read it. That is
enforced by `visible()` plus a replay test that feeds the monitor pure noise in this field and
asserts bit-identical output. If that test ever fails, every lead time the project reports is
circular and worthless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

ORACLE_FIELDS = frozenset({"heldout_accuracy"})
"""Fields the monitor must never see. Enforced by `visible()`, tested by replay."""

TRL_KEY_MAP = {
    # TRL's own log keys, so a real `on_log` payload maps across with no translation layer.
    "reward": "reward_mean",
    "reward_std": "reward_std",
    "frac_reward_zero_std": "frac_reward_zero_std",
    "entropy": "entropy",
    "kl": "kl",
    "grad_norm": "grad_norm",
    "learning_rate": "learning_rate",
    "clip_ratio/low_mean": "clip_low",
    "clip_ratio/high_mean": "clip_high",
    "clip_ratio/region_mean": "clip_region",
    "completions/mean_length": "completion_len_mean",
    "completions/clipped_ratio": "completion_clipped_ratio",
    "importance_ratio/max": "importance_ratio_max",
    "importance_ratio/log_std": "importance_ratio_log_std",
    "num_tokens": "num_tokens",
}


@dataclass(frozen=True)
class StepRecord:
    """One optimizer step, as much or as little of it as the trainer reported."""

    step: int

    # --- T0: available from a stock TRL run with logging_steps=1 -------------------------------
    reward_mean: float | None = None
    reward_std: float | None = None
    frac_reward_zero_std: float | None = None
    entropy: float | None = None
    kl: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None
    clip_low: float | None = None
    clip_high: float | None = None
    clip_region: float | None = None
    completion_len_mean: float | None = None
    completion_clipped_ratio: float | None = None
    importance_ratio_max: float | None = None
    importance_ratio_log_std: float | None = None
    num_tokens: float | None = None

    # --- T1: per-group detail, from log_completions=True or the testbed ------------------------
    group_rewards: np.ndarray | None = None
    """(n_groups, G). The only way to compute a pass-rate distribution rather than its mean."""
    advantages: np.ndarray | None = None

    # --- T2: token-level, from an instrumented trainer ----------------------------------------
    token_entropy_hist: np.ndarray | None = None
    """Fixed 32 bins. A histogram rather than raw per-token values so the record stays O(1) in
    completion length -- otherwise a long-CoT run would make each record grow without bound."""

    # --- oracle, quarantined ------------------------------------------------------------------
    heldout_accuracy: float | None = None
    """Ground truth for labeling. Never an input to any signal. See ORACLE_FIELDS."""

    source: str = "unknown"
    """One of "trl", "testbed", "sim". Traces tagged "sim" are refused by the report generator, so
    a simulated trace can never end up inside a published number."""

    def visible(self) -> StepRecord:
        """This record with every oracle field cleared. What the monitor is allowed to receive."""
        return StepRecord(**{**_shallow(self), **dict.fromkeys(ORACLE_FIELDS)})

    # --- serialization ------------------------------------------------------------------------

    def to_json(self) -> str:
        out: dict[str, Any] = {}
        for k, v in _shallow(self).items():
            if v is None:
                continue
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return json.dumps(out, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> StepRecord:
        raw = json.loads(line)
        array_fields = {"group_rewards", "advantages", "token_entropy_hist"}
        kwargs: dict[str, Any] = {
            k: (np.asarray(v, dtype=float) if k in array_fields else v) for k, v in raw.items()
        }
        return cls(**kwargs)

    @classmethod
    def from_trl_log(cls, logs: dict[str, Any], step: int | None = None) -> StepRecord:
        """Build a record from a TRL `on_log` payload.

        Unknown keys are ignored rather than rejected: TRL shipped nine releases in two months and
        a monitor that dies on a new log key is worse than one that ignores it.
        """
        kwargs: dict[str, Any] = {}
        for trl_key, field_name in TRL_KEY_MAP.items():
            if trl_key in logs and logs[trl_key] is not None:
                kwargs[field_name] = float(logs[trl_key])
        resolved = step if step is not None else int(logs.get("step", logs.get("global_step", 0)))
        return cls(step=resolved, source="trl", **kwargs)


def _shallow(rec: StepRecord) -> dict[str, Any]:
    """`asdict` deep-copies numpy arrays into lists; we want the fields as they are."""
    return {f.name: getattr(rec, f.name) for f in fields(rec)}


def write_jsonl(path: str, records: list[StepRecord]) -> None:
    with open(path, "w") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")


def read_jsonl(path: str) -> list[StepRecord]:
    with open(path) as fh:
        return [StepRecord.from_json(line) for line in fh if line.strip()]


__all__ = ["ORACLE_FIELDS", "TRL_KEY_MAP", "StepRecord", "read_jsonl", "write_jsonl"]
