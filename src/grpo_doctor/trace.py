"""Reading a record stream off disk, and mapping the testbed's column names onto `StepRecord`.

Its own module rather than part of the CLI because everything downstream needs it -- the evaluation
harness reads the same traces the `replay` command does, and having `eval/` import from `cli.py`
would point the dependency arrow backwards and break the moment the CLI grows a report command.

The name mapping is the load-bearing part. The testbed logs TRL's column names verbatim
(`clip_ratio/low_mean`, `importance_ratio/max`, and so on) precisely so that a trace captured from a
real TRL run and a trace produced here arrive in the same shape, and a key that gets renamed on one
side and not the other reads as a metric that was never logged. `StepRecord` treats missing as
`None` rather than zero, so that failure is quiet: the signal simply stops contributing and coverage
drops, which is why the mapping lives in one place with the fallbacks written out.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO, Any

from grpo_doctor.record import StepRecord


def _open(path: Path) -> IO[str]:
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a testbed trace or a plain JSONL record stream.

    Tolerates both because a corpus trace carries a `_spec` header line and a trace someone
    captured from their own trainer will not.
    """
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with _open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "_spec" in obj:
                header = obj["_spec"]
                continue
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path} contained no records")
    return header, rows


def to_records(rows: list[dict[str, Any]]) -> list[StepRecord]:
    """Map testbed metric dicts onto StepRecord.

    The oracle columns are carried through so `label` can use them; `Monitor.update` strips them at
    its own boundary, which is where the guarantee belongs.
    """
    out = []
    for r in rows:
        out.append(
            StepRecord(
                step=int(r.get("step", 0)),
                reward_mean=r.get("reward"),
                reward_std=r.get("reward_std"),
                frac_reward_zero_std=r.get("frac_reward_zero_std"),
                entropy=r.get("entropy"),
                grad_norm=r.get("grad_norm"),
                learning_rate=r.get("learning_rate"),
                clip_low=r.get("clip_ratio/low_mean"),
                clip_high=r.get("clip_ratio/high_mean"),
                clip_region=r.get("clip_ratio/region_mean"),
                completion_len_mean=r.get("completions/mean_length"),
                completion_clipped_ratio=r.get("completions/clipped_ratio"),
                importance_ratio_max=r.get("importance_ratio/max"),
                importance_ratio_log_std=r.get("importance_ratio/log_std"),
                heldout_accuracy=r.get("oracle/heldout_accuracy"),
                source=r.get("source", "unknown"),
            )
        )
    return out


__all__ = ["load_trace", "to_records"]
