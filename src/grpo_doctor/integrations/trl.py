"""Drop-in monitoring for a real TRL `GRPOTrainer`.

    from grpo_doctor.integrations.trl import VitalsCallback

    cfg = GRPOConfig(logging_steps=1, num_iterations=2, output_dir="out")
    trainer = GRPOTrainer(..., args=cfg, callbacks=[VitalsCallback(on_alarm="warn")])

No subclassing and no monkeypatching. `GRPOTrainer.log()` does `logs.update(metrics)` and then
calls `super().log(logs)`, which ends in `callback_handler.on_log(...)`, so a plain
`TrainerCallback` sees every GRPO metric. That is deliberately the whole integration: anything
requiring a trainer subclass would break on TRL's next release, and TRL shipped nine of those in
two months.

Two configuration mistakes are fatal to the measurement and neither announces itself, so both are
checked at `on_train_begin` and raised rather than warned about.

**`logging_steps` must be 1.** `GRPOTrainer.log()` averages its accumulated `_metrics` over the
logging window and then clears them. At the default of 500 the monitor would receive 500-step
means, and every signal would be a smoothed version of something it was calibrated on unsmoothed.
It would still produce numbers.

**`num_iterations` must be >= 2 for the clipping signals to exist.** At 1 the rollout policy *is*
the policy being updated, `old_per_token_logps` is None, the ratio is 1 by construction, and both
clip fractions are exactly 0.0 for the entire run. That is a structurally absent signal, not a
quiet one. This is a warning rather than an error -- mu=1 is a legitimate way to train -- but the
affected signals are marked unavailable rather than being fed zeros a monitor would read as calm.
"""

from __future__ import annotations

import warnings
from typing import Any

from grpo_doctor.monitor import Monitor
from grpo_doctor.record import StepRecord
from grpo_doctor.snapshot import Level, VitalsSnapshot

try:  # pragma: no cover - exercised only in the TRL environment
    from transformers import TrainerCallback as _TrainerCallback

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _TrainerCallback = object  # type: ignore[assignment,misc]
    HAVE_TRANSFORMERS = False


ON_ALARM = ("warn", "stop", "raise", "ignore")


class VitalsCallback(_TrainerCallback):  # type: ignore[misc,valid-type]
    """Watch a GRPO run and optionally halt it.

    Args:
        monitor: an existing Monitor, or None to build the default panel.
        on_alarm: what to do when the level reaches ALARM.
            "warn"   emit a warning and keep training (default)
            "stop"   set `control.should_training_stop`, ending the run cleanly
            "raise"  raise, for CI
            "ignore" record only
        min_level: the level at which `on_alarm` fires.

    `on_alarm="stop"` is opt-in and never the default. Halting someone's training run on a
    heuristic is a decision for them to make, and a monitor that does it uninvited will be removed
    the first time it is wrong.
    """

    def __init__(
        self,
        monitor: Monitor | None = None,
        *,
        on_alarm: str = "warn",
        min_level: Level = Level.ALARM,
        verbose: bool = False,
    ) -> None:
        if on_alarm not in ON_ALARM:
            raise ValueError(f"on_alarm must be one of {ON_ALARM}, got {on_alarm!r}")
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                "VitalsCallback needs transformers/trl. Install with: pip install 'grpo-doctor[trl]'"
            )
        self.monitor = monitor or Monitor()
        self.on_alarm = on_alarm
        self.min_level = min_level
        self.verbose = verbose
        self.snapshots: list[VitalsSnapshot] = []
        self.alarm_step: int | None = None

    # --- lifecycle ---------------------------------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        check_config(args)
        return control

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        if not logs:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        rec = StepRecord.from_trl_log(logs, step=step)
        snap = self.monitor.update(rec)
        self.snapshots.append(snap)

        if self.verbose:
            print(
                f"[grpo-doctor] step {snap.step} {snap.level.name} score={snap.score:.2f} coverage={snap.coverage:.0%}"
            )

        if snap.level >= self.min_level:
            if self.alarm_step is None:
                self.alarm_step = snap.step
            self._react(snap, control)
        return control

    def _react(self, snap: VitalsSnapshot, control: Any) -> None:
        detail = "; ".join(a.message for a in snap.alerts) or f"score {snap.score:.2f}"
        message = (
            f"grpo-doctor: {snap.level.name} at step {snap.step} -- {detail} "
            f"(panel coverage {snap.coverage:.0%})"
        )
        if self.on_alarm == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        elif self.on_alarm == "stop":
            warnings.warn(message + " -- stopping training", RuntimeWarning, stacklevel=2)
            control.should_training_stop = True
        elif self.on_alarm == "raise":
            raise RuntimeError(message)

    @property
    def last(self) -> VitalsSnapshot | None:
        return self.monitor.last


def check_config(args: Any) -> None:
    """Validate a TRL GRPOConfig. Raises on what breaks the measurement, warns on what limits it."""
    logging_steps = getattr(args, "logging_steps", None)
    if logging_steps is not None and logging_steps != 1:
        raise ValueError(
            f"grpo-doctor needs logging_steps=1, got {logging_steps}. TRL averages its metrics "
            "over the logging window and then clears them, so at any other value the monitor "
            "receives window means rather than per-step values -- and would still produce "
            "plausible numbers from them."
        )

    num_iterations = getattr(args, "num_iterations", None)
    if num_iterations is not None and num_iterations < 2:
        warnings.warn(
            f"num_iterations={num_iterations}: the importance ratio is 1 by construction, so "
            "clip_ratio/low_mean and clip_ratio/high_mean will be exactly 0.0 for the whole run. "
            "Those signals are structurally absent, not calm; grpo-doctor marks them unavailable "
            "rather than reading the zeros as healthy.",
            UserWarning,
            stacklevel=2,
        )

    # A sentinel, not `False`: a config object that simply lacks the attribute is a different
    # situation from one that has it turned off, and warning about the former would train users to
    # ignore the warning.
    log_completions = getattr(args, "log_completions", None)
    if log_completions is False:
        warnings.warn(
            "log_completions=False: per-group rewards are unavailable, so the starvation signal "
            "falls back to TRL's frac_reward_zero_std, whose torch.isclose test misses groups with "
            "a small but nonzero reward spread.",
            UserWarning,
            stacklevel=2,
        )


__all__ = ["HAVE_TRANSFORMERS", "ON_ALARM", "VitalsCallback", "check_config"]
