"""The gradient panel must report what happened, including when nothing happened.

Two of these tests pin claims the project makes in writing:

- `test_logged_norm_is_pre_clip_and_diverges_from_the_applied_one` is the concrete demonstration
  that HF's logged `grad_norm` can climb while the update it produced is pinned at `max_grad_norm`.
  If S6 turns out to be uninformative in the ablation, this test is the mechanism.
- `test_zero_gradient_still_moves_the_policy` pins the silent-death combination: degenerate groups
  produce `grad_norm == 0` while Adam keeps stepping from stale momentum, so "no gradient" does not
  mean "no drift". A monitor watching only the norm sees a healthy-looking zero.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from testbed.core.model import ModelConfig, TinyGPT  # noqa: E402
from testbed.core.optim import InstrumentedAdamW, OptimConfig  # noqa: E402

pytestmark = pytest.mark.torch

PANEL_KEYS = {
    "grad_norm",
    "grad_norm_postclip",
    "grad_norm_clip_active",
    "update_norm",
    "update_ratio",
    "learning_rate",
    "nonfinite_grad",
}


def _model(seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(ModelConfig(vocab_size=13, d_model=32, n_layer=2, n_head=4, max_seq_len=32))


def _fill_grads(model: TinyGPT, value: float) -> None:
    for p in model.parameters():
        p.grad = torch.full_like(p, value)


def test_panel_always_has_every_key() -> None:
    """Ragged traces are indistinguishable from unavailable signals downstream."""
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig())
    _fill_grads(model, 1e-3)
    assert set(opt.step()) == PANEL_KEYS


def test_logged_norm_is_pre_clip_and_diverges_from_the_applied_one() -> None:
    """The headline distinction. A 10x larger gradient logs 10x larger and applies the same."""
    small, big = {}, {}
    for g, out in ((1e-2, small), (1e-1, big)):
        model = _model()
        opt = InstrumentedAdamW(model, OptimConfig(max_grad_norm=1.0))
        _fill_grads(model, g)
        out.update(opt.step())

    assert big["grad_norm"] > small["grad_norm"] > 1.0, "both should exceed the clip threshold"
    assert big["grad_norm"] == pytest.approx(10 * small["grad_norm"], rel=1e-4)
    # ...yet the norm that actually scaled the update is identical and pinned at the threshold.
    assert big["grad_norm_postclip"] == pytest.approx(1.0, rel=1e-4)
    assert small["grad_norm_postclip"] == pytest.approx(1.0, rel=1e-4)
    assert big["grad_norm_clip_active"] == 1.0


def test_clip_inactive_leaves_the_gradient_alone() -> None:
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(max_grad_norm=1.0))
    _fill_grads(model, 1e-6)
    m = opt.step()
    assert m["grad_norm"] < 1.0
    assert m["grad_norm_clip_active"] == 0.0
    assert m["grad_norm_postclip"] == pytest.approx(m["grad_norm"], rel=1e-5)


def test_clipping_can_be_disabled() -> None:
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(max_grad_norm=0.0))
    _fill_grads(model, 1.0)
    m = opt.step()
    assert m["grad_norm_postclip"] == pytest.approx(m["grad_norm"], rel=1e-5)
    assert m["grad_norm_clip_active"] == 0.0


def test_zero_gradient_still_moves_the_policy() -> None:
    """Silent death: grad_norm == 0 while Adam keeps stepping from momentum.

    This is why "no gradient" is not "no drift", and why a norm threshold cannot detect advantage
    starvation. Finding #1, stated as an executable fact.
    """
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(lr=1e-2))

    _fill_grads(model, 1e-2)
    opt.step()  # builds momentum

    _fill_grads(model, 0.0)
    m = opt.step()
    assert m["grad_norm"] == 0.0
    assert m["update_ratio"] > 0.0, "stale momentum must still move the weights"


def test_nonfinite_gradient_skips_the_step_entirely() -> None:
    """A NaN run is a lost run, not a labeled failure. Parameters and Adam state stay untouched."""
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig())
    before = [p.detach().clone() for p in model.parameters()]

    _fill_grads(model, float("nan"))
    m = opt.step()

    assert m["nonfinite_grad"] == 1.0
    assert m["update_ratio"] == 0.0
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p.detach(), b)
    assert len(opt.opt.state) == 0, "no Adam state should have been created"


def test_inf_gradient_is_caught_too() -> None:
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig())
    _fill_grads(model, float("inf"))
    assert opt.step()["nonfinite_grad"] == 1.0


def test_warmup_ramps_then_holds() -> None:
    """Linear warmup only. A decaying LR would make falling entropy ambiguous between F4 and the
    schedule, which would confound the entropy-collapse family."""
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(lr=1e-3, warmup_steps=4))
    lrs = []
    for _ in range(6):
        _fill_grads(model, 1e-4)
        lrs.append(opt.step()["learning_rate"])
    assert lrs == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3])


def test_no_warmup_is_constant() -> None:
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(lr=3e-4, warmup_steps=0))
    _fill_grads(model, 1e-4)
    assert opt.step()["learning_rate"] == pytest.approx(3e-4)


def test_weight_decay_skips_norms_and_biases() -> None:
    """Decaying LayerNorm gains suppresses entropy for reasons unrelated to the policy, which we
    would then misread as F4."""
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(weight_decay=0.1))
    decay_group, no_decay_group = opt.opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(decay_group["params"]) + len(no_decay_group["params"]) == n_params


def test_update_ratio_grows_with_learning_rate() -> None:
    ratios = []
    for lr in (1e-4, 1e-3):
        model = _model()
        opt = InstrumentedAdamW(model, OptimConfig(lr=lr))
        _fill_grads(model, 1e-3)
        ratios.append(opt.step()["update_ratio"])
    assert 0.0 < ratios[0] < ratios[1]


def test_update_ratio_can_be_turned_off() -> None:
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig(track_update_ratio=False))
    _fill_grads(model, 1e-3)
    m = opt.step()
    assert m["update_ratio"] == 0.0 and m["update_norm"] == 0.0


def test_grads_are_cleared_after_a_step() -> None:
    """Otherwise the next step's norm silently accumulates two batches."""
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig())
    _fill_grads(model, 1e-3)
    opt.step()
    assert all(p.grad is None for p in model.parameters())


def test_step_count_advances_even_on_a_skipped_step() -> None:
    model = _model()
    opt = InstrumentedAdamW(model, OptimConfig())
    _fill_grads(model, float("nan"))
    opt.step()
    assert opt.step_count == 1
