"""Shape, determinism and gradient checks for the CNN and Transformer models."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_models import CNN1DRegressor, TransformerRegressor, build_model  # noqa: E402

BATCH, WINDOW, FEATURES = 6, 30, 14

CONFIGS = [
    ("cnn1d", {}),
    ("cnn1d", {"layers": 3, "dilation": "exp", "norm": "batch", "residual": True, "pool": "max"}),
    ("cnn1d", {"norm": "layer", "pool": "last", "kernel": 3}),
    ("transformer", {}),
    ("transformer", {"pos": "learned", "pool": "last", "d_model": 64, "layers": 3}),
]


def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, WINDOW, FEATURES)


@pytest.mark.parametrize("name,kwargs", CONFIGS)
def test_shapes_and_registry(name, kwargs):
    model = build_model(name, **kwargs)
    x = batch()
    assert model(x).shape == (BATCH,)
    z = model.encode(x)
    assert z.ndim == 2 and z.shape[0] == BATCH
    assert z.shape[1] == (kwargs.get("channels", 64) if name == "cnn1d" else kwargs.get("d_model", 32))
    assert torch.isfinite(model(x)).all()


@pytest.mark.parametrize("name,kwargs", CONFIGS)
def test_eval_is_deterministic(name, kwargs):
    model = build_model(name, **kwargs).eval()
    x = batch()
    assert torch.equal(model(x), model(x))


@pytest.mark.parametrize("name,kwargs", CONFIGS)
def test_gradient_reaches_every_parameter(name, kwargs):
    model = build_model(name, **kwargs).train()
    model(batch()).pow(2).mean().backward()
    for pname, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), pname
        assert p.grad.abs().sum() > 0, f"zero gradient: {pname}"


def test_no_future_information_leaks_between_samples():
    # Samples in a batch must not influence each other (eval mode, batch norm uses running stats).
    model = build_model("cnn1d", norm="batch").eval()
    x = batch()
    y = model(x)
    x2 = x.clone()
    x2[0] += 5.0
    assert torch.allclose(model(x2)[1:], y[1:], atol=1e-6)


def test_receptive_field():
    assert CNN1DRegressor().receptive_field == 1 + 2 * 4  # 2 layers, kernel 5
    assert CNN1DRegressor(layers=3, dilation="exp", kernel=3).receptive_field == 1 + 2 * (1 + 2 + 4)


def test_cnn_rejects_even_kernel_and_bad_options():
    with pytest.raises(ValueError):
        CNN1DRegressor(kernel=4)
    with pytest.raises(ValueError):
        CNN1DRegressor(pool="median")
    with pytest.raises(ValueError):
        TransformerRegressor(pos="rotary")


def test_attention_weights_are_a_distribution():
    model = TransformerRegressor().eval()
    w = model.attention_weights(batch())
    assert w.shape == (BATCH, WINDOW, WINDOW)
    assert torch.allclose(w.sum(-1), torch.ones(BATCH, WINDOW), atol=1e-5)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("nope")
