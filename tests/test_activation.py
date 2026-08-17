import torch
import torch.nn as nn

from vnorm import VNorm


def test_forward_shape_and_parameter_count():
    layer = VNorm(8)
    output = layer(torch.randn(4, 8))

    assert output.shape == (4, 8)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 17


def test_backward_produces_finite_gradients():
    layer = VNorm(4)
    inputs = torch.randn(6, 4, requires_grad=True)

    layer(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_batchnorm_vnorm_composition():
    model = nn.Sequential(nn.BatchNorm1d(4), VNorm(4))
    output = model(torch.randn(8, 4))

    assert output.shape == (8, 4)


def test_rejects_wrong_feature_dimension():
    layer = VNorm(4)

    try:
        layer(torch.randn(2, 5))
    except ValueError as error:
        assert "last dimension 4" in str(error)
    else:
        raise AssertionError("VNorm should reject an incompatible feature dimension")
