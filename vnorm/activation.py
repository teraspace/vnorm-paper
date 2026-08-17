"""PyTorch implementation of the VNorm activation."""

from __future__ import annotations

import torch
from torch import nn


class VNorm(nn.Module):
    """VNorm v0.1: a parametric activation intended after BatchNorm.

    The final input dimension is interpreted as the feature dimension. For an
    input shaped ``[batch, features]``, this is the usual MLP layout.
    """

    def __init__(
        self,
        features: int,
        denominator_init: float = 1.0,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError("features must be a positive integer")
        self.features = features
        self.gamma = nn.Parameter(torch.zeros(features))
        self.denominator = nn.Parameter(torch.full((features,), denominator_init))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0 or x.shape[-1] != self.features:
            raise ValueError(
                f"expected input with last dimension {self.features}, "
                f"got shape {tuple(x.shape)}"
            )
        u = (x - self.gamma) / (self.denominator.abs() + 1e-6)
        gate = torch.sigmoid(u)
        return x * gate + self.alpha * x * (1.0 - gate)

    def extra_repr(self) -> str:
        return f"features={self.features}"
