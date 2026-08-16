import torch
from torch import nn


class VNorm(nn.Module):
    """VNorm v1: a parametric activation intended after BatchNorm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(width))
        self.denominator = nn.Parameter(torch.ones(width))
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = (x - self.gamma) / (self.denominator.abs() + 1e-6)
        gate = torch.sigmoid(u)
        return x * gate + self.alpha * x * (1.0 - gate)
