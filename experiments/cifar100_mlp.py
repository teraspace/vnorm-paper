import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vnorm.activation import VNorm


DATA_DIR = os.getenv("DATA_DIR", "./data")
EPOCHS = int(os.getenv("EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
SEEDS = [int(value) for value in os.getenv("SEEDS", "1,7,42").split(",")]
ACTIVATIONS = tuple(
    value.strip()
    for value in os.getenv("ACTIVATIONS", "relu,celu,silu,prelu,vnorm").split(",")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "cifar100_mlp_results.csv"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])
    train = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=transform)

    split_generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=split_generator)
    validation_size = 5000
    train_indices = order[:-validation_size].tolist()
    validation_indices = order[-validation_size:].tolist()

    loader_generator = torch.Generator().manual_seed(seed + 1)
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": 2,
        "pin_memory": DEVICE.type == "cuda",
    }
    return (
        DataLoader(Subset(train, train_indices), shuffle=True, generator=loader_generator, **common),
        DataLoader(Subset(train, validation_indices), shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


class SingleHiddenLayerMLP(nn.Module):
    def __init__(self, activation_name: str, hidden_width: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, hidden_width)
        self.bn1 = nn.BatchNorm1d(hidden_width)
        self.activation = self.make_activation(activation_name, hidden_width)
        self.head = nn.Linear(hidden_width, 100)

    @staticmethod
    def make_activation(name: str, width: int) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "celu":
            return nn.CELU()
        if name == "silu":
            return nn.SiLU()
        if name == "prelu":
            return nn.PReLU(1, 0.25)
        if name == "vnorm":
            return VNorm(width)
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        return self.head(x)


def paired_init(model: SingleHiddenLayerMLP, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed + 10000)
    with torch.no_grad():
        limit = 1.0 / math.sqrt(model.fc1.in_features)
        model.fc1.weight.uniform_(-limit, limit, generator=generator)
        model.fc1.bias.uniform_(-limit, limit, generator=generator)
        limit = 1.0 / math.sqrt(model.head.in_features)
        model.head.weight.uniform_(-limit, limit, generator=generator)
        model.head.bias.uniform_(-limit, limit, generator=generator)


def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(DEVICE)).argmax(1).cpu()
            correct += int((outputs == targets).sum())
            total += len(targets)
    return 100.0 * correct / total


def run(seed: int, activation_name: str) -> dict:
    seed_all(seed)
    train_loader, validation_loader, test_loader = build_loaders(seed)
    model = SingleHiddenLayerMLP(activation_name).to(DEVICE)
    paired_init(model, seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_validation = -1.0
    best_epoch = 0
    best_state = None
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        scheduler.step()
        validation = accuracy(model, validation_loader)
        average_loss = total_loss / len(train_loader.dataset)
        print(
            f"seed={seed} {activation_name:5s} ep={epoch:02d} "
            f"loss={average_loss:.4f} val={validation:.2f}%"
        )
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_accuracy = accuracy(model, test_loader)
    elapsed = time.time() - start
    print(
        f"TEST={test_accuracy:.2f}% params={parameter_count:,} "
        f"best_epoch={best_epoch} seconds={elapsed:.1f}"
    )
    return {
        "seed": seed,
        "activation": activation_name,
        "params": parameter_count,
        "val": best_validation,
        "test": test_accuracy,
        "best_epoch": best_epoch,
        "seconds": elapsed,
    }


def main() -> None:
    print(f"device={DEVICE} seeds={SEEDS} epochs={EPOCHS} activations={ACTIVATIONS}")
    rows = [run(seed, activation) for seed in SEEDS for activation in ACTIVATIONS]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
