# VNorm

VNorm is a parametric activation intended to be used after BatchNorm:

```text
Linear/Conv -> BatchNorm -> VNorm
```

This repository contains the public, minimal version of the method and the
CIFAR-100 MLP benchmark used in the paper draft. Private ideas such as
Prisma-Rombo and unpublished extensions are intentionally not included.

## Install

```bash
pip install torch torchvision numpy
```

## Run the benchmark

The default experiment runs CIFAR-100 with three seeds and 30 epochs:

```bash
python experiments/cifar100_mlp.py
```

For a quick smoke run:

```bash
EPOCHS=5 SEEDS=42 python experiments/cifar100_mlp.py
```

Results are written to `results/cifar100_mlp_results.csv`.

## Current claim

The intended claim is narrow: VNorm is a BatchNorm-compatible activation and
shows its strongest behavior in the `BatchNorm -> VNorm` regime. The benchmark
does not claim universal superiority without BatchNorm.
