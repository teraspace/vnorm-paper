# VNorm paper draft

## Working title

VNorm: A BatchNorm-Compatible Parametric Activation Function for Shallow Neural Networks

## Central hypothesis

VNorm is designed to operate after BatchNorm. Its learnable threshold,
denominator and negative-path coefficient complement the standardized feature
distribution produced by BatchNorm.

## Main claim

On the CIFAR-100 MLP benchmark, VNorm is competitive with or superior to
standard activation functions when used in the recommended
`BatchNorm -> VNorm` configuration.

## Scope

This first paper intentionally does not disclose Prisma-Rombo, unpublished
theoretical extensions, or future applications. It evaluates VNorm as a
standalone activation and reports the limitations of the BatchNorm-dependent
operating regime.

## Results currently available

The existing two-hidden-layer CIFAR-100 benchmark with BatchNorm reports:

| Activation | Test accuracy |
| --- | ---: |
| VNorm | 26.31 +/- 0.83% |
| SiLU | 25.85 +/- 0.20% |
| ReLU | 25.57 +/- 0.45% |
| PReLU | 25.32 +/- 0.49% |
| CELU | 25.18 +/- 0.18% |

The one-hidden-layer benchmark in `experiments/cifar100_mlp.py` is the
confirmatory experiment for the first public version.
