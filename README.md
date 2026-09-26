# DUET-Grow — PyTorch Starter

This repository implements the **corrected DUET-Grow design** from the project review as a research prototype.

## What is implemented now

1. Expressivity residual `R = Δ - δW* A_l`.
2. Projection of `A^(l-1)` off the row-space of `A^l` before width scoring.
3. Whitened-SVD width proposal.
4. Actual nonlinear feature construction and least-squares refit of outgoing weights.
5. Random nonlinear depth lift with greedy feature selection (OMP-style).
6. Held-out realized gain so width and depth are compared in the same loss units.
7. FLOP-normalized score `Γ = realized_gain / extra_FLOPs`.
8. Permutation-based signal test instead of relying only on a clean Marchenko–Pastur null.
9. A growable dense MLP with physically resized tensors and gated birth for width and depth.
10. CIFAR-10/CIFAR-100 loaders and an ImageNet `ImageFolder` loader.

## Important scope

The first implementation is intentionally **dense/MLP + real image data**, because every DUET equation can be unit-tested there. The project review recommends this order before moving to convolution/ResNet growth.

The next research stage is a channel-growing CNN/ResNet18 implementation using im2col statistics and residual-stream depth blocks.

## Run tests

```bash
pip install -r requirements.txt
pytest -q
```

## Run the 1-D falsification experiment

```bash
python scripts/falsification.py
```

## Run a static DUET score study on CIFAR

```bash
python scripts/static_cifar.py --dataset cifar10 --epochs 5 --batch-size 256
```

For CIFAR-100:

```bash
python scripts/static_cifar.py --dataset cifar100 --epochs 5 --batch-size 256
```

## ImageNet

Use an ImageNet directory in standard `ImageFolder` form:

```text
imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

Then:

```bash
python scripts/static_imagenet.py --data /path/to/imagenet --epochs 1 --batch-size 64
```

For a first experiment, use a subset of ImageNet/classes rather than the complete dataset.
