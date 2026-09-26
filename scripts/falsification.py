"""Corrected DUET-Grow falsification experiment.

Uses a two-hidden-layer MLP so the 1-D input does not rank-limit the width
test. It collects true preactivation gradients for both hidden layers.

Targets:
  * depth: iterated tent-map composition
  * width: sum of many localized bumps

This is a diagnostic/falsification experiment, not a universal claim that one
operator must win on every target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from duet_grow.scoring import GrowthScorer


class TinyMLP(nn.Module):
    def __init__(self, h1: int = 8, h2: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(1, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward_with_intermediates(self, x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        z2 = self.fc2(a1)
        a2 = torch.relu(z2)
        out = self.fc3(a2)
        return z1, a1, z2, a2, out


def tent(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.abs(2.0 * x - 1.0)


def dataset(kind: str, n: int = 2048):
    if kind == "depth":
        u = torch.linspace(0.0, 1.0, n).unsqueeze(1)
        y01 = tent(tent(tent(u)))
        return 2.0 * u - 1.0, 2.0 * y01 - 1.0

    if kind == "width":
        x = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
        centers = torch.linspace(-1.7, 1.7, 32).view(1, -1)
        coeff = torch.sin(torch.arange(32, dtype=torch.float64)).view(1, -1)
        xx = x.double() @ torch.ones(1, 32, dtype=torch.float64)
        y = (
            coeff
            * torch.exp(-((xx - centers.double()) ** 2) / (2.0 * 0.045**2))
        ).sum(dim=1, keepdim=True)
        y = y / y.abs().max().clamp_min(1e-12)
        return x, y.float()

    raise ValueError(f"Unknown target: {kind}")


def train_model(kind: str, seed: int, steps: int = 1000):
    torch.manual_seed(seed)
    x, y = dataset(kind)
    model = TinyMLP(8, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(steps):
        *_, pred = model.forward_with_intermediates(x)
        loss = ((pred - y) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, x, y


def score_model(model: TinyMLP, x: torch.Tensor, y: torch.Tensor):
    model.zero_grad(set_to_none=True)

    z1, a1, z2, a2, out = model.forward_with_intermediates(x)
    z2.retain_grad()
    out.retain_grad()

    loss = ((out - y) ** 2).mean()
    loss.backward()

    # DUET convention: Delta = -dL/dz.
    delta_layer1 = -z2.grad.T.detach()
    delta_layer2 = -out.grad.T.detach()

    scorer = GrowthScorer()
    results = []

    for layer, (A_prev, A_l, Delta) in enumerate(
        [
            (x.T.detach(), a1.T.detach(), delta_layer1),
            (a1.T.detach(), a2.T.detach(), delta_layer2),
        ],
        start=1,
    ):
        width_actions = scorer.width(A_prev, A_l, Delta, K_max=4)
        depth_actions = scorer.depth(A_l, Delta, h_max=4)

        for action in width_actions + depth_actions:
            action.layer = layer
            results.append(action)

    return loss.item(), results


def print_results(kind: str, seed: int, loss: float, actions):
    width = [a for a in actions if a.kind == "width"]
    depth = [a for a in actions if a.kind == "depth"]

    best_width = max(width, key=lambda a: a.efficiency, default=None)
    best_depth = max(depth, key=lambda a: a.efficiency, default=None)

    print()
    print("=" * 72)
    print(f"Target: {kind} | seed={seed} | final MSE={loss:.6e}")
    print("=" * 72)

    if best_width:
        print(
            f"best WIDTH : layer={best_width.layer} K={best_width.units} "
            f"gain={best_width.gain:.6e} FLOPs={best_width.flops:.0f} "
            f"efficiency={best_width.efficiency:.6e}"
        )
    else:
        print("best WIDTH : none")

    if best_depth:
        print(
            f"best DEPTH : layer={best_depth.layer} h={best_depth.units} "
            f"gain={best_depth.gain:.6e} FLOPs={best_depth.flops:.0f} "
            f"efficiency={best_depth.efficiency:.6e}"
        )
    else:
        print("best DEPTH : none")

    if best_width and best_depth:
        if best_width.efficiency > best_depth.efficiency:
            print("BEST ACTION: WIDTH")
        elif best_depth.efficiency > best_width.efficiency:
            print("BEST ACTION: DEPTH")
        else:
            print("BEST ACTION: TIE")


def main():
    all_results = []

    for kind in ("depth", "width"):
        for seed in (0, 1, 2):
            model, x, y = train_model(kind, seed)
            loss, actions = score_model(model, x, y)
            print_results(kind, seed, loss, actions)

            width = [a.efficiency for a in actions if a.kind == "width"]
            depth = [a.efficiency for a in actions if a.kind == "depth"]

            all_results.append(
                {
                    "target": kind,
                    "seed": seed,
                    "width_efficiency": max(width) if width else 0.0,
                    "depth_efficiency": max(depth) if depth else 0.0,
                }
            )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for kind in ("depth", "width"):
        rows = [r for r in all_results if r["target"] == kind]
        w = torch.tensor([r["width_efficiency"] for r in rows])
        d = torch.tensor([r["depth_efficiency"] for r in rows])
        print(
            f"{kind:>5} target | "
            f"width mean={w.mean().item():.6e} | "
            f"depth mean={d.mean().item():.6e}"
        )


if __name__ == "__main__":
    main()
