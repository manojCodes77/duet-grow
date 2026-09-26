import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from duet_grow.scoring import GrowthScorer


def make_model(hidden=64):
    return nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def dataset(kind, n=2048):
    x = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
    if kind == "depth":
        y = torch.sin(torch.sin(torch.sin(x)))
    else:
        centers = torch.linspace(-1.5, 1.5, 16).view(1, -1)
        coeff = torch.sin(torch.arange(16).float()).view(1, -1)
        xx = x @ torch.ones(1, 16)
        y = (coeff * torch.exp(-((xx - centers) ** 2) / (2 * 0.08 ** 2))).sum(1, keepdim=True)
    return x, y


def fit(kind):
    torch.manual_seed(0)
    x, y = dataset(kind)
    model = make_model(64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(1500):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Construct a synthetic residual target from the output gradient of MSE.
    with torch.no_grad():
        z1 = model[0](x)
        a1 = torch.relu(z1)
        pred = model[2](a1)
    # dL/dz_out for mean squared loss
    Delta = (y - pred).T
    A_l = z1.T * 0.0 + a1.T
    A_prev = x.T
    scorer = GrowthScorer()
    W = scorer.width(A_prev, A_l, Delta, K_max=8)
    D = scorer.depth(A_l, Delta, h_max=8)
    print(kind, "best width gamma=", max(a.efficiency for a in W) if W else 0.0,
          "best depth gamma=", max(a.efficiency for a in D) if D else 0.0)


if __name__ == "__main__":
    fit("depth")
    fit("width")
