from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .linalg import least_squares_map, residual_energy, rowspace_project_out, whitened_svd


@dataclass
class GrowthAction:
    kind: str
    layer: int
    units: int
    gain: float
    flops: float
    efficiency: float
    payload: dict


class GrowthScorer:
    """Corrected DUET width/depth scoring for a dense feed-forward layer."""

    def __init__(self, activation: str = "relu", ridge: float = 1e-8, H_multiplier: int = 8):
        self.ridge = ridge
        self.H_multiplier = H_multiplier
        if activation != "relu":
            raise ValueError("Starter currently supports relu for the nonlinear lift")

    @staticmethod
    def act(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def expressivity_residual(self, A_l: torch.Tensor, Delta: torch.Tensor):
        deltaW = least_squares_map(Delta, A_l, ridge=self.ridge)
        R = Delta.detach().cpu().double() - deltaW @ A_l.detach().cpu().double()
        return deltaW, R

    def width(self, A_prev, A_l, Delta, K_max: Optional[int] = None):
        """Return width proposals from whitened SVD + actual nonlinear refit."""
        _, R = self.expressivity_residual(A_l, Delta)
        A_prev64 = A_prev.detach().cpu().double()
        A_l64 = A_l.detach().cpu().double()
        A_proj = rowspace_project_out(A_prev64, A_l64)

        s, U, Vh, Sinv2, _ = whitened_svd(R, A_proj)
        rank = int((s > 1e-12).sum().item())
        Kmax = min(rank, s.numel()) if K_max is None else min(K_max, s.numel())
        if Kmax <= 0:
            return []

        actions = []
        # Proposed incoming vectors. These are only proposals; actual gains use ReLU features.
        alpha_all = []
        for k in range(Kmax):
            lam = float(s[k].item())
            alpha = (lam ** 0.5) * (Sinv2 @ Vh.T[:, k])
            alpha_all.append(alpha)

        # Greedily add proposed units and refit outgoing weights on this batch.
        F_parts = []
        A_prev_eval = A_prev64
        for K in range(1, Kmax + 1):
            alpha = alpha_all[K - 1]
            Fk = self.act(alpha.unsqueeze(0) @ A_prev_eval)
            F_parts.append(Fk)
            F = torch.cat(F_parts, dim=0)
            omega = least_squares_map(R, F, ridge=self.ridge)
            err0 = (R.square().mean()).item()
            err1 = ((R - omega @ F).square().mean()).item()
            gain = max(0.0, float(err0 - err1))
            dout = R.shape[0]
            dprev = A_prev.shape[0]
            # Linear-layer convention: one multiply + add = 2 FLOPs.
            flops = float(2 * K * (dprev + dout))
            actions.append(
                GrowthAction(
                    kind="width",
                    layer=-1,
                    units=K,
                    gain=gain,
                    flops=flops,
                    efficiency=gain / max(flops, 1.0),
                    payload={"alpha": torch.stack(alpha_all[:K]), "omega_target": omega},
                )
            )
        return actions

    def depth(self, A_l, Delta, h_max: int, theta: Optional[torch.Tensor] = None):
        """Random nonlinear lift + OMP-style feature selection + realized gain."""
        _, R = self.expressivity_residual(A_l, Delta)
        A = A_l.detach().cpu().double()
        dout, N = R.shape
        H = self.H_multiplier * h_max
        if theta is None:
            g = torch.Generator(device="cpu").manual_seed(12345)
            theta = torch.randn(H, A.shape[0], generator=g, dtype=torch.float64) / max(A.shape[0], 1) ** 0.5
        else:
            theta = theta.detach().cpu().double()
        Phi = self.act(theta @ A)

        selected = []
        residual = R.clone()
        F_sel = None
        actions = []
        max_h = min(h_max, H)
        for h in range(1, max_h + 1):
            scores = []
            norms = Phi.square().sum(dim=1).sqrt().clamp_min(1e-12)
            for j in range(H):
                if j in selected:
                    scores.append(float("-inf"))
                    continue
                corr = torch.linalg.norm(residual @ Phi[j].unsqueeze(1)) / norms[j]
                scores.append(float(corr.item()))
            j = int(torch.tensor(scores).argmax().item())
            selected.append(j)
            F_sel = Phi[selected]
            Wout = least_squares_map(R, F_sel, ridge=self.ridge)
            err0 = R.square().mean().item()
            err1 = (R - Wout @ F_sel).square().mean().item()
            gain = max(0.0, float(err0 - err1))
            residual = R - Wout @ F_sel
            dl = A.shape[0]
            flops = float(2 * h * (dl + dout))
            actions.append(
                GrowthAction(
                    kind="depth",
                    layer=-1,
                    units=h,
                    gain=gain,
                    flops=flops,
                    efficiency=gain / max(flops, 1.0),
                    payload={"theta": theta[selected].clone(), "Wout_target": Wout, "selected": selected.copy()},
                )
            )
        return actions

    def score_layer(self, A_prev, A_l, Delta, hidden_dim_next: int, width_max: int, depth_max: int):
        width = self.width(A_prev, A_l, Delta, K_max=width_max)
        depth = self.depth(A_l, Delta, h_max=depth_max)
        for a in width + depth:
            a.layer = -1
        return width + depth


def permutation_signal_test(A_prev, A_l, Delta, scorer: GrowthScorer, n_perm: int = 8, seed: int = 0):
    """Permutation null for width lambda_1; returns observed, null 95th percentile."""
    _, R = scorer.expressivity_residual(A_l, Delta)
    obs_s, *_ = whitened_svd(R, rowspace_project_out(A_prev, A_l))
    observed = float(obs_s[0].item()) if obs_s.numel() else 0.0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    null = []
    for _ in range(n_perm):
        perm = torch.randperm(R.shape[1], generator=gen)
        Rp = R[:, perm]
        s, *_ = whitened_svd(Rp, rowspace_project_out(A_prev, A_l))
        null.append(float(s[0].item()) if s.numel() else 0.0)
    q95 = float(torch.tensor(null).quantile(0.95).item()) if null else 0.0
    return observed, q95, null
