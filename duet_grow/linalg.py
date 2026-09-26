from __future__ import annotations

import torch


def to_cpu64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64)


def psd_inverse_sqrt(S: torch.Tensor, rel_cutoff: float = 1e-8, ridge: float = 0.0):
    """Return S^{-1/2} for a PSD matrix, zeroing tiny eigenvalues.

    Eigen-decomposition is deliberately done in float64 on CPU, as recommended by
    the implementation review.
    """
    S = to_cpu64(S)
    if ridge:
        S = S + ridge * torch.eye(S.shape[0], dtype=S.dtype)
    evals, evecs = torch.linalg.eigh(S)
    max_eval = float(evals.max().item()) if evals.numel() else 0.0
    cutoff = max_eval * rel_cutoff
    inv_sqrt = torch.where(
        evals > cutoff,
        torch.rsqrt(torch.clamp(evals, min=1e-30)),
        torch.zeros_like(evals),
    )
    inv_sqrt_mat = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T
    return inv_sqrt_mat, evals


def pseudoinverse(A: torch.Tensor, rel_cutoff: float = 1e-8, ridge: float = 0.0):
    A64 = to_cpu64(A)
    if ridge:
        return torch.linalg.solve(A64 + ridge * torch.eye(A64.shape[-1], dtype=A64.dtype), torch.eye(A64.shape[-1], dtype=A64.dtype))
    return torch.linalg.pinv(A64, rtol=rel_cutoff)


def least_squares_map(target: torch.Tensor, features: torch.Tensor, ridge: float = 1e-8):
    """Solve W = argmin ||target - W features||_F^2.

    target: [d_out, N], features: [d_in, N]
    """
    target64 = to_cpu64(target)
    feat64 = to_cpu64(features)
    gram = feat64 @ feat64.T / max(1, feat64.shape[1])
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    W = (target64 @ feat64.T / max(1, feat64.shape[1])) @ torch.linalg.pinv(gram + ridge * eye)
    return W


def rowspace_project_out(A_prev: torch.Tensor, A_l: torch.Tensor):
    """Project rows of A_prev off rowspace(A_l).

    A_prev: [d_prev, N]
    A_l:    [d_l, N]
    Returns A_prev @ (I - P_row(A_l)) without constructing the NxN identity.
    """
    X = to_cpu64(A_prev)
    Y = to_cpu64(A_l)
    if Y.numel() == 0:
        return X
    Q, _ = torch.linalg.qr(Y.T, mode="reduced")  # N x r, spans rowspace(Y)
    if Q.numel() == 0:
        return X
    return X - (X @ Q) @ Q.T


def whitened_svd(residual: torch.Tensor, features: torch.Tensor, rel_cutoff: float = 1e-8):
    """Compute S^{-1/2} whitened cross-covariance and its SVD."""
    R = to_cpu64(residual)
    A = to_cpu64(features)
    N = max(1, A.shape[1])
    S = (A @ A.T) / N
    C = (R @ A.T) / N
    Sinv2, evals = psd_inverse_sqrt(S, rel_cutoff=rel_cutoff)
    Ctilde = C @ Sinv2
    U, s, Vh = torch.linalg.svd(Ctilde, full_matrices=False)
    return s, U, Vh, Sinv2, evals


def residual_energy(R: torch.Tensor) -> float:
    R64 = to_cpu64(R)
    return float((R64.square().mean()).item())
