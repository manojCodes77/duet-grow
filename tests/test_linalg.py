import torch

from duet_grow.linalg import least_squares_map, rowspace_project_out, whitened_svd


def test_project_is_orthogonal():
    torch.manual_seed(0)
    A_prev = torch.randn(7, 64, dtype=torch.float64)
    A_l = torch.randn(5, 64, dtype=torch.float64)
    P = rowspace_project_out(A_prev, A_l)
    err = torch.linalg.norm(P @ A_l.T) / torch.linalg.norm(P)
    assert float(err) < 1e-8


def test_whitened_eigenvectors_are_stable():
    torch.manual_seed(0)
    A = torch.randn(4, 128, dtype=torch.float64)
    R = torch.randn(3, 128, dtype=torch.float64)
    s, *_ = whitened_svd(R, A)
    assert s.ndim == 1
    assert torch.all(s[:-1] >= s[1:] - 1e-12)


def test_lstsq_reduces_error():
    torch.manual_seed(0)
    X = torch.randn(3, 100, dtype=torch.float64)
    W0 = torch.randn(5, 3, dtype=torch.float64)
    Y = W0 @ X + 0.1 * torch.randn(5, 100, dtype=torch.float64)
    W = least_squares_map(Y, X)
    before = (Y.square().mean()).item()
    after = ((Y - W @ X).square().mean()).item()
    assert after < before
