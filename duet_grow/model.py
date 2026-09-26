from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinear(nn.Module):
    """Linear layer whose newly added input block is ramped with gamma."""

    def __init__(self, old: nn.Linear, new_target: torch.Tensor):
        super().__init__()
        self.out_features = old.out_features
        self.old_in_features = old.in_features
        self.new_in_features = int(new_target.shape[1])
        self.weight = nn.Parameter(old.weight.detach().clone())
        self.bias = nn.Parameter(old.bias.detach().clone()) if old.bias is not None else None
        self.new_learn = nn.Parameter(torch.zeros_like(new_target))
        self.register_buffer("new_target", new_target.detach().clone())
        self.gamma = 0.0

    @property
    def in_features(self):
        return self.old_in_features + self.new_in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_old = x[..., : self.old_in_features]
        x_new = x[..., self.old_in_features :]
        y = F.linear(x_old, self.weight, self.bias)
        w_new = self.new_learn + float(self.gamma) * self.new_target
        return y + F.linear(x_new, w_new, None)

    def merge(self) -> nn.Linear:
        out = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        w_new = self.new_learn.detach() + self.new_target
        with torch.no_grad():
            out.weight.copy_(torch.cat([self.weight.detach(), w_new], dim=1))
            if self.bias is not None:
                out.bias.copy_(self.bias.detach())
        return out


class DepthBranch(nn.Module):
    """MLP depth side branch:

    z_next <- W_next h + gamma * W_out relu(W_b h)
    """

    def __init__(self, h_in: int, h_out: int, h: int, Wb: torch.Tensor, Wout: torch.Tensor):
        super().__init__()
        if Wb.shape != (h, h_in):
            raise ValueError(f"Wb must be [{h}, {h_in}], got {tuple(Wb.shape)}")
        if Wout.shape != (h_out, h):
            raise ValueError(f"Wout must be [{h_out}, {h}], got {tuple(Wout.shape)}")
        self.h = h
        self.Wb = nn.Parameter(Wb.detach().clone())
        self.Wout = nn.Parameter(Wout.detach().clone())
        self.gamma = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(F.linear(x, self.Wb, None))
        return float(self.gamma) * F.linear(hidden, self.Wout, None)

    def resize_input(self, new_dim: int) -> None:
        old_dim = self.Wb.shape[1]
        if new_dim == old_dim:
            return
        W = torch.zeros(self.h, new_dim, dtype=self.Wb.dtype, device=self.Wb.device)
        W[:, :old_dim] = self.Wb.data
        self.Wb = nn.Parameter(W)

    def resize_output(self, new_dim: int) -> None:
        old_dim = self.Wout.shape[0]
        if new_dim == old_dim:
            return
        W = torch.zeros(new_dim, self.h, dtype=self.Wout.dtype, device=self.Wout.device)
        W[:old_dim] = self.Wout.data
        self.Wout = nn.Parameter(W)


@dataclass
class PendingGrowth:
    kind: str
    layer: int
    gamma: float = 0.0


class GrowableMLP(nn.Module):
    """Dense MLP supporting physical width growth and MLP-side depth growth.

    Hidden layers are indexed 0..L-1. A depth branch at index i feeds the
    preactivation of hidden layer i+1, or the classifier when i is the last
    hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super().__init__()
        if not hidden_dims:
            raise ValueError("At least one hidden layer is required")
        dims = [input_dim, *hidden_dims, num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.hidden_dims = list(hidden_dims)
        self.depth_branches = nn.ModuleList([nn.Identity() for _ in hidden_dims])
        self.pending: Optional[PendingGrowth] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.layers[0](x))

        for i in range(len(self.hidden_dims) - 1):
            z = self.layers[i + 1](h)
            branch = self.depth_branches[i]
            if not isinstance(branch, nn.Identity):
                z = z + branch(h)
            h = torch.relu(z)

        logits = self.layers[-1](h)
        last_branch = self.depth_branches[-1]
        if not isinstance(last_branch, nn.Identity):
            logits = logits + last_branch(h)
        return logits

    @property
    def width(self) -> List[int]:
        return list(self.hidden_dims)

    def add_depth(self, layer: int, Wb: torch.Tensor, Wout: torch.Tensor) -> None:
        if self.pending is not None:
            raise RuntimeError("Finalize the previous growth event first")
        if not (0 <= layer < len(self.hidden_dims)):
            raise IndexError(layer)
        if not isinstance(self.depth_branches[layer], nn.Identity):
            raise RuntimeError("This starter allows one depth branch per connection")

        h_out = self.hidden_dims[layer + 1] if layer + 1 < len(self.hidden_dims) else self.layers[-1].out_features
        branch = DepthBranch(self.hidden_dims[layer], h_out, Wb.shape[0], Wb, Wout)
        self.depth_branches[layer] = branch
        branch.gamma = 0.0
        self.pending = PendingGrowth("depth", layer, 0.0)

    def add_width(self, layer: int, alpha: torch.Tensor, omega_target: torch.Tensor) -> None:
        """Grow hidden layer by K units and gate their outgoing target weights."""
        if self.pending is not None:
            raise RuntimeError("Finalize the previous growth event first")
        if not (0 <= layer < len(self.hidden_dims)):
            raise IndexError(layer)

        alpha = alpha.detach().to(dtype=self.layers[layer].weight.dtype, device=self.layers[layer].weight.device)
        omega_target = omega_target.detach().to(dtype=self.layers[layer].weight.dtype, device=self.layers[layer].weight.device)
        K = int(alpha.shape[0])
        if alpha.ndim != 2 or alpha.shape[1] != self.layers[layer].in_features:
            raise ValueError("alpha must have shape [K, incoming_dim]")
        if omega_target.shape != (self.layers[layer + 1].out_features, K):
            raise ValueError("omega_target must have shape [out_dim_next, K]")
        if not isinstance(self.layers[layer + 1], nn.Linear):
            raise RuntimeError("The neighboring layer is already in a gated-growth state")

        old = self.layers[layer]
        grown = nn.Linear(old.in_features, old.out_features + K, bias=old.bias is not None)
        with torch.no_grad():
            grown.weight[: old.out_features].copy_(old.weight)
            grown.weight[old.out_features:].copy_(alpha)
            if old.bias is not None:
                grown.bias[: old.out_features].copy_(old.bias)
                grown.bias[old.out_features:].zero_()
        self.layers[layer] = grown
        self.layers[layer + 1] = GatedLinear(self.layers[layer + 1], omega_target)
        self.hidden_dims[layer] += K

        # Existing depth branch feeding this layer: add zero rows.
        if layer > 0:
            branch = self.depth_branches[layer - 1]
            if not isinstance(branch, nn.Identity):
                branch.resize_output(self.hidden_dims[layer])

        # Existing depth branch consuming this layer: add zero columns.
        branch = self.depth_branches[layer]
        if not isinstance(branch, nn.Identity):
            branch.resize_input(self.hidden_dims[layer])

        self.pending = PendingGrowth("width", layer, 0.0)

    def set_gamma(self, gamma: float) -> None:
        if self.pending is None:
            return
        gamma = float(gamma)
        self.pending.gamma = gamma
        if self.pending.kind == "width":
            gated = self.layers[self.pending.layer + 1]
            assert isinstance(gated, GatedLinear)
            gated.gamma = gamma
        else:
            branch = self.depth_branches[self.pending.layer]
            assert isinstance(branch, DepthBranch)
            branch.gamma = gamma

    def finalize_growth(self) -> None:
        if self.pending is None:
            return
        if self.pending.kind == "width":
            gated = self.layers[self.pending.layer + 1]
            assert isinstance(gated, GatedLinear)
            self.layers[self.pending.layer + 1] = gated.merge()
        else:
            branch = self.depth_branches[self.pending.layer]
            assert isinstance(branch, DepthBranch)
            branch.gamma = 1.0
        self.pending = None
