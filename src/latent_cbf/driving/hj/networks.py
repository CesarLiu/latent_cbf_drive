from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        layers += [nn.Linear(last, width), nn.SiLU()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class MarginHead(nn.Module):
    """Safety boundary function l(z) in (-1, 1); negative means constraint violated."""

    def __init__(self, latent_dim: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.net = mlp(latent_dim, hidden, 1)

    def raw(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)

    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(self.raw(z))


class QNetwork(nn.Module):
    """Safety critic Q(z, a) over candidate actions described by feature vectors.

    Scoring actions through their features, rather than one output per action
    id, keeps the critic independent of the vocabulary size (SMART has 2048
    motion tokens but only the top-K are ever evaluated).
    """

    def __init__(
        self,
        latent_dim: int,
        action_feat_dim: int,
        hidden: Sequence[int] = (512, 512, 512),
        bounded: bool = True,
    ) -> None:
        super().__init__()
        self.net = mlp(latent_dim + action_feat_dim, hidden, 1)
        self.bounded = bounded

    def forward(self, z: Tensor, action_feat: Tensor) -> Tensor:
        """z: [B, D]; action_feat: [B, F] or [B, K, F] -> [B] or [B, K]."""
        if action_feat.dim() == 3:
            z = z.unsqueeze(1).expand(-1, action_feat.shape[1], -1)
        q = self.net(torch.cat([z, action_feat], dim=-1)).squeeze(-1)
        return torch.tanh(q) if self.bounded else q

    def value(self, z: Tensor, cand_feat: Tensor) -> Tensor:
        """V(z) = max over candidates, [B]."""
        return self.forward(z, cand_feat).amax(-1)
