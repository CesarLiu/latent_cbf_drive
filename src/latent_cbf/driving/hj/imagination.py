"""Interface between a world model and the latent HJ learner.

The learner only needs a batched, resettable rollout of latent states under
discrete candidate actions. ``adapters.smart_imagination`` implements this for
the SMART traffic model; ``toy.car_following`` implements it for unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class Candidates:
    """Candidate actions for the ego at the current step.

    ``idx`` are action ids in the world model's vocabulary (motion tokens for
    SMART). The learner and the filter address actions by their slot ``k`` in
    ``[0, K)``.
    """

    idx: Tensor  # [B, K] long
    feat: Tensor  # [B, K, F] action features fed to the Q network
    logp: Tensor  # [B, K] log-probability under the nominal policy

    def take(self, slot: Tensor) -> Tensor:
        """Features of the chosen slot, [B, F]."""
        index = slot.view(-1, 1, 1).expand(-1, 1, self.feat.shape[-1])
        return self.feat.gather(1, index).squeeze(1)


class Imagination(ABC):
    latent_dim: int
    action_feat_dim: int
    horizon: int

    @abstractmethod
    def reset(self, batch_size: int) -> Tensor:
        """Start ``batch_size`` rollouts and return the latent state [B, D]."""

    @abstractmethod
    def candidates(self) -> Candidates:
        """Candidate ego actions at the current latent state."""

    @abstractmethod
    def step(self, slot: Tensor) -> Tensor:
        """Apply the candidate in ``slot`` [B] and return the next latent [B, D]."""

    def true_margin(self) -> Optional[Tensor]:
        """Ground-truth margin [B] of the current state, if the model knows it."""
        return None
