"""Latent safety filter over discrete candidate actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .hj.imagination import Candidates
from .hj.networks import QNetwork


@dataclass
class FilterDecision:
    slot: Tensor  # [B] chosen candidate slot
    action_idx: Tensor  # [B] chosen action id in the world model's vocabulary
    intervened: Tensor  # [B] bool, the nominal action was replaced
    q_nominal: Tensor  # [B] Q of the nominal action
    q_selected: Tensor  # [B] Q of the executed action
    value: Tensor  # [B] max_a Q(z, a)
    infeasible: Tensor  # [B] bool, no candidate cleared the threshold


class LatentSafetyFilter:
    """Replace the nominal action when its safety value is below a threshold.

    Modes:
        ``cbf``: among candidates with ``Q > threshold`` execute the one closest
            to the nominal action (minimal intervention).
        ``lr``: least-restrictive switching; execute ``argmax_a Q`` whenever the
            nominal action fails the check.

    If no candidate clears the threshold both modes fall back to ``argmax_a Q``
    and flag the step as infeasible. ``threshold`` is 0 for the uncalibrated
    filter and the conformal quantile otherwise (see ``conformal``).
    """

    def __init__(self, q: QNetwork, threshold: float = 0.0, mode: str = "cbf") -> None:
        if mode not in ("cbf", "lr"):
            raise ValueError(f"unknown filter mode: {mode}")
        self.q, self.threshold, self.mode = q, float(threshold), mode

    @torch.no_grad()
    def __call__(
        self,
        z: Tensor,
        cand: Candidates,
        nominal_slot: Tensor,
        distance: Optional[Tensor] = None,
    ) -> FilterDecision:
        """
        Args:
            z: [B, D] latent state.
            cand: candidate actions; must contain the nominal action.
            nominal_slot: [B] slot of the nominal action.
            distance: optional [B, K] distance of every candidate to the nominal
                action (e.g. end-point distance of motion tokens). Defaults to
                the Euclidean distance between action features.
        """
        q = self.q(z, cand.feat)  # [B, K]
        q_nominal = q.gather(1, nominal_slot.unsqueeze(1)).squeeze(1)
        value, best = q.max(-1)
        ok = q > self.threshold
        nominal_ok = q_nominal > self.threshold
        infeasible = ~ok.any(-1)

        if self.mode == "lr":
            fallback = best
        else:
            if distance is None:
                distance = (cand.feat - cand.take(nominal_slot).unsqueeze(1)).norm(dim=-1)
            closest_ok = distance.masked_fill(~ok, float("inf")).argmin(-1)
            fallback = torch.where(infeasible, best, closest_ok)

        slot = torch.where(nominal_ok, nominal_slot, fallback)
        return FilterDecision(
            slot=slot,
            action_idx=cand.idx.gather(1, slot.unsqueeze(1)).squeeze(1),
            intervened=slot != nominal_slot,
            q_nominal=q_nominal,
            q_selected=q.gather(1, slot.unsqueeze(1)).squeeze(1),
            value=value,
            infeasible=infeasible,
        )
