"""Rule monitor: privileged trajectories in, per-step labels out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

from .rules import SafetySpec, Trajectories, compute_signals


@dataclass
class MonitorOutput:
    margin: Tensor  # [T] per-step margin l_t, the HJ boundary function target
    robustness: Tensor  # [T] robustness of G(spec) on the suffix starting at t
    failure: Tensor  # [T] bool, margin < 0
    signals: Dict[str, Tensor]

    @property
    def violated(self) -> bool:
        return bool((self.robustness[0] < 0).item())


class STLMonitor:
    def __init__(self, spec: SafetySpec | None = None, n_circles: int = 3) -> None:
        self.spec = spec or SafetySpec()
        self.n_circles = n_circles

    @torch.no_grad()
    def __call__(self, traj: Trajectories) -> MonitorOutput:
        signals = compute_signals(traj, self.n_circles)
        margin = self.spec.margin(signals)
        # Suffix minimum of the margin equals the robustness of G(state_formula)
        # up to the clipping applied in ``margin``.
        robustness = torch.flip(torch.cummin(torch.flip(margin, [-1]), -1).values, [-1])
        return MonitorOutput(margin, robustness, margin < 0, signals)
