"""Car-following toy problem used to test the latent HJ pipeline end to end.

The ego follows a lead vehicle that occasionally brakes hard. The ego picks one
of a few acceleration "tokens" every 0.5 s. The nominal policy tracks a target
speed and ignores the gap, so it rear-ends the lead when the lead brakes. The
lead's braking mode is hidden from the latent state, which makes the transition
stochastic from the ego's point of view, like other agents in a traffic model.

The latent is a fixed random nonlinear lift of the physical state, so the
margin head and the critic have to work on features rather than on raw state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..hj.imagination import Candidates, Imagination


@dataclass
class CarFollowingConfig:
    dt: float = 0.5
    horizon: int = 16
    accels: tuple = (-4.0, -2.0, 0.0, 1.0, 2.0)  # ego acceleration tokens, m/s^2
    v_max: float = 20.0
    v_target: float = 18.0
    lead_brake: float = -5.0
    p_brake_start: float = 0.08  # per-step probability that the lead starts braking
    brake_steps: int = 6
    gap_scale: float = 10.0  # metres mapped to margin 1
    gap_range: tuple = (4.0, 60.0)
    latent_dim: int = 16
    seed: int = 0


class CarFollowingImagination(Imagination):
    def __init__(self, cfg: Optional[CarFollowingConfig] = None, device="cpu") -> None:
        self.cfg = cfg or CarFollowingConfig()
        self.device = torch.device(device)
        self.horizon = self.cfg.horizon
        self.latent_dim = self.cfg.latent_dim
        self.action_feat_dim = 1
        self.accels = torch.tensor(self.cfg.accels, device=self.device)
        gen = torch.Generator().manual_seed(self.cfg.seed)
        self._lift_w = torch.randn(3, self.latent_dim - 3, generator=gen).to(self.device)
        self._lift_b = torch.randn(self.latent_dim - 3, generator=gen).to(self.device)
        self.gap = self.v = self.u = self.brake_left = None

    # ------------------------------------------------------------- interface
    def reset(self, batch_size: int) -> Tensor:
        c = self.cfg
        rand = lambda lo, hi: lo + (hi - lo) * torch.rand(batch_size, device=self.device)
        self.gap = rand(*c.gap_range)
        self.v = rand(5.0, c.v_max)
        self.u = rand(5.0, c.v_max)
        self.brake_left = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return self.latent()

    def candidates(self) -> Candidates:
        B, K = self.gap.shape[0], self.accels.numel()
        desired = (self.cfg.v_target - self.v).clamp(-4.0, 2.0)
        logp = torch.log_softmax(-2.0 * (self.accels.unsqueeze(0) - desired.unsqueeze(1)).abs(), -1)
        idx = torch.arange(K, device=self.device).expand(B, K)
        feat = (self.accels / 4.0).view(1, K, 1).expand(B, K, 1)
        return Candidates(idx=idx, feat=feat, logp=logp)

    def step(self, slot: Tensor) -> Tensor:
        c = self.cfg
        start = (torch.rand_like(self.gap) < c.p_brake_start) & (self.brake_left == 0)
        self.brake_left = torch.where(start, torch.full_like(self.brake_left, c.brake_steps),
                                      self.brake_left)
        braking = self.brake_left > 0
        lead_acc = torch.where(braking, torch.full_like(self.u, c.lead_brake),
                               0.5 * torch.randn_like(self.u))
        self.brake_left = (self.brake_left - 1).clamp(min=0)

        v_new = (self.v + self.accels[slot] * c.dt).clamp(0.0, c.v_max)
        u_new = (self.u + lead_acc * c.dt).clamp(0.0, c.v_max)
        self.gap = self.gap + 0.5 * ((self.u + u_new) - (self.v + v_new)) * c.dt
        self.v, self.u = v_new, u_new
        return self.latent()

    def true_margin(self) -> Tensor:
        return (self.gap / self.cfg.gap_scale).clamp(-1.0, 1.0)

    # ----------------------------------------------------------------- utils
    def latent(self) -> Tensor:
        x = torch.stack([self.gap / 30.0 - 1.0, self.v / 10.0 - 1.0, self.u / 10.0 - 1.0], -1)
        return torch.cat([x, torch.tanh(x @ self._lift_w + self._lift_b)], dim=-1)

    def nominal_slot(self, cand: Candidates) -> Tensor:
        return cand.logp.argmax(-1)

    def get_state(self) -> dict:
        return {k: getattr(self, k).clone() for k in ("gap", "v", "u", "brake_left")}

    def set_state(self, state: dict) -> Tensor:
        for k, value in state.items():
            setattr(self, k, value.clone())
        return self.latent()
