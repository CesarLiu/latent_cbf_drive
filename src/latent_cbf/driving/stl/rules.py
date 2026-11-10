"""Driving signals and safety rules for the STL monitor.

All geometry is batched torch. Vehicles are covered by ``n_circles`` discs along
their longitudinal axis, which turns box-to-box clearance and time-to-collision
into closed-form disc computations.

Shapes: ego trajectory ``[T, .]``, other agents ``[N, T, .]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor

from .formula import Always, And, Formula, Predicate, Signals

BIG = 1.0e3  # stand-in for "no obstacle" distances, in metres


@dataclass
class Trajectories:
    """Privileged simulator state for one episode."""

    ego_pos: Tensor  # [T, 2]
    ego_heading: Tensor  # [T]
    ego_size: Tensor  # [2] length, width
    other_pos: Tensor  # [N, T, 2]
    other_heading: Tensor  # [N, T]
    other_size: Tensor  # [N, 2] length, width
    other_valid: Tensor  # [N, T] bool
    dt: float
    ego_vel: Optional[Tensor] = None  # [T, 2]; finite-differenced if None
    other_vel: Optional[Tensor] = None  # [N, T, 2]
    road_edge_points: Optional[Tensor] = None  # [M, 2] densely sampled

    def velocities(self) -> tuple[Tensor, Tensor]:
        ego_vel = self.ego_vel
        if ego_vel is None:
            ego_vel = _finite_difference(self.ego_pos, self.dt)
        other_vel = self.other_vel
        if other_vel is None:
            other_vel = _finite_difference(self.other_pos, self.dt)
        return ego_vel, other_vel


def _finite_difference(pos: Tensor, dt: float) -> Tensor:
    """Forward difference along the time axis (second to last), last step held."""
    if pos.shape[-2] < 2:
        return torch.zeros_like(pos)
    vel = (pos[..., 1:, :] - pos[..., :-1, :]) / dt
    return torch.cat([vel, vel[..., -1:, :]], dim=-2)


def circle_cover(pos: Tensor, heading: Tensor, size: Tensor, n_circles: int = 3):
    """Cover boxes with discs.

    Args:
        pos: [..., 2], heading: [...], size: [..., 2] (length, width),
            broadcastable against ``pos[..., 0]``.
    Returns:
        centers [..., n_circles, 2] and radius [...] (half the width).
    """
    length, width = size[..., 0], size[..., 1]
    radius = width / 2
    half_span = torch.clamp(length / 2 - radius, min=0.0)
    offsets = torch.linspace(-1.0, 1.0, n_circles, device=pos.device, dtype=pos.dtype)
    direction = torch.stack([heading.cos(), heading.sin()], dim=-1)
    shift = half_span.unsqueeze(-1) * offsets  # [..., n_circles]
    centers = pos.unsqueeze(-2) + shift.unsqueeze(-1) * direction.unsqueeze(-2)
    return centers, radius.expand(pos.shape[:-1])


def clearance(traj: Trajectories, n_circles: int = 3) -> Tensor:
    """Signed gap [T] between the ego and the nearest valid agent, in metres."""
    T = traj.ego_pos.shape[0]
    if traj.other_pos.shape[0] == 0:
        return traj.ego_pos.new_full((T,), BIG)
    ego_c, ego_r = circle_cover(traj.ego_pos, traj.ego_heading, traj.ego_size, n_circles)
    oth_c, oth_r = circle_cover(
        traj.other_pos, traj.other_heading, traj.other_size.unsqueeze(1), n_circles
    )
    # [N, T, C, C]
    dist = torch.cdist(ego_c.unsqueeze(0).expand(oth_c.shape[0], -1, -1, -1), oth_c)
    gap = dist.flatten(-2).amin(-1) - ego_r.unsqueeze(0) - oth_r
    gap = gap.masked_fill(~traj.other_valid, BIG)
    return gap.amin(0)


def time_to_collision(
    traj: Trajectories, n_circles: int = 3, ttc_max: float = 10.0
) -> Tensor:
    """Constant-velocity time to collision [T] in seconds, capped at ``ttc_max``.

    Returns 0 when already overlapping and ``ttc_max`` when no collision is
    predicted within the cap.
    """
    T = traj.ego_pos.shape[0]
    if traj.other_pos.shape[0] == 0:
        return traj.ego_pos.new_full((T,), ttc_max)
    ego_vel, oth_vel = traj.velocities()
    ego_c, ego_r = circle_cover(traj.ego_pos, traj.ego_heading, traj.ego_size, n_circles)
    oth_c, oth_r = circle_cover(
        traj.other_pos, traj.other_heading, traj.other_size.unsqueeze(1), n_circles
    )
    # relative position of every disc pair: [N, T, C, C, 2]
    p = oth_c.unsqueeze(-3) - ego_c.unsqueeze(0).unsqueeze(-2)
    v = (oth_vel - ego_vel.unsqueeze(0))[..., None, None, :]  # [N, T, 1, 1, 2]
    R = (ego_r.unsqueeze(0) + oth_r)[..., None, None]  # [N, T, 1, 1]

    a = (v * v).sum(-1)
    b = 2 * (p * v).sum(-1)
    c = (p * p).sum(-1) - R * R
    disc = b * b - 4 * a * c
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t_hit = (-b - sqrt_disc) / (2 * a).clamp(min=1e-9)
    hits = (disc > 0) & (b < 0) & (a > 1e-9)
    ttc = torch.where(hits, t_hit, torch.full_like(t_hit, ttc_max))
    ttc = torch.where(c <= 0, torch.zeros_like(ttc), ttc)  # already overlapping
    ttc = ttc.clamp(0.0, ttc_max).flatten(-2).amin(-1)  # [N, T]
    ttc = ttc.masked_fill(~traj.other_valid, ttc_max)
    return ttc.amin(0)


def road_edge_margin(traj: Trajectories, n_circles: int = 3) -> Tensor:
    """Gap [T] between the ego body and the nearest road-edge sample, in metres."""
    T = traj.ego_pos.shape[0]
    if traj.road_edge_points is None or traj.road_edge_points.shape[0] == 0:
        return traj.ego_pos.new_full((T,), BIG)
    ego_c, ego_r = circle_cover(traj.ego_pos, traj.ego_heading, traj.ego_size, n_circles)
    dist = torch.cdist(ego_c.flatten(0, 1), traj.road_edge_points)  # [T*C, M]
    return dist.amin(-1).view(T, n_circles).amin(-1) - ego_r


def compute_signals(traj: Trajectories, n_circles: int = 3, ttc_max: float = 10.0) -> Signals:
    ego_vel, _ = traj.velocities()
    return {
        "clearance": clearance(traj, n_circles),
        "ttc": time_to_collision(traj, n_circles, ttc_max),
        "road_edge": road_edge_margin(traj, n_circles),
        "speed": ego_vel.norm(dim=-1),
    }


@dataclass
class SafetySpec:
    """Safety-fragment specification ``G(state_formula)``.

    HJ reachability consumes the per-step margin of ``state_formula`` and
    handles the temporal "always" itself, so the spec is kept as a state formula
    plus the implied outer ``G``. Each predicate is divided by its scale so the
    predicates are comparable, and the margin is clipped to ``[-clip, clip]``.
    """

    min_clearance: float = 0.0  # metres
    min_road_edge: float = 0.0  # metres
    min_ttc: Optional[float] = None  # seconds; None disables the TTC rule
    clearance_scale: float = 2.0
    road_edge_scale: float = 1.0
    ttc_scale: float = 2.0
    clip: float = 1.0
    extra: Dict[str, Predicate] = field(default_factory=dict)

    def state_formula(self) -> Formula:
        preds = [
            Predicate("clearance", self.min_clearance, 1.0 / self.clearance_scale),
            Predicate("road_edge", self.min_road_edge, 1.0 / self.road_edge_scale),
        ]
        if self.min_ttc is not None:
            preds.append(Predicate("ttc", self.min_ttc, 1.0 / self.ttc_scale))
        preds.extend(self.extra.values())
        return And(*preds)

    def formula(self) -> Formula:
        return Always(self.state_formula())

    def margin(self, signals: Signals) -> Tensor:
        """Per-step safety margin l_t in ``[-clip, clip]``; negative is a violation."""
        return self.state_formula()(signals).clamp(-self.clip, self.clip)
