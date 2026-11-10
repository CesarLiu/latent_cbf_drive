"""Bindings to CtRL-Sim / Nocturne closed-loop evaluation.

CtRL-Sim's evaluators keep one ``vehicle_data_dict`` per episode:

    vehicle_data_dict[veh_id] = {
        "position": [{"x": .., "y": ..}, ...],   # one entry per 10 Hz step
        "velocity": [{"x": .., "y": ..}, ...],
        "heading": [float, ...],
        "existence": [0/1, ...],
        "width": float, "length": float, ...
    }

This module converts that structure into monitor inputs, labels episodes with
the STL spec, and wraps a CtRL-Sim planner so that a safety hook can override
its action. It only needs numpy and torch; Nocturne objects are duck-typed.

Not done here: turning the live Nocturne state into SMART's tokenized input so
that the latent filter can run inside the evaluator. ``FilteredPlanner`` takes
that as an injected ``safety_hook`` (plan stage 2, format bridge).
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Optional, Sequence, Tuple

import numpy as np
import torch

from ..stl import MonitorOutput, STLMonitor, Trajectories

VehicleData = Dict[Hashable, dict]
# hook(vehicle_data_dict, t, ego_id) -> (acceleration, steering) or None to keep
SafetyHook = Callable[[VehicleData, int, Hashable], Optional[Tuple[float, float]]]


def _xy(entries: Sequence[dict], n: int) -> np.ndarray:
    return np.array([[e["x"], e["y"]] for e in entries[:n]], dtype=np.float32).reshape(n, 2)


def densify_polylines(polylines: Sequence[np.ndarray], spacing: float = 0.5) -> np.ndarray:
    """Resample polylines so that consecutive points are at most ``spacing`` apart."""
    points = []
    for line in polylines:
        line = np.asarray(line, dtype=np.float32).reshape(-1, 2)
        if len(line) == 0:
            continue
        points.append(line[:1])
        for a, b in zip(line[:-1], line[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / spacing)), 1)
            ts = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[1:, None]
            points.append(a + ts * (b - a))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(points, axis=0)


def trajectories_from_vehicle_data(
    vehicle_data_dict: VehicleData,
    ego_id: Hashable,
    dt: float = 0.1,
    road_edge_polylines: Optional[Sequence[np.ndarray]] = None,
    edge_spacing: float = 0.5,
    start: int = 0,
) -> Trajectories:
    """Build monitor inputs for the ego from a CtRL-Sim episode record.

    Args:
        start: first step to include, e.g. ``history_steps - 1`` to skip the
            logged history during which the planner is not in control.
    """
    n = min(len(v["position"]) for v in vehicle_data_dict.values())
    if n <= start:
        raise ValueError("episode has no steps after `start`")

    def state(veh_id):
        v = vehicle_data_dict[veh_id]
        pos = _xy(v["position"], n)[start:]
        vel = _xy(v["velocity"], n)[start:] if len(v.get("velocity", [])) >= n else None
        heading = np.asarray(v["heading"][:n], dtype=np.float32)[start:]
        valid = np.asarray(v["existence"][:n], dtype=np.float32)[start:] > 0
        size = np.array([v["length"], v["width"]], dtype=np.float32)
        return pos, vel, heading, valid, size

    ego_pos, ego_vel, ego_heading, _, ego_size = state(ego_id)
    others = [state(k) for k in vehicle_data_dict if k != ego_id]
    T = ego_pos.shape[0]
    if others:
        other_pos = np.stack([o[0] for o in others])
        has_vel = all(o[1] is not None for o in others)
        other_vel = np.stack([o[1] for o in others]) if has_vel else None
        other_heading = np.stack([o[2] for o in others])
        other_valid = np.stack([o[3] for o in others])
        other_size = np.stack([o[4] for o in others])
    else:
        other_pos = np.zeros((0, T, 2), np.float32)
        other_vel = None
        other_heading = np.zeros((0, T), np.float32)
        other_valid = np.zeros((0, T), bool)
        other_size = np.zeros((0, 2), np.float32)

    edges = None
    if road_edge_polylines is not None:
        edges = torch.from_numpy(densify_polylines(road_edge_polylines, edge_spacing))

    return Trajectories(
        ego_pos=torch.from_numpy(ego_pos),
        ego_heading=torch.from_numpy(ego_heading),
        ego_size=torch.from_numpy(ego_size),
        other_pos=torch.from_numpy(other_pos),
        other_heading=torch.from_numpy(other_heading),
        other_size=torch.from_numpy(other_size),
        other_valid=torch.from_numpy(other_valid),
        dt=dt,
        ego_vel=None if ego_vel is None else torch.from_numpy(ego_vel),
        other_vel=None if other_vel is None else torch.from_numpy(other_vel),
        road_edge_points=edges,
    )


def label_episode(
    vehicle_data_dict: VehicleData,
    ego_id: Hashable,
    monitor: Optional[STLMonitor] = None,
    **kwargs,
) -> MonitorOutput:
    """STL labels for one finished (or partial) episode."""
    monitor = monitor or STLMonitor()
    return monitor(trajectories_from_vehicle_data(vehicle_data_dict, ego_id, **kwargs))


def episode_record(out: MonitorOutput, token_stride: int = 5) -> Dict[str, np.ndarray]:
    """Per-step labels plus labels subsampled to SMART's token rate (2 Hz).

    The token-rate margin is the minimum over each token interval, so that a
    violation between two token boundaries is not missed.
    """
    margin = out.margin.cpu().numpy()
    n_tok = len(margin) // token_stride
    token_margin = margin[: n_tok * token_stride].reshape(n_tok, token_stride).min(-1)
    record = {
        "margin": margin,
        "robustness": out.robustness.cpu().numpy(),
        "failure": out.failure.cpu().numpy(),
        "token_margin": token_margin,
        "token_failure": token_margin < 0,
    }
    record.update({f"signal_{k}": v.cpu().numpy() for k, v in out.signals.items()})
    return record


class FilteredPlanner:
    """Wrap a CtRL-Sim policy so that a safety hook can override its action.

    The wrapper forwards the policy interface used by
    ``PlannerAdversaryEvaluator`` (``reset``, ``update_state``, ``predict``,
    ``act``) and every other attribute to the wrapped planner. After the
    planner's ``predict`` it calls ``safety_hook``; a returned
    ``(acceleration, steering)`` replaces the planner's next action.

    Intervention bookkeeping is kept per episode in ``interventions``.
    """

    def __init__(self, planner, safety_hook: SafetyHook, name: Optional[str] = None) -> None:
        self._planner = planner
        self._hook = safety_hook
        self.name = name or f"{planner.name}+filter"
        self.interventions: list = []

    def __getattr__(self, item):
        return getattr(self._planner, item)

    def reset(self, vehicle_data_dict):
        self.interventions = []
        return self._planner.reset(vehicle_data_dict)

    def predict(self, vehicle_data_dict, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t):
        vehicle_data_dict = self._planner.predict(
            vehicle_data_dict, gt_data_dict, preproc_data, dset, vehicles_to_evaluate, t
        )
        acc_key = self._planner.key_dict["next_acceleration"]
        steer_key = self._planner.key_dict["next_steering"]
        for veh_id in vehicles_to_evaluate:
            override = self._hook(vehicle_data_dict, t, veh_id)
            if override is None:
                continue
            nominal = (vehicle_data_dict[veh_id][acc_key], vehicle_data_dict[veh_id][steer_key])
            vehicle_data_dict[veh_id][acc_key] = float(override[0])
            vehicle_data_dict[veh_id][steer_key] = float(override[1])
            self.interventions.append({"t": t, "veh_id": veh_id, "nominal": nominal,
                                       "applied": (float(override[0]), float(override[1]))})
        return vehicle_data_dict

    @property
    def intervention_rate(self) -> float:
        steps = max(self._planner.steps, 1)
        return len(self.interventions) / steps
