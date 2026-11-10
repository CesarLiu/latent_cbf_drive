import numpy as np
import pytest

from driving.adapters.ctrlsim import (
    FilteredPlanner,
    densify_polylines,
    episode_record,
    label_episode,
    trajectories_from_vehicle_data,
)
from driving.stl import SafetySpec, STLMonitor


def vehicle(xs, ys, heading=0.0, length=4.0, width=2.0, exists=None, dt=0.1):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    vx = np.gradient(xs, dt) if len(xs) > 1 else np.zeros_like(xs)
    return {
        "position": [{"x": x, "y": y} for x, y in zip(xs, ys)],
        "velocity": [{"x": v, "y": 0.0} for v in vx],
        "heading": [heading] * len(xs),
        "existence": list(exists) if exists is not None else [1] * len(xs),
        "width": width,
        "length": length,
        # CtRL-Sim appends actions after the state, so these lists can be longer.
        "acceleration": [0.0] * (len(xs) + 1),
    }


def rear_end_episode(T=30):
    t = np.arange(T) * 0.1
    return {
        "ego": vehicle(6.0 * t, np.zeros(T)),
        "lead": vehicle(np.full(T, 10.0), np.zeros(T)),
        "parked": vehicle(np.full(T, 5.0), np.full(T, 20.0)),
    }


def test_conversion_shapes_and_start_offset():
    traj = trajectories_from_vehicle_data(rear_end_episode(), "ego", start=9)
    assert traj.ego_pos.shape == (21, 2)
    assert traj.other_pos.shape == (2, 21, 2) and traj.other_valid.all()
    assert traj.ego_size.tolist() == [4.0, 2.0]
    assert traj.ego_vel is not None and traj.other_vel.shape == (2, 21, 2)
    with pytest.raises(ValueError):
        trajectories_from_vehicle_data(rear_end_episode(), "ego", start=30)


def test_label_episode_detects_rear_end_collision():
    out = label_episode(rear_end_episode(), "ego")
    assert out.violated
    assert 9 <= int(out.failure.float().argmax()) <= 11  # 6 m gap closed at 6 m/s
    record = episode_record(out)
    assert record["token_margin"].shape == (6,)
    # contact from ~1.0 s; by 2.5 s the ego has driven through and is clear again
    assert record["token_failure"].tolist() == [False, False, True, True, True, False]
    assert set(record) >= {"margin", "robustness", "signal_clearance", "signal_ttc"}


def test_removed_vehicles_are_ignored():
    episode = rear_end_episode()
    episode["lead"]["existence"] = [0] * 30
    # CtRL-Sim teleports removed vehicles far away; they must not count either way.
    episode["lead"]["position"] = [{"x": -1e6, "y": -1e6}] * 30
    assert not label_episode(episode, "ego").violated


def test_road_edges_are_densified_and_monitored():
    edge = [np.array([[-5.0, 1.5], [30.0, 1.5]])]
    dense = densify_polylines(edge, spacing=0.5)
    assert len(dense) == 71 and np.allclose(np.diff(dense[:, 0]), 0.5)
    episode = {"ego": vehicle(np.linspace(0, 10, 20), np.zeros(20))}
    spec = STLMonitor(SafetySpec())
    safe = label_episode(episode, "ego", spec, road_edge_polylines=edge)
    assert not safe.violated and safe.signals["road_edge"].min() == pytest.approx(0.5, abs=0.05)
    near = [np.array([[-5.0, 0.7], [30.0, 0.7]])]
    assert label_episode(episode, "ego", spec, road_edge_polylines=near).violated


class DummyPlanner:
    name = "dummy"
    steps = 10
    key_dict = {"next_acceleration": "next_planner_acceleration",
                "next_steering": "next_planner_steering"}

    def __init__(self):
        self.reset_called = False

    def reset(self, vehicle_data_dict):
        self.reset_called = True

    def update_state(self, vehicle_data_dict, vehicles, t):
        return "updated"

    def predict(self, vdd, gt, preproc, dset, vehicles, t):
        for v in vehicles:
            vdd[v]["next_planner_acceleration"] = 2.0
            vdd[v]["next_planner_steering"] = 0.1
        return vdd

    def act(self, veh, t, vdd):
        return veh, (vdd["ego"]["next_planner_acceleration"], vdd["ego"]["next_planner_steering"])


def test_filtered_planner_overrides_and_forwards():
    hook = lambda vdd, t, veh_id: (-4.0, 0.0) if t >= 3 else None
    planner = FilteredPlanner(DummyPlanner(), hook)
    vdd = {"ego": {}}
    planner.reset(vdd)
    assert planner._planner.reset_called and planner.name == "dummy+filter"
    assert planner.update_state(vdd, ["ego"], 0) == "updated"  # forwarded

    planner.predict(vdd, None, None, None, ["ego"], 2)
    assert planner.act(None, 2, vdd)[1] == (2.0, 0.1)
    planner.predict(vdd, None, None, None, ["ego"], 3)
    assert planner.act(None, 3, vdd)[1] == (-4.0, 0.0)
    assert planner.interventions == [
        {"t": 3, "veh_id": "ego", "nominal": (2.0, 0.1), "applied": (-4.0, 0.0)}
    ]
    assert planner.intervention_rate == pytest.approx(0.1)
