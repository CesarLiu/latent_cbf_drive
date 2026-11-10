import math

import torch

from driving.stl import (
    Always, Eventually, GreaterThan, Implies, LessThan, STLMonitor, SafetySpec,
    Trajectories, Until,
)
from driving.stl.rules import BIG, circle_cover, clearance, time_to_collision


def sig(*values):
    return {"x": torch.tensor(values, dtype=torch.float32)}


def test_predicates_and_boolean_ops():
    s = sig(1.0, -2.0, 3.0)
    gt, lt = GreaterThan("x", 0.0), LessThan("x", 2.0)
    assert gt(s).tolist() == [1.0, -2.0, 3.0]
    assert lt(s).tolist() == [1.0, 4.0, -1.0]
    assert (gt & lt)(s).tolist() == [1.0, -2.0, -1.0]
    assert (gt | lt)(s).tolist() == [1.0, 4.0, 3.0]
    assert (~gt)(s).tolist() == [-1.0, 2.0, -3.0]
    assert Implies(gt, lt)(s).tolist() == [1.0, 4.0, -1.0]


def test_always_eventually_unbounded():
    s = sig(3.0, 1.0, 2.0, -1.0, 5.0)
    p = GreaterThan("x", 0.0)
    assert Always(p)(s).tolist() == [-1.0, -1.0, -1.0, -1.0, 5.0]
    assert Eventually(p)(s).tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]


def test_bounded_windows_are_truncated():
    s = sig(3.0, 1.0, 2.0, -1.0, 5.0)
    p = GreaterThan("x", 0.0)
    assert Always(p, 0, 1)(s).tolist() == [1.0, 1.0, -1.0, -1.0, 5.0]
    assert Eventually(p, 1, 2)(s).tolist() == [2.0, 2.0, 5.0, 5.0, -math.inf]
    assert Always(p, 1, 2)(s).tolist() == [1.0, -1.0, -1.0, 5.0, math.inf]


def test_batched_traces():
    s = {"x": torch.tensor([[1.0, -1.0, 2.0], [3.0, 2.0, 1.0]])}
    out = Always(GreaterThan("x", 0.0))(s)
    assert out.tolist() == [[-1.0, -1.0, 2.0], [1.0, 1.0, 1.0]]


def test_until():
    left = GreaterThan("a", 0.0)
    right = GreaterThan("b", 0.0)
    s = {"a": torch.tensor([1.0, 1.0, -1.0, 1.0]), "b": torch.tensor([-1.0, 2.0, -1.0, -1.0])}
    # b holds at t=1 and a holds before it.
    assert Until(left, right)(s).tolist() == [1.0, 2.0, -1.0, -1.0]


def straight_traj(gap0, ego_speed, lead_speed, T=20, dt=0.1, lateral=0.0):
    t = torch.arange(T) * dt
    zeros = torch.zeros(T)
    return Trajectories(
        ego_pos=torch.stack([ego_speed * t, zeros], -1),
        ego_heading=zeros,
        ego_size=torch.tensor([4.0, 2.0]),
        other_pos=torch.stack([gap0 + lead_speed * t, zeros + lateral], -1).unsqueeze(0),
        other_heading=zeros.unsqueeze(0),
        other_size=torch.tensor([[4.0, 2.0]]),
        other_valid=torch.ones(1, T, dtype=torch.bool),
        dt=dt,
    )


def test_circle_cover_spans_the_box():
    centers, radius = circle_cover(torch.zeros(1, 2), torch.zeros(1), torch.tensor([4.0, 2.0]))
    assert radius.item() == 1.0
    assert centers[0, :, 0].tolist() == [-1.0, 0.0, 1.0]


def test_clearance_matches_bumper_gap():
    traj = straight_traj(gap0=10.0, ego_speed=0.0, lead_speed=0.0)
    # centre distance 10 m, two 4 m cars -> 6 m bumper to bumper
    assert torch.allclose(clearance(traj), torch.full((20,), 6.0), atol=1e-5)


def test_ttc_closing_and_opening():
    closing = straight_traj(gap0=10.0, ego_speed=6.0, lead_speed=0.0)
    assert abs(time_to_collision(closing)[0].item() - 1.0) < 1e-4  # 6 m gap at 6 m/s
    opening = straight_traj(gap0=10.0, ego_speed=0.0, lead_speed=5.0)
    assert time_to_collision(opening)[0].item() == 10.0
    adjacent_lane = straight_traj(gap0=10.0, ego_speed=6.0, lead_speed=0.0, lateral=3.5)
    assert time_to_collision(adjacent_lane)[0].item() == 10.0


def test_monitor_flags_collision_and_suffix_robustness():
    traj = straight_traj(gap0=10.0, ego_speed=6.0, lead_speed=0.0, T=20)
    out = STLMonitor(SafetySpec())(traj)
    assert out.violated
    first_fail = int(out.failure.float().argmax())
    assert 9 <= first_fail <= 11  # gap closes after ~1 s
    assert (out.robustness[: first_fail + 1] < 0).all()
    assert torch.all(out.robustness[:-1] <= out.robustness[1:] + 1e-6)
    assert out.margin.max() <= 1.0 and out.margin.min() >= -1.0


def test_monitor_safe_episode_and_invalid_agents():
    traj = straight_traj(gap0=30.0, ego_speed=5.0, lead_speed=5.0)
    assert not STLMonitor()(traj).violated
    crash = straight_traj(gap0=10.0, ego_speed=6.0, lead_speed=0.0)
    crash.other_valid[:] = False
    out = STLMonitor()(crash)
    assert not out.violated and out.signals["clearance"].min() == BIG


def test_ttc_rule_fires_before_contact():
    traj = straight_traj(gap0=10.0, ego_speed=6.0, lead_speed=0.0, T=5)
    assert not STLMonitor(SafetySpec())(traj).violated  # no contact within 0.5 s
    assert STLMonitor(SafetySpec(min_ttc=1.5))(traj).violated


def test_road_edge_margin():
    traj = straight_traj(gap0=50.0, ego_speed=5.0, lead_speed=5.0)
    xs = torch.linspace(-5, 20, 200)
    traj.road_edge_points = torch.stack([xs, torch.full_like(xs, 1.5)], -1)
    out = STLMonitor()(traj)
    assert torch.allclose(out.signals["road_edge"], torch.full((20,), 0.5), atol=0.02)
    traj.road_edge_points = torch.stack([xs, torch.full_like(xs, 0.8)], -1)
    assert STLMonitor()(traj).violated
