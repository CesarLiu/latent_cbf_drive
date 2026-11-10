"""Road-edge rule of the SMART imagination, without CAT-K.

The geometry helpers are tested directly. The methods of ``SmartImagination``
that use them are run on a hand-built state, bypassing the model.
"""

import pytest
import torch

from driving.adapters.smart_imagination import (
    ROAD_EDGE_BOUNDARY,
    ROAD_EDGE_MEDIAN,
    SmartImagination,
    pad_edge_segments,
    road_edge_gap,
)
from driving.stl.rules import BIG, point_segment_distance


def test_point_segment_distance():
    a = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    b = torch.tensor([[10.0, 0.0], [0.0, 5.0]])  # second segment is degenerate
    pts = torch.tensor([[5.0, 2.0], [-3.0, 4.0], [12.0, 0.0]])
    d = point_segment_distance(pts, a, b)
    assert d.shape == (3, 2)
    assert d[:, 0].tolist() == pytest.approx([2.0, 5.0, 2.0])
    assert d[:, 1].tolist() == pytest.approx([(25 + 9) ** 0.5, (9 + 1) ** 0.5, 13.0])


def test_pad_edge_segments_groups_by_scene():
    # 4 pieces of 3 points; pieces 0 and 3 are edges of scene 1, piece 2 of scene 0
    traj = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
    is_edge = torch.tensor([True, False, True, True])
    batch = torch.tensor([1, 0, 0, 1])
    seg_a, seg_b, mask = pad_edge_segments(traj, is_edge, batch, num_graphs=3)
    assert seg_a.shape == (3, 4, 2)
    assert mask.sum(1).tolist() == [2, 4, 0]
    assert torch.equal(seg_a[0, :2], traj[2, :2]) and torch.equal(seg_b[0, :2], traj[2, 1:])
    assert torch.equal(seg_a[1], torch.cat([traj[0, :2], traj[3, :2]]))
    # a scene without edges yields "no obstacle"
    gap = road_edge_gap(torch.zeros(3, 1, 2), torch.ones(3), seg_a, seg_b, mask)
    assert gap[2] == pytest.approx(BIG - 1.0)


def test_pad_edge_segments_without_any_edge():
    seg_a, _, mask = pad_edge_segments(torch.zeros(2, 3, 2), torch.zeros(2, dtype=torch.bool),
                                       torch.zeros(2, dtype=torch.long), num_graphs=2)
    assert seg_a.shape == (2, 1, 2) and not mask.any()


def box(x, y, heading=0.0, length=4.8, width=2.0):
    """Contour (left_front, right_front, right_back, left_back)."""
    c, s = torch.cos(torch.tensor(heading)), torch.sin(torch.tensor(heading))
    local = torch.tensor([[length / 2, width / 2], [length / 2, -width / 2],
                          [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = torch.stack([torch.stack([c, s]), torch.stack([-s, c])])
    return local @ rot + torch.tensor([x, y])


def make_env(ego_y=(0.0, 0.0), **kwargs):
    """Two scenes; agents 0/1 are scene 0 (ego, other), agents 2/3 scene 1 (other, ego).

    Each scene has a boundary edge along y = 3 and a median edge along y = -3.
    """
    env = object.__new__(SmartImagination)
    env.road_edge, env.road_edge_types = True, None
    env.road_edge_scale, env.min_road_edge = 1.0, 0.0
    env.clearance_scale, env.min_clearance, env.n_circles = 2.0, 0.0, 3
    for key, value in kwargs.items():
        setattr(env, key, value)
    env.ego = torch.tensor([0, 3])
    env.pos_a = torch.tensor([[0.0, ego_y[0]], [50.0, 0.0], [50.0, 0.0], [0.0, ego_y[1]]])[:, None]
    env.head_a = torch.zeros(4, 1)
    env.pred_valid = torch.ones(4, 1, dtype=torch.bool)
    env.ta = {"batch": torch.tensor([0, 0, 1, 1]), "num_graphs": 2,
              "token_agent_shape": torch.tensor([[2.0, 4.8]] * 4)}  # (width, length)
    piece = lambda y: torch.tensor([[-20.0, y], [0.0, y], [20.0, y]])
    data = {
        "map_save": {"traj_pos": torch.stack([piece(3.0), piece(-3.0), piece(3.0), piece(-3.0),
                                              piece(0.0)])},
        "pt_token": {"pl_type": torch.tensor([1, 1, 1, 1, 0]),  # last piece is a lane
                     "type": torch.tensor([ROAD_EDGE_BOUNDARY, ROAD_EDGE_MEDIAN] * 2 + [1]),
                     "batch": torch.tensor([0, 0, 1, 1, 0])},
    }
    env._setup_road_edges(data)
    return env


def test_gap_at_reset_and_margin():
    env = make_env(ego_y=(0.0, 1.0))
    assert env.edge_active.all()
    # half width 1 m: 3 - 0 - 1 and 3 - 1 - 1; the lane through the ego is ignored
    assert env.ego_road_edge_gap().tolist() == pytest.approx([2.0, 1.0])
    # clearance is far (margin 1), so the road edge decides: clamp(2), 1
    assert env.true_margin().tolist() == pytest.approx([1.0, 1.0])
    env.road_edge_scale = 4.0
    assert env.true_margin().tolist() == pytest.approx([0.5, 0.25])


def test_sweep_catches_edge_crossed_inside_a_step():
    env = make_env()
    # ego 0 jumps from y=0 to y=8 within one token: start and end are both
    # clear of the edge at y=3, only the intermediate poses touch it
    ys = torch.linspace(0.0, 8.0, 6)
    sweep0 = torch.stack([box(float(i), float(y)) for i, y in enumerate(ys)])
    sweep1 = torch.stack([box(float(i), 0.0) for i in range(6)])
    env._ego_sweep = torch.stack([sweep0, sweep1])
    gap = env.ego_road_edge_gap()
    assert gap[0] < 0 and gap[1] == pytest.approx(2.0)
    assert env.true_margin()[0] < 0 < env.true_margin()[1]
    # the end pose alone would have looked safe
    env._ego_sweep = env._ego_sweep[:, -1:]
    assert env.ego_road_edge_gap()[0] > 0


def test_rule_is_off_for_egos_starting_on_an_edge():
    env = make_env(ego_y=(2.5, 0.0))
    assert env.edge_active.tolist() == [False, True]
    assert env.ego_road_edge_gap()[0] == pytest.approx(BIG)
    assert env.true_margin().tolist() == pytest.approx([1.0, 1.0])


def test_type_filter_and_switch():
    # ego 0 sits on the median at y = -3
    assert make_env(ego_y=(-3.0, 0.0)).edge_active.tolist() == [False, True]
    env = make_env(ego_y=(-3.0, 0.0), road_edge_types=(ROAD_EDGE_BOUNDARY,))
    assert env.edge_active.all() and env.ego_road_edge_gap()[0] == pytest.approx(5.0)
    off = make_env(road_edge=False)
    assert off._edges is None and off.ego_road_edge_gap().tolist() == pytest.approx([BIG, BIG])
