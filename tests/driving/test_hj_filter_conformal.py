import math

import pytest
import torch

from driving.conformal import (
    binomial_tolerance,
    calibrate_threshold,
    evaluate_threshold,
    min_unsafe_samples,
)
from driving.hj import (
    AvoidQConfig,
    AvoidQLearner,
    Candidates,
    MarginHead,
    MarginLossConfig,
    QNetwork,
    avoid_target,
    fit_margin_head,
    margin_loss,
)
from driving.hj.margin import estimate_gradient_scale
from driving.rollouts import calibration_rollouts, evaluate_policy
from driving.safety_filter import LatentSafetyFilter
from driving.toy.car_following import CarFollowingConfig, CarFollowingImagination


# ----------------------------------------------------------------- backup
def test_avoid_target_limits():
    margin = torch.tensor([0.5, 0.5, -0.2, 0.5])
    value = torch.tensor([0.8, 0.1, 0.9, -0.7])
    done = torch.zeros(4, dtype=torch.bool)
    # gamma = 0: the critic regresses the margin (warm-up).
    assert torch.equal(avoid_target(margin, value, done, 0.0), margin)
    # gamma = 1: worst of current margin and best future value.
    assert avoid_target(margin, value, done, 1.0).tolist() == pytest.approx([0.5, 0.1, -0.2, -0.7])
    # at the horizon the future term is dropped.
    assert torch.equal(avoid_target(margin, value, ~done, 1.0), margin)


class ChainEnv:
    """Deterministic 1-D chain: position p in [0, 6], action moves -1/0/+1.

    A wind pushes +1 every step and the margin is positive only for p < 4.5, so
    the ego has to keep choosing -1 to hold position. Values are known exactly.
    """

    latent_dim, action_feat_dim, horizon = 1, 1, 6
    moves = torch.tensor([-1.0, 0.0, 1.0])

    def reset(self, batch_size):
        self.p = torch.randint(0, 7, (batch_size,)).float()
        return self.p.unsqueeze(-1) / 6.0

    def candidates(self):
        B = self.p.shape[0]
        return Candidates(
            torch.arange(3).expand(B, 3),
            self.moves.view(1, 3, 1).expand(B, 3, 1),
            torch.zeros(B, 3),
        )

    def step(self, slot):
        self.p = (self.p + self.moves[slot] + 1.0).clamp(0.0, 6.0)
        return self.p.unsqueeze(-1) / 6.0

    def true_margin(self):
        return (4.5 - self.p) / 6.0  # in [-0.25, 0.75], away from tanh saturation


def test_q_learning_recovers_known_values():
    torch.manual_seed(0)
    env = ChainEnv()
    cfg = AvoidQConfig(
        gamma=1.0, warmup_updates=100, rollout_batch=64, updates_per_rollout=16,
        batch_size=128, tau=0.1, weight_decay=0.0, p_random=0.6, p_nominal=0.0,
    )
    learner = AvoidQLearner(env, cfg=cfg, q=QNetwork(1, 1, (64, 64)))
    learner.train(120)
    moves = env.moves.view(1, 3, 1)
    for p in range(7):
        q = learner.q(torch.tensor([[p / 6.0]]), moves)[0]
        for k, move in enumerate((-1.0, 0.0, 1.0)):
            # After the first move the best policy holds position (-1 against the wind).
            p_next = min(max(p + move + 1.0, 0.0), 6.0)
            expected = (4.5 - p_next) / 6.0
            assert q[k].item() == pytest.approx(expected, abs=0.1), (p, move)


# ------------------------------------------------------------------ margin
def test_margin_head_fits_and_gp_modes():
    torch.manual_seed(0)
    z = torch.randn(4000, 4)
    target = torch.tanh(2.0 * z[:, 0])
    head = MarginHead(4, (64, 64))
    metrics = fit_margin_head(head, z, target, steps=400, batch_size=256)
    assert metrics["sign_acc"] > 0.95 and metrics["gradient_thresh"] > 0
    with torch.no_grad():
        assert (head(z) - target).abs().mean() < 0.1
    # classification labels and the two-sided latent_cbf penalty run as well
    failure = (target < 0).float()
    cfg = MarginLossConfig(mode="classification", gp_mode="two_sided", gradient_thresh=0.5)
    loss, m = margin_loss(MarginHead(4, (32,)), z[:256], failure[:256], cfg)
    assert torch.isfinite(loss) and "gp" in m
    assert estimate_gradient_scale(z, failure) > 0
    assert estimate_gradient_scale(z, torch.ones(4000)) == 1.0


# ------------------------------------------------------------------ filter
class FixedQ(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table)

    def forward(self, z, feat):
        return self.table.expand(z.shape[0], -1)


def make_cand(B=1, K=4):
    feat = torch.arange(K).float().view(1, K, 1).expand(B, K, 1)
    return Candidates(100 + torch.arange(K).expand(B, K), feat, torch.zeros(B, K))


def test_filter_modes_and_fallback():
    z, cand = torch.zeros(1, 2), make_cand()
    nominal = torch.tensor([3])
    q = FixedQ([0.9, 0.2, 0.4, -0.5])

    passthrough = LatentSafetyFilter(q, 0.0, "cbf")(z, cand, torch.tensor([2]))
    assert passthrough.slot.item() == 2 and not passthrough.intervened.item()

    cbf = LatentSafetyFilter(q, 0.0, "cbf")(z, cand, nominal)
    assert cbf.slot.item() == 2 and cbf.intervened.item()  # closest acceptable action
    assert cbf.action_idx.item() == 102 and cbf.q_nominal.item() == pytest.approx(-0.5)

    assert LatentSafetyFilter(q, 0.3, "cbf")(z, cand, nominal).slot.item() == 2
    assert LatentSafetyFilter(q, 0.5, "cbf")(z, cand, nominal).slot.item() == 0

    lr = LatentSafetyFilter(q, 0.0, "lr")(z, cand, nominal)
    assert lr.slot.item() == 0 and lr.value.item() == pytest.approx(0.9)

    stuck = LatentSafetyFilter(q, 0.95, "cbf")(z, cand, nominal)
    assert stuck.infeasible.item() and stuck.slot.item() == 0

    far_first = torch.tensor([[0.0, 5.0, 9.0, 0.0]])  # custom distance prefers slot 0
    custom = LatentSafetyFilter(q, 0.0, "cbf")(z, cand, nominal, distance=far_first)
    assert custom.slot.item() == 0


# --------------------------------------------------------------- conformal
def test_conformal_threshold_order_statistic():
    scores = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8])
    unsafe = torch.tensor([0, 1, 1, 1, 0, 1, 0]).bool()
    # unsafe scores sorted: 0.1 0.2 0.3 0.5; rank = ceil(5 * 0.75) = 4
    cal = calibrate_threshold(scores, unsafe, alpha=0.25)
    assert cal.threshold == pytest.approx(0.5) and cal.n_unsafe == 4
    assert not calibrate_threshold(scores, unsafe, alpha=0.1).attainable
    assert min_unsafe_samples(0.1) == 9 and min_unsafe_samples(0.25) == 3
    report = evaluate_threshold(scores, unsafe, 0.25)
    assert report.false_safe_rate == pytest.approx(0.5)
    assert report.false_unsafe_rate == 0.0
    assert report.unsafe_given_accepted == pytest.approx(2 / 5)


def test_conformal_coverage_on_synthetic_scores():
    torch.manual_seed(0)
    alpha, rates = 0.1, []
    for _ in range(200):
        unsafe = torch.rand(600) < 0.3
        scores = torch.randn(600) + torch.where(unsafe, -0.5, 0.5)
        cal = calibrate_threshold(scores[:300], unsafe[:300], alpha)
        rates.append(evaluate_threshold(scores[300:], unsafe[300:], cal.threshold).false_safe_rate)
    assert sum(rates) / len(rates) <= alpha + 0.01


# ----------------------------------------------------------- full pipeline
def test_pipeline_on_car_following():
    """Margin head -> avoid Q-learning -> filter -> conformal, end to end."""
    torch.manual_seed(0)
    env = CarFollowingImagination(
        CarFollowingConfig(
            p_brake_start=0.04, accels=(-6.0, -3.0, 0.0, 1.0, 2.0), gap_range=(8.0, 60.0)
        )
    )

    # 1. boundary function from labelled random rollouts
    zs, ls = [], []
    env.reset(4096)
    for _ in range(env.horizon):
        zs.append(env.step(torch.randint(0, 5, (4096,))))
        ls.append(env.true_margin())
    head = MarginHead(env.latent_dim, (64, 64))
    metrics = fit_margin_head(head, torch.cat(zs), torch.cat(ls), steps=500, batch_size=256)
    assert metrics["sign_acc"] > 0.97

    # 2. latent HJ on the learned margin
    cfg = AvoidQConfig(
        gamma=0.999, warmup_updates=300, rollout_batch=256, updates_per_rollout=16,
        batch_size=256, tau=0.05, weight_decay=0.0,
    )
    learner = AvoidQLearner(env, margin_fn=head, cfg=cfg, q=QNetwork(env.latent_dim, 1, (64, 64)))
    learner.train(60)

    # 3. filtering reduces violations
    nominal = evaluate_policy(env, 4000)
    uncalibrated = LatentSafetyFilter(learner.q, 0.0, "lr")
    filtered = evaluate_policy(env, 4000, uncalibrated)
    assert nominal.violation_rate > 0.4
    assert filtered.violation_rate < 0.75 * nominal.violation_rate
    assert 0.0 < filtered.intervention_rate < 0.6

    # 4. conformal threshold: bounded false-safe rate on fresh rollouts
    alpha = 0.1
    scores, unsafe = calibration_rollouts(env, learner.q, 6000, 4, uncalibrated)
    cal = calibrate_threshold(scores, unsafe, alpha)
    assert cal.attainable and cal.threshold > 0.0  # the raw critic is optimistic
    test_scores, test_unsafe = calibration_rollouts(env, learner.q, 6000, 4, uncalibrated)
    report = evaluate_threshold(test_scores, test_unsafe, cal.threshold)
    assert report.false_safe_rate <= alpha + binomial_tolerance(alpha, report.n_unsafe)
    raw = evaluate_threshold(test_scores, test_unsafe, 0.0)
    assert raw.false_safe_rate > report.false_safe_rate

    # 5. the calibrated filter is safer than the uncalibrated one
    calibrated = evaluate_policy(env, 4000, LatentSafetyFilter(learner.q, cal.threshold, "lr"))
    assert calibrated.violation_rate < filtered.violation_rate
    assert math.isfinite(calibrated.intervention_rate)
