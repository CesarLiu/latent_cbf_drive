"""Split-conformal calibration of the safety filter threshold.

Setup. A calibration state ``s`` has a score ``v(s)`` (the latent safety value
of the action under test) and a ground-truth label obtained by executing the
safe controller from ``s`` in the real simulator and monitoring the STL spec:
``unsafe(s) = 1`` iff the spec is violated on that rollout.

Guarantee. With ``n`` unsafe calibration states and ``delta`` the
``ceil((n + 1) * (1 - alpha))``-th smallest of their scores, a new unsafe state
exchangeable with them satisfies

    P( v(s_new) > delta ) <= alpha,

i.e. the rate at which truly unsafe situations are declared safe is bounded.
The bound is marginal over the calibration draw and conditional on the unsafe
class. It requires the calibration states to be exchangeable with deployment
states, which is why calibration data must come from closed-loop runs of the
*filtered* policy (and be refreshed when ``delta`` changes the visited states).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Calibration:
    threshold: float  # delta; +inf means alpha is not attainable with this n
    alpha: float
    n_unsafe: int
    n_total: int

    @property
    def attainable(self) -> bool:
        return math.isfinite(self.threshold)


def min_unsafe_samples(alpha: float) -> int:
    """Smallest number of unsafe calibration states for a finite threshold."""
    return math.ceil(1.0 / alpha) - 1


def calibrate_threshold(scores: Tensor, unsafe: Tensor, alpha: float) -> Calibration:
    """
    Args:
        scores: [N] safety values of the calibration states.
        unsafe: [N] bool, ground-truth violation labels.
        alpha: target bound on the false-safe rate, in (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    unsafe = unsafe.bool()
    unsafe_scores = scores[unsafe].float().sort().values
    n = unsafe_scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))  # 1-based order statistic
    threshold = float("inf") if rank > n else float(unsafe_scores[rank - 1])
    return Calibration(threshold, alpha, n, scores.numel())


@dataclass
class FilterReport:
    false_safe_rate: float  # P(score > delta | unsafe): the calibrated quantity
    false_unsafe_rate: float  # P(score <= delta | safe): conservativeness
    unsafe_given_accepted: float  # P(unsafe | score > delta): not guaranteed
    accept_rate: float
    n_unsafe: int
    n_safe: int


def evaluate_threshold(scores: Tensor, unsafe: Tensor, threshold: float) -> FilterReport:
    unsafe = unsafe.bool()
    accepted = scores > threshold
    n_unsafe, n_safe = int(unsafe.sum()), int((~unsafe).sum())

    def rate(mask: Tensor, given: Tensor) -> float:
        total = int(given.sum())
        return float((mask & given).sum()) / total if total else float("nan")

    return FilterReport(
        false_safe_rate=rate(accepted, unsafe),
        false_unsafe_rate=rate(~accepted, ~unsafe),
        unsafe_given_accepted=rate(unsafe, accepted),
        accept_rate=float(accepted.float().mean()) if scores.numel() else float("nan"),
        n_unsafe=n_unsafe,
        n_safe=n_safe,
    )


def binomial_tolerance(alpha: float, n: int, z: float = 3.0) -> float:
    """Slack for comparing an empirical rate on ``n`` test points with ``alpha``."""
    return z * math.sqrt(alpha * (1.0 - alpha) / max(n, 1))


def sweep_alpha(scores: Tensor, unsafe: Tensor, test_scores: Tensor, test_unsafe: Tensor,
                alphas=(0.01, 0.02, 0.05, 0.1, 0.2)) -> list[dict]:
    """Calibrate on one split and report the empirical rates on another."""
    rows = []
    for alpha in alphas:
        cal = calibrate_threshold(scores, unsafe, alpha)
        report = evaluate_threshold(test_scores, test_unsafe, cal.threshold)
        rows.append({"alpha": alpha, "threshold": cal.threshold, **report.__dict__})
    return rows
