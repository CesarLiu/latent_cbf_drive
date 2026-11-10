"""Closed-loop rollouts shared by evaluation and conformal calibration.

These helpers run on anything implementing ``Imagination`` that also exposes a
ground-truth margin: the toy problem, or the SMART imagination when used for a
model-internal sanity check. The Nocturne counterpart, which provides the labels
that actually back the statistical claim, lives in ``adapters.ctrlsim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .hj.imagination import Candidates, Imagination
from .hj.networks import QNetwork
from .safety_filter import LatentSafetyFilter

NominalFn = Callable[[Candidates], Tensor]


def most_likely(cand: Candidates) -> Tensor:
    return cand.logp.argmax(-1)


@dataclass
class RolloutStats:
    violation_rate: float
    intervention_rate: float  # fraction of steps where the filter changed the action
    infeasible_rate: float  # fraction of steps with no candidate above threshold
    worst_margin: Tensor  # [B]


@torch.no_grad()
def evaluate_policy(
    env: Imagination,
    batch_size: int,
    safety_filter: Optional[LatentSafetyFilter] = None,
    nominal: NominalFn = most_likely,
) -> RolloutStats:
    z = env.reset(batch_size)  # data-driven imaginations choose their own batch size
    worst = torch.full((z.shape[0],), float("inf"), device=z.device)
    intervened = infeasible = 0.0
    for _ in range(env.horizon):
        cand = env.candidates()
        slot = nominal(cand)
        if safety_filter is not None:
            decision = safety_filter(z, cand, slot)
            slot = decision.slot
            intervened += decision.intervened.float().mean().item()
            infeasible += decision.infeasible.float().mean().item()
        z = env.step(slot)
        worst = torch.minimum(worst, env.true_margin())
    return RolloutStats(
        violation_rate=(worst < 0).float().mean().item(),
        intervention_rate=intervened / env.horizon,
        infeasible_rate=infeasible / env.horizon,
        worst_margin=worst,
    )


@torch.no_grad()
def calibration_rollouts(
    env: Imagination,
    q: QNetwork,
    batch_size: int,
    prefix_steps: int,
    safety_filter: Optional[LatentSafetyFilter] = None,
    nominal: NominalFn = most_likely,
) -> tuple[Tensor, Tensor]:
    """Scores and ground-truth labels for conformal calibration.

    Each rollout follows the (filtered) nominal policy for ``prefix_steps`` so
    that calibration states come from the deployment distribution. At the
    calibration state the nominal action ``a`` is scored with ``Q(z, a)``; the
    label is whether executing ``a`` and then the safe policy ``argmax Q`` for
    the rest of the horizon violates the constraint. This is the event that the
    acceptance test ``Q(z, a) > delta`` of the filter is meant to exclude.

    Returns:
        scores [B], unsafe [B] bool.
    """
    if not 0 <= prefix_steps < env.horizon:
        raise ValueError("prefix_steps must leave at least one step to evaluate")
    z = env.reset(batch_size)
    for _ in range(prefix_steps):
        cand = env.candidates()
        slot = nominal(cand)
        if safety_filter is not None:
            slot = safety_filter(z, cand, slot).slot
        z = env.step(slot)

    cand = env.candidates()
    slot = nominal(cand)
    scores = q(z, cand.take(slot))
    z = env.step(slot)
    worst = env.true_margin().clone()
    for _ in range(env.horizon - prefix_steps - 1):
        cand = env.candidates()
        z = env.step(q(z, cand.feat).argmax(-1))
        worst = torch.minimum(worst, env.true_margin())
    return scores, worst < 0
