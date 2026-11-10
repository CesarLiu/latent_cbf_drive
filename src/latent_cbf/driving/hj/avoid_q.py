"""Discrete-action HJ avoid value learning in a world model's latent space.

The critic is trained on imagined rollouts with the time-discounted avoid
Bellman backup used by latent_cbf (PyHJ ``BasePolicy_Annealing_Avoid``):

    Q(z, a) <- (1 - gamma) * l(z') + gamma * min(l(z'), max_a' Q(z', a'))

where ``z'`` is the imagined successor of ``(z, a)`` and ``l`` is the safety
margin. With gamma -> 1 the fixed point is the worst margin along the best
(safest) continuation, so ``Q > 0`` means the ego can stay safe after taking
``a``. The maximisation ranges over the candidate actions of the imagination
(for SMART: the top-K tokens of the nominal policy), which makes the safe
controller a restriction of the nominal behaviour rather than an arbitrary one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .imagination import Imagination
from .networks import QNetwork

MarginFn = Callable[[Tensor], Tensor]


@dataclass
class AvoidQConfig:
    gamma: float = 0.9999
    warmup_updates: int = 500  # updates with gamma = 0, i.e. Q regresses l(z')
    lr: float = 1e-3
    weight_decay: float = 1e-3
    tau: float = 0.005
    batch_size: int = 512
    buffer_size: int = 200_000
    rollout_batch: int = 64
    updates_per_rollout: int = 16
    # Behaviour mixture used to collect imagined transitions.
    p_random: float = 0.3
    p_nominal: float = 0.4  # remainder acts greedily w.r.t. the current critic
    double_q: bool = True


class TransitionBuffer:
    """Ring buffer of imagined transitions kept on one device."""

    def __init__(self, capacity: int, latent_dim: int, feat_dim: int, n_cand: int, device) -> None:
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.z = torch.zeros(capacity, latent_dim, device=device)
        self.a_feat = torch.zeros(capacity, feat_dim, device=device)
        self.margin_next = torch.zeros(capacity, device=device)
        self.z_next = torch.zeros(capacity, latent_dim, device=device)
        self.cand_next = torch.zeros(capacity, n_cand, feat_dim, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)

    def add(self, z, a_feat, margin_next, z_next, cand_next, done) -> None:
        n = z.shape[0]
        index = (self.ptr + torch.arange(n, device=z.device)) % self.capacity
        self.z[index], self.a_feat[index] = z, a_feat
        self.margin_next[index], self.z_next[index] = margin_next, z_next
        self.cand_next[index], self.done[index] = cand_next, done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        index = torch.randint(0, self.size, (batch_size,), device=self.z.device)
        return (
            self.z[index],
            self.a_feat[index],
            self.margin_next[index],
            self.z_next[index],
            self.cand_next[index],
            self.done[index],
        )


def avoid_target(margin_next: Tensor, value_next: Tensor, done: Tensor, gamma: float) -> Tensor:
    """Time-discounted avoid backup. At the end of the horizon only l(z') counts."""
    future = torch.where(done, margin_next, torch.minimum(margin_next, value_next))
    return (1.0 - gamma) * margin_next + gamma * future


class AvoidQLearner:
    def __init__(
        self,
        env: Imagination,
        margin_fn: Optional[MarginFn] = None,
        cfg: Optional[AvoidQConfig] = None,
        q: Optional[QNetwork] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Args:
            env: imagination to roll out.
            margin_fn: learned boundary function l(z). If None, the imagination
                must provide ``true_margin`` (toy problems and diagnostics).
        """
        self.env = env
        self.cfg = cfg or AvoidQConfig()
        self.device = torch.device(device)
        self.margin_fn = margin_fn
        self.q = (q or QNetwork(env.latent_dim, env.action_feat_dim)).to(self.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.AdamW(
            self.q.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.buffer: Optional[TransitionBuffer] = None
        self.n_updates = 0

    # ------------------------------------------------------------------ data
    def _margin(self, z: Tensor) -> Tensor:
        if self.margin_fn is not None:
            return self.margin_fn(z)
        margin = self.env.true_margin()
        if margin is None:
            raise ValueError("no margin_fn given and the imagination has no true_margin")
        return margin

    @torch.no_grad()
    def _behaviour(self, z: Tensor, cand) -> Tensor:
        B, K = cand.logp.shape
        greedy = self.q(z, cand.feat).argmax(-1)
        nominal = torch.distributions.Categorical(logits=cand.logp).sample()
        random = torch.randint(0, K, (B,), device=z.device)
        u = torch.rand(B, device=z.device)
        slot = torch.where(u < self.cfg.p_random, random, greedy)
        use_nominal = (u >= self.cfg.p_random) & (u < self.cfg.p_random + self.cfg.p_nominal)
        return torch.where(use_nominal, nominal, slot)

    @torch.no_grad()
    def collect(self) -> int:
        """Roll the imagination for one horizon and store the transitions."""
        z = self.env.reset(self.cfg.rollout_batch).to(self.device)
        cand = self.env.candidates()
        if self.buffer is None:
            self.buffer = TransitionBuffer(
                self.cfg.buffer_size, z.shape[-1], cand.feat.shape[-1], cand.feat.shape[1], self.device
            )
        for t in range(self.env.horizon):
            slot = self._behaviour(z, cand)
            a_feat = cand.take(slot)
            z_next = self.env.step(slot).to(self.device)
            margin_next = self._margin(z_next)
            done = torch.full((z.shape[0],), t == self.env.horizon - 1, device=self.device)
            cand_next = cand if t == self.env.horizon - 1 else self.env.candidates()
            self.buffer.add(z, a_feat, margin_next, z_next, cand_next.feat, done)
            z, cand = z_next, cand_next
        return self.env.horizon * self.cfg.rollout_batch

    # --------------------------------------------------------------- updates
    @property
    def gamma(self) -> float:
        return 0.0 if self.n_updates < self.cfg.warmup_updates else self.cfg.gamma

    def update(self) -> Dict[str, float]:
        assert self.buffer is not None and self.buffer.size > 0, "collect() first"
        z, a_feat, margin_next, z_next, cand_next, done = self.buffer.sample(self.cfg.batch_size)
        with torch.no_grad():
            q_next = self.q_target(z_next, cand_next)
            if self.cfg.double_q:
                best = self.q(z_next, cand_next).argmax(-1, keepdim=True)
                value_next = q_next.gather(1, best).squeeze(1)
            else:
                value_next = q_next.amax(-1)
            target = avoid_target(margin_next, value_next, done, self.gamma)
        q = self.q(z, a_feat)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                p_targ.lerp_(p, self.cfg.tau)
        self.n_updates += 1
        return {
            "loss": loss.item(),
            "q_mean": q.mean().item(),
            "target_neg_frac": (target < 0).float().mean().item(),
            "gamma": self.gamma,
        }

    def train(self, n_rollouts: int, log_every: int = 0) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for i in range(n_rollouts):
            self.collect()
            for _ in range(self.cfg.updates_per_rollout):
                metrics = self.update()
            if log_every and (i + 1) % log_every == 0:
                print(f"[avoid-q] rollout {i + 1}/{n_rollouts} " + " ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics

    # ------------------------------------------------------------------ misc
    def state_dict(self) -> dict:
        return {"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                "n_updates": self.n_updates}

    def load_state_dict(self, state: dict) -> None:
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q_target"])
        self.n_updates = state["n_updates"]
