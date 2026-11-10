"""Training losses for the latent safety boundary function l(z).

Two label types are supported:

* ``regression``: fit the clipped STL robustness of the state formula.
* ``classification``: the original latent_cbf objective on binary failure
  labels (sign hinge plus a zero-sum term).

Both add a gradient penalty on interpolates between safe and unsafe latents,
which keeps l(z) smooth across the boundary so that the value function built on
top of it has usable level sets. latent_cbf pins the gradient norm to a fixed
value (``two_sided``), tuned for Dreamer features. That value is not portable:
if it is smaller than the slope the labels require, the head cannot fit the
boundary at all. The default here is therefore a one-sided Lipschitz cap whose
level is estimated from the data (``estimate_gradient_scale``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .networks import MarginHead


@dataclass
class MarginLossConfig:
    mode: str = "regression"  # "regression" or "classification"
    gp_weight: float = 1.0
    gp_mode: str = "one_sided"  # "one_sided" (cap) or "two_sided" (latent_cbf)
    # Gradient-norm level of the penalty, in margin units per latent unit.
    # None lets ``fit_margin_head`` estimate it from the data.
    gradient_thresh: Optional[float] = None
    sign_weight: float = 1.0
    zs_weight: float = 0.1  # classification only


def _resample(x: Tensor, n: int) -> Tensor:
    index = torch.randint(0, x.shape[0], (n,), device=x.device)
    return x[index]


@torch.no_grad()
def estimate_gradient_scale(z: Tensor, target: Tensor, n_pairs: int = 4096,
                            quantile: float = 0.9, slack: float = 2.0) -> float:
    """Slope the labels demand between safe and unsafe latents.

    Samples safe/unsafe pairs, computes ``|target_i - target_j| / ||z_i - z_j||``
    in atanh space (the space the penalty acts on) and returns ``slack`` times a
    high quantile. Falls back to 1.0 when one class is missing.
    """
    unsafe = target < 0 if target.min() < 0 else target > 0.5
    if unsafe.all() or not unsafe.any():
        return 1.0
    index_i = torch.randint(0, int((~unsafe).sum()), (n_pairs,), device=z.device)
    index_j = torch.randint(0, int(unsafe.sum()), (n_pairs,), device=z.device)
    zi, zj = z[~unsafe][index_i], z[unsafe][index_j]
    if target.min() < 0:
        raw = torch.atanh(target.clamp(-0.95, 0.95))
        diff = (raw[~unsafe][index_i] - raw[unsafe][index_j]).abs()
    else:
        diff = torch.full((n_pairs,), 2.0, device=z.device)
    slope = diff / (zi - zj).norm(dim=-1).clamp(min=1e-6)
    return slack * float(torch.quantile(slope, quantile))


def gradient_penalty(head: MarginHead, safe: Tensor, unsafe: Tensor, thresh: float,
                     mode: str = "one_sided") -> Tensor:
    """Penalise ||dl/dz|| on safe/unsafe interpolates.

    ``one_sided`` penalises only norms above ``thresh``; ``two_sided`` pulls the
    norm towards ``thresh`` as in latent_cbf.
    """
    if safe.shape[0] == 0 or unsafe.shape[0] == 0:
        return safe.new_zeros(())
    n = max(safe.shape[0], unsafe.shape[0])
    pos, neg = _resample(safe, n), _resample(unsafe, n)
    alpha = torch.rand(n, 1, device=safe.device)
    mix = (alpha * pos + (1 - alpha) * neg).detach().requires_grad_(True)
    out = head.raw(mix)
    (grad,) = torch.autograd.grad(out.sum(), mix, create_graph=True)
    norm = torch.sqrt((grad**2).sum(-1) + 1e-12)
    excess = norm - thresh
    if mode == "one_sided":
        excess = torch.relu(excess)
    elif mode != "two_sided":
        raise ValueError(f"unknown gp_mode: {mode}")
    return (excess**2).mean()


def margin_loss(
    head: MarginHead, z: Tensor, target: Tensor, cfg: MarginLossConfig
) -> tuple[Tensor, Dict[str, float]]:
    """Args: z [B, D]; target [B], the margin (regression) or failure flag in {0, 1}."""
    raw = head.raw(z)
    if cfg.mode == "regression":
        unsafe_mask = target < 0
        fit = F.mse_loss(torch.tanh(raw), target)
        # Sign errors are what break the filter, so they get an explicit hinge.
        sign = torch.relu(-raw * torch.sign(target)).mean()
        loss = fit + cfg.sign_weight * sign
        metrics = {"fit": fit.item(), "sign": sign.item()}
    elif cfg.mode == "classification":
        unsafe_mask = target > 0.5
        pos, neg = raw[~unsafe_mask], raw[unsafe_mask]
        pos_mean = pos.mean() if pos.numel() else raw.new_zeros(())
        neg_mean = neg.mean() if neg.numel() else raw.new_zeros(())
        zero = raw.new_zeros(())
        sign = (torch.relu(neg).mean() if neg.numel() else zero) + (
            torch.relu(-pos).mean() if pos.numel() else zero
        )
        zero_sum = neg_mean - pos_mean
        loss = cfg.sign_weight * sign + cfg.zs_weight * zero_sum
        metrics = {"sign": float(sign.detach()), "zero_sum": zero_sum.item()}
    else:
        raise ValueError(f"unknown margin mode: {cfg.mode}")

    if cfg.gp_weight > 0 and cfg.gradient_thresh is not None:
        gp = gradient_penalty(
            head, z[~unsafe_mask], z[unsafe_mask], cfg.gradient_thresh, cfg.gp_mode
        )
        loss = loss + cfg.gp_weight * gp
        metrics["gp"] = gp.item()

    with torch.no_grad():
        metrics["sign_acc"] = ((raw < 0) == unsafe_mask).float().mean().item()
    metrics["loss"] = loss.item()
    return loss, metrics


def fit_margin_head(
    head: MarginHead,
    z: Tensor,
    target: Tensor,
    cfg: MarginLossConfig | None = None,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Minibatch training on a fixed dataset of latents and labels."""
    cfg = cfg or MarginLossConfig()
    if cfg.gp_weight > 0 and cfg.gradient_thresh is None:
        cfg = replace(cfg, gradient_thresh=estimate_gradient_scale(z, target))
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    metrics: Dict[str, float] = {}
    for _ in range(steps):
        index = torch.randint(0, z.shape[0], (min(batch_size, z.shape[0]),), device=z.device)
        loss, metrics = margin_loss(head, z[index], target[index], cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    metrics["gradient_thresh"] = float(cfg.gradient_thresh or 0.0)
    return metrics
