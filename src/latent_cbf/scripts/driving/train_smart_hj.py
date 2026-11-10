"""Train the latent HJ safety critic in SMART's imagination.

Stages (select with ``--stage``):

    probe   plan stage 1: can the ego latent predict the clearance margin?
            Collects (latent, margin) pairs from imagined rollouts, fits the
            margin head on a split and reports held-out sign accuracy.
    train   plan stage 3: discrete avoid Q-learning in imagination.
    eval    violation / intervention rates of the filtered ego inside the
            imagination, plus a model-internal conformal sweep. These numbers
            are a sanity check only; the statistical claim needs labels from
            Nocturne (see adapters.ctrlsim).

    python scripts/driving/train_smart_hj.py --stage probe \
        --catk-root /path/to/catk --ckpt /path/to/clsft.ckpt \
        --data-dir /path/to/womd_cache/validation --out runs/smart_hj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from smart_common import add_smart_args, build_imagination

from driving.adapters.smart_imagination import ROAD_EDGE_BOUNDARY
from driving.conformal import sweep_alpha
from driving.hj import AvoidQConfig, AvoidQLearner, MarginHead, QNetwork, fit_margin_head
from driving.rollouts import calibration_rollouts, evaluate_policy
from driving.safety_filter import LatentSafetyFilter


@torch.no_grad()
def collect_latents(env, n_rollouts: int):
    """Latents and analytic margins along rollouts with random candidate tokens."""
    zs, margins, edge_hit, edge_on = [], [], [], []
    for _ in range(n_rollouts):
        z = env.reset()
        edge_on.append(env.edge_active.float())
        for _ in range(env.horizon):
            cand = env.candidates()
            # mostly nominal, sometimes a random candidate to reach unsafe states
            slot = torch.where(
                torch.rand(z.shape[0], device=z.device) < 0.5,
                torch.randint(0, cand.idx.shape[1], (z.shape[0],), device=z.device),
                torch.distributions.Categorical(logits=cand.logp).sample(),
            )
            z = env.step(slot)
            keep = env.ego_valid()
            zs.append(z[keep])
            margins.append(env.true_margin()[keep])
            edge_hit.append((env.ego_road_edge_gap()[keep] < env.min_road_edge).float())
    stats = {"road_edge_rule_active": torch.cat(edge_on).mean().item(),
             "road_edge_violation_fraction": torch.cat(edge_hit).mean().item()}
    return torch.cat(zs), torch.cat(margins), stats


def stage_probe(env, args, out: Path) -> None:
    z, margin, edge_stats = collect_latents(env, args.rollouts)
    n = z.shape[0]
    perm = torch.randperm(n, device=z.device)
    train, test = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    head = MarginHead(env.latent_dim).to(z.device)
    metrics = fit_margin_head(head, z[train], margin[train], steps=args.margin_steps)
    with torch.no_grad():
        pred = head(z[test])
    unsafe = margin[test] < 0
    report = {
        "n_samples": n,
        "unsafe_fraction": (margin < 0).float().mean().item(),
        **edge_stats,
        "train": metrics,
        "test_mae": (pred - margin[test]).abs().mean().item(),
        "test_sign_acc": ((pred < 0) == unsafe).float().mean().item(),
        "test_unsafe_recall": ((pred < 0) & unsafe).sum().item() / max(int(unsafe.sum()), 1),
        "test_safe_recall": ((pred >= 0) & ~unsafe).sum().item() / max(int((~unsafe).sum()), 1),
    }
    print(json.dumps(report, indent=2))
    (out / "probe.json").write_text(json.dumps(report, indent=2))
    torch.save(head.state_dict(), out / "margin_head.pt")


def make_learner(env, args, out: Path) -> AvoidQLearner:
    margin_fn = None  # analytic clearance margin from the imagined positions
    if args.learned_margin:
        head = MarginHead(env.latent_dim).to(args.device)
        head.load_state_dict(torch.load(out / "margin_head.pt", map_location=args.device))
        margin_fn = head.eval().requires_grad_(False)
    cfg = AvoidQConfig(gamma=args.gamma, warmup_updates=args.warmup, tau=args.tau,
                       updates_per_rollout=args.updates_per_rollout)
    q = QNetwork(env.latent_dim, env.action_feat_dim)
    return AvoidQLearner(env, margin_fn=margin_fn, cfg=cfg, q=q, device=args.device)


def stage_train(env, args, out: Path) -> None:
    learner = make_learner(env, args, out)
    ckpt = out / "avoid_q.pt"
    if args.resume and ckpt.exists():
        learner.load_state_dict(torch.load(ckpt, map_location=args.device))
    done = 0
    while done < args.rollouts:
        chunk = min(args.save_every, args.rollouts - done)
        learner.train(chunk, log_every=args.log_every)
        done += chunk
        torch.save(learner.state_dict(), ckpt)
    print(f"saved {ckpt}")


def stage_eval(env, args, out: Path) -> None:
    learner = make_learner(env, args, out)
    learner.load_state_dict(torch.load(out / "avoid_q.pt", map_location=args.device))
    q = learner.q.eval()

    def average(safety_filter):
        stats = [evaluate_policy(env, 0, safety_filter) for _ in range(args.eval_batches)]
        return {k: sum(getattr(s, k) for s in stats) / len(stats)
                for k in ("violation_rate", "intervention_rate", "infeasible_rate")}

    uncalibrated = LatentSafetyFilter(q, 0.0, args.mode)
    report = {"nominal": average(None), "filtered_threshold_0": average(uncalibrated)}

    def gather(n):
        pairs = [calibration_rollouts(env, q, 0, args.prefix_steps, uncalibrated)
                 for _ in range(n)]
        return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])

    cal_scores, cal_unsafe = gather(args.eval_batches)
    test_scores, test_unsafe = gather(args.eval_batches)
    report["conformal_in_imagination"] = sweep_alpha(cal_scores, cal_unsafe, test_scores,
                                                     test_unsafe)
    print(json.dumps(report, indent=2))
    (out / "eval_imagination.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    add_smart_args(parser)
    parser.add_argument("--stage", choices=("probe", "train", "eval"), required=True)
    parser.add_argument("--out", default="runs/smart_hj")
    parser.add_argument("--rollouts", type=int, default=200,
                        help="imagined rollouts (data batches) to process")
    parser.add_argument("--margin-steps", type=int, default=3000)
    parser.add_argument("--learned-margin", action="store_true",
                        help="use the probe's margin head instead of the analytic clearance")
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--updates-per-rollout", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("cbf", "lr"), default="cbf")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--clearance-scale", type=float, default=2.0)
    parser.add_argument("--no-road-edge", action="store_true",
                        help="margin from vehicle clearance only")
    parser.add_argument("--road-edge-scale", type=float, default=1.0)
    parser.add_argument("--boundary-only", action="store_true",
                        help="ignore median road edges (TYPE_ROAD_EDGE_MEDIAN)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env, *_ = build_imagination(
        args, clearance_scale=args.clearance_scale, road_edge=not args.no_road_edge,
        road_edge_scale=args.road_edge_scale,
        road_edge_types=(ROAD_EDGE_BOUNDARY,) if args.boundary_only else None,
    )
    {"probe": stage_probe, "train": stage_train, "eval": stage_eval}[args.stage](env, args, out)


if __name__ == "__main__":
    main()
