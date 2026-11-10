"""Check the step-wise SMART imagination against CAT-K's own rollout.

With greedy sampling for every agent (top-1) the rollout is deterministic, so
``SmartImagination`` driven with the ego's most likely token must reproduce the
positions returned by ``SMARTAgentDecoder.inference``. Run this before training
anything on top of the adapter.

    python scripts/driving/check_smart_imagination.py \
        --catk-root /path/to/catk --ckpt /path/to/clsft.ckpt \
        --data-dir /path/to/womd_cache/validation
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from smart_common import add_smart_args, build_imagination


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    add_smart_args(parser)
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-3, help="metres")
    args = parser.parse_args()

    env, token_processor, decoder, loader = build_imagination(args, shuffle=False, others_top_k=1)
    greedy = OmegaConf.create({"criterium": "topk_prob", "num_k": 1, "temp": 1.0})

    worst = 0.0
    for i, data in zip(range(args.n_batches), loader):
        data = data.to(args.device)
        tokenized_map, tokenized_agent = token_processor(data)
        ref = decoder.inference(tokenized_map, tokenized_agent, greedy)

        z = env.reset(data=data)
        latents = [z]
        for _ in range(env.horizon):
            env.candidates()
            slot = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)  # top-1
            latents.append(env.step(slot))

        valid = ref["pred_valid"]
        err = (env.pos_a - ref["pred_pos"]).norm(dim=-1)[valid].max().item()
        idx_match = (env.pred_idx == ref["pred_idx"])[valid].float().mean().item()
        worst = max(worst, err)
        print(f"batch {i}: max position error {err:.2e} m, token agreement {idx_match:.4f}, "
              f"latent dim {z.shape[-1]}, egos {z.shape[0]}, "
              f"final clearance min {env.ego_clearance().min().item():.2f} m, "
              f"road-edge rule active {int(env.edge_active.sum())}/{z.shape[0]}, "
              f"final road-edge gap min {env.ego_road_edge_gap().min().item():.2f} m")
        assert len(latents) == env.horizon + 1

    if worst > args.tol:
        raise SystemExit(f"FAILED: max position error {worst:.3e} m exceeds {args.tol} m")
    print("OK: step-wise imagination matches CAT-K inference")


if __name__ == "__main__":
    main()
