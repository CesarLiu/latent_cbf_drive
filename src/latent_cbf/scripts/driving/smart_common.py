"""Shared setup for the SMART scripts: paths, model and data loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]  # .../src/latent_cbf
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def add_smart_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catk-root", required=True, help="path to the CAT-K repository")
    parser.add_argument("--ckpt", required=True, help="CAT-K checkpoint (.ckpt)")
    parser.add_argument("--data-dir", required=True,
                        help="cached WOMD split, e.g. <cache_root>/validation")
    parser.add_argument("--batch-size", type=int, default=8, help="scenes per batch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=16, help="ego candidate tokens")


def build_loader(args, shuffle: bool = True):
    """CAT-K validation-style loader (no agent subsampling)."""
    if args.catk_root not in sys.path:
        sys.path.insert(0, args.catk_root)
    from torch_geometric.loader import DataLoader

    from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
    from src.smart.datasets import MultiDataset

    dataset = MultiDataset(args.data_dir, WaymoTargetBuilderVal())
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, drop_last=False)


def build_imagination(args, shuffle: bool = True, **kwargs):
    from driving.adapters.smart_imagination import SmartImagination, load_smart

    token_processor, decoder = load_smart(args.ckpt, args.catk_root, args.device)
    loader = build_loader(args, shuffle)
    env = SmartImagination(token_processor, decoder, loader, top_k=args.top_k,
                           device=args.device, **kwargs)
    return env, token_processor, decoder, loader
