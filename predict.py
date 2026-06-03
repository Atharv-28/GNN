"""
Inference utility for HGCM-Net demo dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import HGCMConfig
from data_loader import load_demo_dataset
from model import HGCMNet


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = HGCMConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    data = load_demo_dataset(args.data_dir, config)
    model = HGCMNet(config).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    with torch.no_grad():
        x_dict = {k: data[k].x.to(device) for k in data.node_types}
        edge_dict = {k: data[k].edge_index.to(device) for k in data.edge_types}
        logits, _, _ = model(x_dict, edge_dict)
        probs = logits.sigmoid().cpu()

    content_ids = list(data["content"].content_id)
    scored = list(zip(content_ids, probs.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    print("Top predictions (higher = more likely label=1):")
    for cid, score in scored[: args.top_k]:
        print(f"{cid}\t{score:.4f}")


if __name__ == "__main__":
    main()
