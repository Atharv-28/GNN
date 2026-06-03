"""
Evaluation utility for HGCM-Net demo dataset.
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


@torch.no_grad()
def compute_metrics(logits, labels):
    preds = (logits.sigmoid() > 0.5).long()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    acc = (preds == labels).float().mean().item()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * prec * rec / (prec + rec + 1e-8))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _split_masks(n_content, train_ratio, val_ratio, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_content, generator=g)
    n_train = int(train_ratio * n_content)
    n_val = int(val_ratio * n_content)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(n_content, dtype=torch.bool)
    val_mask = torch.zeros(n_content, dtype=torch.bool)
    test_mask = torch.zeros(n_content, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


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

    n_content = data["content"].num_nodes
    _train_mask, _val_mask, test_mask = _split_masks(
        n_content, args.train_ratio, args.val_ratio, args.seed
    )

    with torch.no_grad():
        x_dict = {k: data[k].x.to(device) for k in data.node_types}
        edge_dict = {k: data[k].edge_index.to(device) for k in data.edge_types}
        logits, _, _ = model(x_dict, edge_dict)
        labels = data["content"].y.to(device)
        logits_t = logits[test_mask]
        labels_t = labels[test_mask]

    metrics = compute_metrics(logits_t, labels_t)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
