"""
Dataset loader for HGCM-Net demo CSVs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from config import HGCMConfig
from graph_builder import HeterogeneousGraphBuilder


def _read_users(path: Path, config: HGCMConfig) -> Tuple[List[str], torch.Tensor]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("users.csv is empty")

    feature_cols = [f"f{i}" for i in range(config.user_raw_dim)]
    missing = [c for c in feature_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"users.csv missing columns: {missing}")

    user_ids = [row["user_id"] for row in rows]
    feats = [[float(row[c]) for c in feature_cols] for row in rows]
    return user_ids, torch.tensor(feats, dtype=torch.float)


def _read_contents(path: Path) -> Tuple[List[str], List[int], List[str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("contents.csv is empty")

    content_ids = [row["content_id"] for row in rows]
    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    return content_ids, labels, texts


def _read_interactions(
    path: Path,
    user_id_to_idx: Dict[str, int],
    content_id_to_idx: Dict[str, int],
) -> List[Tuple[int, int, float, float]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    interactions: List[Tuple[int, int, float, float]] = []
    for row in rows:
        u = user_id_to_idx[row["user_id"]]
        c = content_id_to_idx[row["content_id"]]
        rho = float(row.get("rho", 1.0))
        tau = float(row.get("tau_hours", 0.0))
        interactions.append((u, c, rho, tau))
    return interactions


def load_demo_dataset(data_dir: str, config: HGCMConfig):
    data_path = Path(data_dir)
    users_path = data_path / "users.csv"
    contents_path = data_path / "contents.csv"
    interactions_path = data_path / "interactions.csv"

    user_ids, user_features = _read_users(users_path, config)
    content_ids, labels, texts = _read_contents(contents_path)

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    content_id_to_idx = {cid: i for i, cid in enumerate(content_ids)}

    interactions = _read_interactions(
        interactions_path, user_id_to_idx, content_id_to_idx
    )

    builder = HeterogeneousGraphBuilder(config)
    data = builder.build(
        content_texts=texts,
        user_features=user_features,
        interactions=interactions,
        content_labels=torch.tensor(labels, dtype=torch.long),
    )

    data["content"].content_id = content_ids
    return data
