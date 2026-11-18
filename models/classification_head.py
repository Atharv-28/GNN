"""
Classification Head — HGCM-Net 2.0
=====================================
Generalised multi-class classifier that supports both the existing
binary (fake-news) task and the new 3-class hate-speech task.

The skip-connection from raw sentence embeddings is preserved exactly
as in the original model.py — it is the mechanism that enables
text-only inference without a graph.

Architecture
------------
  raw_embed  ──[ raw_proj ]──┐
                             ├──[ cat ]──[ MLP ]──[ num_classes logits ]
  gat_embed  ────────────────┘

Usage
-----
    from models.classification_head import ClassificationHead, build_loss

    head = ClassificationHead(gat_dim=64, raw_dim=384, num_classes=3)
    logits = head(gat_out, raw_embed)          # (N, 3)

    criterion = build_loss(class_weights=weights, num_classes=3)
    loss = criterion(logits, labels)           # CrossEntropyLoss
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────
#  Classification Head
# ─────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """Multi-class classifier with skip-connection from raw embeddings.

    Parameters
    ----------
    gat_dim : int
        Output dimension of the GAT backbone (or zero-tensor in text-only
        mode; both paths are handled identically).
    raw_dim : int
        Dimension of the raw sentence-transformer embedding (384 for
        all-MiniLM-L6-v2).
    num_classes : int
        Number of output classes.  2 → binary (fake-news); 3 → hate-speech.
    hidden_dim : int
        Hidden dimension of the MLP classifier.
    dropout : float
        Dropout probability inside the MLP.
    """

    def __init__(
        self,
        gat_dim:    int,
        raw_dim:    int,
        num_classes: int  = 3,
        hidden_dim: int   = 128,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Project raw embedding to same dim as GAT output
        self.raw_proj = nn.Linear(raw_dim, gat_dim)

        # MLP: (gat_dim * 2) → hidden → num_classes
        self.net = nn.Sequential(
            nn.Linear(gat_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, gat_out: torch.Tensor, raw_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        gat_out : Tensor, shape ``(N, gat_dim)``
            GAT-refined content embeddings.  In text-only mode, pass a
            zero tensor of the same shape (automatically handled by
            HGCMNet2 when graph is unavailable).
        raw_embed : Tensor, shape ``(N, raw_dim)``
            Raw sentence-transformer embeddings (the skip-connection).

        Returns
        -------
        Tensor, shape ``(N, num_classes)``
            Raw logits.  Apply softmax for probabilities.
        """
        raw_proj  = self.raw_proj(raw_embed)              # (N, gat_dim)
        combined  = torch.cat([gat_out, raw_proj], dim=-1) # (N, gat_dim*2)
        return self.net(combined)                          # (N, num_classes)


# ─────────────────────────────────────────────────────────────────────
#  Loss factory
# ─────────────────────────────────────────────────────────────────────

def build_loss(
    num_classes:   int             = 3,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float         = 0.0,
) -> nn.Module:
    """Return the appropriate loss for a classification task.

    Parameters
    ----------
    num_classes : int
        2 → BCEWithLogitsLoss (binary); 3+ → CrossEntropyLoss.
    class_weights : Tensor | None
        Per-class weights for CrossEntropyLoss (shape ``(num_classes,)``).
        Computed by :func:`tasks.hate_speech.pipeline.compute_class_weights`.
    label_smoothing : float
        Label smoothing for CrossEntropyLoss (0.0 = off).

    Returns
    -------
    nn.Module
        A loss criterion compatible with the ClassificationHead output.
    """
    if num_classes == 2:
        # Binary: BCEWithLogitsLoss on the single positive-class logit
        pos_weight = None
        if class_weights is not None and class_weights.numel() == 2:
            # Convert class weights to pos_weight (ratio negative/positive)
            pos_weight = class_weights[1] / (class_weights[0] + 1e-8)
            pos_weight = pos_weight.unsqueeze(0)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
