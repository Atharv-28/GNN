"""
HGCM-Net 2.0 — Hate-Speech Model
===================================
Extends the original HGCMNet with:
  • 3-class output (normal / offensive / hate_speech)
  • Dual inference modes:
      text_only      — GAT is bypassed; only skip-connection path runs
      context_aware  — Full GAT + skip-connection (Phase 3+)
  • No breaking changes to MetadataProjector or HeteroGATBackbone

The existing model.py (fake-news, binary) is untouched.

Usage — text-only (Phase 2 baseline)
-------------------------------------
    from models.hgcm_net2 import HGCMNet2
    from config import HGCMConfig
    from config.labels import NUM_HATE_SPEECH_CLASSES

    config = HGCMConfig()
    model  = HGCMNet2(config, num_classes=NUM_HATE_SPEECH_CLASSES)

    # Text-only forward (no graph needed)
    logits, _, _ = model(x_dict, edge_dict, text_only=True)
    # logits: (N_content, 3)

Usage — context-aware (Phase 3)
---------------------------------
    logits, user_emb, content_emb = model(x_dict, edge_dict, text_only=False)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-use existing well-tested components from root model.py
from model import MetadataProjector, HeteroGATBackbone
from models.classification_head import ClassificationHead
from config.labels import NUM_HATE_SPEECH_CLASSES


# ─────────────────────────────────────────────────────────────────────
#  HGCM-Net 2.0
# ─────────────────────────────────────────────────────────────────────

class HGCMNet2(nn.Module):
    """HGCM-Net 2.0 — Hybrid Graph-based Content Moderation Network.

    Parameters
    ----------
    config : HGCMConfig
        Global hyperparameter config (from root config.py).
    num_classes : int
        Output classes.  3 for hate-speech; 2 for fake-news (legacy).
    """

    def __init__(self, config, num_classes: int = NUM_HATE_SPEECH_CLASSES) -> None:
        super().__init__()
        self.config      = config
        self.num_classes = num_classes

        # ── Re-used from original model.py ───────────────────────────
        self.user_projector = MetadataProjector(
            in_dim     = config.user_raw_dim,
            hidden_dim = config.projector_hidden_dim,
            out_dim    = config.user_proj_dim,
            dropout    = config.gat_dropout,
        )

        self.gat = HeteroGATBackbone(
            user_dim       = config.user_proj_dim,
            content_dim    = config.content_embed_dim,
            hidden_dim     = config.gat_hidden_dim,
            out_dim        = config.gat_out_dim,
            heads_1        = config.gat_heads_1,
            heads_2        = config.gat_heads_2,
            dropout        = config.gat_dropout,
            negative_slope = config.leaky_relu_slope,
        )

        # ── New: generalised classification head ─────────────────────
        self.classifier = ClassificationHead(
            gat_dim     = config.gat_out_dim,
            raw_dim     = config.content_embed_dim,
            num_classes = num_classes,
            hidden_dim  = 128,
            dropout     = config.gat_dropout,
        )

        # ── Text-only: a fixed zero tensor replaces the GAT output ───
        # Registered as a buffer so it moves to the right device with .to()
        self.register_buffer(
            "_gat_zero",
            torch.zeros(1, config.gat_out_dim),
        )

    def forward(
        self,
        x_dict:         Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        text_only:      bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node feature dictionary.  Must contain ``"content"`` and ``"user"``
            keys with shapes ``(N_c, content_embed_dim)`` and
            ``(N_u, user_raw_dim)`` respectively.
        edge_index_dict : dict
            Edge index dictionary for HeteroConv.
        text_only : bool
            If True, the GAT backbone is skipped and replaced with zeros.
            The skip-connection (raw ST embeddings → classifier) is the
            only active path.  Set during Phase 2 baseline training.

        Returns
        -------
        (logits, user_embeds, content_embeds) : tuple
            logits        : (N_content, num_classes)
            user_embeds   : (N_user, gat_out_dim)   — zeros in text-only mode
            content_embeds: (N_content, gat_out_dim) — zeros in text-only mode
        """
        x_dict       = dict(x_dict)                   # shallow copy, don't mutate input
        raw_content  = x_dict["content"]               # (N_c, 384) — save for skip-connection
        n_content    = raw_content.size(0)
        n_user       = x_dict["user"].size(0)

        if text_only:
            # ── Text-only mode ────────────────────────────────────────
            # GAT is completely bypassed.  The classifier sees:
            #   gat_out  = zeros  (no graph information)
            #   raw_proj = raw ST embedding  (the entire signal)
            gat_content = self._gat_zero.expand(n_content, -1)  # (N_c, gat_out_dim)
            gat_user    = self._gat_zero.expand(n_user,    -1)  # (N_u, gat_out_dim)
        else:
            # ── Context-aware mode ────────────────────────────────────
            x_dict["user"] = self.user_projector(x_dict["user"])
            out        = self.gat(x_dict, edge_index_dict)
            gat_user   = out.get("user",    x_dict["user"])
            gat_content = out.get("content", x_dict["content"])

        logits = self.classifier(gat_content, raw_content)   # (N_c, num_classes)
        return logits, gat_user, gat_content
