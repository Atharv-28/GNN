"""
Heterogeneous Directed Graph Builder for HGCM-Net
==================================================
Constructs a ``torch_geometric.data.HeteroData`` object with:
    • **User** nodes   – projected metadata features
    • **Content** nodes – Sentence-Transformer embeddings
    • **Edges**        – (User, interacts_with, Content) with temporal-decay
                         weights  w_ij = ρ_ij / (1 + log(1 + τ_ij))
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

from config import HGCMConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Temporal Decay Edge Weight
# ═══════════════════════════════════════════════════════════════════════

def compute_edge_weight(
    rho: float,
    tau_hours: float,
) -> float:
    r"""Compute the temporal-decay edge weight.

    .. math::
        w_{ij} = \frac{\rho_{ij}}{1 + \log(1 + \tau_{ij})}

    Parameters
    ----------
    rho : float
        Interaction strength / credibility score ρ_ij.
    tau_hours : float
        Time gap τ_ij in **hours** between user action and content creation.

    Returns
    -------
    float
        Edge weight w_ij ≥ 0.
    """
    if tau_hours < 0:
        raise ValueError("tau_hours must be non-negative")
    return rho / (1.0 + math.log(1.0 + tau_hours))


def compute_edge_weights_batch(
    rhos: torch.Tensor,
    taus: torch.Tensor,
) -> torch.Tensor:
    r"""Vectorised temporal-decay edge weights.

    Parameters
    ----------
    rhos : Tensor, shape ``(E,)``
        Interaction strengths.
    taus : Tensor, shape ``(E,)``
        Time gaps in hours.

    Returns
    -------
    Tensor, shape ``(E,)``
    """
    return rhos / (1.0 + torch.log(1.0 + taus))


# ═══════════════════════════════════════════════════════════════════════
#  Graph Builder
# ═══════════════════════════════════════════════════════════════════════

class HeterogeneousGraphBuilder:
    """Builds a heterogeneous directed graph for HGCM-Net.

    The builder accepts raw data (texts, user features, interaction records)
    and produces a ``HeteroData`` object ready for training.

    Parameters
    ----------
    config : HGCMConfig
        Global configuration.
    sentence_model : SentenceTransformer | None
        Pre-loaded sentence-transformer.  If ``None`` the model is loaded
        from ``config.sentence_model_name``.
    """

    def __init__(
        self,
        config: HGCMConfig,
        sentence_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.config = config
        self.st_model = sentence_model or SentenceTransformer(
            config.sentence_model_name
        )
        logger.info(
            "HeterogeneousGraphBuilder initialised  "
            "(sentence-model=%s)", config.sentence_model_name,
        )

    # ── Content Embeddings ──────────────────────────────────────────────

    def encode_content(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode raw text into dense vectors using Sentence-Transformers.

        Parameters
        ----------
        texts : list[str]
            Raw content strings (tweets, posts, articles …).
        batch_size : int
            Encoding batch size.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        Tensor, shape ``(N_content, content_embed_dim)``
        """
        embeddings = self.st_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )
        return embeddings.float()

    # ── Build HeteroData ────────────────────────────────────────────────

    def build(
        self,
        content_texts: List[str],
        user_features: torch.Tensor,
        interactions: List[Tuple[int, int, float, float]],
        content_labels: Optional[torch.Tensor] = None,
    ) -> HeteroData:
        """Construct the heterogeneous directed graph.

        Parameters
        ----------
        content_texts : list[str]
            Raw text for each content node.
        user_features : Tensor, shape ``(N_user, user_raw_dim)``
            Raw numerical metadata for every user.
        interactions : list[tuple]
            Each element is ``(user_idx, content_idx, rho, tau_hours)``.
        content_labels : Tensor | None, shape ``(N_content,)``
            Binary labels (0 = genuine, 1 = fake/hate).  May be ``None``
            at inference time.

        Returns
        -------
        HeteroData
        """
        logger.info(
            "Building graph  users=%d  content=%d  edges=%d",
            user_features.size(0), len(content_texts), len(interactions),
        )

        data = HeteroData()

        # ── Content nodes ───────────────────────────────────────────────
        content_x = self.encode_content(content_texts)
        data["content"].x = content_x
        data["content"].num_nodes = content_x.size(0)
        if content_labels is not None:
            data["content"].y = content_labels.long()

        # ── User nodes ──────────────────────────────────────────────────
        data["user"].x = user_features.float()
        data["user"].num_nodes = user_features.size(0)

        # ── Edges: (user → content) with temporal-decay weights ─────────
        if len(interactions) == 0:
            src = torch.zeros(0, dtype=torch.long)
            dst = torch.zeros(0, dtype=torch.long)
            weights = torch.zeros(0, dtype=torch.float)
        else:
            src_list, dst_list, rho_list, tau_list = zip(*interactions)
            src = torch.tensor(src_list, dtype=torch.long)
            dst = torch.tensor(dst_list, dtype=torch.long)
            rhos = torch.tensor(rho_list, dtype=torch.float)
            taus = torch.tensor(tau_list, dtype=torch.float)
            weights = compute_edge_weights_batch(rhos, taus)

        edge_type = ("user", "interacts_with", "content")
        data[edge_type].edge_index = torch.stack([src, dst], dim=0)
        data[edge_type].edge_weight = weights

        # ── Reverse edges (for message passing back to users) ───────────
        rev_edge_type = ("content", "rev_interacts_with", "user")
        data[rev_edge_type].edge_index = torch.stack([dst, src], dim=0)
        data[rev_edge_type].edge_weight = weights

        logger.info(
            "Graph built  content_dim=%s  user_dim=%s  edge_weights∈[%.4f, %.4f]",
            list(content_x.shape),
            list(user_features.shape),
            weights.min().item() if weights.numel() > 0 else 0.0,
            weights.max().item() if weights.numel() > 0 else 0.0,
        )

        return data


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: Quick Synthetic Graph for Testing
# ═══════════════════════════════════════════════════════════════════════

def build_synthetic_graph(
    n_users: int = 200,
    n_contents: int = 500,
    n_edges: int = 2000,
    config: Optional[HGCMConfig] = None,
) -> HeteroData:
    """Generate a synthetic HeteroData graph for unit testing / smoke runs.

    All content embeddings are random (no Sentence-Transformer needed).

    Parameters
    ----------
    n_users, n_contents, n_edges : int
        Graph dimensions.
    config : HGCMConfig | None
        Uses defaults if ``None``.

    Returns
    -------
    HeteroData
    """
    config = config or HGCMConfig()
    data = HeteroData()

    # Nodes
    data["user"].x = torch.randn(n_users, config.user_raw_dim)
    data["user"].num_nodes = n_users
    data["content"].x = torch.randn(n_contents, config.content_embed_dim)
    data["content"].num_nodes = n_contents
    data["content"].y = torch.randint(0, 2, (n_contents,))

    # Edges
    src = torch.randint(0, n_users, (n_edges,))
    dst = torch.randint(0, n_contents, (n_edges,))
    rhos = torch.rand(n_edges) + 0.5          # ρ ∈ [0.5, 1.5]
    taus = torch.rand(n_edges) * 720          # τ ∈ [0, 720] hours
    weights = compute_edge_weights_batch(rhos, taus)

    edge_type = ("user", "interacts_with", "content")
    data[edge_type].edge_index = torch.stack([src, dst], dim=0)
    data[edge_type].edge_weight = weights

    rev_edge_type = ("content", "rev_interacts_with", "user")
    data[rev_edge_type].edge_index = torch.stack([dst, src], dim=0)
    data[rev_edge_type].edge_weight = weights

    return data
