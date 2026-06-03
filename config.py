"""
HGCM-Net Configuration
======================
Centralised hyperparameters and paths for the Hybrid Graph-based
Contrastive Model (Nivedita & Bazila Banu, Scientific Reports 2025/2026).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HGCMConfig:
    """Master configuration for HGCM-Net."""

    # ── Sentence Transformer ────────────────────────────────────────────
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    content_embed_dim: int = 384          # output dim of all-MiniLM-L6-v2

    # ── User Metadata Projector ─────────────────────────────────────────
    user_raw_dim: int = 10                # raw user feature dimension
    user_proj_dim: int = 384              # projected (latent) dimension
    projector_hidden_dim: int = 256       # hidden layer in MLP projector

    # ── GAT Backbone ────────────────────────────────────────────────────
    gat_hidden_dim: int = 128
    gat_out_dim: int = 64
    gat_heads_1: int = 4                  # heads in 1st GAT layer
    gat_heads_2: int = 2                  # heads in 2nd GAT layer
    gat_dropout: float = 0.4             # slightly higher for real data
    leaky_relu_slope: float = 0.2         # LeakyReLU negative slope

    # ── Contrastive Learning ────────────────────────────────────────────
    temperature: float = 0.07             # temperature τ for NT-Xent style
    contrastive_lambda: float = 0.05      # λ — small so BCE dominates

    # ── Training ────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 512
    num_neighbors: list = field(default_factory=lambda: [15, 10])
    patience: int = 30
    grad_accum_steps: int = 4
    scheduler_warmup: int = 20            # warmup epochs for cosine LR

    # ── Dataset ─────────────────────────────────────────────────────────
    n_users: int = 800                    # K-means user clusters
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Temporal Decay ──────────────────────────────────────────────────
    rho_default: float = 1.0             # default interaction strength ρ

    # ── Device ──────────────────────────────────────────────────────────
    device: str = "cuda"                  # "cuda" or "cpu"

    # ── Paths ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
