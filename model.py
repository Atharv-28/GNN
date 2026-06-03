"""
HGCM-Net Model
===============
Hybrid Graph-based Contrastive Model for fake news & hate speech detection.

Architecture:
  1. MetadataProjector – MLP: raw user features → latent space
  2. HeteroGATBackbone – 2-layer GATConv over heterogeneous graph
  3. ClassificationHead – binary classifier
  4. HGCMNet – end-to-end wrapper with skip-connection from raw embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class MetadataProjector(nn.Module):
    """Non-linear MLP: Linear → BN → ReLU → Dropout → Linear."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HeteroGATBackbone(nn.Module):
    """2-layer heterogeneous GATConv with LeakyReLU attention."""

    def __init__(self, user_dim, content_dim, hidden_dim, out_dim,
                 heads_1=4, heads_2=2, dropout=0.3, negative_slope=0.2):
        super().__init__()

        self.conv1 = HeteroConv({
            ("user", "interacts_with", "content"): GATConv(
                (-1, -1), hidden_dim, heads=heads_1,
                dropout=dropout, negative_slope=negative_slope,
                add_self_loops=False,
            ),
            ("content", "rev_interacts_with", "user"): GATConv(
                (-1, -1), hidden_dim, heads=heads_1,
                dropout=dropout, negative_slope=negative_slope,
                add_self_loops=False,
            ),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("user", "interacts_with", "content"): GATConv(
                (-1, -1), out_dim, heads=heads_2,
                concat=False, dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=False,
            ),
            ("content", "rev_interacts_with", "user"): GATConv(
                (-1, -1), out_dim, heads=heads_2,
                concat=False, dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=False,
            ),
        }, aggr="sum")

        self.act = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.act(self.drop(v)) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class ClassificationHead(nn.Module):
    """Binary classifier with skip-connection from raw content embeddings."""

    def __init__(self, gat_dim, raw_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        # Project raw embeddings to same dim as GAT output
        self.raw_proj = nn.Linear(raw_dim, gat_dim)
        self.net = nn.Sequential(
            nn.Linear(gat_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gat_out, raw_embed):
        # Concatenate GAT output with projected raw embedding (skip-connection)
        raw_proj = self.raw_proj(raw_embed)
        combined = torch.cat([gat_out, raw_proj], dim=-1)
        return self.net(combined).squeeze(-1)


class HGCMNet(nn.Module):
    """Full HGCM-Net model with skip-connection from Sentence-Transformer embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.user_projector = MetadataProjector(
            config.user_raw_dim,
            config.projector_hidden_dim,
            config.user_proj_dim,
            dropout=config.gat_dropout,
        )

        self.gat = HeteroGATBackbone(
            user_dim=config.user_proj_dim,
            content_dim=config.content_embed_dim,
            hidden_dim=config.gat_hidden_dim,
            out_dim=config.gat_out_dim,
            heads_1=config.gat_heads_1,
            heads_2=config.gat_heads_2,
            dropout=config.gat_dropout,
            negative_slope=config.leaky_relu_slope,
        )

        # Classifier uses GAT output + raw embedding skip-connection
        self.classifier = ClassificationHead(
            gat_dim=config.gat_out_dim,
            raw_dim=config.content_embed_dim,
            hidden_dim=128,
            dropout=config.gat_dropout,
        )

    def forward(self, x_dict, edge_index_dict):
        """
        Returns
        -------
        logits         : (N_content,)
        user_embeds    : (N_user, gat_out_dim)
        content_embeds : (N_content, gat_out_dim)
        """
        x_dict = dict(x_dict)
        raw_content = x_dict["content"]   # save raw ST embeddings

        x_dict["user"] = self.user_projector(x_dict["user"])

        out = self.gat(x_dict, edge_index_dict)

        user_embeds    = out.get("user",    x_dict["user"])
        content_embeds = out.get("content", x_dict["content"])

        # Skip-connection: classifier sees both GAT features + raw embeddings
        logits = self.classifier(content_embeds, raw_content)
        return logits, user_embeds, content_embeds
