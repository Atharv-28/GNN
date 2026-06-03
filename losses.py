"""
Loss Functions for HGCM-Net
============================
ContrastiveLoss, ContrastiveLossInBatch, and HGCMCombinedLoss.
L_final = L_BCE + lambda * L_contrastive + alpha * L_L2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Temperature-scaled contrastive loss (NT-Xent variant)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors, positives, negatives):
        """
        anchors:   (B, D)
        positives: (B, D)
        negatives: (B, N_neg, D) or (N_neg, D)
        """
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        pos_sim = (anchors * positives).sum(dim=-1) / self.temperature

        if negatives.dim() == 2:
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.mm(anchors, negatives.t()) / self.temperature
        else:
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.bmm(negatives, anchors.unsqueeze(-1)).squeeze(-1) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, labels)


class ContrastiveLossInBatch(nn.Module):
    """In-batch contrastive: other samples serve as implicit negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_embeds, content_embeds):
        user_embeds = F.normalize(user_embeds, dim=-1)
        content_embeds = F.normalize(content_embeds, dim=-1)
        sim = torch.mm(user_embeds, content_embeds.t()) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)


class HGCMCombinedLoss(nn.Module):
    """L = L_BCE + lambda * L_contrastive + alpha * L_L2."""

    def __init__(self, contrastive_lambda=0.5, temperature=0.07):
        super().__init__()
        self.lam = contrastive_lambda
        self.bce = nn.BCEWithLogitsLoss()
        self.contrastive = ContrastiveLossInBatch(temperature)

    def forward(self, logits, labels, user_embeds, content_embeds, l2_term=0.0):
        loss_bce = self.bce(logits, labels.float())
        loss_con = self.contrastive(user_embeds, content_embeds)
        total = loss_bce + self.lam * loss_con + l2_term
        breakdown = {
            "loss_bce": loss_bce.item(),
            "loss_contrastive": loss_con.item(),
            "loss_l2": l2_term if isinstance(l2_term, float) else l2_term,
            "loss_total": total.item(),
        }
        return total, breakdown
