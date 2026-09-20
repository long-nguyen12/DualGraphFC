"""Attention-based multimodal feature fusion."""

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, mask=None, return_attention=False):
        if nodes.ndim != 3:
            raise ValueError("nodes must have shape [batch, nodes, hidden_dim]")

        scores = self.score(nodes).squeeze(-1)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError("pooling mask must have shape [batch, nodes]")
            mask = mask.bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(nodes * weights.unsqueeze(-1), dim=1)

        if return_attention:
            return pooled, weights
        return pooled


class MultimodalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.text_pool = AttentionPool(hidden_dim)
        self.visual_pool = AttentionPool(hidden_dim)
        self.consistency_pool = AttentionPool(hidden_dim)
        self.text_gate = nn.Linear(hidden_dim, hidden_dim)
        self.visual_gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        text_nodes,
        visual_nodes,
        text_consistency_nodes,
        visual_consistency_nodes,
        text_mask=None,
        visual_mask=None,
        return_details=False,
    ):
        text_embedding, text_attention = self.text_pool(
            text_nodes, text_mask, return_attention=True
        )
        visual_embedding, visual_attention = self.visual_pool(
            visual_nodes, visual_mask, return_attention=True
        )
        text_consistency_embedding, text_consistency_attention = (
            self.consistency_pool(
                text_consistency_nodes,
                text_mask,
                return_attention=True,
            )
        )
        visual_consistency_embedding, visual_consistency_attention = (
            self.consistency_pool(
                visual_consistency_nodes,
                visual_mask,
                return_attention=True,
            )
        )
        consistency_embedding = (
            text_consistency_embedding + visual_consistency_embedding
        ) * 0.5

        text_gate = torch.sigmoid(self.text_gate(text_embedding))
        visual_gate = torch.sigmoid(self.visual_gate(visual_embedding))
        fused = torch.cat(
            (
                text_gate * text_embedding,
                visual_gate * visual_embedding,
                consistency_embedding,
            ),
            dim=-1,
        )

        if not return_details:
            return fused
        return {
            "fused": fused,
            "text_embedding": text_embedding,
            "visual_embedding": visual_embedding,
            "consistency_embedding": consistency_embedding,
            "text_gate": text_gate,
            "visual_gate": visual_gate,
            "text_pool_attention": text_attention,
            "visual_pool_attention": visual_attention,
            "text_consistency_pool_attention": text_consistency_attention,
            "visual_consistency_pool_attention": visual_consistency_attention,
        }
