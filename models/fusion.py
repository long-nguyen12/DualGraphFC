"""Claim-conditioned evidence and visual feature fusion."""

import torch
from torch import nn


class ConditionalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.claim_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, claim, mask, return_attention=False):
        mask = mask.to(device=nodes.device, dtype=torch.bool)
        if not mask.any(dim=-1).all():
            raise ValueError("Conditional pooling requires valid nodes in every sample")
        context = self.node_projection(nodes) + self.claim_projection(claim).unsqueeze(1)
        scores = self.score(torch.tanh(context)).squeeze(-1).float()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(nodes.float() * weights.unsqueeze(-1), dim=1).to(nodes.dtype)

        if return_attention:
            return pooled, weights
        return pooled


class MultimodalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.text_pool = ConditionalAttentionPool(hidden_dim)
        self.visual_pool = ConditionalAttentionPool(hidden_dim)
        self.consistency_pool = ConditionalAttentionPool(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.text_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.visual_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

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
        claim = text_nodes[:, 0]
        evidence_mask = text_mask.bool().clone()
        evidence_mask[:, 0] = False
        evidence_embedding, text_attention = self.text_pool(
            text_nodes, claim, evidence_mask, return_attention=True
        )
        text_embedding = self.text_norm(claim + evidence_embedding)
        visual_embedding, visual_attention = self.visual_pool(
            visual_nodes, claim, visual_mask, return_attention=True
        )
        text_consistency_embedding, text_consistency_attention = self.consistency_pool(
            text_consistency_nodes,
            claim,
            evidence_mask,
            return_attention=True,
        )
        visual_consistency_embedding, visual_consistency_attention = (
            self.consistency_pool(
                visual_consistency_nodes,
                claim,
                visual_mask,
                return_attention=True,
            )
        )
        consistency_embedding = (
            text_consistency_embedding + visual_consistency_embedding
        ) * 0.5

        text_gate = torch.sigmoid(self.text_gate(torch.cat((claim, text_embedding), dim=-1)))
        visual_gate = torch.sigmoid(self.visual_gate(torch.cat((claim, visual_embedding), dim=-1)))
        fused = self.dropout(
            torch.cat(
                (
                    text_gate * text_embedding,
                    visual_gate * visual_embedding,
                    consistency_embedding,
                ),
                dim=-1,
            )
        )

        if not return_details:
            return fused
        return {
            "fused": fused,
            "claim_embedding": claim,
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
