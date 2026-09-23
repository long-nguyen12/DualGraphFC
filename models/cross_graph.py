"""Bidirectional attention between text and visual graph nodes."""

import torch
from torch import nn


class CrossGraphReasoner(nn.Module):
    """Align both graphs and form symmetric multimodal consistency nodes."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.text_to_vision = nn.MultiheadAttention(
            hidden_dim,
            config.cross_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.vision_to_text = nn.MultiheadAttention(
            hidden_dim,
            config.cross_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.consistency = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        text_nodes,
        visual_nodes,
        text_mask=None,
        visual_mask=None,
        return_attention=False,
    ):
        if text_mask is None:
            text_mask = torch.ones(
                text_nodes.shape[:2], dtype=torch.bool, device=text_nodes.device
            )
        if visual_mask is None:
            visual_mask = torch.ones(
                visual_nodes.shape[:2], dtype=torch.bool, device=visual_nodes.device
            )
        text_mask = text_mask.to(device=text_nodes.device, dtype=torch.bool)
        visual_mask = visual_mask.to(device=visual_nodes.device, dtype=torch.bool)

        matched_visual, text_to_vision_attention = self.text_to_vision(
            query=text_nodes,
            key=visual_nodes,
            value=visual_nodes,
            key_padding_mask=~visual_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        matched_text, vision_to_text_attention = self.vision_to_text(
            query=visual_nodes,
            key=text_nodes,
            value=text_nodes,
            key_padding_mask=~text_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        text_context = self.text_norm(text_nodes + self.dropout(matched_visual))
        visual_context = self.vision_norm(visual_nodes + self.dropout(matched_text))
        text_consistency_nodes = self.consistency(
            torch.cat(
                (
                    text_nodes,
                    matched_visual,
                    torch.abs(text_nodes - matched_visual),
                    text_nodes * matched_visual,
                ),
                dim=-1,
            )
        )
        visual_consistency_nodes = self.consistency(
            torch.cat(
                (
                    visual_nodes,
                    matched_text,
                    torch.abs(visual_nodes - matched_text),
                    visual_nodes * matched_text,
                ),
                dim=-1,
            )
        )

        text_scale = text_mask.unsqueeze(-1).to(text_nodes.dtype)
        visual_scale = visual_mask.unsqueeze(-1).to(visual_nodes.dtype)
        return {
            "text_nodes": text_context * text_scale,
            "visual_nodes": visual_context * visual_scale,
            "text_consistency_nodes": text_consistency_nodes * text_scale,
            "visual_consistency_nodes": visual_consistency_nodes * visual_scale,
            "text_to_vision_attention": text_to_vision_attention,
            "vision_to_text_attention": vision_to_text_attention,
        }
