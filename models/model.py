"""End-to-end DualGraphFC model."""

import torch
from torch import nn

from models.cross_graph import CrossGraphReasoner
from models.fusion import MultimodalFusion
from models.text_encoder import TextEncoder
from models.text_graph import TextGraph
from models.vision_encoder import CLIPVisionEncoder
from models.vision_graph import VisionGraph


class DualGraphFC(nn.Module):
    def __init__(
        self,
        config,
        text_backbone=None,
        vision_backbone=None,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(config, encoder=text_backbone)
        self.text_role_embedding = nn.Embedding(2, config.hidden_dim)
        nn.init.normal_(self.text_role_embedding.weight, std=0.02)
        self.text_graph = TextGraph(config)
        self.vision_encoder = CLIPVisionEncoder(config, encoder=vision_backbone)
        self.vision_graph = VisionGraph(config, feature_shape=self.vision_encoder.feature_shape)
        self.cross_graph = CrossGraphReasoner(config)
        self.fusion = MultimodalFusion(config)

        hidden_dim = config.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.num_classes),
        )

    def forward(self, batch, return_details=False, return_attention=False):
        if "images" not in batch or "image_mask" not in batch:
            raise ValueError("Every sample requires actual images and image_mask")
        text_mask = batch["text_node_mask"].bool()
        if (
            text_mask.ndim != 2
            or text_mask.size(0) == 0
            or text_mask.size(1) < 2
            or not text_mask[:, 0].all()
            or not text_mask[:, 1:].any(dim=1).all()
        ):
            raise ValueError("Every sample requires a claim and at least one text evidence")
        text_features = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"],
            node_mask=text_mask,
            token_type_ids=batch.get("token_type_ids"),
        )
        role_ids = torch.ones(text_mask.shape, dtype=torch.long, device=text_features.device)
        role_ids[:, 0] = 0
        text_features = (text_features + self.text_role_embedding(role_ids)) * (
            text_mask.unsqueeze(-1).to(text_features.dtype)
        )

        if return_attention:
            text_nodes, text_attention = self.text_graph(
                text_features, text_mask, return_attention=True
            )
        else:
            text_nodes = self.text_graph(text_features, text_mask)
            text_attention = None

        feature_maps = self.vision_encoder(batch["images"], batch["image_mask"])
        vision_result = self.vision_graph(
            feature_maps=feature_maps,
            image_mask=batch["image_mask"],
            return_mask=True,
            return_details=return_attention,
        )
        if return_attention:
            visual_nodes, visual_mask, vision_details = vision_result
        else:
            visual_nodes, visual_mask = vision_result
            vision_details = None

        cross_output = self.cross_graph(
            text_nodes,
            visual_nodes,
            text_mask=text_mask,
            visual_mask=visual_mask,
            return_attention=return_attention,
        )
        fusion_output = self.fusion(
            cross_output["text_nodes"],
            cross_output["visual_nodes"],
            cross_output["text_consistency_nodes"],
            cross_output["visual_consistency_nodes"],
            text_mask=text_mask,
            visual_mask=visual_mask,
            return_details=return_details or return_attention,
        )

        if not (return_details or return_attention):
            return self.classifier(fusion_output)

        details = fusion_output
        details["logits"] = self.classifier(details["fused"])
        details["fusion_text_embedding"] = details["text_embedding"]
        details["fusion_visual_embedding"] = details["visual_embedding"]

        details["text_node_mask"] = text_mask
        details["visual_node_mask"] = visual_mask
        details["text_node_encoding"] = "claim_and_evidence_claim_pairs"

        if return_attention:
            details["text_gat_attention"] = text_attention
            details["vision_graph"] = vision_details
            details["text_to_vision_attention"] = cross_output[
                "text_to_vision_attention"
            ]
            details["vision_to_text_attention"] = cross_output[
                "vision_to_text_attention"
            ]
        return details
