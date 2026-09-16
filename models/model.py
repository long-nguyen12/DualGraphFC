"""End-to-end DualGraphFC model."""

from torch import nn

from models.cross_graph import CrossGraphReasoner
from models.fusion import MultimodalFusion
from models.text_encoder import LongTextEncoder
from models.text_graph import TextGraph
from models.vision_graph import VisionGraph


class DualGraphFC(nn.Module):
    """Dual text/vision graph fact checker."""

    def __init__(self, config, text_backbone=None, vision_backbone=None):
        super().__init__()
        self.text_encoder = LongTextEncoder(config, encoder=text_backbone)
        self.text_graph = TextGraph(config)
        self.vision_graph = VisionGraph(config, encoder=vision_backbone)
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
        text_mask = batch["text_node_mask"].bool()
        text_features = self.text_encoder(
            batch["input_ids"],
            batch["attention_mask"],
            node_mask=text_mask,
        )

        if return_attention:
            text_nodes, text_attention = self.text_graph(
                text_features, text_mask, return_attention=True
            )
        else:
            text_nodes = self.text_graph(text_features, text_mask)
            text_attention = None

        vision_result = self.vision_graph(
            images=batch.get("images"),
            image_mask=batch["image_mask"],
            return_mask=True,
            return_details=return_attention,
            feature_maps=batch.get("image_features"),
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
            cross_output["consistency_nodes"],
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
        
        details["text_embedding"] = self.fusion.text_pool(text_nodes, text_mask)
        details["visual_embedding"] = self.fusion.visual_pool(
            visual_nodes, visual_mask
        )
        details["text_node_mask"] = text_mask
        details["visual_node_mask"] = visual_mask

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
