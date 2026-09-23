"""Spatial feature graph reasoning for one or more evidence images."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class VisionGraphBlock(nn.Module):
    """A small residual ViG-style block using GraphSAGE."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv = SAGEConv(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes, edge_index):
        residual = nodes
        nodes = self.input_projection(nodes)
        nodes = self.graph_conv(nodes, edge_index)
        nodes = self.output_projection(F.gelu(nodes))
        return self.norm(residual + self.dropout(nodes))


class VisionGraph(nn.Module):
    """Reason spatially over precomputed vision feature maps."""

    def __init__(self, config, feature_shape=None):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        num_layers = config.vision_gnn_layers
        dropout = getattr(config, "dropout", 0.1)

        self.feature_shape = tuple(feature_shape)
        feature_channels, feature_height, feature_width = self.feature_shape
        self.feature_grid = (feature_height, feature_width)

        self.projection = nn.Linear(
            feature_channels,
            self.hidden_dim,
        )
        self.register_buffer(
            "grid_edge_index",
            self._build_grid_edges(*self.feature_grid),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            VisionGraphBlock(self.hidden_dim, dropout) for _ in range(num_layers)
        )
        self.no_image_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        nn.init.normal_(self.no_image_token, std=0.02)

    @staticmethod
    def _build_grid_edges(height, width):
        sources = []
        targets = []
        for row in range(height):
            for column in range(width):
                source = row * width + column
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == 0 and column_offset == 0:
                            continue
                        target_row = row + row_offset
                        target_column = column + column_offset
                        if 0 <= target_row < height and 0 <= target_column < width:
                            sources.append(source)
                            targets.append(target_row * width + target_column)

        if not sources:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor((sources, targets), dtype=torch.long)

    def build_edges(self, features):
        return self.grid_edge_index.to(features.device)

    def forward(
        self,
        feature_maps,
        image_mask=None,
        return_mask=False,
        return_details=False,
    ):
        values = feature_maps
        if values.ndim == 4:
            values = values.unsqueeze(1)
            if image_mask is not None and image_mask.ndim == 1:
                image_mask = image_mask.unsqueeze(1)

        batch_size, max_images, channels, height, width = values.shape

        if image_mask is None:
            image_mask = torch.ones(
                (batch_size, max_images), dtype=torch.bool, device=values.device
            )
        else:
            image_mask = image_mask.to(device=values.device, dtype=torch.bool)

        feature_height, feature_width = self.feature_grid
        features_per_image = feature_height * feature_width
        no_image = ~image_mask.any(dim=1)

        # A collator normally keeps M >= 1. Supporting M == 0 here makes the
        # missing-image contract explicit and avoids encoding empty tensors.
        if max_images == 0:
            nodes = self.no_image_token.expand(batch_size, 1, -1)
            visual_node_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=values.device
            )
            details = {
                "patch_grid": (feature_height, feature_width),
                "patches_per_image": features_per_image,
                "image_mask": image_mask,
                "no_image": no_image,
                "edge_indices": [[] for _ in range(batch_size)],
            }
            return self._format_output(
                nodes, visual_node_mask, details, return_mask, return_details
            )

        flat_values = values.reshape(batch_size * max_images, channels, height, width)
        flat_image_mask = image_mask.reshape(-1)
        valid_indices = flat_image_mask.nonzero(as_tuple=False).flatten()
        edge_details = [
            [None for _ in range(max_images)] for _ in range(batch_size)
        ]

        if valid_indices.numel():
            valid_feature_maps = flat_values[valid_indices]
            valid_feature_maps = valid_feature_maps.to(dtype=self.projection.weight.dtype)
            valid_features = valid_feature_maps.flatten(2).transpose(1, 2).contiguous()
            valid_features = self.projection(valid_features)
            encoded_images = []
            for valid_index, feature_nodes in zip(valid_indices, valid_features):
                edge_index = self.build_edges(feature_nodes)
                for block in self.blocks:
                    feature_nodes = block(feature_nodes, edge_index)
                encoded_images.append(feature_nodes)

                if return_details:
                    flat_index = int(valid_index.item())
                    batch_index, image_index = divmod(flat_index, max_images)
                    edge_details[batch_index][image_index] = edge_index

            encoded_images = torch.stack(encoded_images)
            flat_nodes = encoded_images.new_zeros(
                batch_size * max_images,
                features_per_image,
                self.hidden_dim,
            )
            flat_nodes = flat_nodes.index_copy(0, valid_indices, encoded_images)
        else:
            flat_nodes = self.projection.weight.new_zeros(
                batch_size * max_images,
                features_per_image,
                self.hidden_dim,
            )

        nodes = flat_nodes.reshape(
            batch_size, max_images * features_per_image, self.hidden_dim
        )
        visual_node_mask = image_mask.unsqueeze(-1).expand(
            -1, -1, features_per_image
        ).reshape(batch_size, -1)

        # Keep exactly one valid key/value for samples with no image. This
        # prevents all-masked rows (and NaNs) in downstream cross-attention.
        token_positions = torch.zeros_like(visual_node_mask)
        token_positions[:, 0] = no_image
        nodes = nodes + token_positions.unsqueeze(-1).to(nodes.dtype) * self.no_image_token
        visual_node_mask = visual_node_mask | token_positions

        details = {
            # Keep these public names for inference/checkpoint consumers. They
            # describe the backbone's final spatial feature grid.
            "patch_grid": (feature_height, feature_width),
            "patches_per_image": features_per_image,
            "image_mask": image_mask,
            "no_image": no_image,
            "edge_indices": edge_details,
        }
        return self._format_output(
            nodes, visual_node_mask, details, return_mask, return_details
        )

    @staticmethod
    def _format_output(nodes, node_mask, details, return_mask, return_details):
        if return_mask and return_details:
            return nodes, node_mask, details
        if return_mask:
            return nodes, node_mask
        if return_details:
            return nodes, details
        return nodes
