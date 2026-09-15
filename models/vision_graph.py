"""Spatial feature graph reasoning for one or more evidence images."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from transformers import PoolFormerModel


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
    """Encode images with PoolFormer and reason over spatial feature nodes."""

    def __init__(self, config, encoder=None):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.k = config.vision_knn
        num_layers = config.vision_gnn_layers
        dropout = getattr(config, "dropout", 0.1)

        self.encoder = (
            encoder
            if encoder is not None
            else PoolFormerModel.from_pretrained(config.vision_model)
        )
        self.image_size = config.image_size
        self.feature_grid = self._infer_feature_grid(
            self.image_size,
            self.encoder.config,
        )
        self.projection = nn.Linear(
            self.encoder.config.hidden_sizes[-1],
            self.hidden_dim,
        )
        self.encoder.to(dtype=self.projection.weight.dtype)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        if self.k < 0:
            raise ValueError("vision_knn must be non-negative")
        if num_layers < 1:
            raise ValueError("vision_gnn_layers must be at least 1")
        self.blocks = nn.ModuleList(
            VisionGraphBlock(self.hidden_dim, dropout) for _ in range(num_layers)
        )
        self.no_image_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        nn.init.normal_(self.no_image_token, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def _infer_feature_grid(image_size, encoder_config):
        """Return PoolFormer's final spatial grid for a square input."""

        if image_size < 1:
            raise ValueError("image_size must be positive")

        patch_sizes = encoder_config.patch_sizes
        strides = encoder_config.strides
        paddings = encoder_config.padding
        if not (len(patch_sizes) == len(strides) == len(paddings)):
            raise ValueError("PoolFormer stage configuration lengths must match")

        height = width = image_size
        for patch_size, stride, padding in zip(
            patch_sizes,
            strides,
            paddings,
        ):
            patch_height, patch_width = VisionGraph._pair(patch_size)
            stride_height, stride_width = VisionGraph._pair(stride)
            padding_height, padding_width = VisionGraph._pair(padding)
            if min(
                patch_height,
                patch_width,
                stride_height,
                stride_width,
            ) < 1:
                raise ValueError("PoolFormer patch sizes and strides must be positive")
            height = (height + 2 * padding_height - patch_height) // stride_height + 1
            width = (width + 2 * padding_width - patch_width) // stride_width + 1

        if height < 1 or width < 1:
            raise ValueError("image_size is too small for the PoolFormer stages")
        return height, width

    @staticmethod
    def _pair(value):
        if isinstance(value, int):
            return value, value
        if len(value) != 2:
            raise ValueError("PoolFormer spatial parameters must be scalars or pairs")
        return value[0], value[1]

    def build_edges(self, features):
        """Build a safe bidirectional cosine-kNN spatial graph.

        Args:
            features: Spatial features with shape ``[num_nodes, hidden_dim]``.
        Returns:
            A long tensor with shape ``[2, num_edges]``. For a one-node
            image, it is an empty edge set; GraphSAGE still applies its root
            transformation.
        """

        if features.ndim != 2:
            raise ValueError("features must have shape [visual_nodes, hidden_dim]")
        num_nodes = features.size(0)
        if num_nodes < 1:
            raise ValueError("a visual graph must contain at least one node")

        neighbors_per_node = min(self.k, num_nodes - 1)
        if neighbors_per_node == 0:
            return torch.empty((2, 0), dtype=torch.long, device=features.device)

        normalized = F.normalize(features, p=2, dim=-1, eps=1e-12)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(float("-inf"))
        neighbors = similarity.topk(neighbors_per_node, dim=-1).indices
        source_nodes = torch.arange(num_nodes, device=features.device)
        source_nodes = source_nodes.unsqueeze(1).expand_as(neighbors)

        sources = torch.cat((source_nodes.reshape(-1), neighbors.reshape(-1)))
        targets = torch.cat((neighbors.reshape(-1), source_nodes.reshape(-1)))
        edge_index = torch.stack((sources, targets))
        return torch.unique(edge_index.transpose(0, 1), dim=0).transpose(0, 1).contiguous()

    def forward(
        self,
        images=None,
        image_mask=None,
        return_mask=False,
        return_details=False,
        feature_maps=None,
    ):
        """Return padded visual nodes for a batch of image sets.

        ``images`` normally has shape ``[B, M, 3, H, W]``; ``[B, 3, H, W]``
        is accepted as a one-image shorthand. Alternatively, ``feature_maps``
        accepts cached PoolFormer outputs with shape ``[B, M, C, Hf, Wf]``.
        Each fixed kNN graph is built and processed separately through the
        shared graph blocks. A sample with no valid image receives one valid
        learned no-image token.

        Return conventions are deliberately simple: nodes only by default;
        ``(nodes, visual_node_mask)`` with ``return_mask``; ``(nodes, details)``
        with ``return_details``; or all three when both flags are true.
        """

        if (images is None) == (feature_maps is None):
            raise ValueError("Pass exactly one of images or feature_maps")

        values = feature_maps if feature_maps is not None else images
        if values.ndim == 4:
            values = values.unsqueeze(1)
            if image_mask is not None and image_mask.ndim == 1:
                image_mask = image_mask.unsqueeze(1)
        if values.ndim != 5:
            raise ValueError("Vision inputs must have shape [batch, images, channels, H, W]")
        if not torch.is_floating_point(values):
            raise TypeError("Vision inputs must be floating-point tensors")

        batch_size, max_images, channels, height, width = values.shape
        if feature_maps is None:
            if channels != 3:
                raise ValueError("images must have three RGB channels")
            if (height, width) != (self.image_size, self.image_size):
                raise ValueError(
                    "images must match the PoolFormer input size "
                    f"{self.image_size}x{self.image_size}"
                )
        elif (channels, height, width) != (
            self.projection.in_features,
            *self.feature_grid,
        ):
            raise ValueError(
                "feature_maps must match PoolFormer's final feature shape "
                f"{(self.projection.in_features, *self.feature_grid)}"
            )

        if image_mask is None:
            image_mask = torch.ones(
                (batch_size, max_images), dtype=torch.bool, device=values.device
            )
        elif image_mask.shape != (batch_size, max_images):
            raise ValueError("image_mask must have shape [batch, images]")
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
                (batch_size, 1), dtype=torch.bool, device=images.device
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
            if feature_maps is None:
                with torch.no_grad():
                    encoded = self.encoder(pixel_values=flat_values[valid_indices])
                valid_feature_maps = encoded.last_hidden_state
            else:
                valid_feature_maps = flat_values[valid_indices]
            if valid_feature_maps.ndim != 4:
                raise ValueError(
                    "PoolFormer must return [images, channels, height, width]"
                )
            if valid_feature_maps.size(1) != self.projection.in_features:
                raise ValueError("PoolFormer returned an unexpected channel width")
            if tuple(valid_feature_maps.shape[-2:]) != self.feature_grid:
                raise ValueError("PoolFormer returned an unexpected spatial grid")
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
            # now describe PoolFormer's final spatial feature grid.
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
