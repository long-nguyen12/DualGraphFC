"""Patch-level graph reasoning for one or more evidence images."""

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
    """Encode image patches and reason within each image independently."""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.patch_size = config.vision_patch_size
        self.k = config.vision_knn
        num_layers = config.vision_gnn_layers
        dropout = getattr(config, "dropout", 0.1)

        if self.patch_size < 1:
            raise ValueError("vision_patch_size must be positive")
        if self.k < 0:
            raise ValueError("vision_knn must be non-negative")
        if num_layers < 1:
            raise ValueError("vision_gnn_layers must be at least 1")

        self.patch_embedding = nn.Conv2d(
            3,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList(
            VisionGraphBlock(self.hidden_dim, dropout) for _ in range(num_layers)
        )
        self.no_image_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        nn.init.normal_(self.no_image_token, std=0.02)

    def build_edges(self, features):
        """Build a safe bidirectional cosine-kNN patch graph.

        Args:
            features: Patch features with shape ``[num_patches, hidden_dim]``.
        Returns:
            A long tensor with shape ``[2, num_edges]``. For a one-patch
            image, it is an empty edge set; GraphSAGE still applies its root
            transformation.
        """

        if features.ndim != 2:
            raise ValueError("features must have shape [visual_nodes, hidden_dim]")
        num_nodes = features.size(0)
        if num_nodes < 1:
            raise ValueError("a visual graph must contain at least one patch")

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
        images,
        image_mask=None,
        return_mask=False,
        return_details=False,
    ):
        """Return padded visual nodes for a batch of image sets.

        ``images`` normally has shape ``[B, M, 3, H, W]``; ``[B, 3, H, W]``
        is accepted as a one-image shorthand. Valid images are patch-embedded
        together, but each fixed kNN graph is built and processed separately
        through the shared graph blocks. A sample with no valid image receives
        one valid learned no-image token.

        Return conventions are deliberately simple: nodes only by default;
        ``(nodes, visual_node_mask)`` with ``return_mask``; ``(nodes, details)``
        with ``return_details``; or all three when both flags are true.
        """

        if images.ndim == 4:
            images = images.unsqueeze(1)
            if image_mask is not None and image_mask.ndim == 1:
                image_mask = image_mask.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, images, 3, height, width]")
        if images.size(2) != 3:
            raise ValueError("images must have three RGB channels")
        if not torch.is_floating_point(images):
            raise TypeError("images must be floating-point tensors")

        batch_size, max_images, _, height, width = images.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("image height and width must be at least vision_patch_size")

        if image_mask is None:
            image_mask = torch.ones(
                (batch_size, max_images), dtype=torch.bool, device=images.device
            )
        elif image_mask.shape != (batch_size, max_images):
            raise ValueError("image_mask must have shape [batch, images]")
        else:
            image_mask = image_mask.to(device=images.device, dtype=torch.bool)

        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patches_per_image = patch_height * patch_width
        no_image = ~image_mask.any(dim=1)

        # A collator normally keeps M >= 1. Supporting M == 0 here makes the
        # missing-image contract explicit and avoids applying Conv2d to empties.
        if max_images == 0:
            nodes = self.no_image_token.expand(batch_size, 1, -1)
            visual_node_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=images.device
            )
            details = {
                "patch_grid": (patch_height, patch_width),
                "patches_per_image": patches_per_image,
                "image_mask": image_mask,
                "no_image": no_image,
                "edge_indices": [[] for _ in range(batch_size)],
            }
            return self._format_output(
                nodes, visual_node_mask, details, return_mask, return_details
            )

        flat_images = images.reshape(
            batch_size * max_images, 3, height, width
        )
        flat_image_mask = image_mask.reshape(-1)
        valid_indices = flat_image_mask.nonzero(as_tuple=False).flatten()
        edge_details = [
            [None for _ in range(max_images)] for _ in range(batch_size)
        ]

        if valid_indices.numel():
            valid_patches = self.patch_embedding(flat_images[valid_indices])
            valid_patches = valid_patches.flatten(2).transpose(1, 2).contiguous()
            encoded_images = []
            for valid_index, patch_nodes in zip(valid_indices, valid_patches):
                edge_index = self.build_edges(patch_nodes)
                for block in self.blocks:
                    patch_nodes = block(patch_nodes, edge_index)
                encoded_images.append(patch_nodes)

                if return_details:
                    flat_index = int(valid_index.item())
                    batch_index, image_index = divmod(flat_index, max_images)
                    edge_details[batch_index][image_index] = edge_index

            encoded_images = torch.stack(encoded_images)
            flat_nodes = encoded_images.new_zeros(
                batch_size * max_images,
                patches_per_image,
                self.hidden_dim,
            )
            flat_nodes = flat_nodes.index_copy(0, valid_indices, encoded_images)
        else:
            flat_nodes = images.new_zeros(
                batch_size * max_images,
                patches_per_image,
                self.hidden_dim,
            )

        nodes = flat_nodes.reshape(
            batch_size, max_images * patches_per_image, self.hidden_dim
        )
        visual_node_mask = image_mask.unsqueeze(-1).expand(
            -1, -1, patches_per_image
        ).reshape(batch_size, -1)

        # Keep exactly one valid key/value for samples with no image. This
        # prevents all-masked rows (and NaNs) in downstream cross-attention.
        token_positions = torch.zeros_like(visual_node_mask)
        token_positions[:, 0] = no_image
        nodes = nodes + token_positions.unsqueeze(-1).to(nodes.dtype) * self.no_image_token
        visual_node_mask = visual_node_mask | token_positions

        details = {
            "patch_grid": (patch_height, patch_width),
            "patches_per_image": patches_per_image,
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
