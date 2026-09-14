"""Graph reasoning over claim and textual-evidence nodes."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class TextGraph(nn.Module):
    """Apply a GAT independently to each padded text graph in a batch."""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.k = config.text_graph_k
        num_layers = config.text_gnn_layers
        num_heads = config.text_gnn_heads
        dropout = getattr(config, "dropout", 0.1)

        if num_layers < 1:
            raise ValueError("text_gnn_layers must be at least 1")
        if num_heads < 1 or self.hidden_dim % num_heads:
            raise ValueError("text_gnn_heads must evenly divide hidden_dim")
        if self.k < 0:
            raise ValueError("text_graph_k must be non-negative")

        self.layers = nn.ModuleList(
            GATConv(
                self.hidden_dim,
                self.hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def build_edges(self, features):
        """Build one graph's bidirectional claim and cosine-kNN edges.

        ``features`` must contain only real nodes and have shape
        ``[1 + num_evidence, hidden_dim]``. Node zero is the claim. The
        returned ``edge_index`` is safe for a claim-only graph and always has
        shape ``[2, num_edges]``.
        """

        if features.ndim != 2:
            raise ValueError("features must have shape [text_nodes, hidden_dim]")
        if features.size(0) < 1:
            raise ValueError("a text graph must contain its claim node")

        device = features.device
        num_evidence = features.size(0) - 1
        if num_evidence == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        evidence_nodes = torch.arange(1, num_evidence + 1, device=device)
        claim_nodes = torch.zeros_like(evidence_nodes)
        sources = [claim_nodes, evidence_nodes]
        targets = [evidence_nodes, claim_nodes]

        neighbors_per_node = min(self.k, num_evidence - 1)
        if neighbors_per_node:
            evidence = F.normalize(features[1:], p=2, dim=-1, eps=1e-12)
            similarity = evidence @ evidence.transpose(0, 1)
            similarity.fill_diagonal_(float("-inf"))
            neighbors = similarity.topk(neighbors_per_node, dim=-1).indices + 1
            source_nodes = evidence_nodes.unsqueeze(1).expand_as(neighbors)

            # Make the semantic graph explicitly bidirectional. A directed
            # top-k selection alone need not choose the reverse relationship.
            sources.extend((source_nodes.reshape(-1), neighbors.reshape(-1)))
            targets.extend((neighbors.reshape(-1), source_nodes.reshape(-1)))

        edge_index = torch.stack((torch.cat(sources), torch.cat(targets)))
        return torch.unique(edge_index.transpose(0, 1), dim=0).transpose(0, 1).contiguous()

    def forward(self, features, node_mask=None, return_attention=False):
        """Return updated nodes with the same padded shape as ``features``.

        Args:
            features: ``[batch, text_nodes, hidden_dim]`` or an unbatched
                ``[text_nodes, hidden_dim]`` tensor.
            node_mask: Boolean ``[batch, text_nodes]`` mask. Node zero must be
                valid because it represents the claim.
            return_attention: Also return per-layer GAT edges and per-head
                attention weights for each sample.
        """

        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
            if node_mask is not None and node_mask.ndim == 1:
                node_mask = node_mask.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, text_nodes, hidden_dim]")
        if features.size(-1) != self.hidden_dim:
            raise ValueError("the text feature width must equal config.hidden_dim")

        batch_size, num_nodes, _ = features.shape
        if num_nodes < 1:
            raise ValueError("text graphs must contain a claim node")
        if node_mask is None:
            node_mask = torch.ones(
                (batch_size, num_nodes), dtype=torch.bool, device=features.device
            )
        elif node_mask.shape != (batch_size, num_nodes):
            raise ValueError("node_mask must have shape [batch, text_nodes]")
        else:
            node_mask = node_mask.to(device=features.device, dtype=torch.bool)
        if batch_size and not node_mask[:, 0].all():
            raise ValueError("node zero (the claim) must be valid in every text graph")

        output = features.new_zeros(features.shape)
        batch_details = []
        for batch_index in range(batch_size):
            valid = node_mask[batch_index]
            nodes = features[batch_index, valid]
            edge_index = self.build_edges(nodes)
            layer_details = []

            for layer in self.layers:
                if return_attention:
                    nodes, (used_edges, attention) = layer(
                        nodes,
                        edge_index,
                        return_attention_weights=True,
                    )
                    layer_details.append(
                        {"edge_index": used_edges, "attention": attention}
                    )
                else:
                    nodes = layer(nodes, edge_index)
                nodes = self.dropout(F.gelu(nodes))

            output[batch_index, valid] = nodes
            if return_attention:
                batch_details.append(layer_details)

        if unbatched:
            output = output.squeeze(0)
            if return_attention:
                return output, batch_details[0]
        if return_attention:
            return output, batch_details
        return output
