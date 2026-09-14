"""Transformer encoder for claim and evidence graph nodes."""

from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Encode each text node independently and project it to the graph width."""

    def __init__(self, config, encoder=None):
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(config.text_model)
        )
        self.projection = nn.Linear(
            self.encoder.config.hidden_size,
            config.hidden_dim,
        )
        self.encoder.to(dtype=self.projection.weight.dtype)

    def forward(self, input_ids, attention_mask, node_mask=None):
        """Return ``[batch, text_nodes, hidden_dim]`` node features.

        Two-dimensional inputs are also accepted and return one feature per row.
        With batched graph input, padded text nodes are not sent through the
        transformer.
        """

        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")

        if input_ids.ndim == 2:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return self.projection(outputs.last_hidden_state[:, 0])

        if input_ids.ndim != 3:
            raise ValueError("text inputs must have shape [nodes, tokens] or [batch, nodes, tokens]")

        batch_size, num_nodes, sequence_length = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_nodes, sequence_length)
        flat_attention = attention_mask.reshape(batch_size * num_nodes, sequence_length)

        if node_mask is None:
            node_mask = attention_mask.any(dim=-1)
        if node_mask.shape != (batch_size, num_nodes):
            raise ValueError("node_mask must have shape [batch, text_nodes]")
        valid = node_mask.reshape(-1).bool()

        features = self.projection.weight.new_zeros(
            (batch_size * num_nodes, self.projection.out_features)
        )
        if valid.any():
            outputs = self.encoder(
                input_ids=flat_ids[valid],
                attention_mask=flat_attention[valid],
            )
            features[valid] = self.projection(outputs.last_hidden_state[:, 0])

        return features.reshape(batch_size, num_nodes, -1)
