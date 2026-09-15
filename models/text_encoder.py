"""Transformer encoder for claim and evidence graph nodes."""

import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
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
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask, node_mask=None):
        if input_ids.ndim == 2:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            return self.projection(outputs.last_hidden_state[:, 0])

        batch_size, num_nodes, sequence_length = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_nodes, sequence_length)
        flat_attention = attention_mask.reshape(batch_size * num_nodes, sequence_length)

        if node_mask is None:
            node_mask = attention_mask.any(dim=-1)
        
        valid = node_mask.reshape(-1).bool()

        features = self.projection.weight.new_zeros(
            (batch_size * num_nodes, self.projection.out_features)
        )
        if valid.any():
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=flat_ids[valid],
                    attention_mask=flat_attention[valid],
                )
            features[valid] = self.projection(outputs.last_hidden_state[:, 0])

        return features.reshape(batch_size, num_nodes, -1)
