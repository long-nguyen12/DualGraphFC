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


class LongTextEncoder(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(config.long_text_model)
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

    def _encode(self, input_ids, attention_mask):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if getattr(self.encoder.config, "model_type", None) == "longformer":
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = attention_mask[:, 0]
            model_inputs["global_attention_mask"] = global_attention_mask

        with torch.no_grad():
            outputs = self.encoder(**model_inputs)
        return self.projection(outputs.last_hidden_state[:, 0])

    def forward(self, input_ids, attention_mask, node_mask=None):
        if input_ids.ndim == 2:
            return self._encode(input_ids, attention_mask)
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have two or three dimensions")

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
            features[valid] = self._encode(
                flat_ids[valid],
                flat_attention[valid],
            )
        return features.reshape(batch_size, num_nodes, -1)
