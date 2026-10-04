"""Partially fine-tuned text encoding for claim and evidence-pair nodes."""

from contextlib import nullcontext

import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(
                config.text_model,
                revision=getattr(config, "text_model_revision", None),
            )
        )
        self.projection = nn.Linear(
            self.encoder.config.hidden_size,
            config.hidden_dim,
        )
        self.encoder.to(dtype=self.projection.weight.dtype)
        self.encoder.requires_grad_(False)
        count = getattr(config, "text_finetune_layers", 2)
        if not isinstance(count, int) or count < 0:
            raise ValueError("text_finetune_layers must be a non-negative integer")
        layers = self.encoder.encoder.layer if count else ()
        if count > len(layers):
            raise ValueError("text_finetune_layers exceeds the backbone layer count")
        self.finetuned_layers = tuple(layers[-count:]) if count else ()
        for layer in self.finetuned_layers:
            layer.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        for layer in self.finetuned_layers:
            layer.train(mode)
        return self

    def _encode(self, input_ids, attention_mask, token_type_ids=None):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        context = nullcontext() if self.finetuned_layers else torch.no_grad()
        with context:
            outputs = self.encoder(**model_inputs)
        return self.projection(outputs.last_hidden_state[:, 0])

    def forward(self, input_ids, attention_mask, node_mask=None, token_type_ids=None):
        if input_ids.ndim == 2:
            if not attention_mask.bool().any(dim=-1).all():
                raise ValueError("Every encoded text must contain valid tokens")
            return self._encode(input_ids, attention_mask, token_type_ids)

        batch_size, num_nodes, sequence_length = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_nodes, sequence_length)
        flat_attention = attention_mask.reshape(batch_size * num_nodes, sequence_length)

        if node_mask is None:
            node_mask = attention_mask.any(dim=-1)

        valid = node_mask.reshape(-1).bool()
        if not node_mask.bool().any(dim=-1).all():
            raise ValueError("Every sample must contain valid text nodes")
        if not flat_attention[valid].bool().any(dim=-1).all():
            raise ValueError("Every valid text node must contain valid tokens")
        flat_types = None if token_type_ids is None else token_type_ids.reshape(
            batch_size * num_nodes, sequence_length
        )[valid]
        encoded = self._encode(flat_ids[valid], flat_attention[valid], flat_types)
        features = encoded.new_zeros(batch_size * num_nodes, self.projection.out_features)
        features = features.index_copy(0, valid.nonzero(as_tuple=False).flatten(), encoded)
        return features.reshape(batch_size, num_nodes, -1)
