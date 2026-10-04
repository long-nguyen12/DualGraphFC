"""Online CLIP patch encoding with selected vision blocks fine-tuned."""

from contextlib import nullcontext

import torch
from torch import nn
from transformers import CLIPVisionModel


class CLIPVisionEncoder(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else CLIPVisionModel.from_pretrained(
            config.vision_model,
            revision=getattr(config, "vision_model_revision", None),
        )
        self.encoder.float()
        vision_config = getattr(self.encoder.config, "vision_config", self.encoder.config)
        self.image_size = vision_config.image_size
        if config.image_size != self.image_size:
            raise ValueError("image_size must match the CLIP vision backbone")
        grid_size = self.image_size // vision_config.patch_size
        self.feature_shape = (vision_config.hidden_size, grid_size, grid_size)
        self.encoder.requires_grad_(False)
        count = getattr(config, "vision_finetune_layers", 2)
        if not isinstance(count, int) or count < 0:
            raise ValueError("vision_finetune_layers must be a non-negative integer")
        vision_transformer = getattr(self.encoder, "vision_model", self.encoder)
        layers = vision_transformer.encoder.layers if count else ()
        if count > len(layers):
            raise ValueError("vision_finetune_layers exceeds the backbone layer count")
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

    def forward(self, images, image_mask):
        if images.ndim != 5 or tuple(images.shape[2:]) != (3, self.image_size, self.image_size):
            raise ValueError("images must have shape [batch, images, 3, image_size, image_size]")
        batch_size, max_images = images.shape[:2]
        image_mask = image_mask.to(device=images.device, dtype=torch.bool)
        if tuple(image_mask.shape) != (batch_size, max_images):
            raise ValueError("image_mask must match the batch and image dimensions")
        if max_images == 0 or not image_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid image")
        valid_indices = image_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        valid_images = images.reshape(-1, *images.shape[2:])[valid_indices]
        context = nullcontext() if self.finetuned_layers else torch.no_grad()
        with context:
            output = self.encoder(pixel_values=valid_images)
        patch_tokens = output.last_hidden_state[:, 1:]
        channels, height, width = self.feature_shape
        if tuple(patch_tokens.shape[1:]) != (height * width, channels):
            raise ValueError("CLIP produced an unexpected patch-token shape")
        valid_maps = patch_tokens.transpose(1, 2).reshape(-1, channels, height, width)
        feature_maps = valid_maps.new_zeros(batch_size * max_images, channels, height, width)
        feature_maps = feature_maps.index_copy(0, valid_indices, valid_maps)
        return feature_maps.reshape(batch_size, max_images, channels, height, width)
