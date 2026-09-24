"""Small shared training and serialization helpers."""

from contextlib import nullcontext
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Class-weighted multiclass focal loss."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        log_probabilities = F.log_softmax(logits, dim=-1)
        target_log_probabilities = log_probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        focal_factor = (1.0 - target_log_probabilities.exp()).pow(self.gamma)
        losses = -focal_factor * target_log_probabilities

        if self.weight is None:
            return losses.mean()
        sample_weights = self.weight[targets]
        return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)


def build_classification_loss(config, device):
    return FocalLoss(
        gamma=getattr(config, "focal_gamma", 2.0),
        weight=getattr(config, "class_weights", None),
    ).to(device)


def bf16_autocast(device):
    if torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def contrastive_alignment_loss(text, image, temperature=0.07, valid_mask=None):
    """One-way in-batch text-to-image contrastive loss."""

    if text.ndim != 2 or image.shape != text.shape:
        raise ValueError(
            "text and image embeddings must share shape [batch, hidden_dim]"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if valid_mask is not None:
        if valid_mask.shape != (text.size(0),):
            raise ValueError("valid_mask must have shape [batch]")
        valid_mask = valid_mask.to(device=text.device, dtype=torch.bool)
        text = text[valid_mask]
        image = image[valid_mask]
    if text.size(0) < 2:
        return (text.sum() + image.sum()) * 0.0

    text = F.normalize(text, dim=-1)
    image = F.normalize(image, dim=-1)
    similarity = text @ image.transpose(0, 1) / temperature
    targets = torch.arange(similarity.size(0), device=similarity.device)
    return F.cross_entropy(similarity, targets)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
