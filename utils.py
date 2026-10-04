"""Small shared training and serialization helpers."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Class-weighted multiclass focal loss with optional label smoothing."""

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probabilities = F.log_softmax(logits, dim=-1)

        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing / (num_classes - 1)
            one_hot = logits.new_full(logits.shape, smooth)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            target_log_probabilities = (one_hot * log_probabilities).sum(dim=-1)
            # focal factor based on true class probability
            true_log_prob = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_factor = (1.0 - true_log_prob.exp()).pow(self.gamma)
            losses = -focal_factor * target_log_probabilities
        else:
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
        label_smoothing=getattr(config, "label_smoothing", 0.0),
    ).to(device)


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
    if valid_mask is not None:
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
