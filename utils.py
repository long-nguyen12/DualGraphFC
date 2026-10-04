"""Small shared training and serialization helpers."""

import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn


def build_classification_loss(config, device):
    return nn.CrossEntropyLoss().to(device)


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def autocast_dtype(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    with torch.cuda.device(device):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device):
    dtype = autocast_dtype(device)
    return nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=autocast_dtype(device) == torch.float16)


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


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
