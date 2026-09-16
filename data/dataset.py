"""PyTorch input pipeline built on top of :mod:`dataset_mocheg`."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from data.dataset_mocheg import MochegDataset as MochegLoader

LABELS = {
    "supported": 0,
    "refuted": 1,
    "not enough information": 2,
}
ID_TO_LABEL = {label_id: label for label, label_id in LABELS.items()}
FEATURE_CACHE_VERSION = 1
FEATURE_CACHE_METADATA = "metadata.json"


def feature_cache_path(cache_dir, split, claim_id):
    """Return the cached vision-feature path for one claim."""

    claim_id = str(claim_id)
    if Path(claim_id).name != claim_id:
        raise ValueError(f"Invalid claim ID for feature cache: {claim_id!r}")
    return Path(cache_dir) / split / f"{claim_id}.pt"


def load_feature_cache_metadata(cache_dir, vision_model, image_size):
    """Load and validate the metadata shared by cached feature files."""

    path = Path(cache_dir) / FEATURE_CACHE_METADATA
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature-cache metadata was not found at {path}. "
            "Run precompute_vision_features.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    expected = {
        "format_version": FEATURE_CACHE_VERSION,
        "vision_model": vision_model,
        "image_size": image_size,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"Feature cache {name} is {metadata.get(name)!r}; expected {value!r}"
            )

    feature_shape = metadata.get("feature_shape")
    if (
        not isinstance(feature_shape, list)
        or len(feature_shape) != 3
        or any(not isinstance(value, int) or value < 1 for value in feature_shape)
    ):
        raise ValueError("Feature-cache metadata has an invalid feature_shape")
    return tuple(feature_shape)


def _label_to_id(label):
    normalized = str(label).strip().lower()
    if normalized == "nei":
        normalized = "not enough information"
    try:
        return LABELS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown MOCHEG label: {label!r}") from exc


class _ImageTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"][0]


def build_image_transform(image_size=224, vision_model=None):
    """Return preprocessing for a pretrained vision backbone."""

    if vision_model is not None:
        processor = AutoImageProcessor.from_pretrained(vision_model)
        return _ImageTransform(processor)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class MochegDataset(Dataset):
    """Adapt claim dictionaries from ``dataset_mocheg.py`` for PyTorch.

    Each item contains raw text nodes, a list of successfully decoded image
    tensors, an integer target, and metadata. Token padding and image padding
    are deliberately deferred to :class:`MochegCollator`.
    """

    def __init__(
        self,
        root,
        split,
        image_size=224,
        vision_model=None,
        image_transform=None,
        feature_cache_dir=None,
        limit=None,
    ):
        self.split = split
        self.image_size = image_size
        self.feature_cache_dir = (
            Path(feature_cache_dir) if feature_cache_dir is not None else None
        )
        if self.feature_cache_dir is None:
            self.feature_shape = None
            self.image_transform = image_transform or build_image_transform(
                image_size,
                vision_model,
            )
        else:
            self.feature_shape = load_feature_cache_metadata(
                self.feature_cache_dir,
                vision_model,
                image_size,
            )
            self.image_transform = None

        self.samples = MochegLoader(root).load_split(split, limit=limit)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source = self.samples[index]
        claim = source["claim"]
        evidence = list(source["text_evidence"])
        evidence_ids = list(source["text_evidence_ids"])

        if self.feature_cache_dir is None:
            images, image_paths, image_ids, skipped_paths = self._load_images(source)
            image_features = None
        else:
            image_features, image_paths, image_ids, skipped_paths = (
                self._load_cached_features(source)
            )
            images = None
        label = _label_to_id(source["cleaned_truthfulness"])

        metadata = {
            "claim_id": source["claim_id"],
            "split": source["split"],
            "cleaned_truthfulness": source["cleaned_truthfulness"],
            "text_evidence_ids": evidence_ids,
            "image_evidence_ids": image_ids,
            "image_paths": image_paths,
            "skipped_image_paths": skipped_paths,
            "ruling_outline": source["ruling_outline"],
            "origin": source["origin"],
            "snopes_url": source["snopes_url"],
        }

        item = {
            "claim": claim,
            "evidence": evidence,
            "label": label,
            "metadata": metadata,
        }
        if image_features is None:
            item["images"] = images
        else:
            item["image_features"] = image_features
        return item

    def _load_images(self, source):
        images = []
        loaded_paths = []
        loaded_ids = []
        skipped_paths = []
        evidence_ids = source["image_evidence_ids"]

        for index, raw_path in enumerate(source["images"]):
            path = Path(raw_path)
            try:
                with Image.open(path) as image:
                    image_tensor = self.image_transform(image.convert("RGB"))
            except (FileNotFoundError, OSError):
                skipped_paths.append(str(path))
                continue

            if not isinstance(image_tensor, torch.Tensor):
                raise TypeError("image_transform must return a torch.Tensor")
            expected_shape = (3, self.image_size, self.image_size)
            if tuple(image_tensor.shape) != expected_shape:
                raise ValueError(
                    "image_transform returned shape "
                    f"{tuple(image_tensor.shape)}; expected {expected_shape}"
                )

            images.append(image_tensor.to(dtype=torch.float32))
            loaded_paths.append(str(path))
            loaded_ids.append(evidence_ids[index])

        return images, loaded_paths, loaded_ids, skipped_paths

    def _load_cached_features(self, source):
        path = feature_cache_path(
            self.feature_cache_dir,
            self.split,
            source["claim_id"],
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Cached features were not found for claim {source['claim_id']!r}: "
                f"{path}. Re-run precompute_vision_features.py for {self.split!r}."
            )

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "features" not in payload:
            raise ValueError(f"Invalid feature-cache payload: {path}")

        source_ids = list(source["image_evidence_ids"])
        if payload.get("source_image_evidence_ids") != source_ids:
            raise ValueError(
                f"Cached image IDs do not match the dataset for claim "
                f"{source['claim_id']!r}; re-run precomputation"
            )

        features = payload["features"]
        valid_indices = payload.get("valid_indices")
        if not torch.is_tensor(features) or not torch.is_floating_point(features):
            raise ValueError(f"Cached features must be a floating-point tensor: {path}")
        if features.ndim != 4 or tuple(features.shape[1:]) != self.feature_shape:
            raise ValueError(
                f"Cached features have shape {tuple(features.shape)}; "
                f"expected [images, {', '.join(map(str, self.feature_shape))}]"
            )
        if (
            not torch.is_tensor(valid_indices)
            or valid_indices.ndim != 1
            or valid_indices.dtype != torch.long
        ):
            raise ValueError(
                f"Cached valid_indices must be a one-dimensional long tensor: {path}"
            )
        indices = valid_indices.tolist()
        if len(indices) != features.size(0) or any(
            index < 0 or index >= len(source_ids) for index in indices
        ) or indices != sorted(set(indices)):
            raise ValueError(f"Cached valid_indices are invalid: {path}")

        valid_index_set = set(indices)
        image_paths = [source["images"][index] for index in indices]
        image_ids = [source_ids[index] for index in indices]
        skipped_paths = [
            raw_path
            for index, raw_path in enumerate(source["images"])
            if index not in valid_index_set
        ]
        return features, image_paths, image_ids, skipped_paths


class MochegCollator:
    """Tokenize text nodes and pad a list of MOCHEG samples into one batch."""

    def __init__(
        self,
        tokenizer,
        max_text_length=256,
        image_size=224,
        feature_shape=None,
    ):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.feature_shape = feature_shape

    def __call__(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        batch_size = len(samples)
        text_rows = [
            [sample["claim"], *sample["evidence"]]
            for sample in samples
        ]
        num_text_nodes = max(len(row) for row in text_rows)

        text_node_mask = torch.zeros((batch_size, num_text_nodes), dtype=torch.bool)
        flat_texts = []
        for batch_index, row in enumerate(text_rows):
            text_node_mask[batch_index, : len(row)] = True
            flat_texts.extend(row + [""] * (num_text_nodes - len(row)))

        encoded = self.tokenizer(
            flat_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long).reshape(
            batch_size, num_text_nodes, -1
        )
        attention_mask = torch.as_tensor(
            encoded["attention_mask"], dtype=torch.long
        ).reshape(batch_size, num_text_nodes, -1)

        item_key = "image_features" if self.feature_shape is not None else "images"
        if any(item_key not in sample for sample in samples):
            raise ValueError(f"Every sample must contain {item_key!r}")

        max_images = max(len(sample[item_key]) for sample in samples)
        image_mask = torch.zeros((batch_size, max_images), dtype=torch.bool)
        if self.feature_shape is None:
            image_values = torch.zeros(
                (
                    batch_size,
                    max_images,
                    3,
                    self.image_size,
                    self.image_size,
                ),
                dtype=torch.float32,
            )
            for batch_index, sample in enumerate(samples):
                sample_images = sample["images"]
                if sample_images:
                    image_values[batch_index, : len(sample_images)] = torch.stack(
                        sample_images
                    )
                    image_mask[batch_index, : len(sample_images)] = True
        else:
            dtype = samples[0]["image_features"].dtype
            image_values = torch.zeros(
                (batch_size, max_images, *self.feature_shape),
                dtype=dtype,
            )
            for batch_index, sample in enumerate(samples):
                sample_features = sample["image_features"]
                if tuple(sample_features.shape[1:]) != tuple(self.feature_shape):
                    raise ValueError("A sample has an unexpected cached feature shape")
                if sample_features.dtype != dtype:
                    raise ValueError("Cached feature dtypes must match within a batch")
                if len(sample_features):
                    image_values[batch_index, : len(sample_features)] = sample_features
                    image_mask[batch_index, : len(sample_features)] = True

        batch = {
            # Text tensors: [B, Nt, L]; node mask: [B, Nt].
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_node_mask": text_node_mask,
            "image_mask": image_mask,
            "has_image": image_mask.any(dim=1),
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }
        batch[item_key] = image_values
        return batch
