"""PyTorch input pipeline built on top of :mod:`dataset_mocheg`."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from dataset_mocheg import MochegDataset as MochegLoader

LABELS = {
    "supported": 0,
    "refuted": 1,
    "not enough information": 2,
}
ID_TO_LABEL = {label_id: label for label, label_id in LABELS.items()}


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
    """Return image preprocessing for PoolFormer or a custom vision backbone."""

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
        limit=None,
    ):
        self.image_size = image_size
        self.image_transform = image_transform or build_image_transform(
            image_size,
            vision_model,
        )

        self.samples = MochegLoader(root).load_split(split, limit=limit)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source = self.samples[index]
        claim = source["claim"]
        evidence = list(source["text_evidence"])
        evidence_ids = list(source["text_evidence_ids"])

        images, image_paths, image_ids, skipped_paths = self._load_images(source)
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

        return {
            "claim": claim,
            "evidence": evidence,
            "images": images,
            "label": label,
            "metadata": metadata,
        }

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


class MochegCollator:
    """Tokenize text nodes and pad a list of MOCHEG samples into one batch."""

    def __init__(
        self,
        tokenizer,
        max_text_length=256,
        image_size=224,
    ):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size

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

        max_images = max(len(sample["images"]) for sample in samples)
        images = torch.zeros(
            (
                batch_size,
                max_images,
                3,
                self.image_size,
                self.image_size,
            ),
            dtype=torch.float32,
        )
        image_mask = torch.zeros((batch_size, max_images), dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            sample_images = sample["images"]
            if sample_images:
                images[batch_index, : len(sample_images)] = torch.stack(sample_images)
                image_mask[batch_index, : len(sample_images)] = True

        batch = {
            # Text tensors: [B, Nt, L]; node mask: [B, Nt].
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_node_mask": text_node_mask,
            # Image tensor: [B, batch_max_images, 3, H, W].
            "images": images,
            "image_mask": image_mask,
            "has_image": image_mask.any(dim=1),
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }
        return batch
