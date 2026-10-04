import random
from collections import Counter
from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.dataset_mocheg import MochegDataset as MochegLoader
from data.text_normalization import normalize_text

LABELS = {"supported": 0, "refuted": 1, "not enough information": 2}
ID_TO_LABEL = {label_id: label for label, label_id in LABELS.items()}
DEFAULT_VISION_MODEL = "openai/clip-vit-base-patch32"


def _label_to_id(label):
    normalized = str(label).strip().lower()
    if normalized == "nei":
        normalized = "not enough information"
    try:
        return LABELS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown MOCHEG label: {label!r}") from exc


def load_rgb_image(path):
    """Fully decode an image and return RGB pixels independent of its file."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def _process_image(image, image_processor, image_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixels = image_processor(images=image, return_tensors="pt")["pixel_values"]
    expected_shape = (1, 3, image_size, image_size)
    if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected_shape:
        raise ValueError(f"Image processor must return pixels shaped {expected_shape}")
    pixels = pixels[0].to(dtype=torch.float32)
    if not torch.isfinite(pixels).all():
        raise ValueError("Image processor returned non-finite pixels")
    return pixels


def build_image_transform(
    image_size=224,
    image_processor=None,
    vision_model=DEFAULT_VISION_MODEL,
    vision_model_revision=None,
):
    if image_processor is None:
        from transformers import CLIPImageProcessor

        image_processor = CLIPImageProcessor.from_pretrained(
            vision_model, revision=vision_model_revision
        )

    return partial(_process_image, image_processor=image_processor, image_size=image_size)


class MochegDataset(Dataset):
    """Read raw images for claims with usable text and image evidence."""

    def __init__(
        self,
        root,
        split,
        image_size=224,
        vision_model=DEFAULT_VISION_MODEL,
        image_processor=None,
        image_transform=None,
        retrieved_text_dir=None,
        limit=None,
        vision_model_revision=None,
    ):
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        self.split = split
        self.image_size = image_size
        self.image_transform = image_transform or build_image_transform(
            image_size,
            image_processor=image_processor,
            vision_model=vision_model,
            vision_model_revision=vision_model_revision,
        )
        raw_samples = MochegLoader(
            root, retrieved_text_dir=retrieved_text_dir
        ).load_split(split)
        complete = []
        excluded = []
        dropped_images = []
        for source in raw_samples:
            reasons = []
            if not normalize_text(source["claim"]):
                reasons.append("empty_claim")
            texts = []
            text_ids = []
            for text, evidence_id in zip(
                source["text_evidence"], source["text_evidence_ids"]
            ):
                if normalize_text(text) and str(evidence_id).strip():
                    texts.append(text)
                    text_ids.append(evidence_id)
            if not texts:
                reasons.append("missing_text_evidence")

            paths = []
            image_ids = []
            skipped_paths = []
            for path, evidence_id in zip(source["images"], source["image_evidence_ids"]):
                try:
                    load_rgb_image(path)
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    skipped_paths.append(path)
                    dropped_images.append(
                        {"claim_id": source["claim_id"], "path": path, "reason": str(exc)}
                    )
                    continue
                paths.append(path)
                image_ids.append(evidence_id)
            if not paths:
                reasons.append("missing_image_evidence")
            if reasons:
                excluded.append({"claim_id": source["claim_id"], "reasons": reasons})
                continue
            sample = dict(source)
            sample.update(
                text_evidence=texts,
                text_evidence_ids=text_ids,
                images=paths,
                image_evidence_ids=image_ids,
                skipped_image_paths=skipped_paths,
            )
            complete.append(sample)

        self.samples = complete[:limit] if limit is not None else complete
        groups = {"raw": raw_samples, "complete": complete, "retained": self.samples}
        self.subset_report = {
            "subset_rule": "complete_text_image_v2",
            "split": split,
            "limit": limit,
            "raw_claim_count": len(raw_samples),
            "complete_claim_count": len(complete),
            "retained_claim_count": len(self.samples),
            "complete_claim_ids": [sample["claim_id"] for sample in complete],
            "retained_claim_ids": [sample["claim_id"] for sample in self.samples],
            "excluded_claim_ids": [sample["claim_id"] for sample in excluded],
            "excluded_claims": excluded,
            "exclusion_counts": dict(
                Counter(reason for sample in excluded for reason in sample["reasons"])
            ),
            "dropped_images": dropped_images,
            "class_counts": {
                name: dict(
                    Counter(ID_TO_LABEL[_label_to_id(s["cleaned_truthfulness"])] for s in rows)
                )
                for name, rows in groups.items()
            },
            "text_evidence_source_counts": {
                name: dict(Counter(s["text_evidence_source"] for s in rows))
                for name, rows in groups.items()
            },
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source = self.samples[index]
        images = []
        for path in source["images"]:
            try:
                image = load_rgb_image(path)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise RuntimeError(
                    f"Approved image for claim {source['claim_id']} cannot be read: {path}"
                ) from exc
            pixels = self.image_transform(image)
            expected_shape = (3, self.image_size, self.image_size)
            if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected_shape:
                raise ValueError(
                    f"Image transform for claim {source['claim_id']} at {path} "
                    f"must return a tensor shaped {expected_shape}"
                )
            if not torch.isfinite(pixels).all():
                raise ValueError(f"Non-finite image pixels for claim {source['claim_id']}: {path}")
            images.append(pixels.to(dtype=torch.float32))

        metadata = {
            "claim_id": source["claim_id"],
            "split": source["split"],
            "cleaned_truthfulness": source["cleaned_truthfulness"],
            "text_evidence_ids": list(source["text_evidence_ids"]),
            "text_evidence_source": source["text_evidence_source"],
            "image_evidence_ids": list(source["image_evidence_ids"]),
            "image_paths": list(source["images"]),
            "skipped_image_paths": list(source["skipped_image_paths"]),
            "ruling_outline": source["ruling_outline"],
            "origin": source["origin"],
            "snopes_url": source["snopes_url"],
        }
        return {
            "claim": source["claim"],
            "evidence": list(source["text_evidence"]),
            "images": images,
            "label": _label_to_id(source["cleaned_truthfulness"]),
            "metadata": metadata,
        }


class MochegCollator:
    """Encode claim-conditioned evidence nodes and pad real inputs only."""

    def __init__(
        self,
        tokenizer,
        max_text_length=512,
        image_size=224,
        augment=False,
        evidence_drop_prob=0.10,
    ):
        if not 0 <= evidence_drop_prob <= 1:
            raise ValueError("evidence_drop_prob must be between 0 and 1")
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.augment = augment
        self.evidence_drop_prob = evidence_drop_prob

    def __call__(self, samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        claims = []
        evidence_rows = []
        metadata = []
        for sample in samples:
            claim = normalize_text(sample["claim"])
            evidence = [normalize_text(text) for text in sample["evidence"]]
            details = dict(sample["metadata"])
            claim_id = details.get("claim_id", "unknown")
            if not claim or not evidence or any(not text for text in evidence):
                raise ValueError(f"Claim {claim_id} requires a nonempty claim and text evidence")
            evidence_ids = list(details.get("text_evidence_ids", []))
            if len(evidence_ids) != len(evidence) or any(not str(x).strip() for x in evidence_ids):
                raise ValueError(f"Claim {claim_id} has mismatched text evidence IDs")
            if not len(sample.get("images", [])):
                raise ValueError(f"Claim {claim_id} requires at least one image")
            kept = list(range(len(evidence)))
            if self.augment:
                kept = [i for i in kept if random.random() >= self.evidence_drop_prob]
                if not kept:
                    kept = [0]
            evidence_rows.append([evidence[i] for i in kept])
            details["text_evidence_ids"] = [evidence_ids[i] for i in kept]
            claims.append(claim)
            metadata.append(details)

        claim_encoding = self.tokenizer(
            claims, padding=False, truncation=False, return_attention_mask=True
        )
        claim_specials = self.tokenizer.num_special_tokens_to_add(pair=False)
        pair_specials = self.tokenizer.num_special_tokens_to_add(pair=True)
        for index, ids in enumerate(claim_encoding["input_ids"]):
            claim_length = len(ids) - claim_specials
            if claim_length + pair_specials >= self.max_text_length:
                raise ValueError(
                    f"Claim {metadata[index].get('claim_id', 'unknown')} exceeds the "
                    "claim/evidence pair token budget; the claim cannot be truncated"
                )
        flat_evidence = [text for row in evidence_rows for text in row]
        pair_claims = [claim for claim, row in zip(claims, evidence_rows) for _ in row]
        pair_encoding = self.tokenizer(
            flat_evidence,
            text_pair=pair_claims,
            padding=False,
            truncation="only_first",
            max_length=self.max_text_length,
            return_attention_mask=True,
        )
        fields = ["input_ids", "attention_mask"]
        if "token_type_ids" in claim_encoding or "token_type_ids" in pair_encoding:
            fields.append("token_type_ids")
        encoded_nodes = []
        positions = []
        pair_index = 0
        for batch_index, evidence in enumerate(evidence_rows):
            for node_index in range(len(evidence) + 1):
                encoding = claim_encoding if node_index == 0 else pair_encoding
                index = batch_index if node_index == 0 else pair_index
                node = {
                    field: encoding[field][index]
                    if field in encoding else [0] * len(encoding["input_ids"][index])
                    for field in fields
                }
                encoded_nodes.append(node)
                positions.append((batch_index, node_index))
                if node_index:
                    pair_index += 1
        encoded = self.tokenizer.pad(encoded_nodes, padding="longest", return_tensors="pt")
        batch_size = len(samples)
        num_text_nodes = max(len(row) + 1 for row in evidence_rows)
        text_node_mask = torch.zeros((batch_size, num_text_nodes), dtype=torch.bool)
        sequence_length = encoded["input_ids"].shape[-1]
        batch = {}
        for field in fields:
            values = torch.as_tensor(encoded[field], dtype=torch.long)
            padded = torch.zeros((batch_size, num_text_nodes, sequence_length), dtype=torch.long)
            for flat_index, (batch_index, node_index) in enumerate(positions):
                padded[batch_index, node_index] = values[flat_index]
                text_node_mask[batch_index, node_index] = True
            batch[field] = padded

        max_images = max(len(sample["images"]) for sample in samples)
        images = torch.zeros(
            (batch_size, max_images, 3, self.image_size, self.image_size), dtype=torch.float32
        )
        image_mask = torch.zeros((batch_size, max_images), dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            for image_index, pixels in enumerate(sample["images"]):
                expected_shape = (3, self.image_size, self.image_size)
                if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected_shape:
                    raise ValueError(f"Image pixels must be shaped {expected_shape}")
                if not torch.isfinite(pixels).all():
                    raise ValueError("Image pixels must be finite")
                images[batch_index, image_index] = pixels.to(dtype=torch.float32)
                image_mask[batch_index, image_index] = True
        batch.update(
            text_node_mask=text_node_mask,
            images=images,
            image_mask=image_mask,
            labels=torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            metadata=metadata,
        )
        return batch
