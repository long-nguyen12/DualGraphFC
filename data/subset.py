"""Content fingerprints for the fixed complete-modality study subset."""

import hashlib
import json
from pathlib import Path

from data.dataset import _label_to_id
from data.text_normalization import normalize_text


EXPECTED_COUNTS = {"train": 6888, "val": 920, "test": 1655}
VARIABLE_FIELDS = {
    "seed", "text_finetune_layers", "vision_finetune_layers", "num_workers",
    "data_root", "retrieved_text_dir", "run_dir",
}


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_records(samples):
    image_hashes = {}
    records = []
    for sample in samples:
        hashes = []
        for path in sample["images"]:
            if path not in image_hashes:
                image_hashes[path] = file_digest(path)
            hashes.append(image_hashes[path])
        inputs = {
            "claim_id": str(sample["claim_id"]), "split": sample["split"],
            "label_id": _label_to_id(sample["cleaned_truthfulness"]),
            "claim": sample["claim"], "text_evidence": sample["text_evidence"],
            "text_evidence_ids": sample["text_evidence_ids"],
            "image_evidence_ids": sample["image_evidence_ids"], "image_hashes": hashes,
            "website_url": sample.get("snopes_url", ""),
        }
        records.append({
            "claim_id": inputs["claim_id"], "label_id": inputs["label_id"],
            "fingerprint": json_digest(inputs), "image_hashes": hashes,
        })
    return records


def load_subset_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("subset_manifest_version") != 1:
        raise ValueError("Use the content-locked subset manifest created by the study prepare stage")
    if manifest.get("architecture_version") != 2:
        raise ValueError("Subset manifest must describe architecture_version=2")
    for split in EXPECTED_COUNTS:
        ids = manifest["subset_claim_ids"][split]
        records = manifest["subset_records"][split]
        if len(ids) != len(set(ids)) or ids != [row["claim_id"] for row in records]:
            raise ValueError(f"Invalid ordered claim IDs in {split} subset manifest")
    return manifest


def verify_study_config(config, manifest):
    for name, expected in manifest["config"].items():
        if name not in VARIABLE_FIELDS and getattr(config, name) != expected:
            raise ValueError(f"Study configuration changed: {name}")


def verify_subset(dataset, manifest):
    split = dataset.split
    ids = [str(sample["claim_id"]) for sample in dataset.samples]
    if ids != manifest["subset_claim_ids"][split]:
        raise ValueError(f"Ordered claim IDs changed in locked {split} subset")
    records = fingerprint_records(dataset.samples)
    if records != manifest["subset_records"][split]:
        expected = manifest["subset_records"][split]
        changed = [a["claim_id"] for a, b in zip(records, expected) if a != b]
        raise ValueError(f"Input content or labels changed in {split}: {changed[:10]}")
    return records


def verify_study_inputs(config, manifest):
    from transformers import CLIPImageProcessor
    from data.dataset import MochegDataset

    processor = CLIPImageProcessor.from_pretrained(config.vision_model, revision=config.vision_model_revision)
    # Ignore processor bookkeeping, but check every operation affecting pixels.
    pixel_fields = ("do_resize", "size", "resample", "do_center_crop", "crop_size",
                    "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb")
    for name in pixel_fields:
        if processor.to_dict().get(name) != manifest["image_processor_config"].get(name):
            raise ValueError(f"Image preprocessing changed: {name}")
    for split in EXPECTED_COUNTS:
        dataset = MochegDataset(
            config.data_root, split, image_processor=processor,
            image_size=config.image_size, retrieved_text_dir=config.retrieved_text_dir,
        )
        verify_subset(dataset, manifest)


def reference_ids(reference, split):
    if "subset_claim_ids" in reference:
        return [str(value) for value in reference["subset_claim_ids"][split]]
    return [str(value) for value in reference[split]["retained_claim_ids"]]


def normalized_evidence_hash(text):
    return json_digest(normalize_text(text))
