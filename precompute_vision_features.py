"""Precompute frozen PoolFormer feature maps for MOCHEG images."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, PoolFormerModel

from config import Config
from data.dataset import (
    FEATURE_CACHE_METADATA,
    FEATURE_CACHE_VERSION,
    feature_cache_path,
)
from data.dataset_mocheg import MochegDataset
from models.vision_graph import VisionGraph


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_metadata(cache_dir, metadata):
    path = Path(cache_dir) / FEATURE_CACHE_METADATA
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != metadata:
            raise ValueError(
                f"Existing cache metadata at {path} does not match this run. "
                "Use a new output directory."
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    temporary.replace(path)


def _save_claim(cache_dir, split, accumulator, feature_shape):
    features = accumulator["features"]
    if features:
        features = torch.stack(features)
    else:
        features = torch.empty((0, *feature_shape), dtype=torch.float16)

    payload = {
        "features": features,
        "source_image_evidence_ids": accumulator["source_ids"],
        "valid_indices": torch.tensor(
            accumulator["valid_indices"],
            dtype=torch.long,
        ),
    }
    path = feature_cache_path(cache_dir, split, accumulator["claim_id"])
    _atomic_torch_save(payload, path)


@torch.inference_mode()
def _process_records(
    records,
    accumulators,
    processor,
    encoder,
    device,
    cache_dir,
    split,
    feature_shape,
):
    images = []
    valid_records = []
    for record in records:
        try:
            with Image.open(record["path"]) as image:
                images.append(image.convert("RGB"))
            valid_records.append(record)
        except (FileNotFoundError, OSError):
            continue

    encoded_by_index = {}
    if images:
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device)
        feature_maps = encoder(pixel_values=pixel_values).last_hidden_state
        if tuple(feature_maps.shape[1:]) != feature_shape:
            raise ValueError(
                f"PoolFormer returned feature shape {tuple(feature_maps.shape[1:])}; "
                f"expected {feature_shape}"
            )
        feature_maps = feature_maps.detach().to(device="cpu", dtype=torch.float16)
        encoded_by_index = {
            record["record_index"]: feature
            for record, feature in zip(valid_records, feature_maps)
        }

    for record in records:
        key = record["claim_id"]
        feature = encoded_by_index.get(record["record_index"])
        if feature is not None:
            accumulators[key]["features"].append(feature)
            accumulators[key]["valid_indices"].append(record["image_index"])
        if record["is_last"]:
            _save_claim(cache_dir, split, accumulators.pop(key), feature_shape)


def precompute_split(
    loader,
    split,
    cache_dir,
    processor,
    encoder,
    device,
    feature_shape,
    batch_size,
    overwrite=False,
):
    samples = loader.load_split(split)
    pending = []
    accumulators = {}
    record_index = 0
    remaining_images = sum(
        len(sample["images"])
        for sample in samples
        if overwrite
        or not feature_cache_path(cache_dir, split, sample["claim_id"]).is_file()
    )
    progress = tqdm(total=remaining_images, desc=f"Caching {split}", unit="image")

    for sample in samples:
        output_path = feature_cache_path(cache_dir, split, sample["claim_id"])
        if output_path.is_file() and not overwrite:
            continue

        image_paths = list(sample["images"])
        source_ids = list(sample["image_evidence_ids"])
        if len(image_paths) != len(source_ids):
            raise ValueError(
                f"Image paths and IDs differ for claim {sample['claim_id']!r}"
            )

        claim_id = str(sample["claim_id"])
        accumulators[claim_id] = {
            "claim_id": claim_id,
            "source_ids": source_ids,
            "features": [],
            "valid_indices": [],
        }
        if not image_paths:
            _save_claim(cache_dir, split, accumulators.pop(claim_id), feature_shape)
            continue

        for image_index, image_path in enumerate(image_paths):
            pending.append(
                {
                    "record_index": record_index,
                    "claim_id": claim_id,
                    "image_index": image_index,
                    "path": image_path,
                    "is_last": image_index == len(image_paths) - 1,
                }
            )
            record_index += 1
            if len(pending) == batch_size:
                _process_records(
                    pending,
                    accumulators,
                    processor,
                    encoder,
                    device,
                    cache_dir,
                    split,
                    feature_shape,
                )
                progress.update(len(pending))
                pending = []

    if pending:
        _process_records(
            pending,
            accumulators,
            processor,
            encoder,
            device,
            cache_dir,
            split,
            feature_shape,
        )
        progress.update(len(pending))
    progress.close()
    if accumulators:
        raise RuntimeError("Some claims were not written to the feature cache")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute frozen PoolFormer feature maps for MOCHEG"
    )
    parser.add_argument("--data-root", default=Config.data_root)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=MochegDataset.SPLITS,
        help="Split to cache; repeat for multiple splits (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = Config(data_root=args.data_root)
    processor = AutoImageProcessor.from_pretrained(config.vision_model)
    encoder = PoolFormerModel.from_pretrained(config.vision_model).to(device).eval()
    feature_grid = VisionGraph._infer_feature_grid(config.image_size, encoder.config)
    feature_shape = (encoder.config.hidden_sizes[-1], *feature_grid)
    metadata = {
        "format_version": FEATURE_CACHE_VERSION,
        "vision_model": config.vision_model,
        "image_size": config.image_size,
        "feature_shape": list(feature_shape),
        "storage_dtype": "float16",
    }
    _write_metadata(args.output_dir, metadata)

    loader = MochegDataset(config.data_root)
    splits = args.split or MochegDataset.SPLITS
    for split in splits:
        precompute_split(
            loader,
            split,
            args.output_dir,
            processor,
            encoder,
            device,
            feature_shape,
            args.batch_size,
            overwrite=args.overwrite,
        )
    print(f"Saved PoolFormer feature cache to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
