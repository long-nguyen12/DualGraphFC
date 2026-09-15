"""Extract and save frozen PoolFormer feature maps for MOCHEG images."""

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


def _write_metadata(cache_dir, metadata):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / FEATURE_CACHE_METADATA).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


@torch.inference_mode()
def extract_features(
    image_paths,
    processor,
    encoder,
    device,
    feature_shape,
    batch_size,
):
    """Return FP16 feature maps and indices of successfully loaded images."""

    feature_maps = []
    valid_indices = []

    for start in range(0, len(image_paths), batch_size):
        images = []
        batch_indices = []
        for index in range(start, min(start + batch_size, len(image_paths))):
            try:
                with Image.open(image_paths[index]) as image:
                    images.append(image.convert("RGB"))
                batch_indices.append(index)
            except (FileNotFoundError, OSError):
                continue

        if not images:
            continue

        pixel_values = processor(images=images, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        output = encoder(pixel_values=pixel_values).last_hidden_state
        if tuple(output.shape[1:]) != feature_shape:
            raise ValueError(
                f"PoolFormer returned feature shape {tuple(output.shape[1:])}; "
                f"expected {feature_shape}"
            )

        feature_maps.extend(output.cpu().to(torch.float16).unbind(0))
        valid_indices.extend(batch_indices)

    if feature_maps:
        features = torch.stack(feature_maps)
    else:
        features = torch.empty((0, *feature_shape), dtype=torch.float16)
    return features, torch.tensor(valid_indices, dtype=torch.long)


def precompute_split(
    loader,
    split,
    cache_dir,
    processor,
    encoder,
    device,
    feature_shape,
    batch_size,
):
    """Extract and save one feature tensor per claim in a dataset split."""

    for sample in tqdm(loader.load_split(split), desc=f"Caching {split}", unit="claim"):
        features, valid_indices = extract_features(
            sample["images"],
            processor,
            encoder,
            device,
            feature_shape,
            batch_size,
        )
        output_path = feature_cache_path(cache_dir, split, sample["claim_id"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": features,
                "source_image_evidence_ids": list(sample["image_evidence_ids"]),
                "valid_indices": valid_indices,
            },
            output_path,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract PoolFormer feature maps into the dataset folder"
    )
    parser.add_argument("--data-root", default=Config.data_root)
    parser.add_argument(
        "--split",
        action="append",
        choices=MochegDataset.SPLITS,
        help="Split to process; repeat for multiple splits (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
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
    cache_dir = Path(config.data_root) / "vision_feature"
    processor = AutoImageProcessor.from_pretrained(config.vision_model)
    encoder = PoolFormerModel.from_pretrained(config.vision_model).to(device).eval()
    feature_grid = VisionGraph._infer_feature_grid(config.image_size, encoder.config)
    feature_shape = (encoder.config.hidden_sizes[-1], *feature_grid)

    _write_metadata(
        cache_dir,
        {
            "format_version": FEATURE_CACHE_VERSION,
            "vision_model": config.vision_model,
            "image_size": config.image_size,
            "feature_shape": list(feature_shape),
            "storage_dtype": "float16",
        },
    )

    loader = MochegDataset(config.data_root)
    for split in args.split or MochegDataset.SPLITS:
        precompute_split(
            loader,
            split,
            cache_dir,
            processor,
            encoder,
            device,
            feature_shape,
            args.batch_size,
        )

    print(f"Saved PoolFormer features to {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
