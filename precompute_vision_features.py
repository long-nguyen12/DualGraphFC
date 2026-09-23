import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from config import Config, VISION_MODELS, resolve_vision_model
from data.dataset import (
    FEATURE_CACHE_METADATA,
    FEATURE_CACHE_VERSION,
    feature_cache_path,
)
from data.dataset_mocheg import MochegDataset


def _pair(value):
    if isinstance(value, int):
        return value, value
    return value[0], value[1]


def infer_feature_grid(image_size, encoder_config):
    if all(
        hasattr(encoder_config, name) for name in ("patch_sizes", "strides", "padding")
    ):
        patch_sizes = encoder_config.patch_sizes
        strides = encoder_config.strides
        paddings = encoder_config.padding

        height = width = image_size
        for patch_size, stride, padding in zip(
            patch_sizes,
            strides,
            paddings,
        ):
            patch_height, patch_width = _pair(patch_size)
            stride_height, stride_width = _pair(stride)
            padding_height, padding_width = _pair(padding)
            height = (height + 2 * padding_height - patch_height) // stride_height + 1
            width = (width + 2 * padding_width - patch_width) // stride_width + 1
    elif hasattr(encoder_config, "patch_size"):
        patch_height, patch_width = _pair(encoder_config.patch_size)
        height = image_size // patch_height
        width = image_size // patch_width
        if getattr(encoder_config, "model_type", None) in {
            "convnext",
            "convnextv2",
        }:
            num_stages = getattr(
                encoder_config,
                "num_stages",
                len(encoder_config.hidden_sizes),
            )
            height //= 2 ** (num_stages - 1)
            width //= 2 ** (num_stages - 1)
    else:
        raise ValueError("The selected vision model has no supported patch layout")

    return height, width


def infer_feature_width(encoder_config):
    hidden_sizes = getattr(encoder_config, "hidden_sizes", None)
    if hidden_sizes:
        return hidden_sizes[-1]
    hidden_size = getattr(encoder_config, "hidden_size", None)
    return hidden_size


def to_feature_map(last_hidden_state, feature_grid):
    if last_hidden_state.ndim == 4:
        return last_hidden_state

    height, width = feature_grid
    patch_count = height * width
    patch_tokens = last_hidden_state[:, -patch_count:]
    return patch_tokens.transpose(1, 2).reshape(
        last_hidden_state.size(0),
        last_hidden_state.size(2),
        height,
        width,
    )


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

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(
            device
        )
        output = encoder(pixel_values=pixel_values).last_hidden_state
        output = to_feature_map(output, feature_shape[-2:])

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
        description="Extract vision feature maps into the dataset folder"
    )
    parser.add_argument("--data-root", default=Config.data_root)
    parser.add_argument(
        "--vision-model",
        choices=tuple(VISION_MODELS),
        default="poolformer",
        help="Vision backbone preset (default: poolformer)",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=MochegDataset.SPLITS,
        help="Split to process; repeat for multiple splits (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        data_root=args.data_root,
        vision_model=resolve_vision_model(args.vision_model),
    )
    cache_dir = Path(config.data_root) / "vision_feature" / args.vision_model
    processor = AutoImageProcessor.from_pretrained(config.vision_model)
    encoder = AutoModel.from_pretrained(config.vision_model).to(device).eval()
    feature_grid = infer_feature_grid(config.image_size, encoder.config)
    feature_shape = (
        infer_feature_width(encoder.config),
        *feature_grid,
    )

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

    print(f"Saved {args.vision_model} features to {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
