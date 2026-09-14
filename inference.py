"""Run DualGraphFC on a claim and user-supplied ground-truth evidence."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from config import Config
from dataset import ID_TO_LABEL, MochegCollator, build_image_transform
from evaluate import load_checkpoint, read_checkpoint, resolve_device
from model import DualGraphFC
from utils import move_batch_to_device, save_json


def load_inference_bundle(checkpoint_path, requested_device="auto"):
    """Restore the saved configuration, tokenizer, and model."""

    device = resolve_device(requested_device)
    checkpoint = read_checkpoint(checkpoint_path)
    if "config" in checkpoint:
        config = Config.from_dict(checkpoint["config"])
    else:
        config = Config()

    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    model = DualGraphFC(config).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    return model, tokenizer, config, device


def _prepare_images(image_paths, config):
    transform = build_image_transform(config.image_size)
    tensors = []
    loaded_paths = []
    skipped_paths = []
    for raw_path in image_paths:
        if len(tensors) == config.max_images:
            break
        path = Path(raw_path).expanduser()
        try:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
            loaded_paths.append(str(path.resolve()))
        except (FileNotFoundError, OSError):
            skipped_paths.append(str(path))
    return tensors, loaded_paths, skipped_paths


def _valid_values(values, mask):
    if values is None or mask is None:
        return None
    return values[0, mask[0]].detach().cpu().tolist()


def _serialize_text_gat(batch_attention):
    if not batch_attention:
        return None
    layers = []
    for layer in batch_attention[0]:
        layers.append(
            {
                "edges": layer["edge_index"].transpose(0, 1).detach().cpu().tolist(),
                "attention": layer["attention"].detach().cpu().tolist(),
            }
        )
    return layers


def _cross_attention(values, query_mask, key_mask):
    if values is None:
        return None
    # Average heads for a compact node-to-node explanation matrix.
    values = values[0].mean(dim=0)
    values = values[query_mask[0]][:, key_mask[0]]
    return values.detach().cpu().tolist()


@torch.no_grad()
def predict(model, tokenizer, config, claim, evidence=None, image_paths=None, device=None):
    """Return a prediction and the available attention information."""

    model.eval()
    evidence = list(evidence or [])[: config.max_evidence]
    image_tensors, loaded_paths, skipped_paths = _prepare_images(
        image_paths or [], config
    )
    sample = {
        "claim": claim,
        "evidence": evidence,
        "images": image_tensors,
        # The collator requires a label, but inference does not use it.
        "label": 0,
        "metadata": {
            "image_paths": loaded_paths,
            "skipped_image_paths": skipped_paths,
        },
    }
    collator = MochegCollator(
        tokenizer,
        max_text_length=config.max_text_length,
        max_evidence=config.max_evidence,
        max_images=config.max_images,
        image_size=config.image_size,
    )
    batch = collator([sample])
    if device is None:
        device = next(model.parameters()).device
    batch = move_batch_to_device(batch, device)
    outputs = model(batch, return_details=True, return_attention=True)

    probabilities = torch.softmax(outputs["logits"], dim=-1)[0]
    label_id = int(probabilities.argmax().item())
    text_mask = outputs["text_node_mask"]
    visual_mask = outputs.get("visual_node_mask")

    text_to_vision = outputs.get("text_to_vision_attention")
    vision_to_text = outputs.get("vision_to_text_attention")
    result = {
        "label": ID_TO_LABEL[label_id],
        "label_id": label_id,
        "confidence": float(probabilities[label_id].item()),
        "probabilities": {
            ID_TO_LABEL[index]: float(value)
            for index, value in enumerate(probabilities.detach().cpu().tolist())
        },
        "text_nodes": [claim, *evidence],
        "text_attention": _valid_values(
            outputs.get("text_pool_attention"), text_mask
        ),
        "image_attention": _valid_values(
            outputs.get("visual_pool_attention"), visual_mask
        ),
        "consistency_attention": _valid_values(
            outputs.get("consistency_pool_attention"), text_mask
        ),
        "text_gat_attention": _serialize_text_gat(
            outputs.get("text_gat_attention")
        ),
        "cross_attention": {
            "text_to_image": _cross_attention(
                text_to_vision, text_mask, visual_mask
            ),
            "image_to_text": _cross_attention(
                vision_to_text, visual_mask, text_mask
            ),
        },
        "patch_grid": None,
        "loaded_image_paths": loaded_paths,
        "skipped_image_paths": skipped_paths,
    }
    vision_details = outputs.get("vision_graph")
    if vision_details is not None:
        result["patch_grid"] = list(vision_details["patch_grid"])
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run DualGraphFC inference")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--output", help="Optional prediction JSON path")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, config, device = load_inference_bundle(
        args.checkpoint, args.device
    )
    result = predict(
        model,
        tokenizer,
        config,
        args.claim,
        evidence=args.evidence,
        image_paths=args.image,
        device=device,
    )
    if args.output:
        save_json(result, args.output)
        print(f"Saved prediction to {args.output}")
    print(f"{result['label']} ({result['confidence']:.4f})")


if __name__ == "__main__":
    main()
