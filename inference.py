"""Run DualGraphFC on a claim and user-supplied ground-truth evidence."""

import argparse
from html import unescape
from pathlib import Path
import re

from matplotlib.figure import Figure
import networkx as nx
import torch
from PIL import Image
from transformers import AutoTokenizer

from config import Config
from dataset import ID_TO_LABEL, MochegCollator, _label_to_id, build_image_transform
from dataset_mocheg import MochegDataset as MochegLoader
from evaluate import load_checkpoint, read_checkpoint
from model import DualGraphFC
from utils import move_batch_to_device, save_json


def load_inference_bundle(checkpoint_path, requested_device="auto"):
    """Restore the saved configuration, tokenizer, and model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    transform = build_image_transform(
        config.image_size,
        config.vision_model,
    )
    tensors = []
    loaded_paths = []
    skipped_paths = []
    for raw_path in image_paths:
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


def _short_label(text, max_length=48):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = " ".join(unescape(text).split())
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3]}..."


def _save_text_graph(outputs, text_nodes, output_dir):
    batch_attention = outputs.get("text_gat_attention")
    if not batch_attention or not batch_attention[0]:
        return None

    layer = batch_attention[0][-1]
    edges = layer["edge_index"].detach().cpu().transpose(0, 1).tolist()
    weights = layer["attention"].detach().cpu().mean(dim=-1).tolist()
    graph = nx.DiGraph()
    labels = {}
    for index, text in enumerate(text_nodes):
        prefix = "Claim" if index == 0 else f"E{index}"
        labels[index] = f"{prefix}: {_short_label(text)}"
        graph.add_node(index)
    for (source, target), weight in zip(edges, weights):
        graph.add_edge(source, target, weight=float(weight))

    positions = nx.spring_layout(graph, seed=42)
    figure = Figure(figsize=(max(8, min(16, len(text_nodes) * 1.5)), 7))
    axis = figure.subplots()
    edge_widths = [
        0.5 + 4.0 * graph[source][target]["weight"]
        for source, target in graph.edges
    ]
    nx.draw_networkx(
        graph,
        positions,
        labels=labels,
        node_color=["#f4a261" if node == 0 else "#8ecae6" for node in graph],
        node_size=1800,
        font_size=8,
        width=edge_widths,
        arrows=True,
        clip_on=False,
        ax=axis,
    )
    axis.set_title("Text graph: final GAT layer")
    axis.margins(0.35)
    axis.set_axis_off()
    path = output_dir / "text_graph.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.clear()
    return str(path.resolve())


def _save_vision_graphs(outputs, image_paths, output_dir):
    details = outputs.get("vision_graph")
    if details is None or not image_paths:
        return []

    grid_height, grid_width = details["patch_grid"]
    patches_per_image = details["patches_per_image"]
    edge_sets = details["edge_indices"][0]
    positions = {
        node: (node % grid_width, grid_height - 1 - node // grid_width)
        for node in range(patches_per_image)
    }
    paths = []
    for image_index, image_path in enumerate(image_paths):
        if image_index >= len(edge_sets) or edge_sets[image_index] is None:
            continue
        edges = (
            edge_sets[image_index]
            .detach()
            .cpu()
            .transpose(0, 1)
            .tolist()
        )
        graph = nx.Graph()
        graph.add_nodes_from(range(patches_per_image))
        graph.add_edges_from(edges)

        figure = Figure(figsize=(8, 8))
        axis = figure.subplots()
        nx.draw_networkx(
            graph,
            positions,
            node_size=90,
            node_color="#90be6d",
            edge_color="#6c757d",
            width=0.5,
            alpha=0.65,
            with_labels=False,
            ax=axis,
        )
        axis.set_title(f"Visual feature graph: {Path(image_path).name}")
        axis.set_axis_off()
        path = output_dir / f"vision_graph_{image_index + 1}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        figure.clear()
        paths.append(str(path.resolve()))
    return paths


def _save_cross_attention(outputs, text_nodes, image_paths, output_dir):
    attention = outputs.get("text_to_vision_attention")
    if attention is None:
        return None

    text_mask = outputs["text_node_mask"][0]
    visual_mask = outputs["visual_node_mask"][0]
    attention = attention[0].mean(dim=0)
    attention = attention[text_mask][:, visual_mask].detach().float().cpu()
    if attention.numel() == 0:
        return None

    figure = Figure(
        figsize=(max(8, min(24, attention.size(1) / 8)), max(4, len(text_nodes)))
    )
    axis = figure.subplots()
    image = axis.imshow(attention.numpy(), aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(text_nodes)))
    axis.set_yticklabels(
        [
            f"{'Claim' if index == 0 else f'E{index}'}: {_short_label(text)}"
            for index, text in enumerate(text_nodes)
        ],
        fontsize=8,
    )
    axis.set_ylabel("Text nodes")
    axis.set_xlabel("Visual feature nodes")
    axis.set_title("Text-to-image cross-attention (head average)")

    details = outputs.get("vision_graph")
    if image_paths and details is not None:
        patches_per_image = details["patches_per_image"]
        centers = [
            index * patches_per_image + (patches_per_image - 1) / 2
            for index in range(len(image_paths))
        ]
        axis.set_xticks(centers)
        axis.set_xticklabels(
            [Path(path).name for path in image_paths], rotation=30, ha="right"
        )
        for index in range(1, len(image_paths)):
            axis.axvline(index * patches_per_image - 0.5, color="white", linewidth=1)
    else:
        axis.set_xticks([0])
        axis.set_xticklabels(["No image"])

    figure.colorbar(image, ax=axis, label="Attention")
    path = output_dir / "cross_attention.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.clear()
    return str(path.resolve())


def visualize_graphs(outputs, text_nodes, image_paths, output_dir):
    """Save text, vision, and cross-modal graph visualizations."""

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    text_path = _save_text_graph(outputs, text_nodes, output_dir)
    if text_path is not None:
        paths.append(text_path)
    paths.extend(_save_vision_graphs(outputs, image_paths, output_dir))
    cross_path = _save_cross_attention(
        outputs, text_nodes, image_paths, output_dir
    )
    if cross_path is not None:
        paths.append(cross_path)
    return paths


def load_dataset_sample(data_root, split="test", sample_index=0):
    """Load one claim and all of its evidence from a MOCHEG split."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    samples = MochegLoader(data_root).load_split(
        split,
        limit=sample_index + 1,
    )
    if sample_index >= len(samples):
        raise IndexError(
            f"sample_index {sample_index} is outside the {split!r} split "
            f"with {len(samples)} samples"
        )
    return samples[sample_index]


@torch.no_grad()
def predict(
    model,
    tokenizer,
    config,
    claim,
    evidence=None,
    image_paths=None,
    device=None,
    visualization_dir=None,
):
    """Return a prediction and the available attention information."""

    model.eval()
    evidence = list(evidence or [])
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
    if visualization_dir is not None:
        result["visualizations"] = visualize_graphs(
            outputs,
            [claim, *evidence],
            loaded_paths,
            visualization_dir,
        )
    return result


def predict_dataset_sample(
    model,
    tokenizer,
    config,
    data_root,
    split="test",
    sample_index=0,
    device=None,
    visualization_dir=None,
):
    """Run inference on one indexed MOCHEG sample."""

    sample = load_dataset_sample(data_root, split, sample_index)
    result = predict(
        model,
        tokenizer,
        config,
        sample["claim"],
        evidence=sample["text_evidence"],
        image_paths=sample["images"],
        device=device,
        visualization_dir=visualization_dir,
    )
    ground_truth_id = _label_to_id(sample["cleaned_truthfulness"])
    result["dataset_sample"] = {
        "split": split,
        "sample_index": sample_index,
        "claim_id": sample["claim_id"],
        "ground_truth_label": ID_TO_LABEL[ground_truth_id],
        "ground_truth_label_id": ground_truth_id,
        "correct": result["label_id"] == ground_truth_id,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run DualGraphFC inference")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/poolformer_s12_best.pt",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--claim", help="Manual claim text")
    inputs.add_argument(
        "--sample-index",
        type=int,
        help="Zero-based sample index from a MOCHEG split",
    )
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--data-root", help="Directory containing MOCHEG splits")
    parser.add_argument(
        "--split",
        choices=MochegLoader.SPLITS,
        default="test",
        help="Dataset split used with --sample-index (default: test)",
    )
    parser.add_argument("--output", help="Optional prediction JSON path")
    parser.add_argument(
        "--visualize-dir",
        help="Optional directory for text, vision, and cross-attention plots",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_index is not None and (args.evidence or args.image):
        raise ValueError(
            "--evidence and --image cannot be combined with --sample-index"
        )
    model, tokenizer, config, device = load_inference_bundle(
        args.checkpoint, args.device
    )
    if args.sample_index is not None:
        result = predict_dataset_sample(
            model,
            tokenizer,
            config,
            args.data_root or config.data_root,
            split=args.split,
            sample_index=args.sample_index,
            device=device,
            visualization_dir=args.visualize_dir,
        )
    else:
        result = predict(
            model,
            tokenizer,
            config,
            args.claim,
            evidence=args.evidence,
            image_paths=args.image,
            device=device,
            visualization_dir=args.visualize_dir,
        )
    if args.output:
        save_json(result, args.output)
        print(f"Saved prediction to {args.output}")
    for path in result.get("visualizations", []):
        print(f"Saved visualization to {path}")
    if "dataset_sample" in result:
        sample = result["dataset_sample"]
        print(
            f"claim_id={sample['claim_id']} "
            f"ground_truth={sample['ground_truth_label']} "
            f"correct={sample['correct']}"
        )
    print(f"{result['label']} ({result['confidence']:.4f})")


if __name__ == "__main__":
    main()
