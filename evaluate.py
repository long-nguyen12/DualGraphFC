import argparse
from pathlib import Path
from urllib.parse import urlparse

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor

from data.dataset import ID_TO_LABEL, MochegCollator, MochegDataset
from data.tokenization import load_text_tokenizer
from utils import (
    autocast_context,
    build_classification_loss,
    move_batch_to_device,
    resolve_device,
    save_json,
)


def build_dataloader(
    config, split, *, tokenizer=None, image_processor=None, shuffle=False,
    limit=None, num_workers=0,
):
    if tokenizer is None:
        tokenizer = load_text_tokenizer(config)
    if image_processor is None:
        image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_model, revision=config.vision_model_revision,
        )
    dataset = MochegDataset(
        config.data_root, split,
        image_size=config.image_size,
        vision_model=config.vision_model,
        vision_model_revision=config.vision_model_revision,
        image_processor=image_processor,
        retrieved_text_dir=config.retrieved_text_dir,
        limit=limit,
    )
    if not len(dataset):
        raise ValueError(f"MOCHEG split {split!r} contains no complete records")
    print(
        f"{split}: retained {len(dataset)} complete records; "
        f"exclusions={dataset.subset_report['exclusion_counts']}"
    )
    collator = MochegCollator(
        tokenizer, max_text_length=config.max_text_length,
        image_size=config.image_size, augment=(split == "train" and shuffle),
    )
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collator, drop_last=False,
    ), tokenizer


def compute_metrics(labels, predictions, num_classes=3):
    if not len(labels):
        raise ValueError("Cannot compute metrics for an empty evaluation set")
    class_ids = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, average=None, zero_division=0,
    )
    per_class = {
        ID_TO_LABEL[class_id]: {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index, class_id in enumerate(class_ids)
    }
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "supported_f1": float(f1[0]), "refuted_f1": float(f1[1]),
        "nei_f1": float(f1[2]), "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "sample_count": len(labels),
    }


def source_domain(metadata):
    hostname = urlparse(metadata.get("snopes_url", "")).hostname or ""
    for domain in ("snopes.com", "politifact.com"):
        if hostname == domain or hostname.endswith("." + domain):
            return domain
    return hostname or "unknown"


@torch.no_grad()
def evaluate_model(
    model, dataloader, device, num_classes=3, criterion=None, predictions_path=None,
):
    model.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    labels, predictions, records = [], [], []
    total_loss = 0.0
    progress = tqdm(dataloader, desc="Evaluating", unit="batch")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["labels"].size(0)
        if not batch_size:
            raise ValueError("Evaluation batch contains no samples")
        with autocast_context(device):
            logits = model(batch)
            loss = criterion(logits, batch["labels"])
        if not torch.isfinite(loss):
            raise ValueError("Evaluation loss is not finite")
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        batch_labels = batch["labels"].cpu().tolist()
        batch_predictions = logits.argmax(dim=-1).cpu().tolist()
        total_loss += loss.item() * batch_size
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)
        for label, prediction, values, metadata in zip(
            batch_labels, batch_predictions, probabilities, batch["metadata"],
        ):
            records.append({
                "claim_id": metadata["claim_id"], "split": metadata["split"],
                "label_id": label, "label": ID_TO_LABEL[label],
                "prediction_id": prediction, "prediction": ID_TO_LABEL[prediction],
                "probabilities": {
                    ID_TO_LABEL[index]: value for index, value in enumerate(values)
                },
                "source": source_domain(metadata),
                "text_evidence_source": metadata["text_evidence_source"],
            })
        progress.set_postfix(loss=f"{total_loss / len(labels):.4f}")
    metrics = compute_metrics(labels, predictions, num_classes)
    metrics.update(
        loss=total_loss / len(labels), classification_loss=total_loss / len(labels),
        evaluation_scope="complete_text_image_subset",
    )
    for field, name in (("source", "by_source"), ("text_evidence_source", "by_evidence_source")):
        metrics[name] = {}
        for group in sorted({record[field] for record in records}):
            selected = [record for record in records if record[field] == group]
            metrics[name][group] = compute_metrics(
                [record["label_id"] for record in selected],
                [record["prediction_id"] for record in selected], num_classes,
            )
    if predictions_path is not None:
        save_json(records, predictions_path)
    return metrics


def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")
    if checkpoint.get("config", {}).get("architecture_version") != 2:
        raise ValueError(
            "Checkpoint is not DualGraphFC architecture version 2 (architecture_version=2); "
            "load it with its original implementation"
        )
    if checkpoint.get("architecture_version", 2) != 2:
        raise ValueError("Checkpoint architecture version does not match its configuration")


def read_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_checkpoint(checkpoint)
    return checkpoint


def load_checkpoint(model, checkpoint_or_path, device):
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = read_checkpoint(checkpoint_or_path, device)
    else:
        checkpoint = checkpoint_or_path
        validate_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DualGraphFC on complete MOCHEG records")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", help="Metrics JSON; defaults to the checkpoint directory")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--retrieved-text-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    from config import Config
    from models.model import DualGraphFC

    args = parse_args()
    checkpoint = read_checkpoint(args.checkpoint)
    config = Config.from_dict(checkpoint["config"])
    for name in ("data_root", "batch_size", "num_workers", "retrieved_text_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    device = resolve_device(args.device)
    dataloader, _ = build_dataloader(
        config, args.split, limit=args.limit, num_workers=config.num_workers,
    )
    model = DualGraphFC(config).to(device)
    load_checkpoint(model, checkpoint, device)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{args.split}_metrics.json"
    metrics = evaluate_model(
        model, dataloader, device, config.num_classes,
        criterion=build_classification_loss(config, device),
        predictions_path=output.with_name(output.stem + "_predictions.json"),
    )
    save_json(metrics, output)
    save_json(dataloader.dataset.subset_report, output.with_name(output.stem + "_subset.json"))
    print(f"macro_f1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")
    print(f"Saved complete-subset metrics to {output}")


if __name__ == "__main__":
    main()
