import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset import ID_TO_LABEL, MochegCollator, MochegDataset
from utils import move_batch_to_device, save_json


def build_dataloader(
    config,
    split,
    *,
    tokenizer=None,
    shuffle=False,
    limit=None,
    num_workers=0,
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.text_model)

    dataset = MochegDataset(
        config.data_root,
        split,
        image_size=config.image_size,
        limit=limit,
    )
    collator = MochegCollator(
        tokenizer,
        max_text_length=config.max_text_length,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
    return loader, tokenizer


def compute_metrics(labels, predictions, num_classes=3):
    if len(labels) == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")

    class_ids = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    per_class = {}
    for index, class_id in enumerate(class_ids):
        name = ID_TO_LABEL.get(class_id, str(class_id))
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "supported_f1": float(f1[0]),
        "refuted_f1": float(f1[1]),
        "nei_f1": float(f1[2]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_ids
        ).tolist(),
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=3):
    model.eval()
    labels = []
    predictions = []
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc="Evaluating", unit="batch")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        targets = batch["labels"]
        loss = F.cross_entropy(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

    metrics = compute_metrics(labels, predictions, num_classes=num_classes)
    metrics["loss"] = total_loss / total_examples
    metrics["classification_loss"] = metrics["loss"]
    return metrics


def read_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")
    return checkpoint


def load_checkpoint(model, checkpoint_or_path, device):
    """Load checkpoint weights into ``model`` and return its metadata."""

    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = read_checkpoint(checkpoint_or_path, device)
    else:
        checkpoint = checkpoint_or_path
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain 'model_state_dict'")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DualGraphFC on MOCHEG")
    parser.add_argument("--data-root", help="Directory containing MOCHEG splits")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to best.pt")
    parser.add_argument("--output", help="Metrics JSON path")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    from config import Config
    from model import DualGraphFC

    args = parse_args()
    config = Config()
    checkpoint_path = args.checkpoint or str(Path(config.checkpoint_dir) / "best.pt")
    checkpoint = read_checkpoint(checkpoint_path)
    if "config" in checkpoint:
        config = Config.from_dict(checkpoint["config"])

    if args.data_root is not None:
        config.data_root = args.data_root
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, _ = build_dataloader(
        config,
        args.split,
        limit=args.limit,
        num_workers=config.num_workers,
    )
    model = DualGraphFC(config).to(device)
    load_checkpoint(model, checkpoint, device)
    metrics = evaluate_model(model, dataloader, device, num_classes=config.num_classes)

    output_path = args.output or str(
        Path(config.prediction_dir) / f"{args.split}_metrics.json"
    )
    save_json(metrics, output_path)
    print(f"macro_f1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
