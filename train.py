import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from config import VISION_MODELS, resolve_vision_model
from evaluate import build_dataloader, evaluate_model, load_checkpoint
from utils import (
    build_classification_loss,
    contrastive_alignment_loss,
    move_batch_to_device,
    save_json,
    set_seed,
)


def build_optimizer(model, config, weight_decay=0.01):
    transformer_ids = {
        id(parameter) for parameter in model.text_encoder.encoder.parameters()
    }

    transformer_parameters = []
    graph_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        target = (
            transformer_parameters
            if id(parameter) in transformer_ids
            else graph_parameters
        )
        target.append(parameter)

    parameter_groups = []
    if transformer_parameters:
        parameter_groups.append(
            {"params": transformer_parameters, "lr": config.transformer_lr}
        )
    if graph_parameters:
        parameter_groups.append({"params": graph_parameters, "lr": config.graph_lr})
    if not parameter_groups:
        raise ValueError("The model has no trainable parameters")

    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def build_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    config,
    epoch,
    criterion=None,
):
    model.train()
    if criterion is None:
        criterion = build_classification_loss(config, device)
    use_alignment = config.alignment_weight > 0
    totals = {"loss": 0.0, "classification_loss": 0.0, "alignment_loss": 0.0}
    total_examples = 0

    progress = tqdm(dataloader, desc=f"Training (Epoch {epoch})", unit="batch")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["labels"].size(0)
        if batch_size == 0:
            raise ValueError("Training batch contains no samples")

        optimizer.zero_grad(set_to_none=True)

        if use_alignment:
            outputs = model(batch, return_details=True)
            logits = outputs["logits"]
            alignment_loss = contrastive_alignment_loss(
                outputs["text_embedding"],
                outputs["visual_embedding"],
                temperature=config.temperature,
                valid_mask=batch["has_image"],
            )
        else:
            logits = model(batch)
            alignment_loss = logits.new_zeros(())

        classification_loss = criterion(logits, batch["labels"])
        loss = classification_loss + config.alignment_weight * alignment_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        total_examples += batch_size
        totals["loss"] += loss.detach().item() * batch_size
        totals["classification_loss"] += (
            classification_loss.detach().item() * batch_size
        )
        totals["alignment_loss"] += alignment_loss.detach().item() * batch_size
        progress.set_postfix(loss=f"{totals['loss'] / total_examples:.4f}")

    if total_examples == 0:
        raise ValueError("Cannot train on an empty DataLoader")
    return {name: value / total_examples for name, value in totals.items()}


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_macro_f1,
    config,
    path,
    scheduler=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "config": config.to_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    config,
    checkpoint_path,
    history_path=None,
    criterion=None,
    scheduler=None,
):
    """Train and retain the checkpoint with the highest validation macro-F1."""

    if criterion is None:
        criterion = build_classification_loss(config, device)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, config)
    history = []
    best_macro_f1 = float("-inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            epoch=epoch,
            criterion=criterion,
        )
        validation_metrics = evaluate_model(
            model,
            val_loader,
            device,
            num_classes=config.num_classes,
            criterion=criterion,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rates": [
                parameter_group["lr"] for parameter_group in optimizer.param_groups
            ],
        }
        history.append(record)

        macro_f1 = validation_metrics["macro_f1"]
        improved = macro_f1 > best_macro_f1
        if improved:
            best_macro_f1 = macro_f1
        if scheduler is not None:
            scheduler.step()
        if improved:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_macro_f1,
                config,
                checkpoint_path,
                scheduler=scheduler,
            )

        if history_path is not None:
            save_json(history, history_path)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_classification_loss={validation_metrics['classification_loss']:.4f} "
            f"val_macro_f1={macro_f1:.4f}"
        )

    load_checkpoint(model, checkpoint_path, device)
    return history, best_macro_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Train DualGraphFC on MOCHEG")

    parser.add_argument("--feature-cache", help="Precomputed vision-feature directory")
    parser.add_argument(
        "--retrieved-text-dir",
        help="Directory containing split-specific retrieved-text CSV files",
    )
    return parser.parse_args()


def main():
    from config import Config
    from models.model import DualGraphFC

    args = parse_args()
    config = Config()

    if args.feature_cache is None:
        raise ValueError(
            "A vision feature cache is required. Pass --feature-cache or set "
            "vision_feature_cache_dir in config.py."
        )
    config.vision_feature_cache_dir = args.feature_cache

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, tokenizer = build_dataloader(
        config,
        "train",
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader, _ = build_dataloader(
        config,
        "val",
        tokenizer=tokenizer,
        num_workers=config.num_workers,
    )

    model = DualGraphFC(
        config,
        vision_feature_shape=train_loader.dataset.feature_shape,
    ).to(device)
    optimizer = build_optimizer(model, config, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_classification_loss(config, device)
    vision_name = config.vision_model.rsplit("/", 1)[-1]
    checkpoint_path = args.checkpoint or str(
        Path(config.checkpoint_dir) / f"{vision_name}_best.pt"
    )
    history_path = args.history or str(Path(config.log_dir) / "training_history.json")

    _, best_macro_f1 = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config,
        checkpoint_path,
        history_path,
        criterion=criterion,
        scheduler=scheduler,
    )
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
