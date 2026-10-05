import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoConfig, CLIPImageProcessor

from data.tokenization import load_text_tokenizer

from evaluate import build_dataloader, evaluate_model, load_checkpoint
from utils import (
    autocast_context, autocast_dtype, build_classification_loss, make_grad_scaler,
    move_batch_to_device, resolve_device, save_json, set_seed,
)


def build_optimizer(model, config):
    text = [p for p in model.text_encoder.encoder.parameters() if p.requires_grad]
    vision = [p for p in model.vision_encoder.encoder.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in text + vision}
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = [
        {"name": "deberta", "params": text, "lr": config.transformer_lr},
        {"name": "clip", "params": vision, "lr": config.vision_lr},
        {"name": "reasoning", "params": heads, "lr": config.graph_lr},
    ]
    groups = [group for group in groups if group["params"]]
    ids = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if not groups or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Optimizer must contain every trainable parameter exactly once")
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config):
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs - config.warmup_epochs, 1),
        eta_min=config.min_lr,
    )
    if not config.warmup_epochs:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs],
    )


def train_one_epoch(
    model, dataloader, optimizer, device, config, epoch, criterion=None,
    scaler=None, max_steps=None, verify_gradients=False,
):
    model.train()
    criterion = criterion if criterion is not None else build_classification_loss(config, device)
    scaler = scaler if scaler is not None else make_grad_scaler(device)
    total_loss, total_examples, steps, updates = 0.0, 0, 0, 0
    gradient_checks = {}
    progress = tqdm(dataloader, desc=f"Training (Epoch {epoch})", unit="batch")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["labels"].size(0)
        if not batch_size:
            raise ValueError("Training batch contains no samples")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(batch)
            loss = criterion(logits, batch["labels"])
        if not torch.isfinite(loss):
            raise ValueError("Training loss is not finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm, error_if_nonfinite=not scaler.is_enabled(),
        )
        representatives = {}
        if verify_gradients:
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise AssertionError("A frozen parameter received a gradient")
            for name, wrapper in (("deberta", model.text_encoder), ("clip", model.vision_encoder)):
                for index, layer in enumerate(wrapper.finetuned_layers):
                    if not any(p.grad is not None and torch.isfinite(p.grad).all()
                               and p.grad.count_nonzero().item() for p in layer.parameters()):
                        raise AssertionError(f"No finite nonzero gradient in {name} selected layer {index}")
                    gradient_checks[f"{name}_selected_layer_{index}"] = "finite_nonzero_gradient"
            for group in optimizer.param_groups:
                candidates = [
                    p for p in group["params"]
                    if p.grad is not None and torch.isfinite(p.grad).all()
                    and p.grad.count_nonzero().item()
                ]
                if not candidates:
                    raise AssertionError(f"No finite nonzero gradient in {group['name']}")
                parameter = candidates[0]
                representatives[group["name"]] = (parameter, parameter.detach().clone())
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updates += int(scaler.get_scale() >= previous_scale)
        for name, (parameter, before) in representatives.items():
            if torch.equal(parameter.detach(), before):
                raise AssertionError(f"Optimizer did not update {name}")
            gradient_checks[name] = "finite_gradient_and_parameter_update"
        steps += 1
        total_examples += batch_size
        total_loss += loss.detach().item() * batch_size
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")
        if max_steps is not None and steps >= max_steps:
            break
    if not total_examples:
        raise ValueError("Cannot train on an empty DataLoader")
    metrics = {
        "loss": total_loss / total_examples,
        "classification_loss": total_loss / total_examples,
        "examples": total_examples, "optimizer_steps": updates,
        "batches": steps, "amp_skipped_steps": steps - updates,
    }
    if verify_gradients:
        metrics["gradient_checks"] = gradient_checks
        metrics["frozen_gradients"] = "absent"
    return metrics


def save_checkpoint(
    model, optimizer, epoch, best_macro_f1, config, path,
    scheduler=None, scaler=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture_version": 2, "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "best_macro_f1": best_macro_f1,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def fit(
    model, train_loader, val_loader, optimizer, device, config,
    checkpoint_path, history_path=None, criterion=None, scheduler=None,
    train_eval_loader=None,
):
    history, best_macro_f1, epochs_without_improvement = [], float("-inf"), 0
    scaler = make_grad_scaler(device)
    run_path = Path(checkpoint_path).parent
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, config, epoch,
            criterion=criterion, scaler=scaler,
        )
        validation_metrics = evaluate_model(
            model, val_loader, device, num_classes=config.num_classes,
            criterion=criterion,
            predictions_path=run_path / "predictions" / f"validation_epoch_{epoch:02d}.json",
        )
        history.append({
            "epoch": epoch, "train": train_metrics, "validation": validation_metrics,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
        })
        macro_f1 = validation_metrics["macro_f1"]
        improved = macro_f1 > best_macro_f1
        if improved:
            best_macro_f1, epochs_without_improvement = macro_f1, 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if improved:
            save_checkpoint(
                model, optimizer, epoch, best_macro_f1, config, checkpoint_path,
                scheduler=scheduler, scaler=scaler,
            )
        if history_path is not None:
            save_json(history, history_path)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} val_macro_f1={macro_f1:.4f}"
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
            break
    load_checkpoint(model, checkpoint_path, device)
    best_metrics = evaluate_model(
        model, val_loader, device, config.num_classes, criterion=criterion,
        predictions_path=run_path / "predictions" / "validation_best.json",
    )
    save_json(best_metrics, run_path / "validation_best_metrics.json")
    if train_eval_loader is not None:
        train_metrics = evaluate_model(
            model, train_eval_loader, device, config.num_classes, criterion=criterion,
            predictions_path=run_path / "predictions" / "train_best_clean.json",
        )
        save_json(train_metrics, run_path / "train_best_clean_metrics.json")
    return history, best_macro_f1


def head_initialization_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(("text_encoder.encoder.", "vision_encoder.encoder.")):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_run_manifest(run_path, config, model, tokenizer, processor, loaders, device):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--", "."], capture_output=True, text=True, check=False,
        encoding="utf-8",
    )
    if diff.returncode == 0 and diff.stdout:
        (run_path / "working_tree.patch").write_text(diff.stdout, encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False,
    )
    source_paths = [Path(name) for name in (
        "config.py", "utils.py", "train.py", "evaluate.py", "inference.py",
        "diagnose_modalities.py", "run_finetune_study.py",
    )] + list(Path("models").glob("*.py")) + list(Path("data").glob("*.py"))
    for path in source_paths:
        destination = run_path / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copyfile(path, destination)
    save_json(config.to_dict(), run_path / "config.json")
    subsets = {}
    for split, loader in loaders.items():
        report = loader.dataset.subset_report
        save_json(report, run_path / f"{split}_subset.json")
        subsets[split] = report["retained_claim_ids"]
    manifest = {
        "architecture_version": 2, "seed": config.seed,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(status.stdout), "device": str(device),
        "precision": str(autocast_dtype(device) or torch.float32),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "backbones": {
            "text": {"name": config.text_model, "revision": config.text_model_revision},
            "vision": {"name": config.vision_model, "revision": config.vision_model_revision},
        },
        "tokenizer": {"name": tokenizer.name_or_path, "revision": config.text_model_revision},
        "image_processor": {
            "name": config.vision_model, "revision": config.vision_model_revision,
            "config": processor.to_dict(),
        },
        "subset_claim_ids": subsets,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head_initialization_sha256": head_initialization_digest(model),
        "evaluation_scope": "complete_text_image_subset",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        manifest["gpu"] = {"name": properties.name, "total_vram_gib": properties.total_memory / 2**30}
    save_json(manifest, run_path / "manifest.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DualGraphFC v2 on complete MOCHEG records")
    parser.add_argument("--data-root")
    parser.add_argument("--retrieved-text-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--text-finetune-layers", type=int, choices=(0, 2))
    parser.add_argument("--vision-finetune-layers", type=int, choices=(0, 2))
    parser.add_argument("--subset-manifest")
    parser.add_argument("--limit", type=int, help="Apply after complete-record filtering in each split")
    parser.add_argument("--run-name", help="Unique directory name under outputs/runs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=3)
    return parser.parse_args()


def main():
    from config import Config
    from models.model import DualGraphFC

    args = parse_args()
    overrides = {
        name: getattr(args, name) for name in
        ("data_root", "retrieved_text_dir", "batch_size", "epochs", "num_workers", "seed",
         "text_finetune_layers", "vision_finetune_layers")
        if getattr(args, name) is not None
    }
    subset_manifest = None
    if args.subset_manifest:
        from data.subset import load_subset_manifest, verify_study_config

        subset_manifest = load_subset_manifest(args.subset_manifest)
        config = Config.from_dict(subset_manifest["config"])
        for key, value in overrides.items():
            setattr(config, key, value)
        verify_study_config(config, subset_manifest)
        if args.limit is not None:
            raise ValueError("A locked subset cannot be combined with --limit")
    else:
        config = Config(**overrides)
    if config.batch_size < 1 or config.epochs < 1 or args.smoke_steps < 1:
        raise ValueError("Batch size, epochs and smoke steps must be positive")
    set_seed(config.seed)
    device = resolve_device(args.device)
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run-name must be a single directory name")
    run_path = Path(config.run_dir) / name
    run_path.mkdir(parents=True, exist_ok=False)
    for prefix in ("text", "vision"):
        revision_name = f"{prefix}_model_revision"
        backbone_config = AutoConfig.from_pretrained(
            getattr(config, f"{prefix}_model"), revision=getattr(config, revision_name),
        )
        setattr(config, revision_name, getattr(backbone_config, "_commit_hash", None) or getattr(config, revision_name))
    model = DualGraphFC(config)
    config.text_model_revision = getattr(model.text_encoder.encoder.config, "_commit_hash", None) or config.text_model_revision
    config.vision_model_revision = getattr(model.vision_encoder.encoder.config, "_commit_hash", None) or config.vision_model_revision
    tokenizer = load_text_tokenizer(config)
    processor = CLIPImageProcessor.from_pretrained(config.vision_model, revision=config.vision_model_revision)
    train_loader, _ = build_dataloader(
        config, "train", tokenizer=tokenizer, image_processor=processor,
        shuffle=True, limit=args.limit, num_workers=config.num_workers,
        subset_manifest=subset_manifest,
    )
    loaders = {"train": train_loader}
    if not args.smoke_test:
        val_loader, _ = build_dataloader(
            config, "val", tokenizer=tokenizer, image_processor=processor,
            limit=args.limit, num_workers=config.num_workers,
            subset_manifest=subset_manifest,
        )
        loaders["val"] = val_loader
    write_run_manifest(run_path, config, model, tokenizer, processor, loaders, device)
    if subset_manifest is not None:
        save_json(subset_manifest, run_path / "subset_manifest.json")
        initialization_path = Path(args.subset_manifest).parent / f"head_seed_{config.seed}.json"
        initialization = {"sha256": head_initialization_digest(model)}
        if initialization_path.exists():
            with initialization_path.open(encoding="utf-8") as handle:
                if json.load(handle) != initialization:
                    raise ValueError("Initial reasoning weights changed between configurations of the same seed")
        else:
            save_json(initialization, initialization_path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model = model.to(device)
        optimizer = build_optimizer(model, config)
        criterion = build_classification_loss(config, device)
        if args.smoke_test:
            metrics = train_one_epoch(
                model, train_loader, optimizer, device, config, epoch=1,
                criterion=criterion, max_steps=args.smoke_steps, verify_gradients=True,
            )
            if metrics["optimizer_steps"] != args.smoke_steps:
                raise ValueError("Subset is too small for the requested smoke steps; increase --limit")
            metrics["batch_size"] = config.batch_size
            metrics["target_gpu_vram_gib"] = 48
            metrics["target_batch_8_verified"] = (
                device.type == "cuda" and config.batch_size == 8
                and metrics["examples"] == 8 * args.smoke_steps
                and torch.cuda.get_device_properties(device).total_memory / 2**30 >= 47
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            save_json(metrics, run_path / "smoke_metrics.json")
            print(metrics)
        else:
            train_eval_loader, _ = build_dataloader(
                config, "train", tokenizer=tokenizer, image_processor=processor,
                num_workers=config.num_workers, limit=args.limit,
                subset_manifest=subset_manifest,
            )
            _, best_macro_f1 = fit(
                model, train_loader, val_loader, optimizer, device, config,
                run_path / "best.pt", run_path / "training_history.json",
                criterion=criterion, scheduler=build_scheduler(optimizer, config),
                train_eval_loader=train_eval_loader,
            )
            print(f"Best validation macro-F1: {best_macro_f1:.4f}; checkpoint: {run_path / 'best.pt'}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            save_json({
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }, run_path / "memory_metrics.json")
    except torch.cuda.OutOfMemoryError:
        failure = {
            "error": "CUDA out of memory", "config": config.to_dict(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        save_json(failure, run_path / "memory_failure.json")
        raise
    print(f"Run artifacts: {run_path}")


if __name__ == "__main__":
    main()
