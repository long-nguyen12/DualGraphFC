"""Prepare, smoke-test and run the eight-run backbone study in separate stages."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import transformers
from transformers import AutoConfig, CLIPImageProcessor

from config import Config
from data.dataset import MochegDataset
from data.subset import (
    EXPECTED_COUNTS, file_digest, fingerprint_records, json_digest,
    load_subset_manifest, reference_ids, verify_study_inputs,
)
from utils import save_json


SETTINGS = {"F00": (0, 0), "F20": (2, 0), "F02": (0, 2), "F22": (2, 2)}
SOURCE_FILES = ["config.py", "utils.py", "train.py", "evaluate.py", "inference.py",
                "diagnose_modalities.py", "run_finetune_study.py"]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def runtime_signature():
    paths = [Path(name) for name in SOURCE_FILES]
    paths += sorted(Path("data").glob("*.py")) + sorted(Path("models").glob("*.py"))
    return {
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "source_sha256": json_digest({path.as_posix(): file_digest(path) for path in paths}),
    }


def prepare_subset(args):
    reference = read_json(args.reference_manifest) if args.reference_manifest else None
    audit = read_json(args.subset_reference) if args.subset_reference else None
    config = Config()
    config.run_dir = (Path(args.study_dir) / "runs").as_posix()
    for prefix in ("text", "vision"):
        if reference is not None:
            backbone = reference["backbones"][prefix]
            setattr(config, f"{prefix}_model", backbone["name"])
            setattr(config, f"{prefix}_model_revision", backbone["revision"])
            if not backbone["revision"]:
                raise ValueError("The reference must pin both pretrained backbone revisions")
        else:
            backbone_config = AutoConfig.from_pretrained(
                getattr(config, f"{prefix}_model"),
                revision=getattr(config, f"{prefix}_model_revision"),
            )
            revision = getattr(backbone_config, "_commit_hash", None)
            if not revision:
                raise ValueError(f"Cannot resolve an immutable pretrained revision for {prefix}")
            setattr(config, f"{prefix}_model_revision", revision)
    for field in ("data_root", "retrieved_text_dir"):
        if getattr(args, field):
            setattr(config, field, getattr(args, field))
    processor = CLIPImageProcessor.from_pretrained(config.vision_model, revision=config.vision_model_revision)
    ids, records = {}, {}
    for split, count in EXPECTED_COUNTS.items():
        dataset = MochegDataset(
            config.data_root, split, image_processor=processor,
            image_size=config.image_size, retrieved_text_dir=config.retrieved_text_dir,
        )
        ids[split] = [str(sample["claim_id"]) for sample in dataset.samples]
        if len(ids[split]) != count or len(ids[split]) != len(set(ids[split])):
            raise ValueError(f"{split} must contain {count} unique complete records")
        if audit is not None and ids[split] != reference_ids(audit, split):
            raise ValueError(f"{split} differs from the audited complete subset")
        if reference is not None and split in reference["subset_claim_ids"] and ids[split] != reference_ids(reference, split):
            raise ValueError(f"{split} differs from the logged v2 run")
        records[split] = fingerprint_records(dataset.samples)
        print(f"Locked {split}: {len(ids[split])} claims")
    if any(set(ids[a]) & set(ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Claim IDs overlap between splits")
    manifest = {
        "subset_manifest_version": 1, "architecture_version": 2,
        "config": config.to_dict(), "subset_claim_ids": ids, "subset_records": records,
        "image_processor_config": processor.to_dict(),
        "reference_manifest_sha256": file_digest(args.reference_manifest) if reference is not None else None,
        "subset_reference_sha256": file_digest(args.subset_reference) if audit is not None else None,
    }
    path = Path(args.study_dir) / "subset_manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("The prepared subset changed; use a new study directory")
    else:
        save_json(manifest, path)
    print(f"Subset manifest: {path}")


def select_alternative(screen_results):
    candidates = [name for name in ("F00", "F20", "F02")]
    return min(candidates, key=lambda name: (
        -screen_results[name]["macro_f1"], screen_results[name]["loss"],
        screen_results[name]["trainable_parameters"], candidates.index(name),
    ))


def summarize_confirmation(results, alternative):
    summary = {"alternative": alternative, "seeds": [42, 43, 44], "settings": {}}
    for name in ("F22", alternative):
        rows = [results[name][seed] for seed in (42, 43, 44)]
        scalars = ("macro_f1", "accuracy", "loss", "nei_recall")
        summary["settings"][name] = {
            metric: {"mean": float(np.mean([row[metric] for row in rows])),
                     "std": float(np.std([row[metric] for row in rows], ddof=1))}
            for metric in scalars
        }
        summary["settings"][name]["per_seed"] = rows
    deltas = [results[alternative][seed]["macro_f1"] - results["F22"][seed]["macro_f1"]
              for seed in (42, 43, 44)]
    summary["paired_macro_f1_alternative_minus_F22"] = {
        "per_seed": deltas, "mean": float(np.mean(deltas)), "std": float(np.std(deltas, ddof=1)),
    }
    summary["interpretation"] = "Full factorial interaction is exploratory (screening seed 42 only)"
    return summary


def run_command(arguments):
    subprocess.run([sys.executable] + arguments, check=True)


def read_run_result(path):
    metrics = read_json(path / "validation_best_metrics.json")
    manifest = read_json(path / "manifest.json")
    return {
        "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "loss": metrics["loss"],
        "nei_recall": metrics["per_class"]["not enough information"]["recall"],
        "per_class": metrics["per_class"], "by_source": metrics["by_source"],
        "trainable_parameters": manifest["trainable_parameters"], "seed": manifest["seed"],
    }


def run_stage(args):
    study = Path(args.study_dir)
    lock_path = study / "subset_manifest.json"
    manifest = load_subset_manifest(lock_path)
    config = Config.from_dict(manifest["config"])
    if args.stage == "smoke" and (
        not torch.cuda.is_available()
        or torch.device(args.device).type != "cuda"
        or torch.cuda.get_device_properties(torch.device(args.device)).total_memory / 2**30 < 47
    ):
        raise ValueError("Study smoke requires a CUDA GPU with approximately 48 GB VRAM; batch remains 8")
    verify_study_inputs(config, manifest)
    runtime_path = study / "runtime.json"
    runtime = runtime_signature()
    runtime["device"] = args.device
    if runtime_path.exists():
        if read_json(runtime_path) != runtime:
            raise ValueError("Study source or torch/transformers versions changed; use a new study directory")
    elif args.stage == "smoke":
        save_json(runtime, runtime_path)
    else:
        raise ValueError("Run the smoke stage before training")
    paths = {name: Path(config.run_dir) / f"{name}_seed42" for name in SETTINGS}

    def train_run(name, seed, smoke=False):
        suffix = "_smoke" if smoke else ""
        run_name = f"{name}_seed{seed}{suffix}"
        path = Path(config.run_dir) / run_name
        complete = "smoke_metrics.json" if smoke else "train_best_clean_metrics.json"
        if path.exists():
            if not (path / complete).exists():
                raise ValueError(f"Incomplete run {path}; keep its artifacts and use a new study directory")
        else:
            text, vision = SETTINGS[name]
            command = ["train.py", "--subset-manifest", str(lock_path), "--seed", str(seed),
                       "--text-finetune-layers", str(text), "--vision-finetune-layers", str(vision),
                       "--run-name", run_name, "--device", args.device]
            if smoke:
                command += ["--smoke-test", "--smoke-steps", "3"]
            run_command(command)
        if not smoke:
            diagnostic = study / "diagnostics" / run_name
            if not (diagnostic / "summary.json").exists():
                run_command([
                    "diagnose_modalities.py", "--checkpoint", str(path / "best.pt"),
                    "--manifest", str(lock_path), "--split", "val", "--output-dir", str(diagnostic),
                    "--donor-maps-dir", str(study / "donor_maps"), "--device", args.device,
                ])
        return path

    if args.stage == "smoke":
        for name in SETTINGS:
            path = train_run(name, 42, smoke=True)
            if not read_json(path / "smoke_metrics.json")["target_batch_8_verified"]:
                raise ValueError(f"Target smoke verification failed for {name}")
        return
    for name in SETTINGS:
        smoke = Path(config.run_dir) / f"{name}_seed42_smoke" / "smoke_metrics.json"
        if not smoke.exists() or not read_json(smoke)["target_batch_8_verified"]:
            raise ValueError(f"A passing 48 GB / batch 8 smoke is required for {name}")
    if args.stage == "screen":
        results = {name: read_run_result(train_run(name, 42)) for name in SETTINGS}
        save_json(results, study / "screening.json")
        return
    screen = {name: read_run_result(path) for name, path in paths.items()}
    selection_path = study / "selection.json"
    selection = {
        "alternative": select_alternative(screen), "screening": screen,
        "rule": "best clean validation macro-F1, lower CE, fewer parameters, F00/F20/F02 order",
    }
    if selection_path.exists():
        if read_json(selection_path) != selection:
            raise ValueError("Locked screening selection changed")
    elif args.stage == "confirm":
        save_json(selection, selection_path)
    else:
        raise ValueError("Confirm and lock the validation decision before test evaluation")
    alternative = selection["alternative"]
    if args.stage == "confirm":
        results = {name: {42: screen[name]} for name in ("F22", alternative)}
        for seed in (43, 44):
            for name in ("F22", alternative):
                results[name][seed] = read_run_result(train_run(name, seed))
        save_json(summarize_confirmation(results, alternative), study / "confirmation.json")
        return
    if not (study / "confirmation.json").exists():
        raise ValueError("Complete confirmation before touching test predictions")
    results = {name: {} for name in ("F22", alternative)}
    for seed in (42, 43, 44):
        for name in results:
            path = Path(config.run_dir) / f"{name}_seed{seed}"
            output = path / "test_metrics.json"
            if not output.exists():
                run_command(["evaluate.py", "--checkpoint", str(path / "best.pt"), "--split", "test",
                             "--subset-manifest", str(lock_path), "--output", str(output), "--device", args.device])
            metrics = read_json(output)
            results[name][seed] = {
                "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "loss": metrics["loss"],
                "nei_recall": metrics["per_class"]["not enough information"]["recall"],
                "per_class": metrics["per_class"], "by_source": metrics["by_source"], "seed": seed,
            }
    save_json(summarize_confirmation(results, alternative), study / "test_comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "smoke", "screen", "confirm", "test"))
    parser.add_argument("--study-dir", default="outputs/studies/modality_finetune")
    parser.add_argument("--reference-manifest", help="Optional old-run manifest to reuse pretrained revisions and verify IDs")
    parser.add_argument("--subset-reference", help="Optional subset audit to verify ordered IDs")
    parser.add_argument("--data-root")
    parser.add_argument("--retrieved-text-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_subset(args)
    else:
        if args.data_root or args.retrieved_text_dir:
            parser.error("Data paths are set during prepare and locked for subsequent stages")
        run_stage(args)


if __name__ == "__main__":
    main()
