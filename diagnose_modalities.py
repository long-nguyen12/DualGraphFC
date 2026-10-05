"""Paired evidence/image interventions on a fixed v2 checkpoint."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from data.dataset import load_rgb_image
from data.subset import (
    json_digest, load_subset_manifest, normalized_evidence_hash, verify_study_config,
)
from data.text_normalization import normalize_text
from evaluate import (
    build_dataloader, compute_metrics, load_checkpoint, read_checkpoint, source_domain,
)
from utils import autocast_context, move_batch_to_device, resolve_device, save_json


DONOR_SEEDS = (1001, 1002, 1003, 1004, 1005)
CONDITIONS = ("DE", "DI", "DEI")


def describe_samples(dataset, tokenizer, locked_records, max_length):
    """Tokenize real evidence; keep tokens in memory, never as backbone features."""
    descriptions = {}
    for sample, locked in zip(dataset.samples, locked_records):
        claim = normalize_text(sample["claim"])
        evidence = [normalize_text(text) for text in sample["text_evidence"]]
        pairs = tokenizer(
            evidence, text_pair=[claim] * len(evidence), padding=False,
            truncation="only_first", max_length=max_length,
        )
        tokens = tokenizer(evidence, add_special_tokens=False, truncation=False)["input_ids"]
        visible_lengths = []
        for index, donor_tokens in enumerate(tokens):
            visible = [token for token, sequence_id in zip(
                pairs["input_ids"][index], pairs.sequence_ids(index)
            ) if sequence_id == 0]
            if not visible or visible != donor_tokens[:len(visible)]:
                raise ValueError(f"Evidence prefix cannot be isolated for claim {sample['claim_id']}")
            visible_lengths.append(len(visible))
        claim_id = str(sample["claim_id"])
        descriptions[claim_id] = {
            "claim_id": claim_id, "claim_hash": json_digest(claim),
            "source": source_domain(sample), "fingerprint": locked["fingerprint"],
            "tokens": tokens, "visible_lengths": visible_lengths,
            "evidence_hashes": [normalized_evidence_hash(text) for text in evidence],
            "evidence_ids": sample["text_evidence_ids"],
            "image_hashes": locked["image_hashes"],
            "image_ids": sample["image_evidence_ids"], "image_paths": sample["images"],
        }
    return descriptions


def make_donor_mapping(descriptions, seed):
    rng = random.Random(seed)
    mappings = {}
    # Sort by ID so mapping construction is independent of loader/batch order.
    rows = [descriptions[key] for key in sorted(descriptions)]
    for recipient in rows:
        candidates = [donor for donor in rows if (
            donor["claim_id"] != recipient["claim_id"]
            and donor["claim_hash"] != recipient["claim_hash"]
            and donor["source"] == recipient["source"]
        )]
        text_donors = []
        for index, length in enumerate(recipient["visible_lengths"]):
            eligible = [(donor, evidence_index) for donor in candidates
                        for evidence_index, tokens in enumerate(donor["tokens"])
                        if len(tokens) >= length
                        and donor["evidence_hashes"][evidence_index] not in recipient["evidence_hashes"]
                        and tokens[:length] != recipient["tokens"][index][:length]]
            if not eligible:
                text_donors = []
                break
            donor, evidence_index = rng.choice(eligible)
            text_donors.append({
                "claim_id": donor["claim_id"], "evidence_index": evidence_index,
                "evidence_id": donor["evidence_ids"][evidence_index], "visible_length": length,
                "normalized_sha256": donor["evidence_hashes"][evidence_index],
                "token_sha256": json_digest(donor["tokens"][evidence_index][:length]),
            })
        image_candidates = [donor for donor in candidates
                            if len(donor["image_hashes"]) == len(recipient["image_hashes"])
                            and not set(donor["image_hashes"]) & set(recipient["image_hashes"])]
        donor = rng.choice(image_candidates) if image_candidates else None
        mappings[recipient["claim_id"]] = {
            "recipient_fingerprint": recipient["fingerprint"],
            "text_eligible": bool(text_donors), "image_eligible": donor is not None,
            "evidence_donors": text_donors,
            "image_donor": None if donor is None else {
                "claim_id": donor["claim_id"], "image_ids": donor["image_ids"],
                "image_hashes": donor["image_hashes"],
            },
        }
    return {
        "mapping_version": 1, "seed": seed,
        "input_signature": json_digest([
            {"claim_id": row["claim_id"], "fingerprint": row["fingerprint"],
             "tokens": row["tokens"], "visible_lengths": row["visible_lengths"]}
            for row in rows
        ]),
        "records": mappings,
    }


def replace_modalities(batch, mapping, descriptions, image_transform, condition):
    result = dict(batch)
    if condition in ("DE", "DEI"):
        result["input_ids"] = batch["input_ids"].clone()
    if condition in ("DI", "DEI"):
        result["images"] = batch["images"].clone()
    for batch_index, metadata in enumerate(batch["metadata"]):
        claim_id = str(metadata["claim_id"])
        intervention = mapping["records"][claim_id]
        if condition in ("DE", "DEI") and intervention["text_eligible"]:
            for node_index, donor in enumerate(intervention["evidence_donors"], start=1):
                mask = batch["evidence_token_mask"][batch_index, node_index]
                length = int(mask.sum())
                if length != donor["visible_length"]:
                    raise ValueError(f"Evidence length changed for claim {claim_id}, node {node_index}")
                tokens = descriptions[donor["claim_id"]]["tokens"][donor["evidence_index"]][:length]
                result["input_ids"][batch_index, node_index, mask] = torch.tensor(
                    tokens, dtype=result["input_ids"].dtype, device=result["input_ids"].device,
                )
        if condition in ("DI", "DEI") and intervention["image_eligible"]:
            donor = descriptions[intervention["image_donor"]["claim_id"]]
            indices = batch["image_mask"][batch_index].nonzero(as_tuple=False).flatten()
            if len(indices) != len(donor["image_paths"]):
                raise ValueError(f"Image count changed for claim {claim_id}")
            for image_index, path in zip(indices.tolist(), donor["image_paths"]):
                try:
                    pixels = image_transform(load_rgb_image(path))
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"Approved donor image for claim {donor['claim_id']} cannot be read: {path}") from exc
                result["images"][batch_index, image_index] = pixels.to(result["images"].device)
    return result


@torch.no_grad()
def predict_condition(model, dataloader, device, condition="D0", mapping=None, descriptions=None):
    model.eval()
    if dataloader.collate_fn.augment:
        raise ValueError("Diagnostic evaluation must disable evidence dropout")
    records = []
    for batch in tqdm(dataloader, desc=condition, unit="batch"):
        if condition != "D0":
            batch = replace_modalities(
                batch, mapping, descriptions, dataloader.dataset.image_transform, condition,
            )
        batch = move_batch_to_device(batch, device)
        with autocast_context(device):
            logits = model(batch)
        if not torch.isfinite(logits).all():
            raise ValueError("Diagnostic logits are not finite")
        logits = logits.float()
        probabilities = logits.softmax(-1)
        losses = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none")
        for index, metadata in enumerate(batch["metadata"]):
            label = int(batch["labels"][index])
            claim_id = str(metadata["claim_id"])
            donor = None if mapping is None else mapping["records"][claim_id]
            records.append({
                "claim_id": claim_id, "label_id": label,
                "prediction_id": int(logits[index].argmax()),
                "logits": logits[index].cpu().tolist(),
                "probabilities": probabilities[index].cpu().tolist(),
                "true_probability": float(probabilities[index, label]), "ce": float(losses[index]),
                "source": source_domain(metadata),
                "text_evidence_source": metadata["text_evidence_source"],
                "evidence_count": int(batch["text_node_mask"][index].sum()) - 1,
                "image_count": int(batch["image_mask"][index].sum()),
                "condition": condition, "donor_seed": None if mapping is None else mapping["seed"],
                "donors": donor,
            })
    return records


def metrics_for_records(records):
    metrics = compute_metrics(
        [record["label_id"] for record in records], [record["prediction_id"] for record in records],
    )
    metrics["ce"] = float(np.mean([record["ce"] for record in records]))
    metrics["true_probability"] = float(np.mean([record["true_probability"] for record in records]))
    return metrics


def paired_effect(clean, corrupt):
    if [row["claim_id"] for row in clean] != [row["claim_id"] for row in corrupt]:
        raise ValueError("Paired predictions have different claim IDs/order")
    baseline, changed = metrics_for_records(clean), metrics_for_records(corrupt)
    effects = {
        "delta_macro_f1": baseline["macro_f1"] - changed["macro_f1"],
        "delta_ce": changed["ce"] - baseline["ce"],
        "delta_true_probability": baseline["true_probability"] - changed["true_probability"],
        "prediction_flip_rate": float(np.mean([
            a["prediction_id"] != b["prediction_id"] for a, b in zip(clean, corrupt)
        ])),
        "correct_to_wrong": sum(a["prediction_id"] == a["label_id"] and b["prediction_id"] != b["label_id"]
                                for a, b in zip(clean, corrupt)),
        "wrong_to_correct": sum(a["prediction_id"] != a["label_id"] and b["prediction_id"] == b["label_id"]
                                for a, b in zip(clean, corrupt)),
    }
    for name in baseline["per_class"]:
        for metric in ("recall", "f1"):
            effects[f"delta_{name}_{metric}"] = baseline["per_class"][name][metric] - changed["per_class"][name][metric]
    return effects


def average_metrics(metrics):
    result = {
        key: float(np.mean([row[key] for row in metrics]))
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "ce", "true_probability")
    }
    result["per_class"] = {
        name: {key: float(np.mean([row["per_class"][name][key] for row in metrics]))
               for key in ("precision", "recall", "f1", "support")}
        for name in metrics[0]["per_class"]
    }
    return result


def bootstrap_effects(clean, repeats, iterations=1000, seed=2026):
    strata = {}
    for index, row in enumerate(clean):
        strata.setdefault((row["source"], row["label_id"]), []).append(index)
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(iterations):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata.values()])
        baseline = [clean[index] for index in indices]
        effects = [paired_effect(baseline, [repeat[index] for index in indices]) for repeat in repeats]
        for key in effects[0]:
            samples.setdefault(key, []).append(float(np.mean([effect[key] for effect in effects])))
    return {key: np.percentile(values, [2.5, 97.5]).tolist() for key, values in samples.items()}


def summarize_diagnostics(clean, corrupted, mappings, bootstrap_iterations=1000):
    ids = [row["claim_id"] for row in clean]
    for repeats in corrupted.values():
        if any([row["claim_id"] for row in repeat] != ids for repeat in repeats):
            raise ValueError("All diagnostic predictions must cover the same ordered claims")
    eligible = {claim_id for claim_id in ids if all(
        mapping["records"][claim_id]["text_eligible"] and mapping["records"][claim_id]["image_eligible"]
        for mapping in mappings
    )}
    cohort = [row for row in clean if row["claim_id"] in eligible]
    coverage = {}
    for field in ("source", "label_id"):
        coverage[field] = {
            str(value): {"total": sum(row[field] == value for row in clean),
                         "common_eligible": sum(row[field] == value for row in cohort)}
            for value in sorted({row[field] for row in clean})
        }
    summary = {
        "full_sample_count": len(clean), "common_eligible_count": len(cohort),
        "common_eligible_claim_ids": [row["claim_id"] for row in cohort],
        "coverage_by_source_and_label": coverage, "D0_full": metrics_for_records(clean),
        "D0_common": metrics_for_records(cohort) if cohort else None,
        "coverage_per_seed": [{
            "seed": mapping["seed"],
            "text_eligible": sum(row["text_eligible"] for row in mapping["records"].values()),
            "image_eligible": sum(row["image_eligible"] for row in mapping["records"].values()),
        } for mapping in mappings],
        "conditions": {}, "bootstrap_iterations": bootstrap_iterations,
        "interval_scope": "claims stratified by source x label; excludes training-seed uncertainty",
    }
    for condition, repeats in corrupted.items():
        selected = [[row for row in repeat if row["claim_id"] in eligible] for repeat in repeats]
        full_metrics = [metrics_for_records(repeat) for repeat in repeats]
        result = {"full_metrics_per_seed": full_metrics, "mean_full_metrics": average_metrics(full_metrics)}
        if cohort:
            effects = [paired_effect(cohort, repeat) for repeat in selected]
            common_metrics = [metrics_for_records(repeat) for repeat in selected]
            result.update(
                common_metrics_per_seed=common_metrics, mean_common_metrics=average_metrics(common_metrics),
                mean_common_effect={key: float(np.mean([effect[key] for effect in effects])) for key in effects[0]},
                paired_bootstrap_95=bootstrap_effects(cohort, selected, iterations=bootstrap_iterations),
            )
        else:
            result["status"] = "No common eligible cohort; no paired contribution estimate"
        summary["conditions"][condition] = result
    return summary


def main():
    from config import Config
    from models.model import DualGraphFC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Content-locked subset manifest")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--donor-maps-dir", help="Shared mapping directory for all checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--retrieved-text-dir")
    args = parser.parse_args()
    manifest = load_subset_manifest(args.manifest)
    checkpoint = read_checkpoint(args.checkpoint)
    config = Config.from_dict(checkpoint["config"])
    for name in ("data_root", "retrieved_text_dir"):
        if getattr(args, name):
            setattr(config, name, getattr(args, name))
    verify_study_config(config, manifest)
    device = resolve_device(args.device)
    loader, tokenizer = build_dataloader(
        config, args.split, subset_manifest=manifest, diagnostic=True,
        num_workers=config.num_workers,
    )
    descriptions = describe_samples(
        loader.dataset, tokenizer, manifest["subset_records"][args.split], config.max_text_length,
    )
    output = Path(args.output_dir)
    maps_dir = Path(args.donor_maps_dir) if args.donor_maps_dir else output / "donor_maps"
    mappings = []
    for seed in DONOR_SEEDS:
        mapping = make_donor_mapping(descriptions, seed)
        path = maps_dir / f"{args.split}_{seed}.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                if json.load(handle) != mapping:
                    raise ValueError(f"Existing donor mapping differs: {path}")
        else:
            save_json(mapping, path)
        mappings.append(mapping)
    model = DualGraphFC(config).to(device)
    load_checkpoint(model, checkpoint, device)
    clean = predict_condition(model, loader, device)
    save_json(clean, output / "D0_predictions.json")
    corrupted = {condition: [] for condition in CONDITIONS}
    for mapping in mappings:
        for condition in CONDITIONS:
            records = predict_condition(model, loader, device, condition, mapping, descriptions)
            save_json(records, output / f"{condition}_{mapping['seed']}_predictions.json")
            corrupted[condition].append(records)
    summary = summarize_diagnostics(clean, corrupted, mappings)
    summary.update(checkpoint=str(args.checkpoint), checkpoint_epoch=checkpoint["epoch"], split=args.split)
    save_json(summary, output / "summary.json")
    print(f"Diagnostic cohort: {summary['common_eligible_count']}/{summary['full_sample_count']}; output={output}")


if __name__ == "__main__":
    main()
