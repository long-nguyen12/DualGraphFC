"""Report claim-level evidence availability across the full MOCHEG dataset."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from data.dataset_mocheg import MochegDataset


LABELS = ("supported", "refuted", "not enough information")
TEXT_EVIDENCE_SOURCES = ("original", "retrieved", "missing")
DETAIL_FIELDS = (
    "split",
    "claim_id",
    "label",
    "claim",
    "text_evidence_count",
    "text_evidence_source",
    "image_file_count",
    "missing_evidence",
)


def normalize_label(value):
    label = value.strip().lower()
    if label == "nei":
        label = "not enough information"
    if label not in LABELS:
        raise ValueError(f"Unknown MOCHEG label: {label!r}")
    return label


def claim_row(sample):
    text_count = len(sample["text_evidence"])
    image_count = len(sample["images"])
    text_source = sample.get(
        "text_evidence_source", "original" if text_count else "missing"
    )
    if text_count == 0 and image_count == 0:
        missing = "text_and_image"
    elif text_count == 0:
        missing = "text"
    elif image_count == 0:
        missing = "image"
    else:
        missing = "none"
    return {
        "split": sample["split"],
        "claim_id": sample["claim_id"],
        "label": normalize_label(sample["cleaned_truthfulness"]),
        "claim": sample["claim"],
        "text_evidence_count": text_count,
        "text_evidence_source": text_source,
        "image_file_count": image_count,
        "missing_evidence": missing,
    }


def summarize(samples):
    labels = Counter()
    text_sources = Counter()
    text_presence = {False: Counter(), True: Counter()}
    by_label = {
        label: {
            "no_text": 0,
            "no_images": 0,
            "text_items": 0,
            "images": 0,
            "text_sources": Counter(),
        }
        for label in LABELS
    }
    neither = 0

    for sample in samples:
        label = normalize_label(sample["cleaned_truthfulness"])

        text_count = len(sample["text_evidence"])
        image_count = len(sample["images"])
        has_text = text_count > 0
        text_source = sample.get(
            "text_evidence_source", "original" if has_text else "missing"
        )

        labels[label] += 1
        text_sources[text_source] += 1
        text_presence[has_text][label] += 1
        by_label[label]["text_sources"][text_source] += 1
        by_label[label]["no_text"] += not has_text
        by_label[label]["no_images"] += image_count == 0
        by_label[label]["text_items"] += text_count
        by_label[label]["images"] += image_count
        neither += not has_text and image_count == 0

    return labels, text_sources, text_presence, by_label, neither


def print_summary(name, samples):
    labels, text_sources, text_presence, by_label, neither = summarize(samples)
    total = len(samples)
    print(f"\n{name}: {total:,} claims")
    print(
        f"{'Label':<24} {'Claims':>8} {'Share':>7} "
        f"{'No text':>9} {'No text %':>10} {'Mean text':>10} "
        f"{'No images':>10} {'Mean images':>12}"
    )
    for label in LABELS:
        count = labels[label]
        stats = by_label[label]
        print(
            f"{label:<24} {count:>8,} {count / total:>6.1%} "
            f"{stats['no_text']:>9,} {stats['no_text'] / count:>9.1%} "
            f"{stats['text_items'] / count:>10.2f} "
            f"{stats['no_images']:>10,} {stats['images'] / count:>12.2f}"
        )

    print(f"No text and no images: {neither:,} ({neither / total:.1%})")
    source_summary = ", ".join(
        f"{source}: {text_sources[source]:,}" for source in TEXT_EVIDENCE_SOURCES
    )
    print(f"Text evidence source: {source_summary}")
    print("Label distribution conditional on text evidence:")
    for has_text, description in ((False, "Absent"), (True, "Present")):
        group = text_presence[has_text]
        group_total = sum(group.values())
        distribution = ", ".join(
            f"{label}: {group[label]:,} ({group[label] / group_total:.1%})"
            for label in LABELS
        ) if group_total else "no claims"
        print(f"  {description} ({group_total:,}): {distribution}")


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = [sample for samples in splits.values() for sample in samples]
    detail_rows = [claim_row(sample) for sample in all_samples]
    detail_path = output_dir / "mocheg_claims.csv"
    missing_path = output_dir / "mocheg_missing_evidence.csv"
    summary_path = output_dir / "mocheg_summary.csv"

    write_csv(detail_path, detail_rows, DETAIL_FIELDS)
    write_csv(
        missing_path,
        (row for row in detail_rows if row["missing_evidence"] != "none"),
        DETAIL_FIELDS,
    )

    summary_rows = []
    for split, samples in (*splits.items(), ("all splits", all_samples)):
        labels, _, text_presence, by_label, neither = summarize(samples)
        for label in LABELS:
            count = labels[label]
            stats = by_label[label]
            absent_total = sum(text_presence[False].values())
            present_total = sum(text_presence[True].values())
            summary_rows.append(
                {
                    "split": split,
                    "label": label,
                    "claim_count": count,
                    "split_claim_count": len(samples),
                    "label_share_pct": round(100 * count / len(samples), 2),
                    "original_text_claim_count": stats["text_sources"]["original"],
                    "retrieved_text_claim_count": stats["text_sources"]["retrieved"],
                    "no_text_count": stats["no_text"],
                    "no_text_pct_within_label": round(100 * stats["no_text"] / count, 2),
                    "mean_text_evidence": round(stats["text_items"] / count, 2),
                    "no_image_count": stats["no_images"],
                    "mean_image_files": round(stats["images"] / count, 2),
                    "no_text_and_no_image_count_in_split": neither,
                    "label_pct_given_no_text": (
                        round(100 * text_presence[False][label] / absent_total, 2)
                        if absent_total else ""
                    ),
                    "label_pct_given_text": (
                        round(100 * text_presence[True][label] / present_total, 2)
                        if present_total else ""
                    ),
                }
            )
    write_csv(summary_path, summary_rows, summary_rows[0].keys())
    return detail_path, missing_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all MOCHEG claims by label and evidence availability"
    )
    parser.add_argument(
        "--data-root",
        default="dataset/mocheg",
        help="Directory containing train/, val/, and test/",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis",
        help="Directory for detailed, missing-evidence, and summary CSV files",
    )
    parser.add_argument(
        "--retrieved-text-dir",
        help=(
            "Directory containing split-specific retrieved-text CSV files; "
            "defaults to <data-root>/retrieved_text when that directory exists"
        ),
    )
    args = parser.parse_args()

    retrieved_text_dir = args.retrieved_text_dir
    if retrieved_text_dir is None:
        default_retrieved_text_dir = Path(args.data_root) / "retrieved_text"
        if default_retrieved_text_dir.is_dir():
            retrieved_text_dir = default_retrieved_text_dir

    dataset = MochegDataset(
        args.data_root,
        retrieved_text_dir=retrieved_text_dir,
    )
    if dataset.retrieved_text_dir is not None:
        print(f"Using retrieved text from {dataset.retrieved_text_dir}")
    splits = {split: dataset.load_split(split) for split in dataset.SPLITS}
    for split, samples in splits.items():
        print_summary(split, samples)
    print_summary("all splits", [sample for samples in splits.values() for sample in samples])
    for path in write_reports(splits, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
