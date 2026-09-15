"""Dependency-free loader for the local MOCHEG dataset."""

import csv
from pathlib import Path


class MochegDataset:
    """Load MOCHEG rows as claim-level dictionaries."""

    SPLITS = ("train", "val", "test")

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self._splits = {}
        self._claims = {}

    def load_split(self, split, limit=None):
        split = self._check_split(split)
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        if split not in self._splits:
            self._load_split(split)
        samples = self._splits[split]
        if limit is not None:
            samples = samples[:limit]
        return [self._copy(sample) for sample in samples]

    def get_claim(self, claim_id, split=None):
        claim_id = str(claim_id)
        if split is not None:
            split = self._check_split(split)
            if split not in self._splits:
                self._load_split(split)
            return self._copy(self._claims[split][claim_id])

        matches = []
        for candidate_split in self.SPLITS:
            if candidate_split not in self._splits:
                self._load_split(candidate_split)
            sample = self._claims[candidate_split].get(claim_id)
            if sample is not None:
                matches.append(sample)
        if not matches:
            raise KeyError(f"Claim {claim_id!r} was not found in any split")
        if len(matches) > 1:
            locations = ", ".join(sample["split"] for sample in matches)
            raise ValueError(
                f"Claim {claim_id!r} occurs in multiple splits ({locations}); "
                "pass split explicitly"
            )
        return self._copy(matches[0])

    def get_images(self, claim_id, split=None):
        return self.get_claim(claim_id, split=split)["images"]

    def _load_split(self, split):
        path = self.root / split / "Corpus2.csv"
        samples = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                claim_id = row["claim_id"]
                evidence_id = row["evidence_id"]
                if claim_id not in samples:
                    samples[claim_id] = {
                        "claim_id": claim_id,
                        "claim": row["Claim"],
                        "text_evidence": [],
                        "text_evidence_ids": [],
                        "images": [],
                        "image_evidence_ids": [],
                        "cleaned_truthfulness": row["cleaned_truthfulness"],
                        "ruling_outline": row["ruling_outline"],
                        "origin": row["Origin"],
                        "snopes_url": row["Snopes URL"],
                        "split": split,
                    }
                sample = samples[claim_id]
                evidence = row["Evidence"]
                if evidence.strip():
                    sample["text_evidence"].append(evidence)
                    sample["text_evidence_ids"].append(evidence_id)

        image_dir = self.root / split / "images"
        samples_by_numeric_id = {}
        for claim_id, sample in samples.items():
            try:
                samples_by_numeric_id[int(claim_id)] = sample
            except ValueError:
                continue

        image_paths = list(image_dir.iterdir()) if image_dir.is_dir() else []
        for extension in ("jpg", "jpeg", "png"):
            matching_paths = sorted(
                (
                    image_path
                    for image_path in image_paths
                    if image_path.suffix == f".{extension}"
                ),
                key=lambda image_path: image_path.name,
            )
            for image_path in matching_paths:
                prefix = image_path.name.split("-", 1)[0]
                try:
                    numeric_claim_id = int(prefix)
                except ValueError as exc:
                    raise ValueError(
                        f"Image filename {image_path.name!r} does not start with "
                        "a numeric claim ID"
                    ) from exc

                sample = samples_by_numeric_id.get(numeric_claim_id)
                if sample is None:
                    continue
                sample["image_evidence_ids"].append(image_path.name)
                sample["images"].append(str(image_path))

        self._claims[split] = samples
        self._splits[split] = list(samples.values())

    def _check_split(self, split):
        if split not in self.SPLITS:
            expected = ", ".join(self.SPLITS)
            raise ValueError(f"Unknown split {split!r}; expected one of: {expected}")
        return split

    @staticmethod
    def _copy(sample):
        result = dict(sample)
        for key in (
            "text_evidence",
            "text_evidence_ids",
            "images",
            "image_evidence_ids",
        ):
            result[key] = list(sample[key])
        return result
