"""Retrieve text evidence from the full local MOCHEG sentence corpus.

Corpus and claim texts are encoded with a sentence transformer. FAISS performs
exact dense retrieval on the GPU, and a cross-encoder reranks the candidates.
Labels, ruling outlines, and Origin fields from Corpus2 are never used.
"""

import argparse
import csv
import gc
import json
import mmap
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from data.dataset_mocheg import MochegDataset


INDEX_FORMAT_VERSION = 1
OUTPUT_FIELDS = (
    "split",
    "claim_id",
    "rank",
    "score",
    "dense_score",
    "corpus_id",
    "source_claim_id",
    "relevant_document_id",
    "paragraph_id",
    "source_row_id",
    "text",
)


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for dense retrieval. On the Linux GPU server, "
            "install it with: conda install -c pytorch -c nvidia "
            "-c conda-forge faiss-gpu"
        ) from exc
    return faiss


def _set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _missing_claims(loader, splits):
    targets = {}
    for split in splits:
        for sample in loader.load_split(split):
            if sample["text_evidence"]:
                continue
            claim_id = sample["claim_id"]
            if claim_id in targets:
                raise ValueError(f"Claim {claim_id!r} occurs in multiple splits")
            targets[claim_id] = {
                "split": split,
                "claim": sample["claim"],
            }
    return targets


def _corpus_signature(corpus_path):
    stat = corpus_path.stat()
    return {
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _index_paths(index_dir):
    return {
        "index": index_dir / "sentences.faiss",
        "metadata": index_dir / "sentences.jsonl",
        "offsets": index_dir / "sentences.offsets.i64",
        "manifest": index_dir / "manifest.json",
    }


def _load_manifest(path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_is_current(paths, corpus_path, encoder_model):
    manifest = _load_manifest(paths["manifest"])
    expected = {
        "format_version": INDEX_FORMAT_VERSION,
        "corpus_signature": _corpus_signature(corpus_path),
        "encoder_model": encoder_model,
        "normalized": True,
    }
    return (
        manifest is not None
        and all(manifest.get(name) == value for name, value in expected.items())
        and all(paths[name].is_file() for name in ("index", "metadata", "offsets"))
    )


def _normalize_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("The sentence encoder must return a 2-D embedding array")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The sentence encoder produced a zero-length embedding")
    return np.ascontiguousarray(embeddings / norms)


def _encode(encoder, texts, batch_size, show_progress_bar=False):
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return _normalize_embeddings(embeddings)


def _write_index_batch(
    records,
    encoder,
    encoder_batch_size,
    index,
    metadata_handle,
    offsets_handle,
):
    if not records:
        return
    embeddings = _encode(
        encoder,
        [record["text"] for record in records],
        encoder_batch_size,
    )
    if embeddings.shape[1] != index.d:
        raise ValueError(
            f"Encoder width changed from {index.d} to {embeddings.shape[1]}"
        )
    index.add(embeddings)

    offsets = np.empty(len(records), dtype="<i8")
    for position, record in enumerate(records):
        offsets[position] = metadata_handle.tell()
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        metadata_handle.write(line + b"\n")
    offsets_handle.write(offsets.tobytes())


def build_dense_index(
    corpus_path,
    index_dir,
    encoder_model,
    device,
    encoder_batch_size,
    corpus_chunk_size,
    *,
    rebuild=False,
    faiss_module=None,
    encoder=None,
):
    """Build or reuse the exact normalized inner-product corpus index."""

    paths = _index_paths(index_dir)
    if not rebuild and _index_is_current(paths, corpus_path, encoder_model):
        print(f"Reusing dense index {paths['index']}")
        return paths, _load_manifest(paths["manifest"])

    faiss = faiss_module or _import_faiss()
    index_dir.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        encoder = SentenceTransformer(encoder_model, device=str(device))
    dimension = encoder.get_sentence_embedding_dimension()
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Could not determine the sentence-embedding width")
    index = faiss.IndexFlatIP(dimension)

    temporary = {
        name: path.with_name(f"{path.name}.tmp") for name, path in paths.items()
    }
    for path in temporary.values():
        if path.exists():
            path.unlink()

    _set_csv_field_limit()
    pending = []
    source_row_id = 0
    try:
        with corpus_path.open("r", encoding="utf-8-sig", newline="") as corpus:
            reader = csv.DictReader(corpus)
            required = {
                "claim_id",
                "relevant_document_id",
                "paragraph_id",
                "corpus_id",
                "paragraph",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Sentence corpus {corpus_path} must contain: "
                    f"{', '.join(sorted(required))}"
                )

            with temporary["metadata"].open("wb") as metadata_handle, temporary[
                "offsets"
            ].open("wb") as offsets_handle:
                for row in tqdm(reader, desc="Encoding corpus", unit="sentence"):
                    text = row["paragraph"].strip()
                    if not text:
                        continue
                    pending.append(
                        {
                            "source_row_id": source_row_id,
                            "source_claim_id": row["claim_id"],
                            "relevant_document_id": row["relevant_document_id"],
                            "paragraph_id": row["paragraph_id"],
                            "corpus_id": row["corpus_id"],
                            "text": text,
                        }
                    )
                    source_row_id += 1
                    if len(pending) == corpus_chunk_size:
                        _write_index_batch(
                            pending,
                            encoder,
                            encoder_batch_size,
                            index,
                            metadata_handle,
                            offsets_handle,
                        )
                        pending = []
                _write_index_batch(
                    pending,
                    encoder,
                    encoder_batch_size,
                    index,
                    metadata_handle,
                    offsets_handle,
                )

        if index.ntotal != source_row_id:
            raise RuntimeError(
                f"FAISS contains {index.ntotal} vectors for {source_row_id} records"
            )
        faiss.write_index(index, str(temporary["index"]))
        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "corpus_signature": _corpus_signature(corpus_path),
            "encoder_model": encoder_model,
            "normalized": True,
            "count": source_row_id,
            "dimension": dimension,
        }
        with temporary["manifest"].open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

        for name in ("index", "metadata", "offsets"):
            os.replace(temporary[name], paths[name])
        os.replace(temporary["manifest"], paths["manifest"])
        print(f"Indexed {source_row_id:,} sentences in {paths['index']}")
        return paths, manifest
    except BaseException:
        for path in temporary.values():
            if path.exists():
                path.unlink()
        raise


class MetadataStore:
    """Random-access reader for FAISS row metadata."""

    def __init__(self, metadata_path, offsets_path, expected_count):
        self.offsets = np.memmap(offsets_path, mode="r", dtype="<i8")
        if len(self.offsets) != expected_count:
            raise ValueError(
                f"Metadata has {len(self.offsets)} offsets; expected {expected_count}"
            )
        self.handle = metadata_path.open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, row_id):
        if row_id < 0 or row_id >= len(self.offsets):
            raise IndexError(f"FAISS returned invalid row ID {row_id}")
        start = int(self.offsets[row_id])
        end = self.mapping.find(b"\n", start)
        if end < 0:
            end = len(self.mapping)
        return json.loads(self.mapping[start:end])

    def close(self):
        self.mapping.close()
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def dense_search(
    index_path,
    manifest,
    query_embeddings,
    candidate_k,
    device,
    *,
    faiss_module=None,
):
    """Return exact cosine-search scores and corpus row IDs."""

    faiss = faiss_module or _import_faiss()
    cpu_index = faiss.read_index(str(index_path))
    if cpu_index.ntotal != manifest["count"] or cpu_index.d != manifest["dimension"]:
        raise ValueError("FAISS index does not match its manifest; rebuild the index")
    if candidate_k > cpu_index.ntotal:
        raise ValueError(
            f"candidate_k={candidate_k} exceeds corpus size {cpu_index.ntotal}"
        )

    resources = None
    search_index = cpu_index
    if device.type == "cuda":
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "The installed FAISS build has no GPU support. Install faiss-gpu "
                "in the server environment."
            )
        resources = faiss.StandardGpuResources()
        gpu_id = 0 if device.index is None else device.index
        search_index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)

    scores, row_ids = search_index.search(
        np.ascontiguousarray(query_embeddings, dtype=np.float32),
        candidate_k,
    )
    del search_index, cpu_index, resources
    gc.collect()
    return scores, row_ids


def _reranker_scores(reranker, claim, records, batch_size):
    pairs = [(claim, record["text"]) for record in records]
    scores = np.asarray(
        reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    )
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 1 or len(scores) != len(records):
        raise ValueError(
            "The cross-encoder must produce one relevance score per text pair"
        )
    return scores


def rerank_and_write(
    output_dir,
    splits,
    targets,
    dense_scores,
    row_ids,
    metadata_store,
    reranker,
    reranker_batch_size,
    top_k,
):
    """Rerank dense candidates and write split-specific evidence CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    claim_ids = list(targets)
    counts = {split: 0 for split in splits}
    with ExitStack() as stack:
        writers = {}
        for split in splits:
            handle = stack.enter_context(
                (output_dir / f"{split}.csv").open(
                    "w", encoding="utf-8", newline=""
                )
            )
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writers[split] = writer

        for query_index, claim_id in enumerate(
            tqdm(claim_ids, desc="Reranking text", unit="claim")
        ):
            candidate_ids = row_ids[query_index].tolist()
            records = [metadata_store.get(row_id) for row_id in candidate_ids]
            scores = _reranker_scores(
                reranker,
                targets[claim_id]["claim"],
                records,
                reranker_batch_size,
            )
            order = np.argsort(-scores, kind="stable")[:top_k]
            split = targets[claim_id]["split"]
            for rank, candidate_index in enumerate(order.tolist(), start=1):
                record = records[candidate_index]
                writers[split].writerow(
                    {
                        "split": split,
                        "claim_id": claim_id,
                        "rank": rank,
                        "score": f"{float(scores[candidate_index]):.8f}",
                        "dense_score": (
                            f"{float(dense_scores[query_index, candidate_index]):.8f}"
                        ),
                        "corpus_id": record["corpus_id"],
                        "source_claim_id": record["source_claim_id"],
                        "relevant_document_id": record[
                            "relevant_document_id"
                        ],
                        "paragraph_id": record["paragraph_id"],
                        "source_row_id": record["source_row_id"],
                        "text": record["text"],
                    }
                )
            counts[split] += 1

    for split in splits:
        print(f"Saved {counts[split]} claims to {output_dir / f'{split}.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve and rerank local MOCHEG text evidence with FAISS"
    )
    parser.add_argument("--data-root", default="dataset/mocheg")
    parser.add_argument(
        "--sentence-corpus",
        help="Defaults to <data-root>/supplementary/Corpus3_sentence_level.csv",
    )
    parser.add_argument(
        "--index-dir",
        help="Defaults to <data-root>/retrieval/faiss",
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to <data-root>/retrieved_text",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=MochegDataset.SPLITS,
        help="Split to process; repeat as needed (default: train)",
    )
    parser.add_argument(
        "--encoder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--corpus-chunk-size", type=int, default=8192)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in (
        "candidate_k",
        "top_k",
        "encoder_batch_size",
        "corpus_chunk_size",
        "reranker_batch_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be greater than or equal to --top-k")

    data_root = Path(args.data_root).expanduser().resolve()
    corpus_path = (
        Path(args.sentence_corpus).expanduser().resolve()
        if args.sentence_corpus
        else data_root / "supplementary" / "Corpus3_sentence_level.csv"
    )
    index_dir = (
        Path(args.index_dir).expanduser().resolve()
        if args.index_dir
        else data_root / "retrieval" / "faiss"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_root / "retrieved_text"
    )
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")

    splits = tuple(args.split or ("train",))
    targets = _missing_claims(MochegDataset(data_root), splits)
    print(f"Claims without original text evidence: {len(targets):,}")
    if not targets:
        rerank_and_write(
            output_dir,
            splits,
            targets,
            np.empty((0, 0)),
            np.empty((0, 0), dtype=np.int64),
            None,
            None,
            args.reranker_batch_size,
            args.top_k,
        )
        return

    faiss = _import_faiss()
    if device.type == "cuda" and not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError(
            "The installed FAISS build has no GPU support. Install faiss-gpu "
            "in the server environment."
        )
    encoder = SentenceTransformer(args.encoder_model, device=str(device))
    paths, manifest = build_dense_index(
        corpus_path,
        index_dir,
        args.encoder_model,
        device,
        args.encoder_batch_size,
        args.corpus_chunk_size,
        rebuild=args.rebuild_index,
        faiss_module=faiss,
        encoder=encoder,
    )
    claim_ids = list(targets)
    query_embeddings = _encode(
        encoder,
        [targets[claim_id]["claim"] for claim_id in claim_ids],
        args.encoder_batch_size,
        show_progress_bar=True,
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dense_scores, row_ids = dense_search(
        paths["index"],
        manifest,
        query_embeddings,
        args.candidate_k,
        device,
        faiss_module=faiss,
    )
    del query_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reranker = CrossEncoder(args.reranker_model, device=str(device))
    with MetadataStore(
        paths["metadata"],
        paths["offsets"],
        manifest["count"],
    ) as metadata_store:
        rerank_and_write(
            output_dir,
            splits,
            targets,
            dense_scores,
            row_ids,
            metadata_store,
            reranker,
            args.reranker_batch_size,
            args.top_k,
        )


if __name__ == "__main__":
    main()
