"""Load DeBERTa tokenization with its pretrained SentencePiece normalization."""

import json

from transformers import AutoTokenizer


def load_text_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.text_model, revision=config.text_model_revision,
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or "deberta-v3" not in str(config.text_model).lower():
        return tokenizer
    specification = json.loads(backend.to_str())
    normalization = specification.get("normalizer") or {}
    steps = normalization.get("normalizers", [normalization])
    if specification["model"]["type"] != "Unigram" or not any(
        step.get("type") == "NFC" for step in steps
    ):
        return tokenizer

    from tokenizers import Regex, normalizers
    from sentencepiece import sentencepiece_model_pb2
    from transformers.utils.hub import cached_file

    vocab_file = getattr(tokenizer, "vocab_file", None) or cached_file(
        config.text_model, "spm.model", revision=config.text_model_revision,
        local_files_only=True,
    )
    model = sentencepiece_model_pb2.ModelProto()
    with open(vocab_file, "rb") as handle:
        model.ParseFromString(handle.read())
    # Transformers 5 can reconstruct this tokenizer with NFC instead of the
    # pretrained nmt_nfkc map, turning characters such as ellipsis into UNK.
    backend.normalizer = normalizers.Sequence([
        normalizers.Precompiled(model.normalizer_spec.precompiled_charsmap),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Strip(left=True, right=True),
    ])
    return tokenizer
