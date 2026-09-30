"""Cache frozen text embeddings for gold and both ASR transcript sources (plan 11.1).

One text encoder is used for every principal condition. BanglaBERT is preferred;
XLM-R base is the documented fallback if it will not load. The encoder is frozen
- no gradient ever reaches it - so embeddings can be computed once and reused by
every fold, seed and condition.

Representation: normalized transcript -> tokenizer -> frozen transformer ->
attention-mask-aware mean pooling.

Gold text is embedded once per sentence (it is identical across the four
recordings). ASR text is embedded per recording, because each speaker's audio
produces its own transcript and its own errors.

Writes embeddings/text_{gold,whisper,indic}.npz, each holding `ids` and
`embeddings`, plus embeddings/text_encoder.json recording what was used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_common import batched  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
EMBEDDING_DIR = REPO_ROOT / "embeddings"

PRIMARY_ENCODER = "csebuetnlp/banglabert"
FALLBACK_ENCODER = "xlm-roberta-base"
MAX_LENGTH = 64
BATCH_SIZE = 64


def load_encoder() -> tuple[str, AutoTokenizer, AutoModel]:
    for checkpoint in (PRIMARY_ENCODER, FALLBACK_ENCODER):
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModel.from_pretrained(checkpoint)
            print(f"Text encoder: {checkpoint}")
            return checkpoint, tokenizer, model
        except Exception as error:  # noqa: BLE001
            print(f"  {checkpoint} unavailable ({type(error).__name__}); trying fallback")
    raise RuntimeError("no text encoder could be loaded")


@torch.inference_mode()
def embed(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Attention-mask-aware mean pooling over the frozen encoder's last layer."""
    vectors = []
    for chunk in batched(texts, BATCH_SIZE):
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def main() -> None:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_csv(MANIFEST_FILE)
    checkpoint, tokenizer, model = load_encoder()
    model.to(device).eval()

    sources: dict[str, tuple[list[str], list[str]]] = {}

    gold = manifest.drop_duplicates("sentence_id").sort_values("sentence_id")
    sources["gold"] = (
        gold["sentence_id"].tolist(),
        gold["gold_transcript_normalized"].fillna("").astype(str).tolist(),
    )

    for system in ("whisper", "indic"):
        predictions = pd.read_csv(TRANSCRIPT_DIR / f"{system}_normalized.csv")
        predictions = predictions[predictions["utterance_id"].isin(set(manifest["utterance_id"]))]
        predictions = predictions.sort_values("utterance_id")
        sources[system] = (
            predictions["utterance_id"].tolist(),
            predictions["normalized_prediction"].fillna("").astype(str).tolist(),
        )

    dimensions = {}
    for name, (ids, texts) in sources.items():
        print(f"Embedding {len(texts)} {name} transcripts ...")
        matrix = embed(texts, tokenizer, model, device)
        np.savez_compressed(
            EMBEDDING_DIR / f"text_{name}.npz", ids=np.array(ids), embeddings=matrix
        )
        dimensions[name] = list(matrix.shape)
        print(f"  saved {matrix.shape} -> embeddings/text_{name}.npz")

    (EMBEDDING_DIR / "text_encoder.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "used_fallback": checkpoint == FALLBACK_ENCODER,
                "frozen": True,
                "pooling": "attention-mask-aware mean over last_hidden_state",
                "max_length": MAX_LENGTH,
                "gold_keyed_by": "sentence_id",
                "asr_keyed_by": "utterance_id",
                "shapes": dimensions,
                "transformers_version": transformers.__version__,
                "trained_on_banglamuse": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
