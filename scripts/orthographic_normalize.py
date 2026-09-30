"""Externally specified Bengali orthographic canonicalization (corrected plan sections 20-25).

The mapping is NOT derived from BanglaMUSE. Deriving one from this corpus's own
frequent ASR substitutions and then evaluating on the same corpus would be
post-hoc tuning on the errors it is meant to explain (plan section 21).

Source used instead: the `normalizer` package released with BanglaBERT
(Bhattacharjee et al., "BanglaBERT: Language Model Pretraining and Benchmarks for
Low-Resource Language Understanding and Generation in Bangla", Findings of NAACL
2022). Two independent reasons make it the right choice here:

  1. It is published and externally specified - its tables are fixed by the
     authors, not by us, and are dumped verbatim into the frozen mapping file.
  2. It is the preprocessing the BanglaBERT authors specify for BanglaBERT, which
     is the text encoder used throughout this study. Applying it is independently
     motivated as correct encoder preprocessing, not as an ASR-error fix.

Composition. The frozen pipeline normalizer runs unchanged; canonicalization is
inserted *before* it:

    C-conditions:  raw -> normalize_bangla
    N-conditions:  raw -> csebuetnlp normalize -> normalize_bangla

so the only difference between a C and its N counterpart is the external
orthographic canonicalization. It is applied symmetrically to gold and to both ASR
systems (plan section 23) - never to ASR output alone.

Writes:
    artifacts/orthographic_mapping.json     the frozen mapping, verbatim
    transcripts/{gold,whisper,indic}_orthnorm.csv
    embeddings/text_{gold,whisper,indic}_orthnorm.npz
    artifacts/orthographic_normalization_wer.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from normalizer import const as normalizer_const
from normalizer import normalize as canonicalize
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_asr_metrics import edit_operations  # noqa: E402
from extract_text_embeddings import MAX_LENGTH, embed  # noqa: E402
from normalize_bangla import grapheme_clusters, normalize_bangla, word_tokens  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
EMBEDDING_DIR = REPO_ROOT / "embeddings"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

SOURCE = {
    "name": "csebuetnlp/normalizer",
    "citation": (
        "Bhattacharjee et al. (2022), BanglaBERT: Language Model Pretraining and "
        "Benchmarks for Low-Resource Language Understanding and Generation in Bangla, "
        "Findings of NAACL 2022"
    ),
    "repository": "https://github.com/csebuetnlp/normalizer",
    "why_not_derived_from_data": (
        "Deriving a mapping from BanglaMUSE's own frequent ASR substitutions and then "
        "evaluating on BanglaMUSE would be post-hoc tuning on the test set."
    ),
    "unicode_norm": "NFKC",
    "applied_to": ["gold", "whisper", "indic"],
    "composition": "raw -> csebuetnlp normalize -> frozen normalize_bangla",
}


def canonicalized(text: str) -> str:
    """External canonicalization, then the project's frozen normalizer."""
    if not isinstance(text, str) or not text:
        return ""
    return normalize_bangla(canonicalize(text))


def freeze_mapping() -> None:
    """Dump the external tables verbatim so the mapping is fully documented."""
    payload = {
        **SOURCE,
        "char_replacements": {
            f"U+{code:04X} {chr(code)}": replacement
            for code, replacement in sorted(normalizer_const.CHAR_REPLACEMENTS.items())
        },
        "unicode_replacements": dict(normalizer_const.UNICODE_REPLACEMENTS),
        "n_char_replacements": len(normalizer_const.CHAR_REPLACEMENTS),
        "n_unicode_replacements": len(normalizer_const.UNICODE_REPLACEMENTS),
    }
    (ARTIFACT_DIR / "orthographic_mapping.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Froze mapping: {payload['n_char_replacements']} char + "
        f"{payload['n_unicode_replacements']} unicode replacements"
    )


def corpus_rates(references: list[str], hypotheses: list[str]) -> dict:
    word_errors = reference_words = char_errors = reference_chars = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens = word_tokens(reference)
        substitutions, deletions, insertions = edit_operations(
            reference_tokens, word_tokens(hypothesis)
        )
        word_errors += substitutions + deletions + insertions
        reference_words += len(reference_tokens)

        reference_graphemes = grapheme_clusters(reference)
        substitutions, deletions, insertions = edit_operations(
            reference_graphemes, grapheme_clusters(hypothesis)
        )
        char_errors += substitutions + deletions + insertions
        reference_chars += len(reference_graphemes)
    return {
        "wer": word_errors / max(reference_words, 1),
        "cer": char_errors / max(reference_chars, 1),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    freeze_mapping()

    manifest = pd.read_csv(MANIFEST_FILE)
    gold = manifest.drop_duplicates("sentence_id").sort_values("sentence_id")
    gold_original = gold["gold_transcript_normalized"].fillna("").astype(str).tolist()
    gold_canonical = [canonicalized(t) for t in gold["gold_transcript_raw"].fillna("").astype(str)]

    pd.DataFrame(
        {"sentence_id": gold["sentence_id"], "orthnorm_transcript": gold_canonical}
    ).to_csv(TRANSCRIPT_DIR / "gold_orthnorm.csv", index=False, encoding="utf-8")

    changed_gold = sum(a != b for a, b in zip(gold_original, gold_canonical))
    print(f"Gold sentences altered by canonicalization: {changed_gold}/{len(gold_canonical)}")

    gold_lookup = dict(zip(gold["sentence_id"], gold_canonical))
    reference_by_utterance = [gold_lookup[s] for s in manifest["sentence_id"]]
    original_reference = (
        manifest["gold_transcript_normalized"].fillna("").astype(str).tolist()
    )

    wer_report = {"source": SOURCE["name"], "systems": {}}
    canonical_predictions: dict[str, list[str]] = {}

    for system in ("whisper", "indic"):
        raw = pd.read_csv(TRANSCRIPT_DIR / f"{system}_raw.csv").set_index("utterance_id")
        normalized = pd.read_csv(TRANSCRIPT_DIR / f"{system}_normalized.csv").set_index(
            "utterance_id"
        )
        ordered = manifest["utterance_id"].tolist()

        before = [
            str(normalized.loc[u, "normalized_prediction"])
            if pd.notna(normalized.loc[u, "normalized_prediction"])
            else ""
            for u in ordered
        ]
        after = [
            canonicalized(
                str(raw.loc[u, "raw_prediction"]) if pd.notna(raw.loc[u, "raw_prediction"]) else ""
            )
            for u in ordered
        ]
        canonical_predictions[system] = after

        pd.DataFrame({"utterance_id": ordered, "orthnorm_prediction": after}).to_csv(
            TRANSCRIPT_DIR / f"{system}_orthnorm.csv", index=False, encoding="utf-8"
        )

        wer_report["systems"][system] = {
            "original": corpus_rates(original_reference, before),
            "canonicalized": corpus_rates(reference_by_utterance, after),
            "n_hypotheses_altered": sum(a != b for a, b in zip(before, after)),
        }
        entry = wer_report["systems"][system]
        entry["wer_reduction"] = entry["original"]["wer"] - entry["canonicalized"]["wer"]
        entry["cer_reduction"] = entry["original"]["cer"] - entry["canonicalized"]["cer"]
        print(
            f"{system}: WER {entry['original']['wer']:.4f} -> "
            f"{entry['canonicalized']['wer']:.4f}   "
            f"CER {entry['original']['cer']:.4f} -> {entry['canonicalized']['cer']:.4f}   "
            f"altered {entry['n_hypotheses_altered']}/{len(ordered)}"
        )

    (ARTIFACT_DIR / "orthographic_normalization_wer.json").write_text(
        json.dumps(wer_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nEmbedding canonicalized text with the same frozen encoder ...")
    encoder = json.loads((EMBEDDING_DIR / "text_encoder.json").read_text(encoding="utf-8"))
    checkpoint = encoder["checkpoint"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()

    sources = {
        "gold": (gold["sentence_id"].tolist(), gold_canonical),
        "whisper": (manifest["utterance_id"].tolist(), canonical_predictions["whisper"]),
        "indic": (manifest["utterance_id"].tolist(), canonical_predictions["indic"]),
    }
    for name, (ids, texts) in sources.items():
        order = np.argsort(np.array(ids))
        ids_sorted = [ids[i] for i in order]
        texts_sorted = [texts[i] for i in order]
        matrix = embed(texts_sorted, tokenizer, model, device)
        np.savez_compressed(
            EMBEDDING_DIR / f"text_{name}_orthnorm.npz",
            ids=np.array(ids_sorted),
            embeddings=matrix,
        )
        print(f"  saved {matrix.shape} -> embeddings/text_{name}_orthnorm.npz")

    print(f"\nMax token length used for embedding: {MAX_LENGTH}")


if __name__ == "__main__":
    main()
