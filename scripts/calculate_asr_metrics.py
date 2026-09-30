"""ASR quality metrics and Table 1 (plan section 10.3, 10.4).

WER is computed over whitespace tokens and CER over extended grapheme clusters,
both on text passed through the single frozen normalizer that also produced the
gold side. Corpus-level rates are aggregated over edit operations across all
utterances (total errors / total reference length), not averaged per utterance,
so long and short sentences are weighted correctly.

Per-utterance WER and CER are also written out because the WER-bin analysis
(plan 19.1) and the OOF prediction table (plan 18.2) both need them.

Writes:
    artifacts/asr_metrics.json
    artifacts/asr_metrics_by_utterance.csv
    artifacts/table1_asr_quality.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_bangla import (  # noqa: E402
    NORMALIZATION_VERSION,
    cer_is_grapheme_aware,
    grapheme_clusters,
    word_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

SYSTEMS = {"whisper": "Whisper large-v3", "indic": "Bengali XLS-R (CTC, no LM)"}


def edit_operations(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Levenshtein backtrace giving (substitutions, deletions, insertions)."""
    n, m = len(reference), len(hypothesis)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1],  # substitution
                    distance[i - 1][j],  # deletion
                    distance[i][j - 1],  # insertion
                )

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and distance[i][j] == distance[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions


def score_system(manifest: pd.DataFrame, system: str) -> tuple[pd.DataFrame, dict]:
    predictions = pd.read_csv(TRANSCRIPT_DIR / f"{system}_normalized.csv").set_index("utterance_id")
    version = predictions["normalization_version"].iloc[0]
    if version != NORMALIZATION_VERSION:
        raise ValueError(
            f"{system} transcripts were normalized with {version}, "
            f"but the current normalizer is {NORMALIZATION_VERSION}. Re-run transcription."
        )

    missing = set(manifest["utterance_id"]) - set(predictions.index)
    if missing:
        raise ValueError(f"{system} is missing {len(missing)} utterances, e.g. {sorted(missing)[:5]}")

    rows = []
    for record in manifest.itertuples(index=False):
        hypothesis = predictions.loc[record.utterance_id, "normalized_prediction"]
        hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)

        reference_words = word_tokens(record.gold_transcript_normalized)
        hypothesis_words = word_tokens(hypothesis)
        word_sub, word_del, word_ins = edit_operations(reference_words, hypothesis_words)

        reference_chars = grapheme_clusters(record.gold_transcript_normalized)
        char_sub, char_del, char_ins = edit_operations(reference_chars, grapheme_clusters(hypothesis))

        rows.append(
            {
                "utterance_id": record.utterance_id,
                "sentence_id": record.sentence_id,
                "speaker_id": record.speaker_id,
                "asr_system": system,
                "n_reference_words": len(reference_words),
                "n_hypothesis_words": len(hypothesis_words),
                "word_substitutions": word_sub,
                "word_deletions": word_del,
                "word_insertions": word_ins,
                "utterance_wer": (word_sub + word_del + word_ins) / max(len(reference_words), 1),
                "n_reference_chars": len(reference_chars),
                "char_errors": char_sub + char_del + char_ins,
                "utterance_cer": (char_sub + char_del + char_ins) / max(len(reference_chars), 1),
                "is_empty_hypothesis": len(hypothesis_words) == 0,
            }
        )

    frame = pd.DataFrame(rows)
    reference_words_total = frame["n_reference_words"].sum()
    reference_chars_total = frame["n_reference_chars"].sum()
    word_errors = (
        frame["word_substitutions"].sum() + frame["word_deletions"].sum() + frame["word_insertions"].sum()
    )

    summary = {
        "system": system,
        "display_name": SYSTEMS[system],
        "n_utterances": int(len(frame)),
        "wer": word_errors / reference_words_total,
        "cer": frame["char_errors"].sum() / reference_chars_total,
        "substitution_rate": frame["word_substitutions"].sum() / reference_words_total,
        "deletion_rate": frame["word_deletions"].sum() / reference_words_total,
        "insertion_rate": frame["word_insertions"].sum() / reference_words_total,
        "n_empty_hypotheses": int(frame["is_empty_hypothesis"].sum()),
        "n_utterances_with_zero_wer": int((frame["utterance_wer"] == 0).sum()),
        "median_utterance_wer": float(frame["utterance_wer"].median()),
        "by_speaker": {
            speaker: {
                "wer": float(
                    (block["word_substitutions"] + block["word_deletions"] + block["word_insertions"]).sum()
                    / block["n_reference_words"].sum()
                ),
                "cer": float(block["char_errors"].sum() / block["n_reference_chars"].sum()),
                "n_utterances": int(len(block)),
            }
            for speaker, block in frame.groupby("speaker_id")
        },
    }
    return frame, summary


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)

    per_utterance, summaries = [], []
    for system in SYSTEMS:
        frame, summary = score_system(manifest, system)
        per_utterance.append(frame)
        summaries.append(summary)
        print(
            f"{summary['display_name']:<32} WER {summary['wer']:.3f}  CER {summary['cer']:.3f}  "
            f"empty {summary['n_empty_hypotheses']}"
        )

    pd.concat(per_utterance).to_csv(
        ARTIFACT_DIR / "asr_metrics_by_utterance.csv", index=False, encoding="utf-8"
    )
    (ARTIFACT_DIR / "asr_metrics.json").write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "wer_tokenization": "whitespace tokens on normalized text; punctuation removed",
                "cer_tokenization": (
                    "extended grapheme clusters (regex \\X), spaces excluded"
                    if cer_is_grapheme_aware()
                    else "LIMITATION: unicode code points, `regex` unavailable"
                ),
                "cer_grapheme_aware": cer_is_grapheme_aware(),
                "aggregation": "corpus-level: total edit operations / total reference length",
                "systems": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "### Table 1 - ASR quality on BanglaMUSE",
        "",
        "| ASR system | WER | CER | Sub. | Del. | Ins. | Empty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['display_name']} | {summary['wer']:.3f} | {summary['cer']:.3f} | "
            f"{summary['substitution_rate']:.3f} | {summary['deletion_rate']:.3f} | "
            f"{summary['insertion_rate']:.3f} | {summary['n_empty_hypotheses']} |"
        )
    lines += ["", "Per-speaker WER / CER", "", "| ASR system | f1 | f2 | m1 | m2 |", "|---|---:|---:|---:|---:|"]
    for summary in summaries:
        cells = " | ".join(
            f"{summary['by_speaker'][s]['wer']:.3f} / {summary['by_speaker'][s]['cer']:.3f}"
            for s in ("f1", "f2", "m1", "m2")
        )
        lines.append(f"| {summary['display_name']} | {cells} |")

    (ARTIFACT_DIR / "table1_asr_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {ARTIFACT_DIR / 'table1_asr_quality.md'}")


if __name__ == "__main__":
    main()
