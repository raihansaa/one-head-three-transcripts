"""ASR-induced sentiment flips for error analysis (plan section 19.2).

Selects recordings where the oracle-text condition (C2) is correct but the
ASR-substituted text condition (C3 or C4) is wrong, and lays out the aligned
word-level difference between the gold and ASR transcript so each case can be
categorised by hand.

The `category` column is deliberately left blank: plan 19.2 asks for a human
judgement across eight categories, and auto-labelling it would fabricate an
analysis the paper reports as manual. The `auto_signals` column supplies
objective hints only (a negation token disappeared, the edit is orthographic,
and so on).

Writes:
    artifacts/sentiment_flips.csv     every flip
    artifacts/sentiment_flips_sample.csv   ~30 stratified for annotation
    artifacts/sentiment_flips_sample.md    readable version
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_bangla import word_tokens  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREDICTION_DIR = REPO_ROOT / "predictions"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"

SYSTEM_CONDITIONS = {"whisper": "C3", "indic": "C4"}
ORACLE_TEXT = "C2"
SAMPLE_PER_SYSTEM = 15

# Bengali negation and privative markers whose loss can invert polarity.
NEGATION_TOKENS = {"না", "নি", "নেই", "নয়", "নাই", "কখনো", "কখনও", "কোনো", "কোন"}


def describe_edits(gold: str, hypothesis: str) -> tuple[str, str]:
    """Human-readable edit list plus objective signal flags."""
    gold_tokens, hypothesis_tokens = word_tokens(gold), word_tokens(hypothesis)
    matcher = difflib.SequenceMatcher(a=gold_tokens, b=hypothesis_tokens, autojunk=False)

    edits, removed, added = [], [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source = " ".join(gold_tokens[i1:i2])
        target = " ".join(hypothesis_tokens[j1:j2])
        removed.extend(gold_tokens[i1:i2])
        added.extend(hypothesis_tokens[j1:j2])
        if tag == "replace":
            edits.append(f"{source} -> {target}")
        elif tag == "delete":
            edits.append(f"deleted: {source}")
        else:
            edits.append(f"inserted: {target}")

    signals = []
    lost_negation = {t for t in removed if t in NEGATION_TOKENS} - set(added)
    if lost_negation:
        signals.append(f"negation_lost({'/'.join(sorted(lost_negation))})")
    if {t for t in added if t in NEGATION_TOKENS} - set(removed):
        signals.append("negation_added")
    if len(hypothesis_tokens) < len(gold_tokens):
        signals.append("content_deleted")
    if len(hypothesis_tokens) > len(gold_tokens):
        signals.append("content_inserted")
    if any(ord(character) < 128 and character.isalpha() for character in hypothesis):
        signals.append("latin_script_leak")
    if not signals:
        signals.append("substitution_only")

    return "; ".join(edits) if edits else "(no token-level difference)", ", ".join(signals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="oof_predictions.csv")
    arguments = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_FILE).set_index("utterance_id")
    predictions = pd.read_csv(PREDICTION_DIR / arguments.predictions)
    wer_metrics = pd.read_csv(ARTIFACT_DIR / "asr_metrics_by_utterance.csv")

    averaged = (
        predictions.groupby(["condition", "utterance_id"])
        .agg(prob_positive=("prob_positive", "mean"), gold_label=("gold_label", "first"))
        .reset_index()
    )
    averaged["predicted_label"] = (averaged["prob_positive"] >= 0.5).astype(int)
    averaged["correct"] = averaged["predicted_label"] == averaged["gold_label"]
    correctness = averaged.pivot(index="utterance_id", columns="condition", values="correct")
    predicted = averaged.pivot(index="utterance_id", columns="condition", values="predicted_label")

    rows = []
    for system, condition in SYSTEM_CONDITIONS.items():
        transcripts = pd.read_csv(TRANSCRIPT_DIR / f"{system}_normalized.csv").set_index(
            "utterance_id"
        )
        wer_lookup = wer_metrics[wer_metrics["asr_system"] == system].set_index("utterance_id")

        flipped = correctness[correctness[ORACLE_TEXT] & ~correctness[condition]].index
        for utterance_id in flipped:
            gold_text = manifest.loc[utterance_id, "gold_transcript_normalized"]
            hypothesis = transcripts.loc[utterance_id, "normalized_prediction"]
            hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)
            edits, signals = describe_edits(gold_text, hypothesis)

            rows.append(
                {
                    "utterance_id": utterance_id,
                    "asr_system": system,
                    "condition": condition,
                    "speaker_id": manifest.loc[utterance_id, "speaker_id"],
                    "gold_label": int(manifest.loc[utterance_id, "sentiment_label"]),
                    "sentiment_name": manifest.loc[utterance_id, "sentiment_name"],
                    "predicted_with_asr_text": int(predicted.loc[utterance_id, condition]),
                    "utterance_wer": float(wer_lookup.loc[utterance_id, "utterance_wer"]),
                    "utterance_cer": float(wer_lookup.loc[utterance_id, "utterance_cer"]),
                    "gold_transcript": gold_text,
                    "asr_transcript": hypothesis,
                    "edits": edits,
                    "auto_signals": signals,
                    "category": "",  # for manual annotation
                }
            )

    flips = pd.DataFrame(rows)
    if flips.empty:
        print("No ASR-induced flips found (C2 correct while C3/C4 wrong).")
        return

    flips.to_csv(ARTIFACT_DIR / "sentiment_flips.csv", index=False, encoding="utf-8")

    # Stratify the annotation sample across WER so it is not all catastrophic errors:
    # sort by WER, then take an even stride through each system's flips.
    samples = []
    for system in SYSTEM_CONDITIONS:
        block = flips[flips["asr_system"] == system].sort_values("utterance_wer")
        stride = max(1, len(block) // SAMPLE_PER_SYSTEM)
        samples.append(block.iloc[::stride].head(SAMPLE_PER_SYSTEM))
    sample = pd.concat(samples).reset_index(drop=True)
    sample.to_csv(ARTIFACT_DIR / "sentiment_flips_sample.csv", index=False, encoding="utf-8")

    lines = [
        "### ASR-induced sentiment flips (C2 correct, ASR-substituted text wrong)",
        "",
        f"Total flips: {len(flips)}  "
        + "  ".join(
            f"{system}: {int((flips['asr_system'] == system).sum())}"
            for system in SYSTEM_CONDITIONS
        ),
        "",
        "Signal counts across all flips:",
        "",
    ]
    signal_counts = flips["auto_signals"].str.split(", ").explode().value_counts()
    lines += [f"- `{signal}`: {count}" for signal, count in signal_counts.items()]
    lines += ["", f"Stratified annotation sample ({len(sample)} cases):", ""]

    for _, row in sample.iterrows():
        lines += [
            f"**{row['utterance_id']}** ({row['asr_system']}, {row['sentiment_name']}, "
            f"WER {row['utterance_wer']:.2f})",
            "",
            f"- gold: {row['gold_transcript']}",
            f"- asr : {row['asr_transcript']}",
            f"- edits: {row['edits']}",
            f"- signals: {row['auto_signals']}",
            "",
        ]

    (ARTIFACT_DIR / "sentiment_flips_sample.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(flips)} flips total; wrote {len(sample)}-case annotation sample.")
    print(signal_counts.to_string())


if __name__ == "__main__":
    main()
