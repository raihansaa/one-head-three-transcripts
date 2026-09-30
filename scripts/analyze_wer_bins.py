"""WER-bin and per-speaker analyses (plan sections 19.1 and 19.3).

The binning rule is fixed before any downstream performance is inspected:
bin 0 is exactly WER = 0, and the remaining positive-WER utterances are split
into three approximately equal-count tertiles. Observed WER boundaries and bin
sizes are reported alongside the results so the rule is auditable.

Accuracy rather than macro-F1 is reported inside bins, because bins are small
and can be label-imbalanced, which makes macro-F1 unstable there.

The speaker analysis is descriptive only. BanglaMUSE has four speakers, and in
this delivery speaker is confounded with recording condition (see
artifacts/data_summary.json), so no demographic, dialectal or fairness
conclusion may be drawn from it.

Writes:
    artifacts/wer_bin_analysis.csv / .md
    artifacts/speaker_analysis.csv / .md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREDICTION_DIR = REPO_ROOT / "predictions"

# ASR system -> (substituted-text condition, substituted-fusion condition)
SYSTEM_CONDITIONS = {"whisper": ("C3", "C6"), "indic": ("C4", "C7")}
ORACLE_TEXT = "C2"
ORACLE_FUSION = "C5"
SPEECH_ONLY = "C1"


def averaged_predictions(prediction_file: str) -> pd.DataFrame:
    """One row per (utterance, condition) with probabilities averaged over seeds."""
    predictions = pd.read_csv(PREDICTION_DIR / prediction_file)
    averaged = (
        predictions.groupby(["condition", "utterance_id"])
        .agg(
            prob_positive=("prob_positive", "mean"),
            gold_label=("gold_label", "first"),
            speaker_id=("speaker_id", "first"),
        )
        .reset_index()
    )
    averaged["predicted_label"] = (averaged["prob_positive"] >= 0.5).astype(int)
    averaged["correct"] = (averaged["predicted_label"] == averaged["gold_label"]).astype(int)
    return averaged


def accuracy_for(averaged: pd.DataFrame, condition: str, utterance_ids) -> float:
    block = averaged[
        (averaged["condition"] == condition) & (averaged["utterance_id"].isin(utterance_ids))
    ]
    return float(block["correct"].mean()) if len(block) else float("nan")


def assign_bins(wer: pd.Series) -> tuple[pd.Series, list[dict]]:
    """Bin 0 is WER == 0; positive-WER utterances split into three tertiles."""
    bins = pd.Series(index=wer.index, dtype="object")
    bins[wer == 0] = "WER = 0"

    positive = wer[wer > 0]
    edges = np.quantile(positive, [1 / 3, 2 / 3]) if len(positive) else [0, 0]
    bins[(wer > 0) & (wer <= edges[0])] = "low"
    bins[(wer > edges[0]) & (wer <= edges[1])] = "mid"
    bins[wer > edges[1]] = "high"

    description = [
        {"bin": "WER = 0", "range": "0"},
        {"bin": "low", "range": f"(0, {edges[0]:.3f}]"},
        {"bin": "mid", "range": f"({edges[0]:.3f}, {edges[1]:.3f}]"},
        {"bin": "high", "range": f"> {edges[1]:.3f}"},
    ]
    return bins, description


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="oof_predictions.csv")
    arguments = parser.parse_args()

    averaged = averaged_predictions(arguments.predictions)
    metrics = pd.read_csv(ARTIFACT_DIR / "asr_metrics_by_utterance.csv")

    bin_rows, boundaries = [], {}
    for system, (text_condition, fusion_condition) in SYSTEM_CONDITIONS.items():
        block = metrics[metrics["asr_system"] == system].set_index("utterance_id")
        bins, description = assign_bins(block["utterance_wer"])
        boundaries[system] = description

        for name in ("WER = 0", "low", "mid", "high"):
            utterance_ids = set(bins[bins == name].index)
            if not utterance_ids:
                continue
            bin_rows.append(
                {
                    "asr_system": system,
                    "wer_bin": name,
                    "n_utterances": len(utterance_ids),
                    "mean_wer": float(block.loc[list(utterance_ids), "utterance_wer"].mean()),
                    "acc_oracle_text_C2": accuracy_for(averaged, ORACLE_TEXT, utterance_ids),
                    f"acc_asr_text_{text_condition}": accuracy_for(
                        averaged, text_condition, utterance_ids
                    ),
                    f"acc_asr_fusion_{fusion_condition}": accuracy_for(
                        averaged, fusion_condition, utterance_ids
                    ),
                    "acc_speech_only_C1": accuracy_for(averaged, SPEECH_ONLY, utterance_ids),
                }
            )

    bin_frame = pd.DataFrame(bin_rows)
    bin_frame.to_csv(ARTIFACT_DIR / "wer_bin_analysis.csv", index=False, encoding="utf-8")

    lines = ["### WER-bin analysis (accuracy)", "", "Predefined bins:", ""]
    for system, description in boundaries.items():
        lines.append(f"- **{system}**: " + ", ".join(f"{d['bin']} = {d['range']}" for d in description))
    lines += [
        "",
        "| ASR | Bin | n | Mean WER | Oracle text (C2) | ASR text | ASR fusion | Speech only (C1) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bin_rows:
        text_key = next(k for k in row if k.startswith("acc_asr_text_"))
        fusion_key = next(k for k in row if k.startswith("acc_asr_fusion_"))
        lines.append(
            f"| {row['asr_system']} | {row['wer_bin']} | {row['n_utterances']} | "
            f"{row['mean_wer']:.3f} | {row['acc_oracle_text_C2']:.3f} | {row[text_key]:.3f} | "
            f"{row[fusion_key]:.3f} | {row['acc_speech_only_C1']:.3f} |"
        )
    (ARTIFACT_DIR / "wer_bin_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    asr_summary = json.loads((ARTIFACT_DIR / "asr_metrics.json").read_text(encoding="utf-8"))
    by_system = {entry["system"]: entry["by_speaker"] for entry in asr_summary["systems"]}

    speaker_rows = []
    for speaker in sorted(averaged["speaker_id"].unique()):
        utterance_ids = set(averaged[averaged["speaker_id"] == speaker]["utterance_id"])
        row = {"speaker_id": speaker, "n_utterances": len(utterance_ids)}
        for system, (text_condition, fusion_condition) in SYSTEM_CONDITIONS.items():
            row[f"wer_{system}"] = by_system[system][speaker]["wer"]
            row[f"cer_{system}"] = by_system[system][speaker]["cer"]
            row[f"acc_{text_condition}"] = accuracy_for(averaged, text_condition, utterance_ids)
            row[f"acc_{fusion_condition}"] = accuracy_for(averaged, fusion_condition, utterance_ids)
        row["acc_C1"] = accuracy_for(averaged, SPEECH_ONLY, utterance_ids)
        row["acc_C2"] = accuracy_for(averaged, ORACLE_TEXT, utterance_ids)
        row["acc_C5"] = accuracy_for(averaged, ORACLE_FUSION, utterance_ids)
        speaker_rows.append(row)

    speaker_frame = pd.DataFrame(speaker_rows)
    speaker_frame.to_csv(ARTIFACT_DIR / "speaker_analysis.csv", index=False, encoding="utf-8")

    speaker_lines = [
        "### Per-speaker analysis (descriptive; four speakers only)",
        "",
        "| Speaker | n | WER wh. | WER ind. | C1 | C2 | C3 | C4 | C5 | C6 | C7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in speaker_rows:
        speaker_lines.append(
            f"| {row['speaker_id']} | {row['n_utterances']} | {row['wer_whisper']:.3f} | "
            f"{row['wer_indic']:.3f} | {row['acc_C1']:.3f} | {row['acc_C2']:.3f} | "
            f"{row['acc_C3']:.3f} | {row['acc_C4']:.3f} | {row['acc_C5']:.3f} | "
            f"{row['acc_C6']:.3f} | {row['acc_C7']:.3f} |"
        )
    speaker_lines += [
        "",
        "Descriptive only. Speaker is confounded with recording condition in this delivery "
        "(f1 mixes 44.1 kHz MP3 with 16 kHz OGG; f2 is 48 kHz stereo), so a speaker difference "
        "may reflect channel rather than voice. No demographic or fairness claim follows.",
    ]
    (ARTIFACT_DIR / "speaker_analysis.md").write_text("\n".join(speaker_lines) + "\n", encoding="utf-8")

    print(bin_frame.to_string(index=False))
    print()
    print(speaker_frame.to_string(index=False))
    print(f"\nWrote WER-bin and speaker analyses to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
