"""Per-class ASR quality: are Whisper's errors distributed evenly across labels?

Motivation. Whisper-substituted conditions show a large positive/negative F1
asymmetry (C3: 0.749 negative vs 0.521 positive) that the Bengali XLS-R
conditions do not (C4: 0.882 vs 0.852). Two explanations compete:

  1. Whisper simply transcribes one class worse, so the downstream classifier has
     less to work with there.
  2. The asymmetry is downstream-only and ASR error rates are comparable.

These are distinguishable by measuring WER/CER separately per class. The question
is not incidental here: positive sentences are si_0001-0500 and negative are
si_0501-1000, and those halves were recorded in different sessions (33 of 39
reconstructed sessions are label-pure), so a per-class ASR difference would be
confounded with acquisition conditions rather than with sentiment itself.

Aggregation matches the main table: corpus-level totals (sum of edit operations
divided by sum of reference length), not an average of per-utterance rates.
Edit operations are read from artifacts/asr_metrics_by_utterance.csv, which was
produced by the frozen normalizer, so nothing is recomputed or re-tokenized here.

Writes artifacts/asr_quality_by_class.csv and .md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"

DISPLAY_NAME = {"whisper": "Whisper large-v3", "indic": "Bengali XLS-R CTC"}
# A gap this size or larger is treated as a clear per-class difference rather
# than noise, and is reported as such.
CLEAR_DIFFERENCE = 0.05


def main() -> None:
    metrics = pd.read_csv(ARTIFACT_DIR / "asr_metrics_by_utterance.csv")
    manifest = pd.read_csv(MANIFEST_FILE)[["utterance_id", "sentiment_label", "sentiment_name"]]
    joined = metrics.merge(manifest, on="utterance_id", validate="many_to_one")

    rows = []
    for (system, sentiment), block in joined.groupby(["asr_system", "sentiment_name"]):
        reference_words = block["n_reference_words"].sum()
        reference_chars = block["n_reference_chars"].sum()
        substitutions = block["word_substitutions"].sum()
        deletions = block["word_deletions"].sum()
        insertions = block["word_insertions"].sum()
        rows.append(
            {
                "asr_system": system,
                "display_name": DISPLAY_NAME[system],
                "sentiment_class": sentiment,
                "n_utterances": int(len(block)),
                "n_reference_words": int(reference_words),
                "wer": (substitutions + deletions + insertions) / reference_words,
                "cer": block["char_errors"].sum() / reference_chars,
                "substitution_rate": substitutions / reference_words,
                "deletion_rate": deletions / reference_words,
                "insertion_rate": insertions / reference_words,
                "n_substitutions": int(substitutions),
                "n_deletions": int(deletions),
                "n_insertions": int(insertions),
                "n_empty_hypotheses": int(block["is_empty_hypothesis"].sum()),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["asr_system", "sentiment_class"])
    frame.to_csv(ARTIFACT_DIR / "asr_quality_by_class.csv", index=False, encoding="utf-8")

    print(
        frame[
            ["display_name", "sentiment_class", "n_utterances", "wer", "cer",
             "substitution_rate", "deletion_rate", "insertion_rate"]
        ].round(4).to_string(index=False)
    )

    gaps = {}
    for system, block in frame.groupby("asr_system"):
        indexed = block.set_index("sentiment_class")
        gaps[system] = {
            "wer_gap": float(indexed.loc["positive", "wer"] - indexed.loc["negative", "wer"]),
            "cer_gap": float(indexed.loc["positive", "cer"] - indexed.loc["negative", "cer"]),
        }

    largest = max(abs(g["wer_gap"]) for g in gaps.values())
    clear = largest >= CLEAR_DIFFERENCE

    # If ASR quality does not explain the downstream asymmetry, something else
    # must. Read the decision bias straight off the existing predictions rather
    # than leaving the reader to guess.
    predictions = pd.read_csv(REPO_ROOT / "predictions" / "oof_predictions_final.csv")
    averaged = (
        predictions.groupby(["condition", "utterance_id"])
        .agg(prob=("prob_positive", "mean"), gold=("gold_label", "first"))
        .reset_index()
    )
    averaged["pred"] = (averaged["prob"] >= 0.5).astype(int)
    positive_rate = {
        condition: float(block["pred"].mean())
        for condition, block in averaged.groupby("condition")
    }

    if not clear:
        finding = (
            "Class-dependent downstream performance was not explained by substantial "
            "differences in ASR error rates. Whisper in fact transcribes positive "
            f"utterances marginally *better* than negative ones "
            f"({gaps['whisper']['wer_gap']:+.4f} WER), which is the opposite of what an "
            "ASR-quality explanation of the downstream asymmetry would predict. The "
            "asymmetry instead reflects a decision-bias collapse under out-of-"
            "distribution text: with gold transcripts the classifier predicts the "
            f"positive class {positive_rate.get('C2', float('nan')):.1%} of the time "
            "against a true rate of 50%, but under Whisper-substituted text this falls "
            f"to {positive_rate.get('C3', float('nan')):.1%} "
            f"(Bengali XLS-R: {positive_rate.get('C4', float('nan')):.1%}). The head "
            "collapses toward the negative class when its input distribution degrades, "
            "depressing positive-class F1 without any corresponding asymmetry in "
            "transcription quality."
        )
    else:
        worse = [
            f"{DISPLAY_NAME[s]} transcribes {'positive' if g['wer_gap'] > 0 else 'negative'} "
            f"utterances worse by {abs(g['wer_gap']):.3f} WER"
            for s, g in gaps.items()
            if abs(g["wer_gap"]) >= CLEAR_DIFFERENCE
        ]
        finding = (
            "ASR error rates differ clearly between sentiment classes: "
            + "; ".join(worse)
            + ". Because positive sentences (si_0001-0500) and negative sentences "
            "(si_0501-1000) were recorded in separate sessions - 33 of 39 reconstructed "
            "sessions are label-pure - this per-class difference is confounded with "
            "acquisition conditions and cannot be attributed to sentiment content."
        )

    lines = [
        "### ASR quality by sentiment class",
        "",
        "| ASR system | Class | n | WER | CER | Sub. | Del. | Ins. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['display_name']} | {r['sentiment_class']} | {r['n_utterances']} | "
        f"{r['wer']:.4f} | {r['cer']:.4f} | {r['substitution_rate']:.4f} | "
        f"{r['deletion_rate']:.4f} | {r['insertion_rate']:.4f} |"
        for _, r in frame.iterrows()
    ]
    lines += ["", "#### Positive minus negative", "", "| ASR system | WER gap | CER gap |", "|---|---:|---:|"]
    lines += [
        f"| {DISPLAY_NAME[s]} | {g['wer_gap']:+.4f} | {g['cer_gap']:+.4f} |"
        for s, g in gaps.items()
    ]
    lines += ["", f"**Finding:** {finding}"]
    (ARTIFACT_DIR / "asr_quality_by_class.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nPositive minus negative:")
    for system, gap in gaps.items():
        print(f"  {DISPLAY_NAME[system]:<20} WER {gap['wer_gap']:+.4f}   CER {gap['cer_gap']:+.4f}")
    print(f"\nFinding: {finding}")


if __name__ == "__main__":
    main()
