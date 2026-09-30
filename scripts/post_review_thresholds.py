"""Threshold-only adaptation (post-review analysis plan, E2).

Question: how much of the gain from ASR-matched retraining does a changed decision
threshold alone achieve? Weights and probabilities are never touched. Per outer
fold, the three seeds' inner-validation probabilities are averaged (matching the
ensemble evaluation), one threshold is chosen on validation macro-F1, and that
threshold is applied to the same fold's seed-averaged test probabilities. No test
label selects anything.

    T3/T4/T6/T7         gold-trained head, threshold from ASR validation input
    T3-goldval ...      same head, threshold from GOLD validation input (control:
                        is it ASR-specific adaptation or generic cutoff tuning?)
    A1-threshold ...    ASR-matched head with its own ASR-validation threshold
                        (so both sides of the retraining comparison get a tuned cutoff)

The inner-validation split was already used for early stopping; choosing a
threshold on it is a second, pre-specified model-selection step, not a fresh
validation study.

Reads results/original_baseline_validation.csv (post_review_heads.py) and the
published test probabilities. Writes:
    results/thresholds_by_fold.csv
    results/threshold_predictions.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cluster_bootstrap import macro_f1_from_counts

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results"
PUBLISHED_PREDICTIONS = REPO_ROOT / "predictions" / "oof_predictions_crossasr.csv"

# Thresholds in thousandths, so ties and distances to 0.5 are exact integers.
GRID_THOUSANDTHS = np.arange(0, 1001)

# setting -> (head, validation transcript source, test condition whose probabilities are reused)
SETTINGS = {
    "T3": ("gold_text", "whisper", "C3"),
    "T4": ("gold_text", "indic", "C4"),
    "T6": ("gold_fusion", "whisper", "C6"),
    "T7": ("gold_fusion", "indic", "C7"),
    "T3-goldval": ("gold_text", "gold", "C3"),
    "T4-goldval": ("gold_text", "gold", "C4"),
    "T6-goldval": ("gold_fusion", "gold", "C6"),
    "T7-goldval": ("gold_fusion", "gold", "C7"),
    "A1-threshold": ("whisper_text", "whisper", "A1"),
    "A2-threshold": ("indic_text", "indic", "A2"),
    "A3-threshold": ("whisper_fusion", "whisper", "A3"),
    "A4-threshold": ("indic_fusion", "indic", "A4"),
}


def select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float, int]:
    """Best validation macro-F1 on the grid; ties -> closest to 0.5, then the smaller threshold."""
    scores = np.array(
        [macro_f1_from_counts(labels, (probabilities >= k / 1000).astype(np.int64)) for k in GRID_THOUSANDTHS]
    )
    best = scores.max()
    tied = GRID_THOUSANDTHS[scores == best]
    distance = np.abs(tied - 500)
    chosen = int(tied[distance == distance.min()].min())
    return chosen / 1000, float(best), int(len(tied))


def seed_averaged(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean probability over the three seeds, checking every row has all three."""
    grouped = frame.groupby(keys).agg(
        prob_positive=("prob_positive", "mean"),
        n_seeds=("seed", "nunique"),
        gold_label=("gold_label", "first"),
    )
    if (grouped["n_seeds"] != 3).any():
        raise SystemExit("some rows do not have all three seeds")
    return grouped.reset_index()


def main() -> None:
    validation = pd.read_csv(RESULT_DIR / "original_baseline_validation.csv")
    published = pd.read_csv(PUBLISHED_PREDICTIONS)

    threshold_rows, prediction_rows = [], []
    for setting, (head, source, test_condition) in SETTINGS.items():
        block = validation[(validation["head"] == head) & (validation["validation_text_source"] == source)]
        validation_ensemble = seed_averaged(block, ["outer_fold", "utterance_id"])
        test_ensemble = seed_averaged(
            published[published["condition"] == test_condition], ["outer_fold", "utterance_id"]
        )

        for fold, fold_validation in validation_ensemble.groupby("outer_fold"):
            labels = fold_validation["gold_label"].to_numpy().astype(np.int64)
            probabilities = fold_validation["prob_positive"].to_numpy()
            threshold, validation_f1, n_tied = select_threshold(probabilities, labels)
            threshold_rows.append(
                {
                    "setting": setting,
                    "head": head,
                    "validation_text_source": source,
                    "test_condition": test_condition,
                    "outer_fold": int(fold),
                    "threshold": threshold,
                    "validation_macro_f1_at_threshold": validation_f1,
                    "validation_macro_f1_at_0.5": macro_f1_from_counts(
                        labels, (probabilities >= 0.5).astype(np.int64)
                    ),
                    "n_grid_points_tied_at_best": n_tied,
                    "n_validation_recordings": int(len(labels)),
                }
            )
            fold_test = test_ensemble[test_ensemble["outer_fold"] == fold]
            prediction_rows += [
                {
                    "utterance_id": row.utterance_id,
                    "outer_fold": int(fold),
                    "condition": setting,
                    "source_condition": test_condition,
                    "gold_label": int(row.gold_label),
                    "prob_positive": float(row.prob_positive),
                    "threshold": threshold,
                    "predicted_label": int(row.prob_positive >= threshold),
                }
                for row in fold_test.itertuples(index=False)
            ]

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(RESULT_DIR / "thresholds_by_fold.csv", index=False, encoding="utf-8")
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(RESULT_DIR / "threshold_predictions.csv", index=False, encoding="utf-8")

    print("Validation-selected thresholds per outer fold")
    print(thresholds.pivot(index="setting", columns="outer_fold", values="threshold").to_string())
    print(f"\nWrote {RESULT_DIR / 'thresholds_by_fold.csv'} and threshold_predictions.csv "
          f"({len(predictions)} rows)")


if __name__ == "__main__":
    main()
