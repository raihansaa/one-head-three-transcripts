"""Statistics for the secondary analyses (secondary-analysis plan: P0 check, E1-E3 comparisons, E6, E7).

Secondary analysis. Everything uses the primary estimator: pooled
recording-level out-of-fold macro-F1 on seed-averaged probabilities, and the
original 10,000-iteration paired sentence-group bootstrap (seed 20260807). The
row order and resampling are identical to cluster_bootstrap.py, so every
condition - old and new - is scored on the same resamples, and the published
intervals are reproduced exactly as a check.

Recovery ratios (E7) are computed inside each resample from unrounded
predictions, never by dividing interval endpoints, and are neither clipped nor
filtered. They are conditional on the fitted heads, splits and selected
thresholds; seed variability is reported separately (E6).

Reads the published predictions plus results/ from secondary_heads.py,
secondary_thresholds.py and secondary_lexical.py. Writes:
    results/p0_published_baseline_check.json
    results/condition_scores.csv
    results/seed_scores.csv
    results/seed_paired_differences.csv
    results/paired_gains_and_recovery_intervals.csv
    results/threshold_comparison.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_bootstrap import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    bootstrap_indices,
    build_condition_arrays,
    group_index_structure,
    macro_f1_from_counts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results"
PUBLISHED_PREDICTIONS = REPO_ROOT / "predictions" / "oof_predictions_crossasr.csv"

# Submitted-paper values the stored predictions must reproduce (Tables 2, 4, 7; A.5).
PAPER_SCORES = {
    "C1": 0.9397, "C2": 0.9760, "C3": 0.6352, "C4": 0.8672, "C5": 0.9812, "C6": 0.6767, "C7": 0.8948,
    "A1": 0.8503, "A2": 0.9297, "A3": 0.9244, "A4": 0.9567, "X1": 0.9088, "X2": 0.8148,
}
PAPER_GAP_INTERVALS = {
    ("C2", "C3"): (0.3408, 0.3164, 0.3653), ("C2", "C4"): (0.1088, 0.0947, 0.1239),
    ("C5", "C6"): (0.3045, 0.2821, 0.3276), ("C5", "C7"): (0.0864, 0.0743, 0.0995),
}
PAPER_POSITIVE_RATES = {"C2": 50.0, "C3": 18.7, "C4": 38.9}

COMPARISON_TYPE = {
    "C": "controlled transcript substitution (same head, test transcript changes)",
    "T": "threshold-only decision-rule adaptation (same weights and probabilities)",
    "system": "system-level comparison (different fitted heads and/or inputs)",
    "lexical": "system-level comparison with a lexical baseline",
}

# (setting, oracle O, gold-head baseline B, threshold-only T, gold-val control, matched A, matched-threshold)
RECOVERY_SETTINGS = (
    ("Whisper text", "C2", "C3", "T3", "T3-goldval", "A1", "A1-threshold"),
    ("XLS-R text", "C2", "C4", "T4", "T4-goldval", "A2", "A2-threshold"),
    ("Whisper fusion", "C5", "C6", "T6", "T6-goldval", "A3", "A3-threshold"),
    ("XLS-R fusion", "C5", "C7", "T7", "T7-goldval", "A4", "A4-threshold"),
)

# (a, b, family, comparison type key)
PAIRED_COMPARISONS = (
    ("C2", "C3", "published check", "C"), ("C2", "C4", "published check", "C"),
    ("C5", "C6", "published check", "C"), ("C5", "C7", "published check", "C"),
    ("C6", "C3", "E1 reference", "system"), ("C7", "C4", "E1 reference", "system"),
    ("N6", "C3", "E1 nuisance fusion vs text", "system"), ("N7", "C4", "E1 nuisance fusion vs text", "system"),
    ("C6", "N6", "E1 speech vs nuisance fusion", "system"), ("C7", "N7", "E1 speech vs nuisance fusion", "system"),
    ("N5", "C2", "E1 gold-text control", "system"), ("C5", "N5", "E1 gold-text control", "system"),
    ("N5", "N6", "E1 nuisance substitution gap", "C"), ("N5", "N7", "E1 nuisance substitution gap", "C"),
    ("NA3", "A1", "E1 matched nuisance gain", "system"), ("NA4", "A2", "E1 matched nuisance gain", "system"),
    ("A3", "A1", "E1 matched speech gain", "system"), ("A4", "A2", "E1 matched speech gain", "system"),
    ("A3", "NA3", "E1 matched speech vs nuisance", "system"), ("A4", "NA4", "E1 matched speech vs nuisance", "system"),
    ("TFIDF-LR", "C2", "E3 lexical baseline", "lexical"),
)

SEED_DIFFERENCES = (
    ("C2", "C3"), ("C2", "C4"), ("C5", "C6"), ("C5", "C7"),
    ("A1", "C3"), ("A2", "C4"), ("A3", "C6"), ("A4", "C7"),
    ("N6", "C3"), ("N7", "C4"), ("C6", "N6"), ("C7", "N7"), ("N5", "C2"), ("C5", "N5"),
    ("NA3", "A1"), ("NA4", "A2"), ("A3", "A1"), ("A4", "A2"), ("A3", "NA3"), ("A4", "NA4"),
)


def confusion(true_labels: np.ndarray, predicted: np.ndarray) -> dict:
    tn, fp, fn, tp = np.bincount(true_labels * 2 + predicted, minlength=4)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return {
        "macro_f1": (f1_pos + f1_neg) / 2, "accuracy": (tp + tn) / len(true_labels),
        "f1_negative": f1_neg, "f1_positive": f1_pos,
        "precision_negative": tn / max(tn + fn, 1), "recall_negative": tn / max(tn + fp, 1),
        "precision_positive": tp / max(tp + fp, 1), "recall_positive": tp / max(tp + fn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_positive_rate": (tp + fp) / len(true_labels),
        "actual_positive_rate": (tp + fn) / len(true_labels),
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    published = pd.read_csv(PUBLISHED_PREDICTIONS)
    new_trained = pd.concat(
        [pd.read_csv(RESULT_DIR / f) for f in ("nuisance_fusion_oof.csv", "nuisance_matched_fusion_oof.csv")],
        ignore_index=True,
    )
    lexical = pd.read_csv(RESULT_DIR / "lexical_baseline_oof.csv")
    threshold_predictions = pd.read_csv(RESULT_DIR / "threshold_predictions.csv")
    thresholds_by_fold = pd.read_csv(RESULT_DIR / "thresholds_by_fold.csv")

    seeded = pd.concat([published, new_trained], ignore_index=True)
    spine, predicted = build_condition_arrays(pd.concat([seeded, lexical], ignore_index=True))
    for setting, block in threshold_predictions.groupby("condition"):
        predicted[setting] = block.set_index("utterance_id").loc[spine["utterance_id"], "predicted_label"].to_numpy().astype(np.int64)
    true_labels = spine["gold_label"].to_numpy().astype(np.int64)
    conditions = sorted(predicted)
    observed = {c: confusion(true_labels, predicted[c]) for c in conditions}

    # E6: each seed scored on its own (rule p >= 0.5), plus within-seed differences.
    seed_score = {}
    for (condition, seed), block in seeded.groupby(["condition", "seed"]):
        block = block.set_index("utterance_id").loc[spine["utterance_id"]]
        seed_score[(condition, seed)] = macro_f1_from_counts(true_labels, (block["prob_positive"].to_numpy() >= 0.5).astype(np.int64))
    seeds = sorted(seeded["seed"].unique())
    seed_rows = []
    for condition in sorted(seeded["condition"].unique()):
        values = np.array([seed_score[(condition, s)] for s in seeds])
        seed_rows.append({"condition": condition, **{f"seed_{s}": v for s, v in zip(seeds, values)},
                          "seed_mean": values.mean(), "seed_sd": values.std(ddof=1),
                          "ensemble_macro_f1": observed[condition]["macro_f1"]})
    pd.DataFrame(seed_rows).to_csv(RESULT_DIR / "seed_scores.csv", index=False, encoding="utf-8")
    difference_rows = []
    for a, b in SEED_DIFFERENCES:
        values = np.array([seed_score[(a, s)] - seed_score[(b, s)] for s in seeds])
        difference_rows.append({"comparison": f"{a} - {b}", **{f"seed_{s}": v for s, v in zip(seeds, values)},
                                "seed_mean": values.mean(), "seed_sd": values.std(ddof=1),
                                "ensemble_difference": observed[a]["macro_f1"] - observed[b]["macro_f1"]})
    pd.DataFrame(difference_rows).to_csv(RESULT_DIR / "seed_paired_differences.csv", index=False, encoding="utf-8")

    # One set of paired sentence-group resamples for every condition (E1, E2, E3, E7).
    flat_positions, starts, lengths = group_index_structure(spine)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = {c: np.empty(N_BOOTSTRAP) for c in conditions}
    for iteration in range(N_BOOTSTRAP):
        index = bootstrap_indices(flat_positions, starts, lengths, rng)
        resampled_true = true_labels[index]
        for c in conditions:
            resampled[c][iteration] = macro_f1_from_counts(resampled_true, predicted[c][index])

    rows = []

    def add(family: str, quantity: str, kind: str, value: float, draws: np.ndarray, note: str = "") -> None:
        low, high = interval(draws)
        rows.append({"family": family, "quantity": quantity, "comparison_type": COMPARISON_TYPE.get(kind, kind),
                     "observed": value, "ci_low": low, "ci_high": high,
                     "excludes_zero": bool(low > 0 or high < 0), "note": note})

    score = {c: observed[c]["macro_f1"] for c in conditions}
    for a, b, family, kind in PAIRED_COMPARISONS:
        add(family, f"{a} - {b}", kind, score[a] - score[b], resampled[a] - resampled[b])

    for setting, o, b, t, tg, a, at in RECOVERY_SETTINGS:
        gap = resampled[o] - resampled[b]
        family = f"E2/E7 {setting}"
        add(family, f"{t} - {b} (threshold-only gain)", "T", score[t] - score[b], resampled[t] - resampled[b])
        add(family, f"{a} - {t} (additional matched gain)", "system", score[a] - score[t], resampled[a] - resampled[t])
        add(family, f"{a} - {b} (total matched gain)", "system", score[a] - score[b], resampled[a] - resampled[b])
        add(family, f"{o} - {a} (residual after matching)", "system", score[o] - score[a], resampled[o] - resampled[a])
        add(family, f"{o} - {b} (original gap)", "C", score[o] - score[b], gap)
        add(family, f"{at} - {t} (matched-threshold vs threshold-only)", "system", score[at] - score[t], resampled[at] - resampled[t])
        add(family, f"{at} - {a} (threshold gain for the matched head)", "T", score[at] - score[a], resampled[at] - resampled[a])
        add(family, f"{tg} - {b} (gold-validation threshold gain)", "T", score[tg] - score[b], resampled[tg] - resampled[b])
        add(family, f"{t} - {tg} (ASR-validation vs gold-validation threshold)", "T", score[t] - score[tg], resampled[t] - resampled[tg])
        note = f"min resampled O-B {gap.min():.4f}; resamples with O-B < 0.01: {int((gap < 0.01).sum())}"
        observed_gap = score[o] - score[b]
        add(family, f"({a} - {b}) / ({o} - {b}) total recovery ratio", "ratio",
            (score[a] - score[b]) / observed_gap, (resampled[a] - resampled[b]) / gap, note)
        add(family, f"({t} - {b}) / ({o} - {b}) threshold-only ratio", "ratio",
            (score[t] - score[b]) / observed_gap, (resampled[t] - resampled[b]) / gap, note)
        add(family, f"({a} - {t}) / ({o} - {b}) additional matched ratio", "ratio",
            (score[a] - score[t]) / observed_gap, (resampled[a] - resampled[t]) / gap, note)
    intervals = pd.DataFrame(rows)
    intervals.to_csv(RESULT_DIR / "paired_gains_and_recovery_intervals.csv", index=False, encoding="utf-8")

    # Table B: decision rule versus retraining, with positive rates and fold thresholds.
    table_b = []
    for setting, o, b, t, tg, a, at in RECOVERY_SETTINGS:
        entry = {"setting": setting, "oracle": o, "oracle_macro_f1": score[o]}
        for label, c in (("gold_head_default", b), ("gold_head_asr_val_threshold", t),
                         ("gold_head_gold_val_threshold", tg), ("matched_head_default", a),
                         ("matched_head_asr_val_threshold", at)):
            entry[f"{label}_condition"] = c
            entry[f"{label}_macro_f1"] = score[c]
            entry[f"{label}_predicted_positive_rate"] = observed[c]["predicted_positive_rate"]
        for label, c in (("asr_val", t), ("gold_val", tg), ("matched_head", at)):
            fold_thresholds = thresholds_by_fold[thresholds_by_fold["setting"] == c].sort_values("outer_fold")
            entry[f"fold_thresholds_{label}"] = "/".join(f"{v:.3f}" for v in fold_thresholds["threshold"])
        entry["actual_positive_rate"] = observed[b]["actual_positive_rate"]
        table_b.append(entry)
    pd.DataFrame(table_b).to_csv(RESULT_DIR / "threshold_comparison.csv", index=False, encoding="utf-8")

    frame = pd.DataFrame([{"condition": c, **observed[c]} for c in conditions])
    frame.to_csv(RESULT_DIR / "condition_scores.csv", index=False, encoding="utf-8")

    # P0: the stored predictions must reproduce the submitted paper.
    checks = {}
    for c, v in PAPER_SCORES.items():
        value = round(float(score[c]), 4)
        checks[f"score {c}"] = {"paper": v, "recomputed": value, "ok": value == v}
    for (a, b), (gap, low, high) in PAPER_GAP_INTERVALS.items():
        match = intervals[intervals["quantity"] == f"{a} - {b}"].iloc[0]
        values = [round(float(match[k]), 4) for k in ("observed", "ci_low", "ci_high")]
        checks[f"gap {a}-{b}"] = {"paper": [gap, low, high], "recomputed": values, "ok": values == [gap, low, high]}
    for c, rate in PAPER_POSITIVE_RATES.items():
        value = round(100 * float(observed[c]["predicted_positive_rate"]), 1)
        checks[f"predicted positive rate {c}"] = {"paper": rate, "recomputed": value, "ok": value == rate}
    counts = {"recordings": int(len(spine)), "sentence_groups": int(spine["sentence_group_id"].nunique()),
              "positive": int(true_labels.sum()), "negative": int((1 - true_labels).sum())}
    checks["counts"] = {"paper": {"recordings": 3996, "sentence_groups": 998, "positive": 1997, "negative": 1999},
                        "recomputed": counts}
    checks["counts"]["ok"] = checks["counts"]["paper"] == counts
    all_ok = all(entry["ok"] for entry in checks.values())
    (RESULT_DIR / "p0_published_baseline_check.json").write_text(
        json.dumps({"all_checks_pass": all_ok, "decision_rule": "seed-averaged prob_positive >= 0.5",
                    "checks": checks}, indent=2), encoding="utf-8")

    print(f"P0 published-baseline checks pass: {all_ok}")
    for name, entry in checks.items():
        if not entry["ok"]:
            print(f"  MISMATCH {name}: {entry}")
    print("\nCondition scores")
    print(frame[["condition", "macro_f1", "accuracy", "predicted_positive_rate"]].round(4).to_string(index=False))
    print("\nPaired intervals")
    print(intervals[["family", "quantity", "observed", "ci_low", "ci_high"]].round(4).to_string(index=False))
    print("\nSeed variability")
    print(pd.DataFrame(seed_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
