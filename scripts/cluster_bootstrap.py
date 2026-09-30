"""Condition metrics and the paired sentence-cluster bootstrap (plan sections 15, 18).

Recordings of the same sentence are not independent: four speakers read the same
words, so their errors correlate. Bootstrapping recordings individually would
therefore understate uncertainty. Here sentence GROUPS are resampled with
replacement and all of a sampled group's recordings enter together.

The same resample is shared by every condition within an iteration, which makes
each reported interval a *paired* interval on the difference - the quantity the
predeclared comparisons in plan section 15 are about.

Writes:
    artifacts/condition_metrics.csv
    artifacts/bootstrap_comparisons.json
    artifacts/table2_principal_results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_DIR = REPO_ROOT / "predictions"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260807

CONDITION_NAMES = {
    "C1": "Speech only",
    "C2": "Oracle text",
    "C3": "Whisper-substituted text",
    "C4": "Bengali XLS-R-substituted text",
    "C5": "Oracle fusion",
    "C6": "Whisper-substituted fusion",
    "C7": "Bengali XLS-R-substituted fusion",
    "A1": "Whisper-adapted text",
    "A2": "Bengali XLS-R-adapted text",
    "A3": "Whisper-adapted fusion",
    "A4": "Bengali XLS-R-adapted fusion",
}

# Declared before looking at any result (plan section 15).
PREDECLARED_COMPARISONS = [
    ("delta_text_whisper", "C2", "C3", "Oracle-to-Whisper text degradation"),
    ("delta_text_indic", "C2", "C4", "Oracle-to-Indic text degradation"),
    ("delta_fusion_whisper", "C5", "C6", "Oracle-to-Whisper fusion degradation"),
    ("delta_fusion_indic", "C5", "C7", "Oracle-to-Indic fusion degradation"),
    ("gain_speech_whisper", "C6", "C3", "Speech contribution under Whisper text"),
    ("gain_speech_indic", "C7", "C4", "Speech contribution under Indic text"),
]

ADAPTATION_COMPARISONS = [
    ("recovery_text_whisper", "A1", "C3", "Whisper-adapted text recovery"),
    ("recovery_text_indic", "A2", "C4", "Bengali XLS-R-adapted text recovery"),
    ("recovery_fusion_whisper", "A3", "C6", "Whisper-adapted fusion recovery"),
    ("recovery_fusion_indic", "A4", "C7", "Bengali XLS-R-adapted fusion recovery"),
]

# POST-HOC, not predeclared. These test whether substituted fusion is actually
# below the speech-only baseline - i.e. whether adding a noisy transcript to
# speech does net harm. They were added after seeing C6 and C7 fall below C1, so
# they are reported as exploratory and labelled as such wherever they appear.
NEGATIVE_TRANSFER_COMPARISONS = [
    ("negative_transfer_whisper", "C1", "C6", "Speech-only minus Whisper-substituted fusion"),
    ("negative_transfer_bengali_xlsr", "C1", "C7", "Speech-only minus Bengali XLS-R-substituted fusion"),
]


def macro_f1_from_counts(true_labels: np.ndarray, predicted: np.ndarray) -> float:
    """Binary macro-F1 straight from the 2x2 confusion counts (fast enough for 10k iterations)."""
    counts = np.bincount(true_labels * 2 + predicted, minlength=4)
    c00, c01, c10, c11 = counts
    f1_negative = 2 * c00 / max(2 * c00 + c01 + c10, 1e-12)
    f1_positive = 2 * c11 / max(2 * c11 + c10 + c01, 1e-12)
    return float((f1_negative + f1_positive) / 2)


def full_metrics(true_labels: np.ndarray, predicted: np.ndarray) -> dict:
    counts = np.bincount(true_labels * 2 + predicted, minlength=4)
    c00, c01, c10, c11 = counts
    f1_negative = 2 * c00 / max(2 * c00 + c01 + c10, 1e-12)
    f1_positive = 2 * c11 / max(2 * c11 + c10 + c01, 1e-12)
    return {
        "macro_f1": (f1_negative + f1_positive) / 2,
        "accuracy": (c00 + c11) / max(counts.sum(), 1),
        "f1_positive": f1_positive,
        "f1_negative": f1_negative,
    }


def build_condition_arrays(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Average probabilities across seeds, then align every condition to one row order."""
    averaged = (
        predictions.groupby(["condition", "utterance_id"])
        .agg(
            prob_positive=("prob_positive", "mean"),
            gold_label=("gold_label", "first"),
            sentence_group_id=("sentence_group_id", "first"),
            speaker_id=("speaker_id", "first"),
            outer_fold=("outer_fold", "first"),
        )
        .reset_index()
    )

    reference = (
        averaged[averaged["condition"] == averaged["condition"].iloc[0]]
        .sort_values("utterance_id")
        .reset_index(drop=True)
    )
    spine = reference[["utterance_id", "sentence_group_id", "speaker_id", "outer_fold", "gold_label"]]

    predicted_by_condition = {}
    for condition, block in averaged.groupby("condition"):
        block = block.set_index("utterance_id").loc[spine["utterance_id"]]
        predicted_by_condition[condition] = (block["prob_positive"].to_numpy() >= 0.5).astype(np.int64)
    return spine, predicted_by_condition


def group_index_structure(spine: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat row positions grouped by sentence group, with per-group starts and lengths."""
    groups = spine["sentence_group_id"].to_numpy()
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    _, starts, lengths = np.unique(sorted_groups, return_index=True, return_counts=True)
    return order.astype(np.int64), starts.astype(np.int64), lengths.astype(np.int64)


def bootstrap_indices(
    flat_positions: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One cluster resample: draw groups with replacement, take all their recordings."""
    sampled = rng.integers(0, len(starts), size=len(starts))
    sampled_lengths = lengths[sampled]
    total = int(sampled_lengths.sum())
    within = np.arange(total) - np.repeat(np.cumsum(sampled_lengths) - sampled_lengths, sampled_lengths)
    return flat_positions[np.repeat(starts[sampled], sampled_lengths) + within]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="oof_predictions.csv")
    parser.add_argument("--iterations", type=int, default=N_BOOTSTRAP)
    arguments = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTION_DIR / arguments.predictions)
    spine, predicted_by_condition = build_condition_arrays(predictions)
    true_labels = spine["gold_label"].to_numpy().astype(np.int64)
    conditions = [c for c in CONDITION_NAMES if c in predicted_by_condition]

    metric_rows = []
    for condition in conditions:
        predicted = predicted_by_condition[condition]
        metrics = full_metrics(true_labels, predicted)

        fold_scores = [
            macro_f1_from_counts(true_labels[mask], predicted[mask])
            for fold in sorted(spine["outer_fold"].unique())
            if (mask := (spine["outer_fold"].to_numpy() == fold)).any()
        ]
        metric_rows.append(
            {
                "condition": condition,
                "description": CONDITION_NAMES[condition],
                **metrics,
                "fold_mean_macro_f1": float(np.mean(fold_scores)),
                "fold_std_macro_f1": float(np.std(fold_scores, ddof=1)),
                "n_recordings": int(len(true_labels)),
            }
        )
    print("Pooled out-of-fold results")
    for row in metric_rows:
        print(
            f"  {row['condition']}  {row['description']:<32} macro-F1 {row['macro_f1']:.4f}  "
            f"acc {row['accuracy']:.4f}  fold sd {row['fold_std_macro_f1']:.4f}"
        )

    comparisons = [
        c
        for c in PREDECLARED_COMPARISONS + ADAPTATION_COMPARISONS + NEGATIVE_TRANSFER_COMPARISONS
        if c[1] in predicted_by_condition and c[2] in predicted_by_condition
    ]
    post_hoc = {name for name, _, _, _ in NEGATIVE_TRANSFER_COMPARISONS}

    flat_positions, starts, lengths = group_index_structure(spine)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = {name: np.empty(arguments.iterations) for name, _, _, _ in comparisons}
    per_condition = {condition: np.empty(arguments.iterations) for condition in conditions}

    print(f"\nRunning {arguments.iterations} sentence-clustered bootstrap iterations ...")
    for iteration in range(arguments.iterations):
        index = bootstrap_indices(flat_positions, starts, lengths, rng)
        resampled_true = true_labels[index]
        scores = {
            condition: macro_f1_from_counts(resampled_true, predicted_by_condition[condition][index])
            for condition in conditions
        }
        for condition, score in scores.items():
            per_condition[condition][iteration] = score
        for name, better, worse, _ in comparisons:
            differences[name][iteration] = scores[better] - scores[worse]

    # Per-condition intervals describe one score's precision; they are NOT a
    # significance test between conditions - the paired intervals below are.
    for row in metric_rows:
        low, high = np.percentile(per_condition[row["condition"]], [2.5, 97.5])
        row["bootstrap_ci_low"] = float(low)
        row["bootstrap_ci_high"] = float(high)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(ARTIFACT_DIR / "condition_metrics.csv", index=False, encoding="utf-8")

    results = {}
    print("\nPaired sentence-clustered bootstrap, 95% intervals")
    for name, better, worse, description in comparisons:
        observed = (
            macro_f1_from_counts(true_labels, predicted_by_condition[better])
            - macro_f1_from_counts(true_labels, predicted_by_condition[worse])
        )
        low, high = np.percentile(differences[name], [2.5, 97.5])
        results[name] = {
            "description": description,
            "comparison": f"{better} - {worse}",
            "observed_difference": float(observed),
            "ci_low": float(low),
            "ci_high": float(high),
            "excludes_zero": bool(low > 0 or high < 0),
            "predeclared": name not in post_hoc,
        }
        print(
            f"  {better} - {worse}  {description:<46} "
            f"{observed:+.4f}  [{low:+.4f}, {high:+.4f}]"
            f"{'  *' if (low > 0 or high < 0) else '   '}"
            f"{'  [post-hoc]' if name in post_hoc else ''}"
        )

    (ARTIFACT_DIR / "bootstrap_comparisons.json").write_text(
        json.dumps(
            {
                "n_iterations": arguments.iterations,
                "seed": BOOTSTRAP_SEED,
                "resampling_unit": "sentence_group_id (all recordings of a sampled group included)",
                "note": "intervals apply to the paired difference, not to a single condition's score",
                "comparisons": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    delta_of = {
        "C3": "delta_text_whisper", "C4": "delta_text_indic",
        "C6": "delta_fusion_whisper", "C7": "delta_fusion_indic",
    }
    lines = [
        "### Table 2 - Principal sentiment results (recording-level out-of-fold)",
        "",
        "| ID | Condition | Macro-F1 | Accuracy | Relevant delta | 95% CI for delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        key = delta_of.get(row["condition"])
        if key and key in results:
            entry = results[key]
            delta = f"{entry['comparison']} = {entry['observed_difference']:+.4f}"
            interval = f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}]"
        else:
            delta = interval = "-"
        lines.append(
            f"| {row['condition']} | {row['description']} | {row['macro_f1']:.4f} | "
            f"{row['accuracy']:.4f} | {delta} | {interval} |"
        )
    lines += [
        "",
        "Intervals are paired sentence-clustered bootstrap intervals on the *difference*, "
        f"{arguments.iterations} iterations, resampling sentence groups with all their recordings.",
    ]
    (ARTIFACT_DIR / "table2_principal_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {ARTIFACT_DIR / 'table2_principal_results.md'}")

    # C1 is an independently trained head, so C1-vs-C6/C7 is a SYSTEM-LEVEL
    # deployment comparison ("is this ASR-fusion system better than just using
    # speech?"), not a controlled test-time substitution contrast like C5->C6.
    # The distinction has to travel with the number or it will be read as
    # the latter.
    SYSTEM_LEVEL_CAVEAT = (
        "System-level deployment comparison: C1 is a separately trained speech-only "
        "head, so this is NOT a controlled test-time substitution contrast "
        "(unlike C5-C6 / C5-C7, which share one trained fusion head)."
    )
    negative_transfer = [
        {
            "comparison": results[name]["comparison"],
            "description": results[name]["description"],
            "observed_delta": round(results[name]["observed_difference"], 4),
            "ci_low": round(results[name]["ci_low"], 4),
            "ci_high": round(results[name]["ci_high"], 4),
            "excludes_zero": results[name]["excludes_zero"],
            "predeclared": False,
            "comparison_type": "system-level deployment (separately trained heads)",
            "caveat": SYSTEM_LEVEL_CAVEAT,
        }
        for name, _, _, _ in NEGATIVE_TRANSFER_COMPARISONS
        if name in results
    ]
    if negative_transfer:
        pd.DataFrame(negative_transfer).to_csv(
            ARTIFACT_DIR / "negative_transfer_bootstrap.csv", index=False, encoding="utf-8"
        )
        print(f"\nWrote {ARTIFACT_DIR / 'negative_transfer_bootstrap.csv'}")
        print(f"  NOTE: {SYSTEM_LEVEL_CAVEAT}")

    # Oracle-gap decomposition: how much of each oracle-to-ASR gap does matched
    # ASR training recover, and how much remains? The remainder is the residual
    # gap after THIS adaptation strategy - never call it irrecoverable, since a
    # stronger adaptation method could close more of it.
    decomposition = []
    for setting, oracle, substituted, adapted in [
        ("Whisper text", "C2", "C3", "A1"),
        ("Whisper fusion", "C5", "C6", "A3"),
        ("Bengali XLS-R text", "C2", "C4", "A2"),
        ("Bengali XLS-R fusion", "C5", "C7", "A4"),
    ]:
        if not all(c in predicted_by_condition for c in (oracle, substituted, adapted)):
            continue
        score = {
            c: macro_f1_from_counts(true_labels, predicted_by_condition[c])
            for c in (oracle, substituted, adapted)
        }
        total = score[oracle] - score[substituted]
        recovered = score[adapted] - score[substituted]
        decomposition.append(
            {
                "setting": setting,
                "oracle_condition": oracle,
                "substituted_condition": substituted,
                "adapted_condition": adapted,
                "total_oracle_gap": round(total, 4),
                "recovered_by_matched_training": round(recovered, 4),
                "recovered_fraction": round(recovered / total, 4) if total else float("nan"),
                "residual_gap_after_matched_training": round(score[oracle] - score[adapted], 4),
            }
        )
    if decomposition:
        frame = pd.DataFrame(decomposition)
        frame.to_csv(ARTIFACT_DIR / "oracle_gap_decomposition.csv", index=False, encoding="utf-8")
        lines = [
            "### Oracle-gap decomposition",
            "",
            "| Setting | Total oracle gap | Recovered by ASR-matched training | Residual gap |",
            "|---|---:|---:|---:|",
        ]
        for row in decomposition:
            lines.append(
                f"| {row['setting']} | {row['total_oracle_gap']:.4f} | "
                f"{row['recovered_by_matched_training']:.4f} "
                f"({row['recovered_fraction']:.1%}) | "
                f"{row['residual_gap_after_matched_training']:.4f} |"
            )
        lines += [
            "",
            "The last column is the residual gap after *this* adaptation strategy (A1-A4), "
            "not an irrecoverable floor: a stronger adaptation method could close more of it.",
        ]
        (ARTIFACT_DIR / "oracle_gap_decomposition.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print("\nOracle-gap decomposition")
        print(frame.to_string(index=False))

    recovery_rows = [
        (name, better, worse, description)
        for name, better, worse, description in ADAPTATION_COMPARISONS
        if name in results
    ]
    if recovery_rows:
        by_condition = {row["condition"]: row for row in metric_rows}
        table3 = [
            "### Table 3 - ASR-aware adaptation",
            "",
            "| Condition | Macro-F1 | Recovery over controlled substitution | 95% CI |",
            "|---|---:|---:|---:|",
        ]
        for name, better, worse, _ in recovery_rows:
            entry = results[name]
            table3.append(
                f"| {better} {CONDITION_NAMES[better]} | {by_condition[better]['macro_f1']:.4f} | "
                f"{better} - {worse} = {entry['observed_difference']:+.4f} | "
                f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] |"
            )
        table3 += [
            "",
            "Adaptation heads are separately trained on matched ASR transcripts. They are a "
            "secondary result; the controlled C2-C7 comparison remains the clean audit of "
            "test-time substitution.",
        ]
        (ARTIFACT_DIR / "table3_adaptation.md").write_text("\n".join(table3) + "\n", encoding="utf-8")
        print(f"Wrote {ARTIFACT_DIR / 'table3_adaptation.md'}")


if __name__ == "__main__":
    main()
