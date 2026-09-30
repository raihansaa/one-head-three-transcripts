"""Robustness check: recompute the four controlled gaps with near-duplicate sentences excluded.

The corpus contains paraphrase families - 60 sentence pairs exceed 0.7 token
Jaccard and 84 sentences take part in one. They inflate absolute scores in every
condition alike, so the paired differences should be largely protected, but that
is an argument, not evidence. This drops every sentence involved in such a pair
and recomputes C2-C3, C2-C4, C5-C6 and C5-C7 on what remains.

The Bengali XLS-R gaps (C2-C4, C5-C7) are the ones small enough that removing
~8% of items could realistically move them.

Everything about the estimator is unchanged: same seed-averaged probabilities,
same 10,000-iteration paired sentence-grouped bootstrap, same functions imported
from cluster_bootstrap.

Writes:
    artifacts/robustness_high_similarity_excluded.csv
    artifacts/robustness_high_similarity_excluded.json  (exclusion counts, for the paper text)
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
ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREDICTION_DIR = REPO_ROOT / "predictions"

JACCARD_THRESHOLD = 0.7

COMPARISONS = [
    ("C2", "C3", "Oracle-to-Whisper text degradation"),
    ("C2", "C4", "Oracle-to-Bengali XLS-R text degradation"),
    ("C5", "C6", "Oracle-to-Whisper fusion degradation"),
    ("C5", "C7", "Oracle-to-Bengali XLS-R fusion degradation"),
]


def excluded_sentences_and_groups() -> tuple[set[str], set[str], int]:
    """Sentences pairing with another at >= 0.7 Jaccard, their groups, and the pair count."""
    pairs = pd.read_csv(ARTIFACT_DIR / "near_duplicate_sentences.csv")
    high = pairs[pairs["jaccard"] >= JACCARD_THRESHOLD]
    sentences = set(high["sentence_id_a"]) | set(high["sentence_id_b"])

    manifest = pd.read_csv(
        REPO_ROOT / "manifests" / "banglamuse.csv", usecols=["sentence_id", "sentence_group_id"]
    ).drop_duplicates()
    group_of = dict(zip(manifest["sentence_id"], manifest["sentence_group_id"]))

    unmapped = sentences - set(group_of)
    assert not unmapped, f"sentence ids absent from the manifest: {sorted(unmapped)}"
    # Fewer groups than sentences: the two token-multiset-identical pairs were
    # already merged into single groups during corpus preparation.
    return sentences, {group_of[s] for s in sentences}, len(high)


def paired_intervals(
    spine: pd.DataFrame, predicted_by_condition: dict[str, np.ndarray]
) -> dict[tuple[str, str], dict]:
    """Observed difference and 95% paired interval for each comparison, on the rows given."""
    true_labels = spine["gold_label"].to_numpy().astype(np.int64)
    flat_positions, starts, lengths = group_index_structure(spine)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    conditions = sorted({c for pair in COMPARISONS for c in pair[:2]})
    differences = {(a, b): np.empty(N_BOOTSTRAP) for a, b, _ in COMPARISONS}

    for iteration in range(N_BOOTSTRAP):
        index = bootstrap_indices(flat_positions, starts, lengths, rng)
        resampled_true = true_labels[index]
        scores = {
            condition: macro_f1_from_counts(resampled_true, predicted_by_condition[condition][index])
            for condition in conditions
        }
        for a, b, _ in COMPARISONS:
            differences[(a, b)][iteration] = scores[a] - scores[b]

    results = {}
    for a, b, description in COMPARISONS:
        low, high = np.percentile(differences[(a, b)], [2.5, 97.5])
        results[(a, b)] = {
            "description": description,
            "observed": float(
                macro_f1_from_counts(true_labels, predicted_by_condition[a])
                - macro_f1_from_counts(true_labels, predicted_by_condition[b])
            ),
            "ci_low": float(low),
            "ci_high": float(high),
            "macro_f1_a": float(macro_f1_from_counts(true_labels, predicted_by_condition[a])),
            "macro_f1_b": float(macro_f1_from_counts(true_labels, predicted_by_condition[b])),
        }
    return results


def main() -> None:
    predictions = pd.read_csv(PREDICTION_DIR / "oof_predictions_final.csv")
    spine, predicted_by_condition = build_condition_arrays(predictions)

    sentences, excluded, n_pairs = excluded_sentences_and_groups()
    keep = ~spine["sentence_group_id"].isin(excluded).to_numpy()
    filtered_spine = spine[keep].reset_index(drop=True)
    filtered_predictions = {c: p[keep] for c, p in predicted_by_condition.items()}

    print(
        f"Sentence groups: {spine['sentence_group_id'].nunique()} total, "
        f"{len(excluded)} excluded, {filtered_spine['sentence_group_id'].nunique()} kept"
    )
    print(f"Recordings:      {len(spine)} total -> {len(filtered_spine)} kept "
          f"({100 * (1 - len(filtered_spine) / len(spine)):.1f}% removed)")

    print(f"\nRunning {N_BOOTSTRAP} iterations on the full set ...")
    full = paired_intervals(spine, predicted_by_condition)
    print(f"Running {N_BOOTSTRAP} iterations on the filtered set ...")
    filtered = paired_intervals(filtered_spine, filtered_predictions)

    rows = []
    for a, b, description in COMPARISONS:
        f, r = full[(a, b)], filtered[(a, b)]
        rows.append(
            {
                "comparison": f"{a} - {b}",
                "description": description,
                "gap_full": round(f["observed"], 4),
                "ci_low_full": round(f["ci_low"], 4),
                "ci_high_full": round(f["ci_high"], 4),
                "gap_filtered": round(r["observed"], 4),
                "ci_low_filtered": round(r["ci_low"], 4),
                "ci_high_filtered": round(r["ci_high"], 4),
                "change": round(r["observed"] - f["observed"], 4),
                "excludes_zero_filtered": bool(r["ci_low"] > 0 or r["ci_high"] < 0),
                f"macro_f1_{a}_filtered": round(r["macro_f1_a"], 4),
                f"macro_f1_{b}_filtered": round(r["macro_f1_b"], 4),
            }
        )

    frame = pd.DataFrame(rows)
    output = ARTIFACT_DIR / "robustness_high_similarity_excluded.csv"
    frame.to_csv(output, index=False, encoding="utf-8")

    changes = [abs(row["change"]) for row in rows]
    (ARTIFACT_DIR / "robustness_high_similarity_excluded.json").write_text(
        json.dumps(
            {
                "jaccard_threshold": JACCARD_THRESHOLD,
                "n_pairs": n_pairs,
                "n_sentences_excluded": len(sentences),
                "n_groups_excluded": len(excluded),
                "n_groups_total": int(spine["sentence_group_id"].nunique()),
                "n_recordings_excluded": int(len(spine) - len(filtered_spine)),
                "n_recordings_total": int(len(spine)),
                "fraction_recordings_excluded": round(1 - len(filtered_spine) / len(spine), 4),
                "n_iterations": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "min_absolute_change": round(min(changes), 4),
                "max_absolute_change": round(max(changes), 4),
                "all_filtered_intervals_exclude_zero": all(r["excludes_zero_filtered"] for r in rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nControlled gaps, full set vs high-similarity sentences excluded")
    print(f"{'':<9} {'full':>26}   {'filtered':>26}   change")
    for row in rows:
        print(
            f"  {row['comparison']:<7} "
            f"{row['gap_full']:+.4f} [{row['ci_low_full']:+.4f}, {row['ci_high_full']:+.4f}]   "
            f"{row['gap_filtered']:+.4f} [{row['ci_low_filtered']:+.4f}, {row['ci_high_filtered']:+.4f}]   "
            f"{row['change']:+.4f}"
        )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
