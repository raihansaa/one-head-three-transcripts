"""Whole-pipeline permutation null for the nuisance classifier (post-review analysis plan, E4).

Post-review. The Table 5 classifier reaches 0.6687 mean-fold
macro-F1 from acquisition features. The paper compared that to "chance 0.500";
this replaces that with an empirical null.

Null hypothesis (group exchangeability): which label a sentence group carries is
unrelated to the acquisition features of its recordings. Each permutation
therefore shuffles the 998 group labels BETWEEN sentence groups and broadcasts
each permuted label to every recording of its group.

Note scikit-learn's permutation_test_score(groups=...) shuffles labels WITHIN
groups; with one constant label per sentence group that randomizes nothing, so a
custom loop is used.

The splitter is label-stratified, so every permutation regenerates the outer
folds with the original rule (build_folds.assign_outer_folds: StratifiedKFold,
5 folds, shuffle, seed 42, over the group table) and refits the whole pipeline.
The observed statistic runs the same procedure on the true labels, which
reproduces the frozen folds (asserted). The pipeline has no validation-based
model selection, so no inner split is involved.

Interpretation limit: a small p-value supports predictability relative to this
null only. It does not remove session artifacts, show which signal the speech
encoder uses, or license inference to independently sampled sessions.

Writes:
    results/nuisance_permutation_scores.csv
    results/nuisance_permutation_null.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_folds import assign_outer_folds, group_table
from recording_condition_audit import N_OUTER_FOLDS, PRIMARY_CATEGORICAL, PRIMARY_NUMERIC
from run_conditions import MANIFEST_FILE, SPLIT_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RESULT_DIR = REPO_ROOT / "results"

N_PERMUTATIONS = 1000
PERMUTATION_SEED = 20260922


def design_matrix(manifest: pd.DataFrame) -> np.ndarray:
    """The Table 5 primary design, built exactly as recording_condition_audit.py builds it."""
    features = pd.read_csv(ARTIFACT_DIR / "nuisance_features.csv")
    provenance = pd.read_csv(ARTIFACT_DIR / "acquisition_provenance.csv")
    features = features.drop(columns=["codec", "bit_rate"], errors="ignore").merge(
        provenance[["utterance_id", "codec", "bit_rate"]], on="utterance_id", validate="one_to_one"
    )
    features["bit_rate"] = features["bit_rate"].fillna(features["bit_rate"].median())
    features = features.set_index("utterance_id").loc[manifest["utterance_id"]]
    design = pd.get_dummies(
        features[PRIMARY_NUMERIC + PRIMARY_CATEGORICAL], columns=PRIMARY_CATEGORICAL, drop_first=False
    )
    return design.astype(float).fillna(0.0).to_numpy()


def outer_folds(manifest: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Regenerate the outer folds from a label vector with the original splitter."""
    relabelled = manifest.assign(sentiment_label=labels)
    groups = assign_outer_folds(group_table(relabelled))
    return relabelled["sentence_group_id"].map(dict(zip(groups["sentence_group_id"], groups["outer_fold"]))).to_numpy()


def run_pipeline(design: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    """Pooled out-of-fold macro-F1 (the test statistic) and the mean of fold macro-F1."""
    predicted = np.empty_like(labels)
    fold_scores = []
    for fold in range(N_OUTER_FOLDS):
        test = folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        model.fit(design[~test], labels[~test])
        predicted[test] = model.predict(design[test])
        fold_scores.append(f1_score(labels[test], predicted[test], average="macro", zero_division=0))
    return float(f1_score(labels, predicted, average="macro", zero_division=0)), float(np.mean(fold_scores))


def permuted_labels(
    group_ids: np.ndarray, group_label: pd.Series, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle labels between sentence groups; broadcast to every recording of a group."""
    shuffled = dict(zip(group_label.index, rng.permutation(group_label.to_numpy())))
    return np.array([shuffled[g] for g in group_ids])


def run_permutation(manifest: pd.DataFrame, design: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return run_pipeline(design, labels, outer_folds(manifest, labels))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)
    labels = manifest["sentiment_label"].to_numpy()
    group_ids = manifest["sentence_group_id"].to_numpy()

    group_label = manifest.groupby("sentence_group_id")["sentiment_label"].agg(["first", "nunique"])
    if (group_label["nunique"] != 1).any():
        raise SystemExit("a sentence group carries more than one label")
    group_label = group_label["first"]

    frozen = pd.read_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv")
    frozen_folds = manifest["sentence_group_id"].map(
        dict(zip(frozen["sentence_group_id"], frozen["outer_fold"]))
    ).to_numpy()
    observed_folds = outer_folds(manifest, labels)
    if not np.array_equal(observed_folds, frozen_folds):
        raise SystemExit("regenerated folds differ from the frozen split file")

    design = design_matrix(manifest)
    observed_pooled, observed_mean_fold = run_pipeline(design, labels, observed_folds)
    print(f"Observed: pooled OOF macro-F1 {observed_pooled:.4f}, mean fold macro-F1 {observed_mean_fold:.4f}")

    rng = np.random.default_rng(PERMUTATION_SEED)
    label_sets = [permuted_labels(group_ids, group_label, rng) for _ in range(N_PERMUTATIONS)]
    print(f"Running {N_PERMUTATIONS} whole-pipeline permutations ...", flush=True)
    null = Parallel(n_jobs=-1)(delayed(run_permutation)(manifest, design, y) for y in label_sets)

    scores = pd.DataFrame(null, columns=["pooled_oof_macro_f1", "mean_fold_macro_f1"])
    scores.insert(0, "permutation", np.arange(1, N_PERMUTATIONS + 1))
    scores["positive_recordings"] = [int(y.sum()) for y in label_sets]
    scores.to_csv(RESULT_DIR / "nuisance_permutation_scores.csv", index=False, encoding="utf-8")

    pooled = scores["pooled_oof_macro_f1"].to_numpy()
    exceed = int((pooled >= observed_pooled).sum())
    summary = {
        "feature_set": "primary Table 5 acquisition features",
        "pipeline": "get_dummies + StandardScaler + LogisticRegression(C=1.0, max_iter=2000), "
                    "fitted on outer-training rows of each fold",
        "statistic": "pooled recording-level out-of-fold macro-F1",
        "null": "labels shuffled between sentence groups (group exchangeability), broadcast "
                "to all recordings of a group; outer folds regenerated per permutation with "
                "the original splitter; whole pipeline refitted",
        "n_permutations": N_PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "observed_pooled_oof_macro_f1": observed_pooled,
        "observed_mean_fold_macro_f1": observed_mean_fold,
        "observed_folds_match_frozen_splits": True,
        "null_mean": float(pooled.mean()),
        "null_sd": float(pooled.std(ddof=1)),
        "null_quantiles": {
            q: float(np.quantile(pooled, float(q))) for q in ("0.025", "0.5", "0.975", "0.99")
        },
        "null_max": float(pooled.max()),
        "n_null_at_or_above_observed": exceed,
        "p_value": (1 + exceed) / (N_PERMUTATIONS + 1),
        "interpretation_limit": (
            "Supports predictability relative to this group-exchangeability null only; does "
            "not remove session artifacts, identify the signal used by the speech encoder, "
            "or support inference to independently sampled recording sessions."
        ),
    }
    (RESULT_DIR / "nuisance_permutation_null.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Null pooled macro-F1: mean {summary['null_mean']:.4f}, sd {summary['null_sd']:.4f}, "
          f"max {summary['null_max']:.4f}; p = {summary['p_value']:.4f}")


if __name__ == "__main__":
    main()
