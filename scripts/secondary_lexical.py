"""Lexical baseline: TF-IDF + logistic regression on gold text (secondary-analysis plan, E3).

Secondary analysis. Puts the near-ceiling oracle score (C2) in
context: how far do plain word and bigram counts get under the same protocol?

Everything mirrors C2 so the paired difference is meaningful:
- same outer folds and inner-validation groups;
- same normalized gold text (`gold_transcript_normalized`, normalizer v1);
- training rows are one per sentence group, using the same representative
  sentence as run_conditions.FeatureBuilder.build_group_level, so repeated gold
  text is not quadruple-counted; inner validation is group-level too;
- test rows are recordings (each recording's own sentence text), so a
  sentence's prediction counts once per recording, exactly as C2 is scored.

Tokenization is explicit whitespace splitting - the same tokens WER uses - rather
than scikit-learn's default regex, which is not designed for Bangla graphemes.
The vectorizer vocabulary and IDF are fitted on the training partition only.

Writes:
    results/lexical_baseline_oof.csv
    results/lexical_baseline_folds.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cluster_bootstrap import full_metrics, macro_f1_from_counts
from run_conditions import MANIFEST_FILE, N_OUTER_FOLDS, SPLIT_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results"

CONDITION = "TFIDF-LR"
C_GRID = (0.1, 1.0, 10.0)
MAX_ITER = 10_000


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )


def group_level(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence group - the representative build_group_level uses."""
    return rows.drop_duplicates("sentence_group_id")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["text"] = manifest["gold_transcript_normalized"].fillna("").astype(str)
    outer = pd.read_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv")
    inner = pd.read_csv(SPLIT_DIR / "inner_validation_groups.csv")
    manifest["outer_fold"] = manifest["sentence_group_id"].map(
        dict(zip(outer["sentence_group_id"], outer["outer_fold"]))
    )

    oof_rows, fold_rows = [], []
    for fold in range(N_OUTER_FOLDS):
        test = manifest[manifest["outer_fold"] == fold]
        training = manifest[manifest["outer_fold"] != fold]
        block = inner[inner["outer_fold"] == fold]
        validation_groups = set(block[block["is_inner_validation"] == 1]["sentence_group_id"])
        fit = group_level(training[~training["sentence_group_id"].isin(validation_groups)])
        validate = group_level(training[training["sentence_group_id"].isin(validation_groups)])

        vectorizer = make_vectorizer().fit(fit["text"])
        train_x = vectorizer.transform(fit["text"])
        validation_x = vectorizer.transform(validate["text"])
        train_y = fit["sentiment_label"].to_numpy()
        validation_y = validate["sentiment_label"].to_numpy().astype(np.int64)

        candidates = []
        for c in C_GRID:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER).fit(train_x, train_y)
            validation_predicted = (model.predict_proba(validation_x)[:, 1] >= 0.5).astype(np.int64)
            candidates.append(
                {
                    "C": c,
                    "model": model,
                    "validation_macro_f1": macro_f1_from_counts(validation_y, validation_predicted),
                    "n_iter": int(model.n_iter_[0]),
                    "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                }
            )
        best_score = max(candidate["validation_macro_f1"] for candidate in candidates)
        chosen = min(
            (candidate for candidate in candidates if candidate["validation_macro_f1"] == best_score),
            key=lambda candidate: candidate["C"],
        )

        probabilities = chosen["model"].predict_proba(vectorizer.transform(test["text"]))[:, 1]
        oof_rows += [
            {
                "utterance_id": row.utterance_id,
                "sentence_group_id": row.sentence_group_id,
                "speaker_id": row.speaker_id,
                "outer_fold": fold,
                "condition": CONDITION,
                "seed": 0,
                "gold_label": int(row.sentiment_label),
                "prob_negative": float(1 - probabilities[i]),
                "prob_positive": float(probabilities[i]),
                "predicted_label": int(probabilities[i] >= 0.5),
            }
            for i, row in enumerate(test.itertuples(index=False))
        ]
        fold_rows.append(
            {
                "outer_fold": fold,
                "n_training_groups": len(fit),
                "n_validation_groups": len(validate),
                "n_test_recordings": len(test),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "selected_C": chosen["C"],
                **{f"validation_macro_f1_C{c:g}": cand["validation_macro_f1"] for c, cand in zip(C_GRID, candidates)},
                **{f"n_iter_C{c:g}": cand["n_iter"] for c, cand in zip(C_GRID, candidates)},
                "all_converged": all(cand["converged"] for cand in candidates),
                "solver": "lbfgs",
                "max_iter": MAX_ITER,
            }
        )

    oof = pd.DataFrame(oof_rows)
    oof.to_csv(RESULT_DIR / "lexical_baseline_oof.csv", index=False, encoding="utf-8")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RESULT_DIR / "lexical_baseline_folds.csv", index=False, encoding="utf-8")

    metrics = full_metrics(oof["gold_label"].to_numpy(), oof["predicted_label"].to_numpy())
    print(folds[["outer_fold", "vocabulary_size", "selected_C", "all_converged"]].to_string(index=False))
    print(f"\n{CONDITION} pooled OOF: macro-F1 {metrics['macro_f1']:.4f}  accuracy {metrics['accuracy']:.4f}  "
          f"F1- {metrics['f1_negative']:.4f}  F1+ {metrics['f1_positive']:.4f}")


if __name__ == "__main__":
    main()
