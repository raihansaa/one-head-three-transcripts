"""Heads for the secondary analyses (secondary-analysis plan: P0, E1, input for E2).

One pass over the original folds and seeds that does three things:

1. Re-trains every original head with the unchanged recipe and checks that its
   test probabilities are bit-identical to the published predictions for
   C1-C7, A1-A4 and X1-X2. The published baseline must be recovered before any
   new condition is interpreted (plan P0).
2. Saves each head's inner-validation probabilities. The original run used them
   only for early stopping and never stored them; threshold-only adaptation (E2)
   needs them.
3. Trains the E1 nuisance-fusion control. The speech embedding is replaced by the
   primary Table 5 acquisition feature vector:
       gold_nuisance    -> N5 / N6 / N7 (gold, Whisper, XLS-R test text)
       whisper_nuisance -> NA3,  indic_nuisance -> NA4 (matched test text)

Unlike the original heads, the nuisance branch needs fitted statistics. Imputation,
one-hot encoding and standardization are fitted on each fold's training partition
only and frozen for validation, test and every transcript substitution; the
branch is then L2-normalized per row, like the embeddings.

Writes:
    results/original_baseline_validation.csv
    results/original_baseline_rerun_check.json
    results/nuisance_fusion_oof.csv
    results/nuisance_matched_fusion_oof.csv
    results/nuisance_pipeline_checks.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recording_condition_audit import PRIMARY_CATEGORICAL, PRIMARY_NUMERIC
from run_conditions import (
    ADAPTATION_EVALUATIONS,
    CROSS_ASR_EVALUATIONS,
    FINAL_SEEDS,
    HEAD_SPECIFICATIONS,
    MANIFEST_FILE,
    N_OUTER_FOLDS,
    PREDICTION_DIR,
    PRINCIPAL_EVALUATIONS,
    SENTENCE_KEYED_TEXT,
    SPLIT_DIR,
    FeatureBuilder,
    Head,
    l2_normalize,
    load_embeddings,
    predict_probabilities,
    train_head,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RESULT_DIR = REPO_ROOT / "results"
PUBLISHED_PREDICTIONS = PREDICTION_DIR / "oof_predictions_crossasr.csv"

ORIGINAL_EVALUATIONS = PRINCIPAL_EVALUATIONS + ADAPTATION_EVALUATIONS + CROSS_ASR_EVALUATIONS
ORIGINAL_HEADS = sorted({evaluation.head for evaluation in ORIGINAL_EVALUATIONS})

# Inner-validation transcript sources whose probabilities are saved, per head.
VALIDATION_SOURCES = {
    "gold_text": ("gold", "whisper", "indic"),
    "gold_fusion": ("gold", "whisper", "indic"),
    "whisper_text": ("whisper",),
    "indic_text": ("indic",),
    "whisper_fusion": ("whisper",),
    "indic_fusion": ("indic",),
}

# Nuisance head -> the transcript source it is trained and early-stopped on.
NUISANCE_HEADS = {"gold_nuisance": "gold", "whisper_nuisance": "whisper", "indic_nuisance": "indic"}
NUISANCE_EVALUATIONS = (
    ("N5", "gold_nuisance", "gold", "nuisance_fusion_oof.csv"),
    ("N6", "gold_nuisance", "whisper", "nuisance_fusion_oof.csv"),
    ("N7", "gold_nuisance", "indic", "nuisance_fusion_oof.csv"),
    ("NA3", "whisper_nuisance", "whisper", "nuisance_matched_fusion_oof.csv"),
    ("NA4", "indic_nuisance", "indic", "nuisance_matched_fusion_oof.csv"),
)


def load_nuisance_table() -> pd.DataFrame:
    """Primary Table 5 features, merged exactly as recording_condition_audit.py does."""
    features = pd.read_csv(ARTIFACT_DIR / "nuisance_features.csv")
    provenance = pd.read_csv(ARTIFACT_DIR / "acquisition_provenance.csv")
    features = features.drop(columns=["codec", "bit_rate"], errors="ignore").merge(
        provenance[["utterance_id", "codec", "bit_rate"]], on="utterance_id", validate="one_to_one"
    )
    return features.set_index("utterance_id")[PRIMARY_NUMERIC + PRIMARY_CATEGORICAL]


class NuisanceEncoder:
    """Nuisance branch with every statistic fitted on one fold's training rows."""

    def __init__(self, table: pd.DataFrame, training_ids: list[str]) -> None:
        self.table = table
        training = table.loc[training_ids]
        self.medians = training[PRIMARY_NUMERIC].median()
        self.one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.one_hot.fit(training[PRIMARY_CATEGORICAL].astype(str))
        self.scaler = StandardScaler().fit(self._raw(training))

    def _raw(self, rows: pd.DataFrame) -> np.ndarray:
        numeric = rows[PRIMARY_NUMERIC].fillna(self.medians).to_numpy(dtype=float)
        categorical = self.one_hot.transform(rows[PRIMARY_CATEGORICAL].astype(str))
        return np.concatenate([numeric, categorical], axis=1)

    def transform(self, ids: list[str]) -> np.ndarray:
        return l2_normalize(self.scaler.transform(self._raw(self.table.loc[ids])).astype(np.float32))

    def checks(self, ids: list[str]) -> dict:
        rows = self.table.loc[ids]
        one_hot = self.one_hot.transform(rows[PRIMARY_CATEGORICAL].astype(str))
        standardized = self.scaler.transform(self._raw(rows))
        return {
            "n_rows": len(ids),
            "missing_values": int(rows[PRIMARY_NUMERIC].isna().sum().sum()),
            "rows_with_unseen_category": int((one_hot.sum(axis=1) < len(PRIMARY_CATEGORICAL)).sum()),
            "zero_norm_rows": int((np.linalg.norm(standardized, axis=1) < 1e-9).sum()),
        }


def nuisance_only_sanity_check(manifest: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Table 5's logistic regression re-run on the E1 branch (training-only preprocessing).

    Same row set as Table 5 (all outer-training recordings), so a large departure
    from its 0.6687 mean-fold macro-F1 would point to a pipeline error.
    """
    labels = manifest["sentiment_label"].to_numpy()
    predicted = np.empty_like(labels)
    fold_scores = []
    for fold in range(N_OUTER_FOLDS):
        test = (manifest["outer_fold"] == fold).to_numpy()
        training_ids = manifest.loc[~test, "utterance_id"].tolist()
        encoder = NuisanceEncoder(table, training_ids)
        model = LogisticRegression(max_iter=2000, C=1.0).fit(encoder.transform(training_ids), labels[~test])
        predicted[test] = model.predict(encoder.transform(manifest.loc[test, "utterance_id"].tolist()))
        fold_scores.append(f1_score(labels[test], predicted[test], average="macro", zero_division=0))
    return {
        "pooled_oof_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "mean_fold_macro_f1": float(np.mean(fold_scores)),
        "table5_mean_fold_reference": 0.6687,
    }


def edge_window_levels() -> dict:
    """Descriptive: 0.15 s edge-window RMS relative to the whole utterance, in dB.

    Windows near 0 dB carry energy comparable to the utterance and so probably
    contain speech; the vector is therefore called acquisition-focused, not
    content-free.
    """
    features = pd.read_csv(ARTIFACT_DIR / "nuisance_features.csv")
    full = features["full_rms"].clip(lower=1e-12)
    levels = {}
    for edge in ("leading", "trailing"):
        decibels = 20 * np.log10(features[f"{edge}_rms"].clip(lower=1e-12) / full)
        levels[edge] = {
            "quantiles_db": {q: float(decibels.quantile(float(q))) for q in ("0.1", "0.5", "0.9")},
            "fraction_above_minus_10_db": float((decibels > -10).mean()),
            "fraction_above_minus_20_db": float((decibels > -20).mean()),
        }
    return levels


def nuisance_text_features(
    builder: FeatureBuilder, encoder: NuisanceEncoder, ids: list[str], text_source: str
) -> np.ndarray:
    text = np.stack([builder.text_vector(u, text_source) for u in ids])
    return np.concatenate([encoder.transform(ids), text], axis=1).astype(np.float32)


def original_training_data(
    builder: FeatureBuilder, head_name: str, fit: pd.DataFrame, validate: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly the rows and features run_conditions.py trains each head on."""
    text_source, use_speech = HEAD_SPECIFICATIONS[head_name]
    if text_source in SENTENCE_KEYED_TEXT and not use_speech:
        train_x, train_y = builder.build_group_level(fit, text_source)
        validation_x, validation_y = builder.build_group_level(validate, text_source)
    else:
        train_x = builder.build(fit["utterance_id"].tolist(), text_source, use_speech)
        train_y = fit["sentiment_label"].to_numpy()
        validation_x = builder.build(validate["utterance_id"].tolist(), text_source, use_speech)
        validation_y = validate["sentiment_label"].to_numpy()
    return train_x, train_y, validation_x, validation_y


def parameter_count(input_dim: int) -> int:
    return sum(p.numel() for p in Head(input_dim).parameters())


def prediction_rows(
    test: pd.DataFrame, fold: int, condition: str, seed: int, probabilities: np.ndarray
) -> list[dict]:
    return [
        {
            "utterance_id": row.utterance_id,
            "sentence_group_id": row.sentence_group_id,
            "speaker_id": row.speaker_id,
            "outer_fold": fold,
            "condition": condition,
            "seed": seed,
            "gold_label": int(row.sentiment_label),
            "prob_negative": float(probabilities[i, 0]),
            "prob_positive": float(probabilities[i, 1]),
        }
        for i, row in enumerate(test.itertuples(index=False))
    ]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_csv(MANIFEST_FILE)
    outer = pd.read_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv")
    inner = pd.read_csv(SPLIT_DIR / "inner_validation_groups.csv")
    manifest["outer_fold"] = manifest["sentence_group_id"].map(
        dict(zip(outer["sentence_group_id"], outer["outer_fold"]))
    )
    builder = FeatureBuilder(
        manifest,
        {
            "speech": load_embeddings("speech"),
            "text_gold": load_embeddings("text_gold"),
            "text_whisper": load_embeddings("text_whisper"),
            "text_indic": load_embeddings("text_indic"),
        },
    )
    nuisance_table = load_nuisance_table()

    rerun_rows: list[dict] = []
    validation_rows: list[dict] = []
    nuisance_rows: dict[str, list[dict]] = {}
    fold_checks: list[dict] = []
    nuisance_dim = None

    for seed in FINAL_SEEDS:
        for fold in range(N_OUTER_FOLDS):
            test = manifest[manifest["outer_fold"] == fold]
            training = manifest[manifest["outer_fold"] != fold]
            block = inner[inner["outer_fold"] == fold]
            validation_groups = set(block[block["is_inner_validation"] == 1]["sentence_group_id"])
            fit = training[~training["sentence_group_id"].isin(validation_groups)]
            validate = training[training["sentence_group_id"].isin(validation_groups)]
            fit_ids, validate_ids = fit["utterance_id"].tolist(), validate["utterance_id"].tolist()
            test_ids = test["utterance_id"].tolist()

            heads: dict[str, Head] = {}
            for head_name in ORIGINAL_HEADS:
                heads[head_name] = train_head(
                    *original_training_data(builder, head_name, fit, validate), seed, device
                )

            for evaluation in ORIGINAL_EVALUATIONS:
                features = builder.build(test_ids, evaluation.text_source, evaluation.use_speech)
                probabilities = predict_probabilities(heads[evaluation.head], features, device)
                rerun_rows += prediction_rows(test, fold, evaluation.condition, seed, probabilities)

            for head_name, sources in VALIDATION_SOURCES.items():
                use_speech = HEAD_SPECIFICATIONS[head_name][1]
                for source in sources:
                    features = builder.build(validate_ids, source, use_speech)
                    probabilities = predict_probabilities(heads[head_name], features, device)
                    validation_rows += [
                        {
                            "seed": seed,
                            "outer_fold": fold,
                            "head": head_name,
                            "validation_text_source": source,
                            "utterance_id": row.utterance_id,
                            "sentence_group_id": row.sentence_group_id,
                            "gold_label": int(row.sentiment_label),
                            "prob_positive": float(probabilities[i, 1]),
                        }
                        for i, row in enumerate(validate.itertuples(index=False))
                    ]

            encoder = NuisanceEncoder(nuisance_table, fit_ids)
            if seed == FINAL_SEEDS[0]:
                variance = encoder.scaler.var_
                fold_checks.append(
                    {
                        "outer_fold": fold,
                        "fitted_on": "training partition (outer training minus inner validation)",
                        "one_hot_categories": {
                            name: [str(v) for v in values]
                            for name, values in zip(PRIMARY_CATEGORICAL, encoder.one_hot.categories_)
                        },
                        "zero_variance_training_columns": int((variance == 0).sum()),
                        "training": encoder.checks(fit_ids),
                        "validation": encoder.checks(validate_ids),
                        "test": encoder.checks(test_ids),
                    }
                )

            nuisance_heads = {}
            for head_name, source in NUISANCE_HEADS.items():
                train_x = nuisance_text_features(builder, encoder, fit_ids, source)
                validation_x = nuisance_text_features(builder, encoder, validate_ids, source)
                nuisance_dim = train_x.shape[1] - builder.text_dim
                nuisance_heads[head_name] = train_head(
                    train_x,
                    fit["sentiment_label"].to_numpy(),
                    validation_x,
                    validate["sentiment_label"].to_numpy(),
                    seed,
                    device,
                )
            for condition, head_name, source, output in NUISANCE_EVALUATIONS:
                features = nuisance_text_features(builder, encoder, test_ids, source)
                probabilities = predict_probabilities(nuisance_heads[head_name], features, device)
                nuisance_rows.setdefault(output, [])
                nuisance_rows[output] += prediction_rows(test, fold, condition, seed, probabilities)
                predicted = probabilities.argmax(axis=1)
                accuracy = (predicted == test["sentiment_label"].to_numpy()).mean()
                print(f"  seed {seed} fold {fold} {condition}: accuracy {accuracy:.4f}", flush=True)

    # P0: the re-run must reproduce the published predictions bit for bit. The
    # default CSV float parser can be one ulp off (~1e-16), so a bitwise check
    # must parse the file exactly.
    published = pd.read_csv(PUBLISHED_PREDICTIONS, float_precision="round_trip")
    rerun = pd.DataFrame(rerun_rows)
    merged = published.merge(
        rerun, on=["utterance_id", "condition", "seed"], suffixes=("_published", "_rerun"),
        validate="one_to_one",
    )
    difference = (merged["prob_positive_published"] - merged["prob_positive_rerun"]).abs()
    rerun_labels = (merged["prob_positive_rerun"] > merged["prob_negative_rerun"]).astype(int)
    check = {
        "published_file": PUBLISHED_PREDICTIONS.name,
        "rows_published": int(len(published)),
        "rows_rerun": int(len(rerun)),
        "rows_matched": int(len(merged)),
        "max_abs_prob_positive_difference": float(difference.max()),
        "rows_with_any_difference": int((difference > 0).sum()),
        "identical_argmax_labels": int((rerun_labels == merged["predicted_label"]).sum()),
        "per_condition_max_difference": {
            c: float(d) for c, d in difference.groupby(merged["condition"]).max().items()
        },
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    check["bit_identical"] = bool(
        check["rows_matched"] == check["rows_published"] == check["rows_rerun"]
        and check["rows_with_any_difference"] == 0
    )
    (RESULT_DIR / "original_baseline_rerun_check.json").write_text(
        json.dumps(check, indent=2), encoding="utf-8"
    )
    print(f"\nRe-run vs published: {check['rows_matched']} rows, max |diff| "
          f"{check['max_abs_prob_positive_difference']:.3e}, bit-identical: {check['bit_identical']}")
    if not check["bit_identical"]:
        raise SystemExit("Published baseline NOT recovered - stop and explain before interpreting E1/E2.")

    pd.DataFrame(validation_rows).to_csv(
        RESULT_DIR / "original_baseline_validation.csv", index=False, encoding="utf-8"
    )
    for output, rows in nuisance_rows.items():
        frame = pd.DataFrame(rows)
        frame["predicted_label"] = (frame["prob_positive"] > frame["prob_negative"]).astype(int)
        frame.to_csv(RESULT_DIR / output, index=False, encoding="utf-8")

    text_dim, speech_dim = builder.text_dim, builder.speech_dim
    (RESULT_DIR / "nuisance_pipeline_checks.json").write_text(
        json.dumps(
            {
                "numeric_features": PRIMARY_NUMERIC,
                "categorical_features": PRIMARY_CATEGORICAL,
                "preprocessing": (
                    "median imputation, one-hot (handle_unknown=ignore), StandardScaler over "
                    "all columns, per-row L2 normalization; all fitted on the fold's training "
                    "partition only"
                ),
                "input_dimensions": {
                    "text_only_head": text_dim,
                    "speech_fusion_head": speech_dim + text_dim,
                    "nuisance_fusion_head": nuisance_dim + text_dim,
                    "nuisance_branch": nuisance_dim,
                },
                "parameter_counts": {
                    "text_only_head": parameter_count(text_dim),
                    "speech_fusion_head": parameter_count(speech_dim + text_dim),
                    "nuisance_fusion_head": parameter_count(nuisance_dim + text_dim),
                },
                "folds": fold_checks,
                "nuisance_only_sanity_check": nuisance_only_sanity_check(manifest, nuisance_table),
                "edge_window_levels_re_full_utterance": edge_window_levels(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote validation probabilities, N5-N7 and NA3-NA4 predictions to {RESULT_DIR}")


if __name__ == "__main__":
    main()
