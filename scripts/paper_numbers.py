"""Every number the paper prints, rounded once from full precision.

The paper prints three decimals throughout. Rounding values that were stored at
four decimals would round twice (0.3445 could become 0.344 or 0.345 depending
on the unrounded value), so every number here is recomputed or read at full
precision and rounded exactly once:
    scores, differences, WER/CER    3 decimals
    rates, recovery ratios          percent with 1 decimal

Two published sources were stored only at four decimals - the cross-recognizer
comparisons and the high-similarity robustness check - so their intervals are
recomputed from the stored predictions with the unchanged bootstrap (same seed,
same resamples, so they match the published values when rounded to four places).

Writes results/paper_numbers.json  {key: printed string}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cluster_bootstrap import BOOTSTRAP_SEED, N_BOOTSTRAP, bootstrap_indices, build_condition_arrays, group_index_structure, macro_f1_from_counts
from recording_condition_audit import EXPLORATORY_NUMERIC, PRIMARY_CATEGORICAL, PRIMARY_NUMERIC
from robustness_high_similarity import excluded_sentences_and_groups

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results"
ARTIFACT_DIR = REPO_ROOT / "artifacts"


def f3(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def paired(spine: pd.DataFrame, predicted: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> dict:
    """Observed difference and 95% paired interval, exactly as cluster_bootstrap.py computes them."""
    true_labels = spine["gold_label"].to_numpy().astype(np.int64)
    flat_positions, starts, lengths = group_index_structure(spine)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    conditions = sorted({c for pair in pairs for c in pair})
    draws = {c: np.empty(N_BOOTSTRAP) for c in conditions}
    for iteration in range(N_BOOTSTRAP):
        index = bootstrap_indices(flat_positions, starts, lengths, rng)
        for c in conditions:
            draws[c][iteration] = macro_f1_from_counts(true_labels[index], predicted[c][index])
    out = {}
    for a, b in pairs:
        low, high = np.percentile(draws[a] - draws[b], [2.5, 97.5])
        observed = macro_f1_from_counts(true_labels, predicted[a]) - macro_f1_from_counts(true_labels, predicted[b])
        out[f"{a}-{b}"] = (observed, low, high)
    return out


def table5_pooled() -> dict[str, tuple[float, float, float]]:
    """Table 5 logistic regressions scored as pooled out-of-fold predictions.

    Same features, folds and model as recording_condition_audit.evaluate(); only the
    scoring changes from the mean of fold macro-F1 (as originally reported) to the
    pooled recording-level score used everywhere else. The fold mean is returned too,
    as a check that the original numbers are reproduced.
    """
    features = pd.read_csv(ARTIFACT_DIR / "nuisance_features.csv")
    provenance = pd.read_csv(ARTIFACT_DIR / "acquisition_provenance.csv")[["utterance_id", "codec", "bit_rate"]]
    features = features.drop(columns=["codec", "bit_rate"], errors="ignore").merge(
        provenance, on="utterance_id", validate="one_to_one"
    )
    features["bit_rate"] = features["bit_rate"].fillna(features["bit_rate"].median())
    labels = features["sentiment_label"].to_numpy().astype(np.int64)
    feature_sets = {
        "metadata": PRIMARY_CATEGORICAL,
        "primary": PRIMARY_NUMERIC + PRIMARY_CATEGORICAL,
        "exploratory": PRIMARY_NUMERIC + PRIMARY_CATEGORICAL + EXPLORATORY_NUMERIC,
    }
    scores = {}
    for name, columns in feature_sets.items():
        design = pd.get_dummies(features[columns], columns=PRIMARY_CATEGORICAL, drop_first=False)
        design = design.astype(float).fillna(0.0)
        predicted = np.empty_like(labels)
        fold_scores = []
        for fold in range(5):
            test = (features["outer_fold"] == fold).to_numpy()
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            predicted[test] = model.fit(design[~test], labels[~test]).predict(design[test])
            fold_scores.append(macro_f1_from_counts(labels[test], predicted[test]))
        scores[name] = (macro_f1_from_counts(labels, predicted), float((predicted == labels).mean()), float(np.mean(fold_scores)))
    return scores


def main() -> None:
    numbers: dict[str, str] = {}

    def put_interval(key: str, observed: float, low: float, high: float) -> None:
        numbers[key] = f3(observed)
        numbers[f"{key}.ci"] = f"[{f3(low)}, {f3(high)}]"

    scores = pd.read_csv(RESULT_DIR / "condition_scores.csv").set_index("condition")
    for condition, row in scores.iterrows():
        for column in ("macro_f1", "accuracy", "f1_negative", "f1_positive"):
            numbers[f"{column}.{condition}"] = f3(row[column])
        numbers[f"pospred.{condition}"] = pct(row["predicted_positive_rate"])
        for column in ("precision_positive", "recall_positive"):
            numbers[f"{column}.{condition}"] = f3(row[column])

    intervals = pd.read_csv(RESULT_DIR / "paired_gains_and_recovery_intervals.csv")
    ratio_names = {"total recovery ratio": "total", "threshold-only ratio": "threshold", "additional matched ratio": "additional"}
    for row in intervals.itertuples(index=False):
        kind = next((name for suffix, name in ratio_names.items() if row.quantity.endswith(suffix)), None)
        if kind is None:
            put_interval("diff." + row.quantity.split(" (")[0].replace(" ", ""), row.observed, row.ci_low, row.ci_high)
            continue
        key = f"ratio.{row.family.split(' ', 1)[1].replace(' ', '_')}.{kind}"
        numbers[key] = pct(row.observed)
        numbers[f"{key}.ci"] = f"[{pct(row.ci_low)}, {pct(row.ci_high)}]"

    published = json.loads((ARTIFACT_DIR / "bootstrap_comparisons.json").read_text(encoding="utf-8"))["comparisons"]
    for entry in published.values():
        put_interval("diff." + entry["comparison"].replace(" ", ""), entry["observed_difference"], entry["ci_low"], entry["ci_high"])

    predictions = pd.read_csv(REPO_ROOT / "predictions" / "oof_predictions_crossasr.csv")
    spine, predicted = build_condition_arrays(predictions)
    for key, value in paired(spine, predicted, [("A1", "X2"), ("A2", "X1"), ("X2", "C3"), ("X1", "C4")]).items():
        put_interval(f"diff.{key}", *value)

    _, excluded, _ = excluded_sentences_and_groups()
    keep = ~spine["sentence_group_id"].isin(excluded).to_numpy()
    kept_spine = spine[keep].reset_index(drop=True)
    kept = {c: p[keep] for c, p in predicted.items()}
    robust_pairs = [("C2", "C3"), ("C2", "C4"), ("C5", "C6"), ("C5", "C7")]
    full_set = paired(spine, predicted, robust_pairs)
    for key, value in paired(kept_spine, kept, robust_pairs).items():
        put_interval(f"robust.{key}", *value)
        numbers[f"robust.change.{key}"] = f"{value[0] - full_set[key][0]:+.3f}"

    asr = json.loads((ARTIFACT_DIR / "asr_metrics.json").read_text(encoding="utf-8"))["systems"]
    for system in asr:
        for column in ("wer", "cer", "substitution_rate", "deletion_rate", "insertion_rate"):
            numbers[f"asr.{system['system']}.{column}"] = f3(system[column])
    by_class = pd.read_csv(ARTIFACT_DIR / "asr_quality_by_class.csv")
    for row in by_class.itertuples(index=False):
        for column in ("wer", "cer", "substitution_rate", "deletion_rate", "insertion_rate"):
            numbers[f"asrclass.{row.asr_system}.{row.sentiment_class}.{column}"] = f3(getattr(row, column))
    bins = pd.read_csv(ARTIFACT_DIR / "wer_bin_analysis.csv")
    for row in bins.itertuples(index=False):
        prefix = f"bin.{row.asr_system}.{row.wer_bin}"
        numbers[f"{prefix}.n"] = str(row.n_utterances)
        numbers[f"{prefix}.wer"] = f3(row.mean_wer)
        text, fusion = ("acc_asr_text_C3", "acc_asr_fusion_C6") if row.asr_system == "whisper" else ("acc_asr_text_C4", "acc_asr_fusion_C7")
        for label, column in (("oracle", "acc_oracle_text_C2"), ("text", text), ("fusion", fusion), ("speech", "acc_speech_only_C1")):
            numbers[f"{prefix}.{label}"] = f3(getattr(row, column))

    nuisance = pd.read_csv(ARTIFACT_DIR / "nuisance_only_classifier.csv").groupby("feature_set")[["macro_f1", "accuracy"]].mean()
    for feature_set, row in nuisance.iterrows():
        numbers[f"nuisance.{feature_set.split()[0]}.macro_f1"] = f3(row["macro_f1"])
        numbers[f"nuisance.{feature_set.split()[0]}.accuracy"] = f3(row["accuracy"])
    for name, (pooled, accuracy, fold_mean) in table5_pooled().items():
        if abs(fold_mean - nuisance.loc[next(i for i in nuisance.index if i.startswith(name)), "macro_f1"]) > 1e-9:
            raise SystemExit(f"Table 5 {name}: re-run does not reproduce the published fold mean")
        numbers[f"nuisance.{name}.pooled_macro_f1"] = f3(pooled)
        numbers[f"nuisance.{name}.pooled_accuracy"] = f3(accuracy)

    null = json.loads((RESULT_DIR / "nuisance_permutation_null.json").read_text(encoding="utf-8"))
    numbers["perm.observed"] = f3(null["observed_pooled_oof_macro_f1"])
    numbers["perm.null_mean"] = f3(null["null_mean"])
    numbers["perm.null_sd"] = f3(null["null_sd"])
    numbers["perm.null_range"] = f"{f3(null['null_quantiles']['0.025'])}--{f3(null['null_quantiles']['0.975'])}"
    numbers["perm.null_max"] = f3(null["null_max"])
    numbers["perm.p"] = f3(null["p_value"])

    whisper = pd.read_csv(RESULT_DIR / "whisper_same_sample_diagnostic.csv")
    beam2 = whisper[whisper["configuration"].str.startswith("beam2")].iloc[0]
    numbers["beam2.wer"], numbers["beam2.cer"] = f3(beam2["wer"]), f3(beam2["cer"])
    numbers["beam2.latin"], numbers["beam2.empty"] = pct(beam2["romanization_rate"]), pct(beam2["empty_rate"])

    seeds = pd.read_csv(RESULT_DIR / "seed_scores.csv").set_index("condition")
    for condition, row in seeds.iterrows():
        numbers[f"seed.{condition}"] = " / ".join(f3(row[c]) for c in ("seed_13", "seed_42", "seed_87"))
        numbers[f"seedsd.{condition}"] = f3(row["seed_sd"])
    seed_diffs = pd.read_csv(RESULT_DIR / "seed_paired_differences.csv").set_index("comparison")
    for comparison, row in seed_diffs.iterrows():
        key = comparison.replace(" ", "")
        numbers[f"seeddiff.{key}"] = " / ".join(f"{row[c]:.3f}" for c in ("seed_13", "seed_42", "seed_87"))
        numbers[f"seeddiffsd.{key}"] = f3(row["seed_sd"])

    holdout = pd.read_csv(RESULT_DIR / "speaker_holdout_results.csv")
    for row in holdout.itertuples(index=False):
        protocol = "joint" if row.protocol.startswith("joint") else "shared"
        model = "speech" if row.model.startswith("speech") else "nuisance"
        numbers[f"holdout.{model}.{protocol}.{row.speaker}"] = f3(row.macro_f1)

    oof = pd.read_csv(RESULT_DIR / "speaker_holdout_oof.csv")
    lr = oof[oof["model"] == "nuisance_classifier"].assign(positive=lambda f: f["prob_positive"] > 0.5)
    for (speaker, label), rate in lr.groupby(["speaker_id", "gold_label"])["positive"].mean().items():
        numbers[f"holdout.nuisance.pospred.{speaker}.{'pos' if label == 1 else 'neg'}"] = pct(rate)

    checks = json.loads((RESULT_DIR / "nuisance_pipeline_checks.json").read_text(encoding="utf-8"))
    numbers["nuisance.sanity.pooled"] = f3(checks["nuisance_only_sanity_check"]["pooled_oof_macro_f1"])
    edges = checks["edge_window_levels_re_full_utterance"]
    numbers["edge.leading.median_db"] = f"{-edges['leading']['quantiles_db']['0.5']:.0f}"
    numbers["edge.trailing.within20db"] = f"{100 * edges['trailing']['fraction_above_minus_20_db']:.0f}"
    numbers["edge.trailing.within10db"] = f"{100 * edges['trailing']['fraction_above_minus_10_db']:.0f}"

    validity = json.loads((RESULT_DIR / "whisper_decoding_validity.json").read_text(encoding="utf-8"))
    audit = validity["main_run_audit"]["whisper"]
    numbers["whisper.latin_share"] = pct(audit["latin_script_outputs"] / audit["n_utterances"])

    thresholds = pd.read_csv(RESULT_DIR / "thresholds_by_fold.csv")
    for setting, block in thresholds.groupby("setting"):
        values = block["threshold"]
        numbers[f"threshold.{setting}.range"] = f"{values.min():.3f}--{values.max():.3f}"

    output = RESULT_DIR / "paper_numbers.json"
    output.write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}: {len(numbers)} printed values")


if __name__ == "__main__":
    main()
