"""Joint speaker-and-sentence holdout (post-review analysis plan, E8).

Post-review speaker-transfer diagnostic. Plain
leave-one-speaker-out would still share sentences: the other three speakers read
the same 1,000 sentences, so the held-out speaker's sentences would be in
training. Here a recording is tested only when BOTH its speaker and its sentence
group were unseen in training:

    for speaker s and original outer fold f (4 x 5 = 20 partitions):
        test        speaker s, fold f's test sentence groups
        training    the other three speakers, fold f's training groups
        validation  the other three speakers, fold f's inner-validation groups

Every recording receives exactly one joint-held-out prediction. Nothing is tuned
on the held-out speaker or the test sentence groups.

Two models, both unchanged recipes:
  - the speech-only head (C1 recipe, seeds 13/42/87, seed-averaged);
  - the Table 5 primary nuisance classifier (no validation step, so it is fitted
    on the other speakers' recordings in all of fold f's outer-training groups).

The shared-speaker scores of the original protocol are reported beside them for
reference; they come from a different protocol and are not LOSO results. Four
speakers support no population-level claim about speaker generalization.

Writes:
    results/speaker_holdout_oof.csv
    results/speaker_holdout_results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cluster_bootstrap import full_metrics
from post_review_permutation import design_matrix
from run_conditions import (
    FINAL_SEEDS,
    MANIFEST_FILE,
    N_OUTER_FOLDS,
    SPLIT_DIR,
    FeatureBuilder,
    load_embeddings,
    predict_probabilities,
    train_head,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results"
PUBLISHED_PREDICTIONS = REPO_ROOT / "predictions" / "oof_predictions_crossasr.csv"


def nuisance_classifier() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def scores(labels: np.ndarray, predicted: np.ndarray) -> dict:
    metrics = full_metrics(labels.astype(np.int64), predicted.astype(np.int64))
    return {"macro_f1": float(metrics["macro_f1"]), "accuracy": float(metrics["accuracy"]), "n": int(len(labels))}


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
        manifest, {"speech": load_embeddings("speech"), "text_gold": load_embeddings("text_gold")}
    )
    design = design_matrix(manifest)
    labels = manifest["sentiment_label"].to_numpy()
    speakers = sorted(manifest["speaker_id"].unique())

    rows = []
    for speaker in speakers:
        for fold in range(N_OUTER_FOLDS):
            block = inner[inner["outer_fold"] == fold]
            validation_groups = set(block[block["is_inner_validation"] == 1]["sentence_group_id"])
            test_mask = ((manifest["outer_fold"] == fold) & (manifest["speaker_id"] == speaker)).to_numpy()
            train_mask = ((manifest["outer_fold"] != fold) & (manifest["speaker_id"] != speaker)).to_numpy()
            validation_mask = train_mask & manifest["sentence_group_id"].isin(validation_groups).to_numpy()
            fit_mask = train_mask & ~validation_mask

            ids = manifest["utterance_id"].to_numpy()
            test_x = builder.build(ids[test_mask].tolist(), None, True)
            for seed in FINAL_SEEDS:
                head = train_head(
                    builder.build(ids[fit_mask].tolist(), None, True), labels[fit_mask],
                    builder.build(ids[validation_mask].tolist(), None, True), labels[validation_mask],
                    seed, device,
                )
                probabilities = predict_probabilities(head, test_x, device)[:, 1]
                rows += [
                    {"utterance_id": u, "speaker_id": speaker, "outer_fold": fold, "model": "speech_head",
                     "seed": seed, "gold_label": int(y), "prob_positive": float(p)}
                    for u, y, p in zip(ids[test_mask], labels[test_mask], probabilities)
                ]

            model = nuisance_classifier().fit(design[train_mask], labels[train_mask])
            probabilities = model.predict_proba(design[test_mask])[:, 1]
            rows += [
                {"utterance_id": u, "speaker_id": speaker, "outer_fold": fold, "model": "nuisance_classifier",
                 "seed": 0, "gold_label": int(y), "prob_positive": float(p)}
                for u, y, p in zip(ids[test_mask], labels[test_mask], probabilities)
            ]
            print(f"  held-out speaker {speaker}, fold {fold}: {int(test_mask.sum())} test recordings", flush=True)

    oof = pd.DataFrame(rows)
    oof.to_csv(RESULT_DIR / "speaker_holdout_oof.csv", index=False, encoding="utf-8")
    joint = (
        oof.groupby(["model", "utterance_id"])
        .agg(speaker_id=("speaker_id", "first"), gold_label=("gold_label", "first"),
             prob_positive=("prob_positive", "mean"))
        .reset_index()
    )
    # speech_head: seed-averaged; nuisance classifier: LogisticRegression.predict convention (p > 0.5).
    joint["predicted"] = np.where(
        joint["model"] == "speech_head", joint["prob_positive"] >= 0.5, joint["prob_positive"] > 0.5
    ).astype(int)
    if joint.groupby("model")["utterance_id"].nunique().ne(len(manifest)).any():
        raise SystemExit("some recording did not receive exactly one joint-held-out prediction")

    # Reference: the original shared-speaker protocol on the same models.
    published = pd.read_csv(PUBLISHED_PREDICTIONS)
    c1 = published[published["condition"] == "C1"].groupby("utterance_id").agg(
        speaker_id=("speaker_id", "first"), gold_label=("gold_label", "first"), prob_positive=("prob_positive", "mean")
    )
    c1["predicted"] = (c1["prob_positive"] >= 0.5).astype(int)
    shared_nuisance = np.empty_like(labels)
    for fold in range(N_OUTER_FOLDS):
        test_mask = (manifest["outer_fold"] == fold).to_numpy()
        shared_nuisance[test_mask] = nuisance_classifier().fit(design[~test_mask], labels[~test_mask]).predict(design[test_mask])
    shared = manifest[["utterance_id", "speaker_id"]].assign(gold_label=labels, predicted=shared_nuisance)

    results = []
    for model, protocol, frame in (
        ("speech_head", "joint speaker-and-sentence holdout (post-review)", joint[joint["model"] == "speech_head"]),
        ("nuisance_classifier", "joint speaker-and-sentence holdout (post-review)", joint[joint["model"] == "nuisance_classifier"]),
        ("speech_head (C1)", "shared-speaker sentence-grouped CV (original protocol)", c1.reset_index()),
        ("nuisance_classifier", "shared-speaker sentence-grouped CV (original protocol)", shared),
    ):
        for speaker in speakers + ["pooled"]:
            subset = frame if speaker == "pooled" else frame[frame["speaker_id"] == speaker]
            results.append({"model": model, "protocol": protocol, "speaker": speaker,
                            **scores(subset["gold_label"].to_numpy(), subset["predicted"].to_numpy())})
    table = pd.DataFrame(results)
    table.to_csv(RESULT_DIR / "speaker_holdout_results.csv", index=False, encoding="utf-8")
    print(table.pivot_table(index=["model", "protocol"], columns="speaker", values="macro_f1").round(4).to_string())


if __name__ == "__main__":
    main()
