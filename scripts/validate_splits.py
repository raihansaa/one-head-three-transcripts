"""Mandatory split-leakage assertions (plan section 8.4).

Run after build_folds.py and again before the Friday gate. A non-zero exit means
the evaluation protocol is unsound and no result from it may be reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
SPLIT_DIR = REPO_ROOT / "splits"

N_OUTER_FOLDS = 5

_failures: list[str] = []
_checks = 0


def check(condition: bool, description: str, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  PASS  {description}")
        return
    print(f"  FAIL  {description}" + (f"  [{detail}]" if detail else ""))
    _failures.append(description)


def main() -> int:
    manifest = pd.read_csv(MANIFEST_FILE)
    outer = pd.read_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv")
    inner = pd.read_csv(SPLIT_DIR / "inner_validation_groups.csv")

    fold_of_group = dict(zip(outer["sentence_group_id"], outer["outer_fold"]))
    manifest["outer_fold"] = manifest["sentence_group_id"].map(fold_of_group)
    text_of_group = (
        manifest.groupby("sentence_group_id")["gold_transcript_normalized"].apply(set).to_dict()
    )

    print("Structure")
    all_groups = set(manifest["sentence_group_id"])
    check(
        manifest["outer_fold"].notna().all(),
        "every recording maps to an outer fold",
        f"{int(manifest['outer_fold'].isna().sum())} unmapped",
    )
    check(
        set(outer["sentence_group_id"]) == all_groups,
        "split file covers exactly the manifest's sentence groups",
    )
    check(
        outer.groupby("sentence_group_id")["outer_fold"].nunique().eq(1).all(),
        "each sentence group has exactly one outer fold",
    )
    check(
        manifest.groupby("sentence_group_id")["outer_fold"].nunique().eq(1).all(),
        "all recordings of a sentence group share one outer fold",
    )
    check(
        manifest.groupby("sentence_id")["outer_fold"].nunique().eq(1).all(),
        "all recordings of a sentence share one outer fold",
    )
    check(
        sorted(outer["outer_fold"].unique()) == list(range(N_OUTER_FOLDS)),
        f"exactly {N_OUTER_FOLDS} outer folds present",
    )

    print("\nOuter-fold disjointness")
    for fold in range(N_OUTER_FOLDS):
        test_groups = set(outer[outer["outer_fold"] == fold]["sentence_group_id"])
        train_groups = all_groups - test_groups
        check(
            not (test_groups & train_groups),
            f"fold {fold}: zero train/test sentence-group overlap",
        )
        train_text = {t for g in train_groups for t in text_of_group[g]}
        test_text = {t for g in test_groups for t in text_of_group[g]}
        shared = train_text & test_text
        check(
            not shared,
            f"fold {fold}: zero normalized-transcript overlap between train and test",
            f"{len(shared)} shared strings",
        )

    print("\nInner validation")
    for fold in range(N_OUTER_FOLDS):
        test_groups = set(outer[outer["outer_fold"] == fold]["sentence_group_id"])
        block = inner[inner["outer_fold"] == fold]
        validation = set(block[block["is_inner_validation"] == 1]["sentence_group_id"])
        training = set(block[block["is_inner_validation"] == 0]["sentence_group_id"])

        check(not (validation & training), f"fold {fold}: inner train/validation disjoint")
        check(not (validation & test_groups), f"fold {fold}: inner validation excludes outer test")
        check(
            validation | training == all_groups - test_groups,
            f"fold {fold}: inner split covers the whole outer-training partition",
        )
        validation_text = {t for g in validation for t in text_of_group[g]}
        training_text = {t for g in training for t in text_of_group[g]}
        check(
            not (validation_text & training_text),
            f"fold {fold}: zero normalized-transcript overlap between inner train and validation",
        )

    print("\nCoverage")
    test_counts = outer.groupby("sentence_group_id").size()
    check(
        (test_counts >= 1).all(),
        "every sentence group appears in the test partition exactly once across folds",
    )
    recordings_per_fold = manifest.groupby("outer_fold").size()
    check(
        int(recordings_per_fold.sum()) == len(manifest),
        "fold recording counts sum to the manifest size",
        f"{int(recordings_per_fold.sum())} vs {len(manifest)}",
    )
    check(
        manifest["utterance_id"].is_unique,
        "utterance ids are unique, so one OOF prediction per recording is well defined",
    )

    print("\nLabel balance by fold")
    for fold in range(N_OUTER_FOLDS):
        test = manifest[manifest["outer_fold"] == fold]
        rate = test["sentiment_label"].mean()
        print(f"  fold {fold}: {len(test):>4} recordings, positive rate {rate:.3f}")
        check(0.40 <= rate <= 0.60, f"fold {fold}: test positive rate within [0.40, 0.60]")

    print(f"\n{_checks - len(_failures)}/{_checks} checks passed")
    if _failures:
        print("\nFAILED:")
        for description in _failures:
            print(f"  - {description}")
        return 1
    print("All split-leakage assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
