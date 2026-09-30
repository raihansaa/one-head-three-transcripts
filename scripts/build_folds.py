"""Create the frozen sentence-grouped cross-validation splits (plan section 8).

Splitting happens at the level of `sentence_group_id`, never at the level of a
recording. Every sentence group carries one sentiment label and all of its
recordings (normally four, one per speaker), so assigning a group to a fold
automatically keeps its four recordings together and makes the folds
sentence-disjoint. An utterance-random split would put the same sentence in
training via one speaker and in test via another - severe lexical leakage.

Writes:
    splits/five_fold_sentence_grouped.csv   sentence-group -> outer fold
    splits/inner_validation_groups.csv      per outer fold, the 10% held out
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
SPLIT_DIR = REPO_ROOT / "splits"

N_OUTER_FOLDS = 5
INNER_VALIDATION_FRACTION = 0.10
OUTER_SEED = 42
INNER_SEED_BASE = 1000


def group_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence group, with its label and recording count."""
    conflicts = manifest.groupby("sentence_group_id")["sentiment_label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(
            f"sentence groups with inconsistent labels: {conflicts[conflicts > 1].index.tolist()}"
        )

    groups = (
        manifest.groupby("sentence_group_id")
        .agg(
            label=("sentiment_label", "first"),
            n_recordings=("utterance_id", "count"),
            n_sentences=("sentence_id", "nunique"),
            sentence_ids=("sentence_id", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values("sentence_group_id")
        .reset_index(drop=True)
    )
    return groups


def assign_outer_folds(groups: pd.DataFrame) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    groups = groups.copy()
    groups["outer_fold"] = -1
    for fold, (_, test_index) in enumerate(splitter.split(groups, groups["label"])):
        groups.loc[test_index, "outer_fold"] = fold

    if (groups["outer_fold"] < 0).any():
        raise ValueError("some sentence group was never assigned to a test fold")
    return groups


def assign_inner_validation(groups: pd.DataFrame) -> pd.DataFrame:
    """Deterministic 10% validation split of each outer-training partition."""
    records = []
    for fold in range(N_OUTER_FOLDS):
        training = groups[groups["outer_fold"] != fold]
        train_ids, validation_ids = train_test_split(
            training["sentence_group_id"].tolist(),
            test_size=INNER_VALIDATION_FRACTION,
            stratify=training["label"].tolist(),
            random_state=INNER_SEED_BASE + fold,
            shuffle=True,
        )
        records.extend(
            {"outer_fold": fold, "sentence_group_id": gid, "is_inner_validation": 0}
            for gid in sorted(train_ids)
        )
        records.extend(
            {"outer_fold": fold, "sentence_group_id": gid, "is_inner_validation": 1}
            for gid in sorted(validation_ids)
        )
    return pd.DataFrame(records)


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)

    groups = assign_outer_folds(group_table(manifest))
    inner = assign_inner_validation(groups)

    label_of_group = dict(zip(groups["sentence_group_id"], groups["label"]))
    fold_of_group = dict(zip(groups["sentence_group_id"], groups["outer_fold"]))

    outer_rows = (
        manifest[["sentence_group_id", "sentence_id", "sentiment_label"]]
        .drop_duplicates()
        .sort_values(["sentence_group_id", "sentence_id"])
        .rename(columns={"sentiment_label": "label"})
    )
    outer_rows["outer_fold"] = outer_rows["sentence_group_id"].map(fold_of_group)
    outer_rows.to_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv", index=False, encoding="utf-8")

    inner["label"] = inner["sentence_group_id"].map(label_of_group)
    inner.to_csv(SPLIT_DIR / "inner_validation_groups.csv", index=False, encoding="utf-8")

    recordings_per_group = dict(zip(groups["sentence_group_id"], groups["n_recordings"]))
    print(f"{len(groups)} sentence groups, {int(groups['n_recordings'].sum())} recordings\n")
    print("fold  test_groups  test_recordings  test_positive_rate  inner_val_groups")
    for fold in range(N_OUTER_FOLDS):
        test = groups[groups["outer_fold"] == fold]
        n_validation = int(
            inner[(inner["outer_fold"] == fold) & (inner["is_inner_validation"] == 1)].shape[0]
        )
        print(
            f"{fold:>4}  {len(test):>11}  "
            f"{sum(recordings_per_group[g] for g in test['sentence_group_id']):>15}  "
            f"{test['label'].mean():>18.3f}  {n_validation:>16}"
        )
    print(f"\nWrote splits to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
