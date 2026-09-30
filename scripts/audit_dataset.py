"""Build the immutable unified manifest and run the dataset audit (plan section 7).

Resolves the delivered folder's real-world messiness into one canonical
recording per (sentence, speaker):

  * m1 ships 736 extra files named `NNNN_m1(k).mp3`. Every one checked here must
    be byte-identical to its base file; identical copies are dropped, and any
    copy that differs is escalated to problematic_samples.csv rather than
    silently picked.
  * m1 also ships `00100_m1.mp3`, a five-digit spelling of 0100.
  * m2 spells four ids without zero padding (`590_m2.mp3`).
  * f1 delivers two recordings as .ogg rather than .mp3.
  * m1 has no recording at all for sentences 0174 and 0942.

Numeric ids are parsed as integers and re-padded, so padding variants collapse
automatically instead of being treated as distinct sentences.

Outputs:
    manifests/banglamuse.csv
    artifacts/data_audit.csv
    artifacts/data_summary.json
    artifacts/problematic_samples.csv
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_bangla import NORMALIZATION_VERSION, normalize_bangla  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "Multimodal Dataset"
LABEL_FILE = DATASET_ROOT / "Text_folder.xlsx"
VOICE_ROOT = DATASET_ROOT / "Voice_folder"

MANIFEST_DIR = REPO_ROOT / "manifests"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

SPEAKERS = ("f1", "f2", "m1", "m2")
EXPECTED_SENTENCES = 1000
EXPECTED_RECORDINGS_PER_SENTENCE = 4
AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")

# `0007_f1.mp3`, `00100_m1.mp3`, `590_m2.mp3`, `0004_m1(3).mp3`
FILENAME_RE = re.compile(
    r"^(?P<number>\d+)_(?P<speaker>f1|f2|m1|m2)(?:\((?P<copy>\d+)\))?$",
    re.IGNORECASE,
)

SILENCE_PEAK_THRESHOLD = 1e-4
SHORT_DURATION_THRESHOLD = 0.30


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels() -> pd.DataFrame:
    """Load the sentence/label sheet, tolerating stray whitespace in headers."""
    raw = pd.read_excel(LABEL_FILE)
    renamed = {c: str(c).strip().rstrip("\t").strip().lower() for c in raw.columns}
    df = raw.rename(columns=renamed)

    missing = {"sentence_id", "text", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"label sheet is missing columns: {sorted(missing)}")

    notes_column = next((c for c in df.columns if c.startswith("unnamed")), None)
    df["annotation_note"] = df[notes_column].fillna("") if notes_column else ""

    df = df[["sentence_id", "text", "sentiment", "annotation_note"]].copy()
    df["sentence_id"] = df["sentence_id"].astype(str).str.strip()
    df["gold_transcript_raw"] = df["text"].astype(str)
    df["gold_transcript_normalized"] = df["gold_transcript_raw"].map(normalize_bangla)

    sentiment = df["sentiment"].astype(str).str.strip().str.lower()
    unknown = sorted(set(sentiment) - {"positive", "negative"})
    if unknown:
        raise ValueError(f"unexpected sentiment values: {unknown}")
    df["sentiment_label"] = (sentiment == "positive").astype(int)
    df["sentiment_name"] = sentiment

    return df.drop(columns=["text", "sentiment"])


def scan_audio_files() -> tuple[dict[tuple[str, str], list[dict]], list[dict]]:
    """Return candidate recordings keyed by (sentence_id, speaker), plus rejects."""
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejected: list[dict] = []

    for speaker in SPEAKERS:
        folder = VOICE_ROOT / f"{speaker}_voice_folder"
        if not folder.is_dir():
            raise FileNotFoundError(f"missing speaker folder: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                rejected.append(
                    {"path": str(path.relative_to(REPO_ROOT)), "issue": "unsupported_extension"}
                )
                continue

            match = FILENAME_RE.match(path.stem)
            if match is None:
                rejected.append(
                    {"path": str(path.relative_to(REPO_ROOT)), "issue": "unparsable_filename"}
                )
                continue

            folder_speaker = speaker
            name_speaker = match.group("speaker").lower()
            if name_speaker != folder_speaker:
                rejected.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "issue": f"speaker_mismatch_folder_{folder_speaker}_name_{name_speaker}",
                    }
                )
                continue

            number = int(match.group("number"))
            sentence_id = f"si_{number:04d}"
            candidates[(sentence_id, folder_speaker)].append(
                {
                    "path": path,
                    "copy_index": int(match.group("copy")) if match.group("copy") else 0,
                    "padding_variant": len(match.group("number")) != 4,
                    "extension": path.suffix.lower(),
                }
            )

    return candidates, rejected


def choose_canonical(entries: list[dict]) -> tuple[dict, list[dict], bool]:
    """Pick the base-named copy; report whether the discarded copies differ."""
    for entry in entries:
        entry["file_hash"] = file_md5(entry["path"])

    ranked = sorted(entries, key=lambda e: (e["copy_index"], e["padding_variant"], e["path"].name))
    canonical = ranked[0]
    duplicates = ranked[1:]
    all_identical = all(d["file_hash"] == canonical["file_hash"] for d in duplicates)
    return canonical, duplicates, all_identical


def read_audio_properties(path: Path) -> dict:
    """Header metadata plus peak/RMS amplitude, used for corruption checks."""
    try:
        info = sf.info(str(path))
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        return {"read_error": f"info: {type(error).__name__}: {error}"}

    properties = {
        "duration_seconds": round(float(info.duration), 4),
        "sample_rate": int(info.samplerate),
        "num_channels": int(info.channels),
        "audio_format": info.format,
        "read_error": "",
    }

    try:
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as error:  # noqa: BLE001
        properties["read_error"] = f"read: {type(error).__name__}: {error}"
        return properties

    if data.size == 0:
        properties.update({"peak_amplitude": 0.0, "rms_amplitude": 0.0})
        return properties

    mono = data.mean(axis=1)
    properties["peak_amplitude"] = round(float(abs(mono).max()), 6)
    properties["rms_amplitude"] = round(float((mono**2).mean() ** 0.5), 6)
    return properties


def build_manifest(labels: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    candidates, problems = scan_audio_files()
    known_sentences = set(labels["sentence_id"])
    rows: list[dict] = []

    for (sentence_id, speaker), entries in sorted(candidates.items()):
        if sentence_id not in known_sentences:
            problems.append(
                {
                    "sentence_id": sentence_id,
                    "speaker_id": speaker,
                    "issue": "audio_without_label_row",
                    "detail": ", ".join(e["path"].name for e in entries),
                }
            )
            continue

        canonical, duplicates, identical = choose_canonical(entries)
        if duplicates:
            problems.append(
                {
                    "sentence_id": sentence_id,
                    "speaker_id": speaker,
                    "issue": "duplicate_copies_identical" if identical else "DUPLICATE_COPIES_DIFFER",
                    "detail": f"kept {canonical['path'].name}; dropped "
                    + ", ".join(d["path"].name for d in duplicates),
                }
            )

        properties = read_audio_properties(canonical["path"])
        rows.append(
            {
                "utterance_id": f"{sentence_id}__{speaker}",
                "sentence_id": sentence_id,
                "speaker_id": speaker,
                "audio_path": canonical["path"].relative_to(REPO_ROOT).as_posix(),
                "file_hash": canonical["file_hash"],
                "n_duplicate_copies_dropped": len(duplicates),
                "duplicate_copies_identical": identical,
                **properties,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest = manifest.merge(labels, on="sentence_id", how="left", validate="many_to_one")
    manifest = manifest.sort_values(["sentence_id", "speaker_id"]).reset_index(drop=True)
    return manifest, problems


def drop_sentence_ambiguous_audio(manifest: pd.DataFrame, problems: list[dict]) -> pd.DataFrame:
    """Remove recordings whose audio file is shared by two different sentences.

    m2 ships one file under both 0225 and 0226 (a near-paraphrase pair differing
    by a single word). Exactly one of the two gold transcripts describes that
    audio and there is no way to tell which, so both recordings are excluded
    rather than allowed to contribute a knowingly wrong WER reference. This rule
    is content-derived, not hard-coded to those ids.
    """
    hash_groups = manifest.groupby("file_hash")["sentence_id"].nunique()
    ambiguous = set(hash_groups[hash_groups > 1].index)
    if not ambiguous:
        return manifest

    affected = manifest[manifest["file_hash"].isin(ambiguous)]
    for _, row in affected.iterrows():
        partners = affected[
            (affected["file_hash"] == row["file_hash"])
            & (affected["utterance_id"] != row["utterance_id"])
        ]["utterance_id"].tolist()
        problems.append(
            {
                "sentence_id": row["sentence_id"],
                "speaker_id": row["speaker_id"],
                "issue": "EXCLUDED_audio_shared_by_multiple_sentences",
                "detail": f"{row['audio_path']} is byte-identical to {partners}; "
                "correct transcript cannot be determined",
            }
        )
    return manifest[~manifest["file_hash"].isin(ambiguous)].reset_index(drop=True)


def _token_key(text: str) -> tuple[str, ...]:
    """Order-insensitive identity of a sentence, used for duplicate grouping."""
    return tuple(sorted(text.split()))


def assign_sentence_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate normalized sentences into one splitting unit (plan 7.3).

    Two sentences share a group when their normalized text is identical, or when
    their token multisets are identical and only word order differs. The second
    rule is objective and declared in advance; it catches pairs such as
    si_0023/si_0176 that a bag-of-words-pooled text encoder cannot distinguish,
    and which would otherwise leak lexically across folds. Looser paraphrases are
    deliberately NOT merged - they are reported instead (plan 7.3).
    """
    sentences = (
        manifest[["sentence_id", "gold_transcript_normalized"]]
        .drop_duplicates("sentence_id")
        .sort_values("sentence_id")
    )

    representative: dict[tuple[str, ...], str] = {}
    group_of_sentence: dict[str, str] = {}
    for row in sentences.itertuples(index=False):
        key = _token_key(row.gold_transcript_normalized)
        representative.setdefault(key, row.sentence_id)
        group_of_sentence[row.sentence_id] = f"sg_{representative[key][3:]}"

    manifest["sentence_group_id"] = manifest["sentence_id"].map(group_of_sentence)
    return manifest


def report_near_duplicates(labels: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Token-Jaccard near-duplicate pairs, for reporting and manual inspection."""
    import itertools

    sentences = labels.drop_duplicates("sentence_id").reset_index(drop=True)
    token_sets = [set(t.split()) for t in sentences["gold_transcript_normalized"]]

    rows = []
    for i, j in itertools.combinations(range(len(sentences)), 2):
        union = token_sets[i] | token_sets[j]
        if not union:
            continue
        jaccard = len(token_sets[i] & token_sets[j]) / len(union)
        if jaccard < threshold:
            continue
        rows.append(
            {
                "jaccard": round(jaccard, 4),
                "sentence_id_a": sentences.sentence_id[i],
                "sentence_id_b": sentences.sentence_id[j],
                "label_a": int(sentences.sentiment_label[i]),
                "label_b": int(sentences.sentiment_label[j]),
                "same_label": bool(sentences.sentiment_label[i] == sentences.sentiment_label[j]),
                "text_a": sentences.gold_transcript_normalized[i],
                "text_b": sentences.gold_transcript_normalized[j],
            }
        )
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)


def run_audit(manifest: pd.DataFrame, labels: pd.DataFrame, problems: list[dict]) -> dict:
    """Every check listed in plan section 7.2, as pass/fail facts."""
    per_sentence = manifest.groupby("sentence_id").size()
    duplicate_hashes = (
        manifest.groupby("file_hash")["utterance_id"].apply(list).loc[lambda s: s.map(len) > 1]
    )

    normalized_counts = (
        labels.drop_duplicates("sentence_id")["gold_transcript_normalized"].value_counts()
    )
    duplicate_texts = normalized_counts[normalized_counts > 1]

    label_conflicts = (
        manifest.groupby("sentence_id")["sentiment_label"].nunique().loc[lambda s: s > 1]
    )

    empty_gold = labels[labels["gold_transcript_normalized"].str.len() == 0]["sentence_id"].tolist()
    missing_pairs = [
        f"{sid}__{sp}"
        for sid in labels["sentence_id"]
        for sp in SPEAKERS
        if f"{sid}__{sp}" not in set(manifest["utterance_id"])
    ]

    unreadable = manifest[manifest["read_error"].astype(str).str.len() > 0]
    silent = manifest[manifest.get("peak_amplitude", pd.Series(dtype=float)) < SILENCE_PEAK_THRESHOLD]
    too_short = manifest[manifest["duration_seconds"] < SHORT_DURATION_THRESHOLD]

    for _, row in pd.concat([unreadable, silent, too_short]).drop_duplicates("utterance_id").iterrows():
        problems.append(
            {
                "sentence_id": row["sentence_id"],
                "speaker_id": row["speaker_id"],
                "issue": "unreadable_or_degenerate_audio",
                "detail": f"duration={row['duration_seconds']}s peak={row.get('peak_amplitude')} "
                f"error={row['read_error']}",
            }
        )
    for utterance_id in missing_pairs:
        sentence_id, speaker = utterance_id.split("__")
        problems.append(
            {
                "sentence_id": sentence_id,
                "speaker_id": speaker,
                "issue": "missing_recording",
                "detail": "no audio file found for this sentence/speaker pair",
            }
        )

    non_bengali = [
        sid
        for sid, text in labels.set_index("sentence_id")["gold_transcript_normalized"].items()
        if not any("ঀ" <= ch <= "৿" for ch in text)
    ]

    return {
        "normalization_version": NORMALIZATION_VERSION,
        "n_label_rows": int(len(labels)),
        "n_unique_sentence_ids": int(labels["sentence_id"].nunique()),
        "n_sentence_groups": int(manifest["sentence_group_id"].nunique()),
        "n_usable_recordings": int(len(manifest)),
        "n_expected_recordings": EXPECTED_SENTENCES * EXPECTED_RECORDINGS_PER_SENTENCE,
        "recordings_per_sentence": {
            str(k): int(v) for k, v in per_sentence.value_counts().sort_index().items()
        },
        "sentences_with_fewer_than_four": sorted(per_sentence[per_sentence < 4].index.tolist()),
        "missing_recordings": sorted(missing_pairs),
        "label_balance": {
            "positive": int((labels["sentiment_label"] == 1).sum()),
            "negative": int((labels["sentiment_label"] == 0).sum()),
        },
        "recordings_per_speaker": {
            k: int(v) for k, v in manifest["speaker_id"].value_counts().sort_index().items()
        },
        "n_label_conflicts_within_sentence": int(len(label_conflicts)),
        "n_duplicate_audio_paths": int(manifest["audio_path"].duplicated().sum()),
        "n_duplicate_file_hashes_across_utterances": int(len(duplicate_hashes)),
        "duplicate_hash_examples": {k: v for k, v in list(duplicate_hashes.items())[:5]},
        "n_redundant_copies_dropped": int(manifest["n_duplicate_copies_dropped"].sum()),
        "all_dropped_copies_byte_identical": bool(manifest["duplicate_copies_identical"].all()),
        "n_exact_duplicate_normalized_sentences": int(duplicate_texts.sum() - len(duplicate_texts))
        if len(duplicate_texts)
        else 0,
        "n_empty_gold_transcripts": len(empty_gold),
        "n_sentences_without_bengali_characters": len(non_bengali),
        "sample_rates": {str(k): int(v) for k, v in manifest["sample_rate"].value_counts().items()},
        "channel_counts": {str(k): int(v) for k, v in manifest["num_channels"].value_counts().items()},
        "audio_formats": {str(k): int(v) for k, v in manifest["audio_format"].value_counts().items()},
        "n_unreadable_audio": int(len(unreadable)),
        "n_silent_audio": int(len(silent)),
        "n_shorter_than_threshold": int(len(too_short)),
        "duration_seconds": {
            "min": round(float(manifest["duration_seconds"].min()), 3),
            "p05": round(float(manifest["duration_seconds"].quantile(0.05)), 3),
            "median": round(float(manifest["duration_seconds"].median()), 3),
            "p95": round(float(manifest["duration_seconds"].quantile(0.95)), 3),
            "max": round(float(manifest["duration_seconds"].max()), 3),
            "total_hours": round(float(manifest["duration_seconds"].sum()) / 3600, 3),
        },
        "annotation_notes": {
            str(k): int(v)
            for k, v in labels["annotation_note"].replace("", pd.NA).dropna().value_counts().items()
        },
    }


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading labels ...")
    labels = load_labels()

    print(f"Scanning and hashing audio under {VOICE_ROOT} ...")
    manifest, problems = build_manifest(labels)
    manifest = drop_sentence_ambiguous_audio(manifest, problems)
    manifest = assign_sentence_groups(manifest)

    print("Detecting near-duplicate sentences ...")
    near_duplicates = report_near_duplicates(labels)
    near_duplicates.to_csv(
        ARTIFACT_DIR / "near_duplicate_sentences.csv", index=False, encoding="utf-8"
    )

    print("Running audit checks ...")
    summary = run_audit(manifest, labels, problems)
    high = near_duplicates[near_duplicates["jaccard"] >= 0.7]
    summary["near_duplicates"] = {
        "n_pairs_jaccard_ge_0.5": int(len(near_duplicates)),
        "n_pairs_jaccard_ge_0.7": int(len(high)),
        "n_sentences_in_a_pair_ge_0.7": int(
            len(set(high["sentence_id_a"]) | set(high["sentence_id_b"]))
        ),
        "n_pairs_ge_0.7_with_opposite_labels": int((~high["same_label"]).sum()),
        "n_sentence_groups_merged_by_token_identity": int(
            labels["sentence_id"].nunique() - manifest["sentence_group_id"].nunique()
        ),
    }

    manifest_columns = [
        "utterance_id",
        "sentence_id",
        "sentence_group_id",
        "speaker_id",
        "audio_path",
        "gold_transcript_raw",
        "gold_transcript_normalized",
        "sentiment_label",
        "sentiment_name",
        "duration_seconds",
        "sample_rate",
        "num_channels",
        "audio_format",
        "peak_amplitude",
        "rms_amplitude",
        "file_hash",
        "annotation_note",
    ]
    manifest[manifest_columns].to_csv(
        MANIFEST_DIR / "banglamuse.csv", index=False, encoding="utf-8"
    )
    manifest.to_csv(ARTIFACT_DIR / "data_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame(problems or [{"issue": "none"}]).to_csv(
        ARTIFACT_DIR / "problematic_samples.csv", index=False, encoding="utf-8"
    )
    (ARTIFACT_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {MANIFEST_DIR / 'banglamuse.csv'} with {len(manifest)} recordings.")


if __name__ == "__main__":
    main()
