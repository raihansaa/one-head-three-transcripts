"""Shared helpers for the two ASR transcription runs (plan sections 9 and 10.1)."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_bangla import NORMALIZATION_VERSION, normalize_bangla  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

TARGET_SAMPLE_RATE = 16_000
RESAMPLING_METHOD = "librosa.load(sr=16000, mono=True, res_type='soxr_hq')"


def load_manifest() -> pd.DataFrame:
    """Recordings ordered by duration so batches pad as little as possible."""
    manifest = pd.read_csv(MANIFEST_FILE)
    return manifest.sort_values("duration_seconds").reset_index(drop=True)


def load_audio(relative_path: str) -> np.ndarray:
    """Decode to 16 kHz mono float32.

    The delivered folder mixes 44.1/48/16 kHz, mono/stereo, MP3/OGG. Both ASR
    systems and the speech encoder expect 16 kHz mono, so every recording goes
    through this one function and the differences never reach a model.
    """
    waveform, _ = librosa.load(
        REPO_ROOT / relative_path, sr=TARGET_SAMPLE_RATE, mono=True, res_type="soxr_hq"
    )
    return waveform.astype(np.float32)


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_outputs(
    system: str,
    rows: list[dict],
    run_metadata: dict,
    *,
    force: bool = False,
) -> None:
    """Write raw and normalized transcripts plus the run record.

    Raw predictions are treated as immutable (plan 10.1): an existing raw file is
    never silently replaced.
    """
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = TRANSCRIPT_DIR / f"{system}_raw.csv"
    if raw_file.exists():
        if not force:
            raise FileExistsError(
                f"{raw_file} already exists. Raw ASR output is immutable; "
                "pass --force to supersede it (the existing file is archived, not deleted)."
            )
        # --force must never destroy raw output. Archive the existing pair under
        # the next free index instead, so a stray or concurrent re-run can be
        # reverted rather than silently desynchronising transcripts from the
        # embeddings and predictions computed off them.
        index = 1
        while (TRANSCRIPT_DIR / f"{system}_superseded{index}_raw.csv").exists():
            index += 1
        for stem in ("raw", "normalized"):
            existing = TRANSCRIPT_DIR / f"{system}_{stem}.csv"
            if existing.exists():
                existing.rename(TRANSCRIPT_DIR / f"{system}_superseded{index}_{stem}.csv")
        run_record = ARTIFACT_DIR / f"asr_{system}_run.json"
        if run_record.exists():
            run_record.rename(ARTIFACT_DIR / f"asr_{system}_superseded{index}_run.json")
        print(f"Archived previous {system} output as {system}_superseded{index}_*.csv")

    frame = pd.DataFrame(rows)
    frame["normalized_prediction"] = frame["raw_prediction"].map(normalize_bangla)
    frame["normalization_version"] = NORMALIZATION_VERSION
    frame = frame.sort_values("utterance_id").reset_index(drop=True)

    frame[["utterance_id", "raw_prediction", "inference_seconds", "error"]].to_csv(
        raw_file, index=False, encoding="utf-8"
    )
    frame[["utterance_id", "normalized_prediction", "normalization_version"]].to_csv(
        TRANSCRIPT_DIR / f"{system}_normalized.csv", index=False, encoding="utf-8"
    )

    n_empty = int((frame["normalized_prediction"].str.len() == 0).sum())
    n_failed = int((frame["error"].astype(str).str.len() > 0).sum())
    run_metadata.update(
        {
            "normalization_version": NORMALIZATION_VERSION,
            "resampling_method": RESAMPLING_METHOD,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "n_utterances": len(frame),
            "n_empty_normalized_outputs": n_empty,
            "n_failed_utterances": n_failed,
            "failed_utterance_ids": frame.loc[
                frame["error"].astype(str).str.len() > 0, "utterance_id"
            ].tolist()[:50],
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
    )
    (ARTIFACT_DIR / f"asr_{system}_run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote {raw_file} and the normalized companion ({len(frame)} rows).")
    print(f"empty normalized outputs: {n_empty}   failures: {n_failed}")
