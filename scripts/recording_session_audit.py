"""Acquisition provenance: codec, bitrate and recording-session order (corrected plan sections 5 and 9).

Completes two checks the first confound pass left unfinished.

Section 5 lists six metadata fields to cross-tabulate against label. The earlier
audit covered sample rate, channels, container and speaker but not codec or
bitrate, both of which are recoverable with ffprobe.

Section 9 asks whether recording order correlates with label. Modification times
are useless here - every file was copied in a 30-second window - but on Windows
`st_ctime` preserves the original creation time, and those span November 2025 with
a distinct date per speaker. That gives real acquisition order, so sessions can be
reconstructed and tested against the label directly rather than inferred from
filename order alone.

This matters because the corpus is perfectly label-blocked (si_0001-0500 positive,
si_0501-1000 negative). If recording sessions also align with that split, then
session-level acoustic drift is confounded with the label by construction.

Writes:
    artifacts/acquisition_provenance.csv
    artifacts/recording_session_audit.md
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

SESSION_GAP_MINUTES = 30.0


def probe(path: Path) -> dict:
    """codec_name and bit_rate via ffprobe; empty strings when unavailable."""
    try:
        output = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,bit_rate",
                "-show_entries", "format=bit_rate,format_name",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        payload = json.loads(output)
        stream = (payload.get("streams") or [{}])[0]
        container = payload.get("format", {})
        bit_rate = stream.get("bit_rate") or container.get("bit_rate") or ""
        return {
            "codec": stream.get("codec_name", ""),
            "container_format": container.get("format_name", ""),
            "bit_rate": float(bit_rate) if bit_rate else float("nan"),
        }
    except Exception:  # noqa: BLE001 - recorded as missing, not fatal
        return {"codec": "", "container_format": "", "bit_rate": float("nan")}


def collect(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, record in enumerate(manifest.itertuples(index=False)):
        path = REPO_ROOT / record.audio_path
        stat = os.stat(path)
        rows.append(
            {
                "utterance_id": record.utterance_id,
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "file_bytes": stat.st_size,
                **probe(path),
            }
        )
        if position % 500 == 0:
            print(f"  {position}/{len(manifest)}", flush=True)
    return manifest.merge(pd.DataFrame(rows), on="utterance_id", validate="one_to_one")


def assign_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Split each speaker's recordings into sessions at gaps in creation time."""
    frame = frame.sort_values(["speaker_id", "created_at"]).copy()
    session_ids, order_within = [], []
    for _, block in frame.groupby("speaker_id", sort=False):
        gaps = block["created_at"].diff().dt.total_seconds().fillna(0) / 60.0
        index = (gaps > SESSION_GAP_MINUTES).cumsum()
        session_ids.extend(f"{block['speaker_id'].iloc[0]}_s{i}" for i in index)
        order_within.extend(range(len(block)))
    frame["session_id"] = session_ids
    frame["order_within_speaker"] = order_within
    return frame


def label_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    label = frame["sentiment_label"].map({1: "positive", 0: "negative"})
    table = pd.crosstab(frame[column], label)
    for value in ("positive", "negative"):
        if value not in table:
            table[value] = 0
    table = table.reset_index().rename(columns={column: "value"})
    table.insert(0, "feature", column)
    table["positive_rate"] = (
        table["positive"] / (table["positive"] + table["negative"])
    ).round(4)
    return table


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)

    cache = ARTIFACT_DIR / "acquisition_provenance.csv"
    if cache.exists():
        print(f"Reusing {cache.name}")
        frame = pd.read_csv(cache, parse_dates=["created_at", "modified_at"])
    else:
        print(f"Probing {len(manifest)} files with ffprobe ...")
        frame = assign_sessions(collect(manifest))
        frame.to_csv(cache, index=False, encoding="utf-8")

    frame["bitrate_kbps_bin"] = pd.cut(
        frame["bit_rate"] / 1000,
        [0, 64, 96, 128, 192, 1e9],
        labels=["<=64", "64-96", "96-128", "128-192", ">192"],
    ).astype(str)

    tables = pd.concat(
        [label_table(frame, c) for c in ("codec", "container_format", "bitrate_kbps_bin", "session_id")],
        ignore_index=True,
    )
    tables.to_csv(ARTIFACT_DIR / "acquisition_metadata_by_label.csv", index=False, encoding="utf-8")

    # Does acquisition order track the label within a speaker?
    order_rows = []
    for speaker, block in frame.groupby("speaker_id"):
        block = block.sort_values("created_at")
        correlation = float(
            np.corrcoef(block["order_within_speaker"], block["sentiment_label"])[0, 1]
        )
        labels = block["sentiment_label"].to_numpy()
        switches = int((labels[1:] != labels[:-1]).sum())
        order_rows.append(
            {
                "speaker_id": speaker,
                "n": len(block),
                "n_sessions": block["session_id"].nunique(),
                "first_recording": block["created_at"].min(),
                "last_recording": block["created_at"].max(),
                "span_hours": round(
                    (block["created_at"].max() - block["created_at"].min()).total_seconds() / 3600, 2
                ),
                "corr_order_vs_label": round(correlation, 4),
                "label_switches_in_order": switches,
            }
        )
    order = pd.DataFrame(order_rows)

    sessions = frame.groupby("session_id").agg(
        speaker=("speaker_id", "first"),
        n=("utterance_id", "count"),
        positive_rate=("sentiment_label", "mean"),
        start=("created_at", "min"),
        end=("created_at", "max"),
    ).reset_index()
    sessions["positive_rate"] = sessions["positive_rate"].round(3)
    pure = int(((sessions["positive_rate"] == 0) | (sessions["positive_rate"] == 1)).sum())

    print("\nAcquisition order by speaker")
    print(order.to_string(index=False))
    print(f"\nSessions (gap > {SESSION_GAP_MINUTES:.0f} min): {len(sessions)}, "
          f"label-pure: {pure}")
    print(sessions.head(20).to_string(index=False))
    print("\ncodec / bitrate by label")
    print(tables[tables["feature"].isin(["codec", "bitrate_kbps_bin"])].to_string(index=False))

    worst = order["corr_order_vs_label"].abs().max()
    if worst >= 0.7:
        verdict = (
            f"CONFIRMED - within-speaker acquisition order tracks the label almost "
            f"perfectly (max |r| = {worst:.3f}). Positive and negative material was "
            "recorded in separate blocks, so any session-level drift in gain, room, "
            "microphone or codec settings is confounded with the label by construction."
        )
    elif worst >= 0.3:
        verdict = (
            f"PARTIAL - acquisition order carries moderate label structure "
            f"(max |r| = {worst:.3f}). Report it and keep the confound caveat."
        )
    else:
        verdict = (
            f"CLEAN - acquisition order is not aligned with the label "
            f"(max |r| = {worst:.3f}), so block-recording is not indicated."
        )

    lines = [
        "### Acquisition provenance and recording-session audit",
        "",
        "Completes plan section 5 (codec, bitrate) and section 9 (recording order).",
        "",
        "#### Timestamp availability",
        "",
        "Modification times are uninformative - every file was copied within a "
        "30-second window - but creation times survive and span November 2025, with a "
        "distinct date per speaker. Acquisition order is therefore recoverable.",
        "",
        "#### Recording order vs label, by speaker",
        "",
        "| Speaker | n | Sessions | First | Last | Span (h) | corr(order, label) | Label switches |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['speaker_id']} | {r['n']} | {r['n_sessions']} | "
        f"{r['first_recording']:%Y-%m-%d %H:%M} | {r['last_recording']:%Y-%m-%d %H:%M} | "
        f"{r['span_hours']} | {r['corr_order_vs_label']:+.4f} | {r['label_switches_in_order']} |"
        for _, r in order.iterrows()
    ]
    lines += [
        "",
        "A perfectly interleaved 500/500 recording order would produce roughly 500 "
        "label switches per speaker and a correlation near zero.",
        "",
        f"**Verdict: {verdict}**",
        "",
        "#### Sessions",
        "",
        f"Splitting each speaker's recordings at creation-time gaps greater than "
        f"{SESSION_GAP_MINUTES:.0f} minutes yields {len(sessions)} sessions, of which "
        f"**{pure} contain only one sentiment class**.",
        "",
        "| Session | Speaker | n | Positive rate | Start | End |",
        "|---|---|---:|---:|---|---|",
    ]
    lines += [
        f"| {r['session_id']} | {r['speaker']} | {r['n']} | {r['positive_rate']:.3f} | "
        f"{r['start']:%Y-%m-%d %H:%M} | {r['end']:%Y-%m-%d %H:%M} |"
        for _, r in sessions.iterrows()
    ]
    lines += [
        "",
        "#### Codec and bitrate by label",
        "",
        "| Feature | Value | Positive | Negative | Positive rate |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['feature']} | {r['value']} | {int(r['positive'])} | {int(r['negative'])} "
        f"| {r['positive_rate']:.3f} |"
        for _, r in tables[tables["feature"].isin(["codec", "container_format", "bitrate_kbps_bin"])].iterrows()
    ]
    (ARTIFACT_DIR / "recording_session_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{verdict}")
    print(f"\nWrote {ARTIFACT_DIR / 'recording_session_audit.md'}")


if __name__ == "__main__":
    main()
