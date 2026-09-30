"""Recording-condition confound audit for the speech-only baseline (corrected plan sections 4-9).

C1 reaches 0.9397 macro-F1. Speaker identity cannot explain that - every speaker
reads all 1,000 sentences at 500/500 balance, so speaker is balanced against label
by construction. The real risk is different: if positive and negative material was
recorded in separate sessions, then gain, codec, room, microphone state or noise
floor could correlate with the label, and a frozen speech encoder would happily
read that off the waveform.

This dataset is in fact perfectly label-blocked - si_0001-0500 are all positive and
si_0501-1000 all negative - so any session-level drift maps exactly onto the label.
That makes this audit load-bearing rather than routine.

Two checks:
  A. acquisition metadata cross-tabulated against label
  B. a nuisance-only classifier that sees ONLY acquisition/nuisance features and
     never lexical or prosodic content

Primary features are deliberately restricted to acquisition properties. Duration,
full-utterance RMS and spectral centroid are excluded from the primary set and
reported separately, because they legitimately carry sentence and vocal
information - if they predict sentiment that is not evidence of a recording
artifact.

Writes:
    artifacts/channel_metadata_by_label.csv
    artifacts/nuisance_only_classifier.csv
    artifacts/recording_condition_audit.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_common import TARGET_SAMPLE_RATE, load_audio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
SPLIT_DIR = REPO_ROOT / "splits"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

N_OUTER_FOLDS = 5
EDGE_SECONDS = 0.15  # leading/trailing window treated as near-silence
CLIP_THRESHOLD = 0.99

PRIMARY_NUMERIC = [
    "bytes_per_second",
    "bit_rate",
    "clipping_fraction",
    "dc_offset",
    "leading_rms",
    "trailing_rms",
    "leading_noise_floor",
    "trailing_noise_floor",
]
PRIMARY_CATEGORICAL = ["sample_rate", "num_channels", "audio_format", "codec"]
EXPLORATORY_NUMERIC = ["duration_seconds", "full_rms", "spectral_centroid"]


def signal_features(relative_path: str) -> dict:
    """Nuisance descriptors computed on the 16 kHz mono signal the encoder sees."""
    waveform = load_audio(relative_path)
    if waveform.size == 0:
        return dict.fromkeys(
            PRIMARY_NUMERIC[1:] + ["full_rms", "spectral_centroid"], 0.0
        )

    edge = max(int(EDGE_SECONDS * TARGET_SAMPLE_RATE), 1)
    leading, trailing = waveform[:edge], waveform[-edge:]
    magnitude = np.abs(np.fft.rfft(waveform))
    frequencies = np.fft.rfftfreq(len(waveform), 1 / TARGET_SAMPLE_RATE)

    return {
        "clipping_fraction": float((np.abs(waveform) >= CLIP_THRESHOLD).mean()),
        "dc_offset": float(waveform.mean()),
        "leading_rms": float(np.sqrt((leading**2).mean())),
        "trailing_rms": float(np.sqrt((trailing**2).mean())),
        "leading_noise_floor": float(np.percentile(np.abs(leading), 10)),
        "trailing_noise_floor": float(np.percentile(np.abs(trailing), 10)),
        "full_rms": float(np.sqrt((waveform**2).mean())),
        "spectral_centroid": float(
            (frequencies * magnitude).sum() / max(magnitude.sum(), 1e-9)
        ),
    }


def build_features(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, record in enumerate(manifest.itertuples(index=False)):
        path = REPO_ROOT / record.audio_path
        rows.append(
            {
                "utterance_id": record.utterance_id,
                "bytes_per_second": path.stat().st_size / max(record.duration_seconds, 1e-6),
                **signal_features(record.audio_path),
            }
        )
        if position % 500 == 0:
            print(f"  {position}/{len(manifest)}", flush=True)
    return manifest.merge(pd.DataFrame(rows), on="utterance_id", validate="one_to_one")


def cross_tabs(features: pd.DataFrame) -> pd.DataFrame:
    label = features["sentiment_label"].map({1: "positive", 0: "negative"})
    blocks = []
    for column in PRIMARY_CATEGORICAL + ["speaker_id"]:
        table = pd.crosstab(features[column], label)
        for value in ("positive", "negative"):
            if value not in table:
                table[value] = 0
        table = table.reset_index().rename(columns={column: "value"})
        table.insert(0, "feature", column)
        table["positive_rate"] = (
            table["positive"] / (table["positive"] + table["negative"])
        ).round(4)
        blocks.append(table)
    return pd.concat(blocks, ignore_index=True)


def evaluate(features: pd.DataFrame, columns: list[str], categorical: list[str]) -> pd.DataFrame:
    """Sentence-grouped CV with the scaler fitted on training folds only."""
    design = features[columns].copy()
    if categorical:
        design = pd.get_dummies(design, columns=categorical, drop_first=False)
    design = design.astype(float).fillna(0.0)

    labels = features["sentiment_label"].to_numpy()
    rows = []
    for fold in range(N_OUTER_FOLDS):
        test_mask = (features["outer_fold"] == fold).to_numpy()
        if not test_mask.any() or test_mask.all():
            continue
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        )
        model.fit(design[~test_mask], labels[~test_mask])
        predicted = model.predict(design[test_mask])
        rows.append(
            {
                "fold": fold,
                "n_test": int(test_mask.sum()),
                "macro_f1": f1_score(labels[test_mask], predicted, average="macro", zero_division=0),
                "accuracy": accuracy_score(labels[test_mask], predicted),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)
    outer = pd.read_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv")
    manifest["outer_fold"] = manifest["sentence_group_id"].map(
        dict(zip(outer["sentence_group_id"], outer["outer_fold"]))
    )

    cache = ARTIFACT_DIR / "nuisance_features.csv"
    if cache.exists():
        print(f"Reusing cached nuisance features from {cache.name}")
        features = pd.read_csv(cache)
    else:
        print(f"Computing nuisance features for {len(manifest)} recordings ...")
        features = build_features(manifest)
        features.to_csv(cache, index=False, encoding="utf-8")

    # Codec and bitrate come from ffprobe (plan section 6 lists both as recommended
    # nuisance features); recording_session_audit.py produces them.
    provenance_file = ARTIFACT_DIR / "acquisition_provenance.csv"
    if provenance_file.exists():
        provenance = pd.read_csv(provenance_file)[["utterance_id", "codec", "bit_rate"]]
        features = features.drop(columns=["codec", "bit_rate"], errors="ignore").merge(
            provenance, on="utterance_id", validate="one_to_one"
        )
        features["bit_rate"] = features["bit_rate"].fillna(features["bit_rate"].median())
    else:
        raise SystemExit(
            "Run scripts/recording_session_audit.py first - codec and bitrate are "
            "required nuisance features (plan section 6)."
        )

    tabs = cross_tabs(features)
    tabs.to_csv(ARTIFACT_DIR / "channel_metadata_by_label.csv", index=False, encoding="utf-8")

    print("\nCheck A - acquisition metadata by label")
    print(tabs.to_string(index=False))

    print("\nCheck B - nuisance-only classifier")
    primary = evaluate(features, PRIMARY_NUMERIC + PRIMARY_CATEGORICAL, PRIMARY_CATEGORICAL)
    primary["feature_set"] = "primary (acquisition only)"
    exploratory = evaluate(
        features,
        PRIMARY_NUMERIC + PRIMARY_CATEGORICAL + EXPLORATORY_NUMERIC,
        PRIMARY_CATEGORICAL,
    )
    exploratory["feature_set"] = "exploratory (+ duration, RMS, centroid)"
    metadata_only = evaluate(features, PRIMARY_CATEGORICAL, PRIMARY_CATEGORICAL)
    metadata_only["feature_set"] = "metadata only (format/rate/channels)"

    results = pd.concat([primary, metadata_only, exploratory], ignore_index=True)
    results.to_csv(ARTIFACT_DIR / "nuisance_only_classifier.csv", index=False, encoding="utf-8")

    summary = results.groupby("feature_set")[["macro_f1", "accuracy"]].mean().round(4)
    print(summary.to_string())

    # Bands are the corrected plan's, not invented here: <0.55 clean,
    # 0.55-0.65 mild, 0.80+ strong. Scores between 0.65 and 0.80 fall in a gap the
    # plan does not name, so say so rather than rounding to whichever label is
    # convenient.
    primary_f1 = float(primary["macro_f1"].mean())
    if primary_f1 < 0.55:
        verdict = (
            "CLEAN - acquisition and nuisance features do not predict sentiment above "
            "chance, giving no evidence that the speech-only baseline is driven by "
            "recording-condition leakage."
        )
    elif primary_f1 < 0.65:
        verdict = (
            "MILD - acquisition features carry some label signal. Keep C1, note the "
            "predictability, and avoid claiming C1 reflects content or prosody alone."
        )
    elif primary_f1 < 0.80:
        verdict = (
            f"SUBSTANTIAL ({primary_f1:.3f}) - above the plan's mild band (0.55-0.65) "
            "but below its strong band (0.80+). Recording conditions carry real label "
            "signal, so C1 cannot be presented as a clean content/prosody baseline and "
            "negative transfer should not lead the paper. The oracle-gap decomposition "
            "is unaffected and should carry the contribution instead."
        )
    else:
        verdict = (
            "STRONG CONFOUND - acquisition features alone predict sentiment well. Do "
            "not make negative transfer the headline; lead with the oracle-gap "
            "decomposition, which is unaffected because C2-C7 all share test audio."
        )

    sentences = manifest.drop_duplicates("sentence_id").copy()
    sentences["index"] = sentences["sentence_id"].str[3:].astype(int)
    correlation = float(np.corrcoef(sentences["index"], sentences["sentiment_label"])[0, 1])

    lines = [
        "### Recording-condition confound audit",
        "",
        "#### Label blocking",
        "",
        f"Sentence order and label are almost perfectly confounded "
        f"(point-biserial r = {correlation:+.3f}): si_0001-0500 are all positive and "
        "si_0501-1000 all negative, i.e. two runs of 500. If recording followed "
        "filename order, all positive material was captured in one contiguous block "
        "and all negative in another, so any session-level drift in gain, room, "
        "microphone or codec settings aligns exactly with the label.",
        "",
        "#### Check A - acquisition metadata by label",
        "",
        "| Feature | Value | Positive | Negative | Positive rate |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['feature']} | {r['value']} | {int(r['positive'])} | {int(r['negative'])} "
        f"| {r['positive_rate']:.3f} |"
        for _, r in tabs.iterrows()
    ]
    lines += [
        "",
        "**All 244 OGG / 16 kHz recordings belong to f1 and are 100% negative** "
        "(sentence ids 0726-0996, entirely inside the negative block). Container "
        "format alone is a perfect label giveaway on 6.1% of the corpus, and those "
        "files differ acoustically from the rest even after resampling to 16 kHz "
        "(no energy above 8 kHz, Vorbis rather than MP3 coding artefacts).",
        "",
        "#### Check B - nuisance-only classifier",
        "",
        "Logistic regression on acquisition features only, same sentence-grouped folds, "
        "scaler fitted on training folds only. Chance is 0.500.",
        "",
        "| Feature set | Macro-F1 | Accuracy |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {name} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} |"
        for name, row in summary.iterrows()
    ]
    lines += [
        "",
        f"**Verdict: {verdict}**",
        "",
        "The exploratory row adds duration, full-utterance RMS and spectral centroid. "
        "It is reported separately and is NOT the confound test: those features "
        "legitimately carry sentence length and vocal information, so predictive power "
        "there does not demonstrate a recording artefact.",
        "",
        "Note that the C2-C7 substitution comparisons are unaffected by this issue in "
        "either direction: every one of them is evaluated on the same recordings, and "
        "only the transcript source changes between them.",
    ]
    (ARTIFACT_DIR / "recording_condition_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\n{verdict}")
    print(f"\nWrote {ARTIFACT_DIR / 'recording_condition_audit.md'}")


if __name__ == "__main__":
    main()
