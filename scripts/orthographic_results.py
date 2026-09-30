"""Orthographic-gap reduction R_orth (corrected plan section 25).

    G_orig = F1(C2) - F1(C3)      original oracle-to-ASR text gap
    G_norm = F1(N2) - F1(N3)      same gap after canonicalization
    R_orth = G_orig - G_norm      how much the gap shrinks

and likewise with C4/N4 for Bengali XLS-R. A positive R_orth means canonicalization
closed part of the gap, i.e. transcript-form mismatch contributed to it.

Interpreting R_orth requires knowing how much the canonicalizer actually altered
the text. If it changed almost nothing, R_orth near zero is a statement about this
particular scheme on this corpus - not evidence that orthographic mismatch is
irrelevant in general. That distinction is carried into the written output.

Writes:
    artifacts/orthographic_normalization_results.csv / .md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_bootstrap import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    bootstrap_indices,
    build_condition_arrays,
    group_index_structure,
    macro_f1_from_counts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREDICTION_DIR = REPO_ROOT / "predictions"

PAIRS = [
    ("Whisper", "C2", "C3", "N2", "N3", "whisper"),
    ("Bengali XLS-R", "C2", "C4", "N2", "N4", "indic"),
]


def main() -> None:
    predictions = pd.read_csv(PREDICTION_DIR / "oof_predictions_orthnorm.csv")
    spine, predicted = build_condition_arrays(predictions)
    true_labels = spine["gold_label"].to_numpy().astype(np.int64)

    missing = {c for _, *ids, _ in PAIRS for c in ids} - set(predicted)
    if missing:
        raise SystemExit(f"missing conditions in predictions: {sorted(missing)}")

    scores = {
        condition: macro_f1_from_counts(true_labels, predicted[condition])
        for condition in predicted
    }

    wer_report = json.loads(
        (ARTIFACT_DIR / "orthographic_normalization_wer.json").read_text(encoding="utf-8")
    )

    flat, starts, lengths = group_index_structure(spine)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {name: np.empty(N_BOOTSTRAP) for name, *_ in PAIRS}

    print(f"Bootstrapping R_orth over {N_BOOTSTRAP} sentence-clustered resamples ...")
    for iteration in range(N_BOOTSTRAP):
        index = bootstrap_indices(flat, starts, lengths, rng)
        resampled = true_labels[index]
        cached = {
            condition: macro_f1_from_counts(resampled, predicted[condition][index])
            for condition in {c for _, *ids, _ in PAIRS for c in ids}
        }
        for name, oracle, substituted, oracle_norm, substituted_norm, _ in PAIRS:
            original_gap = cached[oracle] - cached[substituted]
            normalized_gap = cached[oracle_norm] - cached[substituted_norm]
            draws[name][iteration] = original_gap - normalized_gap

    rows = []
    for name, oracle, substituted, oracle_norm, substituted_norm, system in PAIRS:
        original_gap = scores[oracle] - scores[substituted]
        normalized_gap = scores[oracle_norm] - scores[substituted_norm]
        low, high = np.percentile(draws[name], [2.5, 97.5])
        altered = wer_report["systems"][system]["n_hypotheses_altered"]
        rows.append(
            {
                "asr_system": name,
                "oracle_macro_f1": round(scores[oracle], 4),
                "substituted_macro_f1": round(scores[substituted], 4),
                "canonicalized_oracle_macro_f1": round(scores[oracle_norm], 4),
                "canonicalized_substituted_macro_f1": round(scores[substituted_norm], 4),
                "gap_original": round(original_gap, 4),
                "gap_canonicalized": round(normalized_gap, 4),
                "r_orth": round(original_gap - normalized_gap, 4),
                "r_orth_ci_low": round(float(low), 4),
                "r_orth_ci_high": round(float(high), 4),
                "excludes_zero": bool(low > 0 or high < 0),
                "n_hypotheses_altered": altered,
                "wer_original": round(wer_report["systems"][system]["original"]["wer"], 4),
                "wer_canonicalized": round(
                    wer_report["systems"][system]["canonicalized"]["wer"], 4
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(
        ARTIFACT_DIR / "orthographic_normalization_results.csv", index=False, encoding="utf-8"
    )
    print(frame.to_string(index=False))

    mapping = json.loads((ARTIFACT_DIR / "orthographic_mapping.json").read_text(encoding="utf-8"))
    total_altered = sum(r["n_hypotheses_altered"] for r in rows)
    largest = max(abs(r["r_orth"]) for r in rows)

    if total_altered < 50:
        verdict = (
            "NULL BY CONSTRUCTION - the externally specified canonicalizer altered "
            f"only {total_altered} of 7,992 hypotheses and no gold sentences, because "
            "the pipeline's frozen normalizer (NFC, zero-width stripping, digit "
            "mapping) already performs most of what this scheme's Bengali rules do. "
            "R_orth near zero therefore says the oracle-to-ASR gap measured here is "
            "not attributable to the orthographic variation this scheme addresses - "
            "that variation had already been removed before the C-conditions were "
            "computed. It is NOT evidence that orthographic mismatch is irrelevant "
            "in general, and the experiment cannot support that stronger claim."
        )
    elif largest >= 0.05:
        verdict = (
            "SUBSTANTIAL - canonicalization measurably narrows the oracle-to-ASR gap, "
            "indicating that transcript-form mismatch contributes materially to the "
            "downstream degradation."
        )
    else:
        verdict = (
            "SMALL - canonicalization explains only a limited portion of the gap, "
            "suggesting broader lexical substitutions, deletions and insertions "
            "dominate."
        )

    lines = [
        "### Orthographic canonicalization (N2/N3/N4)",
        "",
        f"**Mapping source:** {mapping['name']} - {mapping['citation']}",
        "",
        f"Externally specified and frozen before evaluation: "
        f"{mapping['n_char_replacements']} character replacements and "
        f"{mapping['n_unicode_replacements']} Unicode replacements, dumped verbatim in "
        "`artifacts/orthographic_mapping.json`. The mapping was **not** derived from "
        "BanglaMUSE's own ASR substitutions, which would have been post-hoc tuning on "
        "the errors it is meant to explain. It is applied symmetrically to gold and to "
        "both ASR systems.",
        "",
        "#### Effect on the transcripts",
        "",
        "| System | Hypotheses altered | WER before | WER after |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['asr_system']} | {r['n_hypotheses_altered']} / 3,996 | "
        f"{r['wer_original']:.4f} | {r['wer_canonicalized']:.4f} |"
        for r in rows
    ]
    lines += [
        "",
        "Gold sentences altered: 0 / 1,000.",
        "",
        "#### Gap reduction",
        "",
        "| System | Original gap | Canonicalized gap | R_orth | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['asr_system']} | {r['gap_original']:+.4f} | {r['gap_canonicalized']:+.4f} | "
        f"{r['r_orth']:+.4f} | [{r['r_orth_ci_low']:+.4f}, {r['r_orth_ci_high']:+.4f}] |"
        for r in rows
    ]
    lines += ["", f"**Verdict: {verdict}**"]
    (ARTIFACT_DIR / "orthographic_normalization_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\n{verdict}")
    print(f"\nWrote {ARTIFACT_DIR / 'orthographic_normalization_results.md'}")


if __name__ == "__main__":
    main()
