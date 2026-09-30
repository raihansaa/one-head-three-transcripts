"""Whisper beam-2 diagnostic row and decoding audit (post-review analysis plan, E5).

Post-review. The decoding diagnostic (paper Table 14) compared greedy and beam 5 on a
300-utterance sample, but the experiments decode with beam 2, which was never
measured on that sample. This:

1. Re-draws the sample exactly as diagnose_whisper.py drew it (75 per speaker,
   random_state 20260807) and confirms recovery by reproducing the published
   greedy row from the cached full-corpus greedy transcripts.
2. Scores the cached main-run beam-2 hypotheses on the same IDs with the same
   normalizer and the same score() function. These ARE the transcripts every
   downstream condition used, so no new ASR run is needed.
3. Audits the main run for decoding failures: token-cap truncation, repetition
   loops, Latin-script output, empty output, audio-loading errors, forced
   task/language - and asks whether Whisper's high insertion rate comes from
   repetition loops.
4. Records the generation settings actually used, including Whisper's
   long-form options, as set / unset / inactive in the short-form path taken.

Writes:
    results/whisper_same_sample_diagnostic.csv
    results/whisper_diagnostic_sample_ids.csv
    results/whisper_decoding_validity.json
    results/whisper_failure_sample.csv
    results/whisper_generation_manifest.json
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import GenerationConfig, WhisperTokenizer

from diagnose_whisper import CHECKPOINT, CONFIGURATIONS, LATIN, PER_SPEAKER, SAMPLE_SEED, score

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
RESULT_DIR = REPO_ROOT / "results"

MAIN_RUN_BUDGET = 200
DIAGNOSTIC_BUDGET = 128
WHISPER_SEGMENT_SECONDS = 30.0
COMPRESSION_RATIO_THRESHOLD = 2.4  # OpenAI Whisper's repetition heuristic, used here descriptively
LOOP_MAX_NGRAM, LOOP_MIN_COPIES = 4, 3
DEFECT_TRUNCATION_FRACTION = 0.01
N_FAILURE_EXAMPLES = 20


def repetition_loop(tokens: list[str]) -> bool:
    """True if some n-gram (n <= 4) appears at least three times back to back."""
    for n in range(1, LOOP_MAX_NGRAM + 1):
        for start in range(len(tokens) - n * LOOP_MIN_COPIES + 1):
            unit = tokens[start : start + n]
            if all(tokens[start + k * n : start + (k + 1) * n] == unit for k in range(1, LOOP_MIN_COPIES)):
                return True
    return False


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def load_hypotheses(stem: str) -> pd.DataFrame:
    normalized = pd.read_csv(TRANSCRIPT_DIR / f"{stem}_normalized.csv")
    raw = pd.read_csv(TRANSCRIPT_DIR / f"{stem}_raw.csv")
    frame = normalized.merge(raw, on="utterance_id", validate="one_to_one")
    frame["normalized_prediction"] = frame["normalized_prediction"].fillna("").astype(str)
    frame["raw_prediction"] = frame["raw_prediction"].fillna("").astype(str)
    return frame.set_index("utterance_id")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_FILE)
    run = json.loads((ARTIFACT_DIR / "asr_whisper_run.json").read_text(encoding="utf-8"))
    greedy_run = json.loads((ARTIFACT_DIR / "asr_whisper_greedy_run.json").read_text(encoding="utf-8"))
    published = json.loads((ARTIFACT_DIR / "whisper_decoding_diagnostic.json").read_text(encoding="utf-8"))
    tokenizer = WhisperTokenizer.from_pretrained(CHECKPOINT, revision=run["revision"], local_files_only=True)

    def token_length(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    # 1. The sample, drawn exactly as diagnose_whisper.py draws it.
    subset = (
        manifest.groupby("speaker_id", group_keys=False)
        .sample(n=PER_SPEAKER, random_state=SAMPLE_SEED)
        .sort_values("utterance_id")
        .reset_index(drop=True)
    )
    subset[["utterance_id", "speaker_id", "sentence_id"]].to_csv(
        RESULT_DIR / "whisper_diagnostic_sample_ids.csv", index=False, encoding="utf-8"
    )
    ids = subset["utterance_id"].tolist()
    references = subset["gold_transcript_normalized"].fillna("").astype(str).tolist()

    beam2 = load_hypotheses("whisper")
    greedy = load_hypotheses("whisper_greedy")

    rows = []
    for name, entry in published["configurations"].items():
        rows.append({"configuration": name, "source": "published Table 10 run (diagnose_whisper.py)",
                     "num_beams": CONFIGURATIONS[name]["num_beams"], "max_new_tokens": DIAGNOSTIC_BUDGET,
                     **{k: entry[k] for k in ("wer", "cer", "romanization_rate", "empty_rate")}})
    for name, frame, config, budget in (
        ("greedy_bn (cached full-corpus greedy run)", greedy, greedy_run, greedy_run["decoding"]["max_new_tokens"]),
        ("beam2_bn (cached main run - the experiments' operating point)", beam2, run, run["decoding"]["max_new_tokens"]),
    ):
        hypotheses = frame.loc[ids, "normalized_prediction"].tolist()
        lengths = [token_length(t) for t in frame.loc[ids, "raw_prediction"]]
        rows.append(
            {
                "configuration": name,
                "source": f"transcripts rescored on the same {len(ids)} IDs",
                "num_beams": config["decoding"]["num_beams"],
                "early_stopping": config["decoding"].get("early_stopping", False),
                "max_new_tokens": budget,
                **score(references, hypotheses),
                "n_retokenized_at_or_above_budget_minus_1": int(sum(n >= budget - 1 for n in lengths)),
                "max_retokenized_length": int(max(lengths)),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(1, "n_utterances", len(ids))
    table.to_csv(RESULT_DIR / "whisper_same_sample_diagnostic.csv", index=False, encoding="utf-8")

    greedy_examples = greedy.loc[ids[:3], "normalized_prediction"].tolist()
    recovered = {
        "greedy_wer_reproduced": bool(
            abs(table.iloc[len(published["configurations"])]["wer"] - published["configurations"]["greedy_bn"]["wer"]) < 1e-12
        ),
        "published_greedy_wer": published["configurations"]["greedy_bn"]["wer"],
        "rescored_cached_greedy_wer": float(table.iloc[len(published["configurations"])]["wer"]),
        "example_outputs_match": greedy_examples == published["configurations"]["greedy_bn"]["example_outputs"],
    }

    # 3. Decoding audit of the full main run (beam 2).
    metrics = pd.read_csv(ARTIFACT_DIR / "asr_metrics_by_utterance.csv")
    audit = manifest[["utterance_id", "speaker_id", "gold_transcript_normalized", "duration_seconds"]].merge(
        metrics[["utterance_id", "asr_system", "n_reference_words", "n_hypothesis_words", "word_insertions",
                 "utterance_wer"]],
        on="utterance_id",
    )
    system_flags = {}
    for system, stem, budget in (("whisper", "whisper", MAIN_RUN_BUDGET), ("indic", "indic", None)):
        block = audit[audit["asr_system"] == system].set_index("utterance_id")
        normalized = pd.read_csv(TRANSCRIPT_DIR / f"{stem}_normalized.csv").set_index("utterance_id")
        text = normalized.loc[block.index, "normalized_prediction"].fillna("").astype(str)
        loop = text.map(lambda t: repetition_loop(t.split()))
        compression = text.map(compression_ratio) > COMPRESSION_RATIO_THRESHOLD
        flagged = loop | compression
        insertions = block["word_insertions"]
        entry = {
            "n_utterances": int(len(block)),
            "repetition_loop_ngram": int(loop.sum()),
            "compression_ratio_above_2.4": int(compression.sum()),
            "flagged_either": int(flagged.sum()),
            "insertion_rate_all": float(insertions.sum() / block["n_reference_words"].sum()),
            "insertion_rate_unflagged": float(
                insertions[~flagged].sum() / block.loc[~flagged, "n_reference_words"].sum()
            ),
            "share_of_insertions_in_flagged": float(insertions[flagged].sum() / max(insertions.sum(), 1)),
            "hypothesis_to_reference_length_ratio_quantiles": {
                q: float(np.quantile(block["n_hypothesis_words"] / block["n_reference_words"], float(q)))
                for q in ("0.5", "0.9", "0.99", "1.0")
            },
            "length_ratio_above_1.5": int((block["n_hypothesis_words"] > 1.5 * block["n_reference_words"]).sum()),
            "latin_script_outputs": int(text.map(lambda t: bool(LATIN.search(t))).sum()),
            "empty_outputs": int((text.str.len() == 0).sum()),
        }
        if budget:
            raw = pd.read_csv(TRANSCRIPT_DIR / f"{stem}_raw.csv").set_index("utterance_id")
            lengths = raw.loc[block.index, "raw_prediction"].fillna("").astype(str).map(token_length)
            entry["retokenized_length_max"] = int(lengths.max())
            entry["retokenized_length_p99"] = float(lengths.quantile(0.99))
            entry["at_or_near_token_cap"] = int((lengths >= budget - 1).sum())
            block = block.assign(loop=loop, compression=compression, latin=text.map(lambda t: bool(LATIN.search(t))),
                                 retokenized_length=lengths, hypothesis=text)
            worst = block.sort_values("utterance_wer", ascending=False).head(N_FAILURE_EXAMPLES)
            worst.reset_index()[["utterance_id", "speaker_id", "gold_transcript_normalized", "hypothesis",
                                 "utterance_wer", "n_reference_words", "n_hypothesis_words", "word_insertions",
                                 "loop", "compression", "latin", "retokenized_length"]].to_csv(
                RESULT_DIR / "whisper_failure_sample.csv", index=False, encoding="utf-8"
            )
        system_flags[system] = entry

    whisper = system_flags["whisper"]
    defects = {
        "truncation_at_token_cap": whisper["at_or_near_token_cap"] >= DEFECT_TRUNCATION_FRACTION * whisper["n_utterances"],
        "task_or_language_not_forced": not (run["decoding"]["language"] == "bn" and run["decoding"]["task"] == "transcribe"),
        "audio_loading_failures": run["n_failed_utterances"] > 0,
    }
    greedy_sample_capped = int(table.iloc[len(published["configurations"])]["n_retokenized_at_or_above_budget_minus_1"])
    (RESULT_DIR / "whisper_decoding_validity.json").write_text(
        json.dumps(
            {
                "sample_recovery": recovered,
                "main_run_audit": system_flags,
                "defect_rule": "(a) >= 1% of main-run hypotheses at the 200-token cap, (b) task/language "
                               "not forced, (c) any audio-loading failure",
                "defects": defects,
                "implementation_defect_found": any(defects.values()),
                "diagnostic_rows_budget_note": (
                    f"Table 10 rows used max_new_tokens={DIAGNOSTIC_BUDGET}; {greedy_sample_capped} of the "
                    f"{len(ids)} sampled greedy hypotheses reach that budget when re-tokenized."
                ),
                "max_clip_seconds": float(manifest["duration_seconds"].max()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 4. Generation manifest: what the main run actually did.
    config = GenerationConfig.from_pretrained(CHECKPOINT, revision=run["revision"], local_files_only=True)
    library_defaults = GenerationConfig._get_default_generation_params()

    def effective(name: str) -> str:
        value = getattr(config, name)
        return f"{value} (set in generation_config)" if value is not None else f"not set; library default {library_defaults[name]}"

    short_form = float(manifest["duration_seconds"].max()) < WHISPER_SEGMENT_SECONDS
    inactive = "inactive: every clip is shorter than 30 s, so generate() takes the short-form path" if short_form else "check"
    (RESULT_DIR / "whisper_generation_manifest.json").write_text(
        json.dumps(
            {
                "code_path": "scripts/transcribe_whisper.py -> transformers WhisperForConditionalGeneration.generate",
                "transformers_version": run["transformers_version"],
                "torch_version": run["torch_version"],
                "checkpoint": run["checkpoint"],
                "checkpoint_revision": run["revision"],
                "task": "transcribe (passed to generate)",
                "language": "bn (passed to generate; forced_decoder_ids cleared)",
                "num_beams": run["decoding"]["num_beams"],
                "early_stopping": run["decoding"]["early_stopping"],
                "do_sample": run["decoding"]["do_sample"],
                "temperature": "not passed; do_sample=False, so beam search with no sampling and no temperature fallback",
                "max_new_tokens": run["decoding"]["max_new_tokens"],
                "generation_config_max_length": config.max_length,
                "repetition_penalty": effective("repetition_penalty"),
                "no_repeat_ngram_size": effective("no_repeat_ngram_size"),
                "length_penalty": effective("length_penalty"),
                "return_timestamps": run["decoding"]["return_timestamps"],
                "prompt_ids": "none",
                "condition_on_prev_tokens": f"not set; long-form option; {inactive}",
                "compression_ratio_threshold": f"not set (generation_config: {getattr(config, 'compression_ratio_threshold', None)}); {inactive}",
                "logprob_threshold": f"not set (generation_config: {getattr(config, 'logprob_threshold', None)}); {inactive}",
                "no_speech_threshold": f"not set (generation_config: {getattr(config, 'no_speech_threshold', None)}); {inactive}",
                "chunking": "none; clips are passed whole (max %.2f s)" % manifest["duration_seconds"].max(),
                "audio": run["resampling_method"],
                "features": "WhisperFeatureExtractor log-Mel, padded to 30 s; no clip truncated",
                "attention_mask": "passed (return_attention_mask=True)",
                "precision": run["precision"],
                "batch_size": run["batch_size"],
                "device": run["device"],
                "normalization": f"normalize_bangla {run['normalization_version']}",
                "failed_utterances": run["n_failed_utterances"],
                "empty_normalized_outputs": run["n_empty_normalized_outputs"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(table[["configuration", "max_new_tokens", "wer", "cer", "romanization_rate", "empty_rate"]].to_string(index=False))
    print(f"\nSample recovered: {recovered}")
    print(f"Whisper audit: {json.dumps(whisper, indent=1)}")
    print(f"Implementation defect found: {any(defects.values())}  {defects}")


if __name__ == "__main__":
    main()
