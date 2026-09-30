"""Is Whisper's 0.744 WER a real property or a decoding-config artifact?

Two questions:

1. Does the Bengali language token actually reach the decoder? The main run passes
   `language="bn"` to generate() while also clearing `forced_decoder_ids`. If those
   interact badly, Whisper would be free-running language detection and the 31.5%
   romanized output would be our bug, not the model's behaviour.
2. Does decoding strategy matter? The main run is greedy. If beam search materially
   lowers WER, the headline oracle-to-ASR gap is inflated and the whole pipeline
   must be re-run.

A control already rules out audio handling: ASR-B reads the same waveforms through
the same load_audio() and scores 0.276 WER, so resampling and decoding of the MP3s
are not the cause.

Evaluated on a speaker-stratified subset so the comparison is cheap but not biased
toward one recording condition.

Writes artifacts/whisper_decoding_diagnostic.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_common import TARGET_SAMPLE_RATE, batched, load_audio  # noqa: E402
from calculate_asr_metrics import edit_operations  # noqa: E402
from normalize_bangla import grapheme_clusters, normalize_bangla, word_tokens  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

CHECKPOINT = "openai/whisper-large-v3"
PER_SPEAKER = 75
SAMPLE_SEED = 20260807
BATCH_SIZE = 8

LATIN = re.compile(r"[A-Za-z]")

CONFIGURATIONS = {
    "greedy_bn": {"language": "bn", "task": "transcribe", "num_beams": 1},
    "beam5_bn": {"language": "bn", "task": "transcribe", "num_beams": 5},
    "greedy_autodetect": {"task": "transcribe", "num_beams": 1},
}


def score(references: list[str], hypotheses: list[str]) -> dict:
    word_errors = reference_words = char_errors = reference_chars = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens, hypothesis_tokens = word_tokens(reference), word_tokens(hypothesis)
        substitutions, deletions, insertions = edit_operations(reference_tokens, hypothesis_tokens)
        word_errors += substitutions + deletions + insertions
        reference_words += len(reference_tokens)

        reference_graphemes = grapheme_clusters(reference)
        substitutions, deletions, insertions = edit_operations(
            reference_graphemes, grapheme_clusters(hypothesis)
        )
        char_errors += substitutions + deletions + insertions
        reference_chars += len(reference_graphemes)

    return {
        "wer": word_errors / max(reference_words, 1),
        "cer": char_errors / max(reference_chars, 1),
        "romanization_rate": sum(bool(LATIN.search(h)) for h in hypotheses) / max(len(hypotheses), 1),
        "empty_rate": sum(not h.strip() for h in hypotheses) / max(len(hypotheses), 1),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    manifest = pd.read_csv(MANIFEST_FILE)
    subset = (
        manifest.groupby("speaker_id", group_keys=False)
        .sample(n=PER_SPEAKER, random_state=SAMPLE_SEED)
        .sort_values("utterance_id")
        .reset_index(drop=True)
    )
    print(f"Diagnostic subset: {len(subset)} recordings, {PER_SPEAKER} per speaker.\n")

    processor = WhisperProcessor.from_pretrained(CHECKPOINT)
    model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=dtype).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None

    waveforms = [load_audio(path) for path in subset["audio_path"]]
    references = subset["gold_transcript_normalized"].fillna("").astype(str).tolist()

    results, prefixes = {}, {}
    for name, options in CONFIGURATIONS.items():
        print(f"--- {name}: {options}")
        hypotheses, started = [], time.time()

        for index, chunk in enumerate(batched(list(range(len(waveforms))), BATCH_SIZE)):
            features = processor(
                [waveforms[i] for i in chunk],
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            with torch.inference_mode():
                generated = model.generate(
                    features.input_features.to(device, dtype=dtype),
                    attention_mask=features.attention_mask.to(device),
                    do_sample=False,
                    return_timestamps=False,
                    max_new_tokens=128,
                    **options,
                )
            hypotheses.extend(
                normalize_bangla(text)
                for text in processor.batch_decode(generated, skip_special_tokens=True)
            )
            # Capture the untrimmed prefix once: this is where the language token shows up.
            if index == 0:
                prefixes[name] = [
                    processor.tokenizer.convert_ids_to_tokens(row[:5].tolist())
                    for row in generated[:3]
                ]

        results[name] = {
            **score(references, hypotheses),
            "seconds": round(time.time() - started, 1),
            "decoder_prefix_sample": prefixes[name],
            "example_outputs": hypotheses[:3],
        }
        entry = results[name]
        print(
            f"    WER {entry['wer']:.3f}  CER {entry['cer']:.3f}  "
            f"romanized {entry['romanization_rate']:.1%}  ({entry['seconds']}s)"
        )
        print(f"    prefix: {prefixes[name][0]}\n")

    baseline = results["greedy_bn"]["wer"]
    verdict = {
        name: {
            "wer_change_vs_greedy_bn": round(entry["wer"] - baseline, 4),
            "relative_change": round((entry["wer"] - baseline) / max(baseline, 1e-9), 3),
        }
        for name, entry in results.items()
        if name != "greedy_bn"
    }

    payload = {
        "checkpoint": CHECKPOINT,
        "n_utterances": len(subset),
        "per_speaker": PER_SPEAKER,
        "sample_seed": SAMPLE_SEED,
        "audio_control": (
            "ASR-B reads identical waveforms via the same load_audio() and scores "
            "0.276 WER, so audio handling is not implicated"
        ),
        "configurations": results,
        "verdict": verdict,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "whisper_decoding_diagnostic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 62)
    print(f"greedy_bn (main run config) WER on subset: {baseline:.3f}")
    for name, change in verdict.items():
        direction = "BETTER" if change["wer_change_vs_greedy_bn"] < 0 else "worse"
        print(
            f"  {name}: {change['wer_change_vs_greedy_bn']:+.4f} WER "
            f"({change['relative_change']:+.1%}) {direction}"
        )
    print(f"\nWrote {ARTIFACT_DIR / 'whisper_decoding_diagnostic.json'}")


if __name__ == "__main__":
    main()
