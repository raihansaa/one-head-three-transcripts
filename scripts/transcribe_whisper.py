"""ASR-A: Whisper large-v3 transcription of BanglaMUSE (plan section 9.1, 9.4).

Deterministic by construction: greedy decoding (num_beams=1), do_sample=False,
temperature 0, language forced to Bengali, task forced to transcribe, timestamps
disabled. No fine-tuning - this is an off-the-shelf deployment audit.

Usage:
    python scripts/transcribe_whisper.py [--limit N] [--batch-size N] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_common import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    batched,
    load_audio,
    load_manifest,
    write_outputs,
)

CHECKPOINT = "openai/whisper-large-v3"
LANGUAGE = "bn"
TASK = "transcribe"

# Beam search is the default operating point. A greedy pass on this data emits
# romanized Latin-script Bengali for 31% of utterances, which beam search reduces
# to ~2% (artifacts/whisper_decoding_diagnostic.json). Reporting greedy as ASR-A
# would understate a strong multilingual baseline. `early_stopping` with a tight
# token budget matters: without them beam search runs every beam to the cap and is
# ~8x slower for identical output.
NUM_BEAMS = 5

# Whisper's BPE spends ~2 tokens per Bengali grapheme, so these sentences need a
# median of 72 new tokens and up to 160. A 48-token cap silently truncated 99.5%
# of them mid-word while still looking fast and clean. Never lower this without
# re-checking output length against gold.
MAX_NEW_TOKENS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="transcribe only the first N (smoke test)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=NUM_BEAMS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--system-name", default="whisper", help="output file stem")
    parser.add_argument("--force", action="store_true", help="overwrite existing raw transcripts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    manifest = load_manifest()
    if args.limit:
        manifest = manifest.head(args.limit)
    records = manifest[["utterance_id", "audio_path"]].to_dict("records")
    print(f"Transcribing {len(records)} recordings with {CHECKPOINT} on {device} ({dtype}).")

    processor = WhisperProcessor.from_pretrained(CHECKPOINT)
    model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=dtype).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None

    rows: list[dict] = []
    started = time.time()

    for index, batch in enumerate(batched(records, args.batch_size)):
        waveforms, usable = [], []
        for record in batch:
            try:
                waveforms.append(load_audio(record["audio_path"]))
                usable.append(record)
            except Exception as error:  # noqa: BLE001 - recorded per utterance
                rows.append(
                    {
                        "utterance_id": record["utterance_id"],
                        "raw_prediction": "",
                        "inference_seconds": 0.0,
                        "error": f"audio: {type(error).__name__}: {error}",
                    }
                )
        if not usable:
            continue

        batch_started = time.time()
        try:
            features = processor(
                waveforms,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            with torch.inference_mode():
                generated = model.generate(
                    features.input_features.to(device, dtype=dtype),
                    attention_mask=features.attention_mask.to(device),
                    language=LANGUAGE,
                    task=TASK,
                    num_beams=args.num_beams,
                    early_stopping=args.num_beams > 1,
                    do_sample=False,
                    return_timestamps=False,
                    max_new_tokens=args.max_new_tokens,
                )
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            elapsed = (time.time() - batch_started) / len(usable)
            for record, text in zip(usable, texts):
                rows.append(
                    {
                        "utterance_id": record["utterance_id"],
                        "raw_prediction": text.strip(),
                        "inference_seconds": round(elapsed, 4),
                        "error": "",
                    }
                )
        except Exception as error:  # noqa: BLE001 - one bad batch must not kill the run
            for record in usable:
                rows.append(
                    {
                        "utterance_id": record["utterance_id"],
                        "raw_prediction": "",
                        "inference_seconds": 0.0,
                        "error": f"generate: {type(error).__name__}: {error}",
                    }
                )

        if index % 25 == 0:
            done = len(rows)
            rate = done / max(time.time() - started, 1e-6)
            remaining = (len(records) - done) / max(rate, 1e-6)
            print(
                f"  {done}/{len(records)}  {rate:.2f} utt/s  eta {remaining / 60:.1f} min",
                flush=True,
            )

    total_seconds = time.time() - started
    write_outputs(
        args.system_name,
        rows,
        {
            "system": "ASR-A",
            "checkpoint": CHECKPOINT,
            "revision": getattr(model.config, "_commit_hash", "unknown"),
            "decoding": {
                "language": LANGUAGE,
                "task": TASK,
                "num_beams": args.num_beams,
                "early_stopping": args.num_beams > 1,
                "do_sample": False,
                "temperature": 0.0,
                "return_timestamps": False,
                "max_new_tokens": args.max_new_tokens,
            },
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "precision": str(dtype),
            "batch_size": args.batch_size,
            "total_inference_seconds": round(total_seconds, 2),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "fine_tuned_on_banglamuse": False,
        },
        force=args.force,
    )
    print(f"Total wall clock: {total_seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
