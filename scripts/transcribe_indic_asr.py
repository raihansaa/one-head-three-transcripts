"""ASR-B: Indic-focused Bengali transcription of BanglaMUSE (plan section 9.2, 9.4).

Checkpoint selection followed the plan's deadline rule (section 9.2). Both
AI4Bharat options were abandoned at the decision point rather than debugged:
`indic-conformer-600m-multilingual` is a gated repository (HTTP 401) shipping
only ONNX/TorchScript assets that require the NeMo toolkit, and
`indicwav2vec_v1_bengali` is likewise gated at file level. No HuggingFace
credentials were available for this run.

`arijitx/wav2vec2-xls-r-300m-bengali` is used instead: an openly accessible
Bengali-specific fine-tune of XLS-R-300M with standard Wav2Vec2ForCTC weights.
Because the checkpoint is language-specific, it may accurately be described as
"Bengali-specialized" in the paper. It is also architecturally complementary to
ASR-A: greedy CTC over a self-supervised encoder versus an autoregressive
sequence-to-sequence decoder.

The repository ships a 5-gram language model, which is deliberately NOT used.
Decoding stays greedy so ASR-B is fully deterministic and reproducible without
kenlm; the resulting WER is therefore a no-LM figure and must be reported as
such rather than compared to published LM-rescored numbers.

Usage:
    python scripts/transcribe_indic_asr.py [--limit N] [--batch-size N] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_common import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    batched,
    load_audio,
    load_manifest,
    write_outputs,
)

CHECKPOINT = "arijitx/wav2vec2-xls-r-300m-bengali"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="transcribe only the first N (smoke test)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="overwrite existing raw transcripts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = load_manifest()
    if args.limit:
        manifest = manifest.head(args.limit)
    records = manifest[["utterance_id", "audio_path"]].to_dict("records")
    print(f"Transcribing {len(records)} recordings with {CHECKPOINT} on {device}.")

    processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT)
    model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT).to(device)
    model.eval()

    # Layer-norm feature encoders need the attention mask; group-norm ones are
    # documented as performing better without it.
    use_attention_mask = model.config.feat_extract_norm == "layer"

    rows: list[dict] = []
    started = time.time()

    for index, batch in enumerate(batched(records, args.batch_size)):
        waveforms, usable = [], []
        for record in batch:
            try:
                waveforms.append(load_audio(record["audio_path"]))
                usable.append(record)
            except Exception as error:  # noqa: BLE001
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
            inputs = processor(
                waveforms,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=use_attention_mask,
            )
            call = {"input_values": inputs.input_values.to(device)}
            if use_attention_mask:
                call["attention_mask"] = inputs.attention_mask.to(device)

            with torch.inference_mode():
                logits = model(**call).logits
            texts = processor.batch_decode(logits.argmax(dim=-1).cpu().numpy())

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
        except Exception as error:  # noqa: BLE001
            for record in usable:
                rows.append(
                    {
                        "utterance_id": record["utterance_id"],
                        "raw_prediction": "",
                        "inference_seconds": 0.0,
                        "error": f"forward: {type(error).__name__}: {error}",
                    }
                )

        if index % 25 == 0:
            done = len(rows)
            rate = done / max(time.time() - started, 1e-6)
            print(
                f"  {done}/{len(records)}  {rate:.2f} utt/s  "
                f"eta {(len(records) - done) / max(rate, 1e-6) / 60:.1f} min",
                flush=True,
            )

    total_seconds = time.time() - started
    write_outputs(
        "indic",
        rows,
        {
            "system": "ASR-B",
            "checkpoint": CHECKPOINT,
            "revision": getattr(model.config, "_commit_hash", "unknown"),
            "selection_note": (
                "ai4bharat/indic-conformer-600m-multilingual is gated (HTTP 401) and NeMo-only; "
                "switched at the plan's decision point per section 9.2"
            ),
            "language_specific": True,
            "decoding": {
                "strategy": "greedy CTC argmax",
                "language_model": None,
                "beam_size": 1,
                "used_attention_mask": use_attention_mask,
                "feat_extract_norm": model.config.feat_extract_norm,
            },
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "precision": "float32",
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
