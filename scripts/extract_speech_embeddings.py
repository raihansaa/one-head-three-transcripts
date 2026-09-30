"""Cache frozen speech embeddings for every recording (plan section 11.2).

XLS-R-300M is used frozen and has never seen BanglaMUSE. Note for disclosure:
ASR-B (arijitx/wav2vec2-xls-r-300m-bengali) is a Bengali fine-tune of this same
self-supervised backbone. That is not leakage - neither model saw BanglaMUSE -
but it does mean the speech channel and the ASR-B text channel derive from a
common pretrained representation, so any complementarity observed in C7 cannot
be attributed to encoder diversity. This belongs in the paper's setup section.

Representation: waveform -> 16 kHz mono resampling -> frozen encoder ->
mask-aware temporal mean and standard-deviation pooling -> 2048-dim embedding.

Writes embeddings/speech.npz (`ids`, `embeddings`) and embeddings/speech_encoder.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoFeatureExtractor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_common import TARGET_SAMPLE_RATE, batched, load_audio, load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = REPO_ROOT / "embeddings"

CHECKPOINT = "facebook/wav2vec2-xls-r-300m"
BATCH_SIZE = 8


@torch.inference_mode()
def main() -> None:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = load_manifest()
    records = manifest[["utterance_id", "audio_path"]].to_dict("records")
    print(f"Embedding {len(records)} recordings with {CHECKPOINT} on {device}.")

    extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    started = time.time()

    for index, batch in enumerate(batched(records, BATCH_SIZE)):
        waveforms = [load_audio(record["audio_path"]) for record in batch]
        inputs = extractor(
            waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        hidden = model(
            input_values=inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
        ).last_hidden_state

        # Padding must not contribute to the pooled statistics.
        frame_mask = model._get_feature_vector_attention_mask(
            hidden.shape[1], inputs.attention_mask.to(device)
        ).unsqueeze(-1).to(hidden.dtype)
        lengths = frame_mask.sum(dim=1).clamp(min=1e-9)

        mean = (hidden * frame_mask).sum(dim=1) / lengths
        variance = ((hidden - mean.unsqueeze(1)) ** 2 * frame_mask).sum(dim=1) / lengths
        pooled = torch.cat([mean, variance.clamp(min=1e-10).sqrt()], dim=-1)

        ids.extend(record["utterance_id"] for record in batch)
        vectors.append(pooled.float().cpu().numpy())

        if index % 50 == 0:
            done = len(ids)
            rate = done / max(time.time() - started, 1e-6)
            print(
                f"  {done}/{len(records)}  {rate:.1f} utt/s  "
                f"eta {(len(records) - done) / max(rate, 1e-6) / 60:.1f} min",
                flush=True,
            )

    matrix = np.concatenate(vectors, axis=0)
    order = np.argsort(np.array(ids))
    np.savez_compressed(
        EMBEDDING_DIR / "speech.npz", ids=np.array(ids)[order], embeddings=matrix[order]
    )

    (EMBEDDING_DIR / "speech_encoder.json").write_text(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "frozen": True,
                "pooling": "mask-aware temporal mean and standard deviation, concatenated",
                "embedding_dim": int(matrix.shape[1]),
                "sample_rate": TARGET_SAMPLE_RATE,
                "batch_size": BATCH_SIZE,
                "total_seconds": round(time.time() - started, 1),
                "transformers_version": transformers.__version__,
                "trained_on_banglamuse": False,
                "shares_backbone_with_asr_b": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {matrix.shape} -> embeddings/speech.npz")


if __name__ == "__main__":
    main()
