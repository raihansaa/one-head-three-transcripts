"""Reproducibility manifest for the secondary analyses.

Records the environment, model revisions, every seed and fixed setting the
secondary_*.py scripts use, the order to run them in, and a SHA-256 for every
file in results/, so a later check can tell whether any reported number moved.

    python scripts/secondary_manifest.py

Writes results/reproducibility_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

import build_folds
import cluster_bootstrap
import secondary_lexical
import secondary_permutation
import run_conditions

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results"
MANIFEST_FILE = RESULT_DIR / "reproducibility_manifest.json"

RUN_ORDER = [
    "python scripts/secondary_heads.py            # P0 bitwise re-run + validation probabilities + E1 heads (GPU)",
    "python scripts/secondary_thresholds.py       # E2",
    "python scripts/secondary_lexical.py          # E3",
    "python scripts/secondary_permutation.py      # E4",
    "python scripts/secondary_whisper.py          # E5 (cached transcripts; no ASR run)",
    "python scripts/secondary_speaker_holdout.py  # E8 (GPU)",
    "python scripts/secondary_statistics.py       # P0 paper check, E1-E3 comparisons, E6, E7",
    "python scripts/secondary_manifest.py         # this manifest",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def read_json(relative: str) -> dict:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    whisper, indic = read_json("artifacts/asr_whisper_run.json"), read_json("artifacts/asr_indic_run.json")
    text, speech = read_json("embeddings/text_encoder.json"), read_json("embeddings/speech_encoder.json")
    manifest = {
        "label": "secondary analyses E1-E8",
        "analysis_specification": "results/SECONDARY_ANALYSIS_PLAN.md",
        "source_commit": git("rev-parse", "HEAD") or "not generated inside a git checkout",
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "transformers": transformers.__version__, "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
            "lock_file": "requirements.txt",
        },
        "models": {
            "asr_a": {"checkpoint": whisper["checkpoint"], "revision": whisper["revision"], "decoding": whisper["decoding"]},
            "asr_b": {"checkpoint": indic["checkpoint"], "revision": indic["revision"], "decoding": indic["decoding"]},
            "text_encoder": {"checkpoint": text["checkpoint"], "revision": text["revision"], "pooling": text["pooling"]},
            "speech_encoder": {"checkpoint": speech["checkpoint"], "revision": speech["revision"], "pooling": speech["pooling"]},
        },
        "protocol": {
            "outer_folds": {"splitter": "StratifiedKFold over sentence groups", "n_splits": build_folds.N_OUTER_FOLDS,
                            "shuffle": True, "random_state": build_folds.OUTER_SEED},
            "inner_validation": {"fraction": build_folds.INNER_VALIDATION_FRACTION,
                                 "random_state": f"{build_folds.INNER_SEED_BASE} + fold", "stratified": True},
            "head": {"hidden_units": run_conditions.HIDDEN_UNITS, "dropout": run_conditions.DROPOUT,
                     "learning_rate": run_conditions.LEARNING_RATE, "weight_decay": run_conditions.WEIGHT_DECAY,
                     "batch_size": run_conditions.BATCH_SIZE, "max_epochs": run_conditions.MAX_EPOCHS,
                     "patience": run_conditions.PATIENCE, "seeds": list(run_conditions.FINAL_SEEDS)},
            "bootstrap": {"iterations": cluster_bootstrap.N_BOOTSTRAP, "seed": cluster_bootstrap.BOOTSTRAP_SEED,
                          "unit": "sentence group, all recordings included, shared across conditions"},
            "threshold_grid": "0.000-1.000 step 0.001; ties -> closest to 0.5, then smaller",
            "tfidf": {"ngram_range": [1, 2], "tokenizer": "str.split", "C_grid": list(secondary_lexical.C_GRID),
                      "max_iter": secondary_lexical.MAX_ITER},
            "permutation": {"n_permutations": secondary_permutation.N_PERMUTATIONS,
                            "seed": secondary_permutation.PERMUTATION_SEED},
        },
        "run_order": RUN_ORDER,
        "result_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): sha256(path)
            for path in sorted(RESULT_DIR.glob("*"))
            if path.is_file() and path != MANIFEST_FILE
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST_FILE} ({len(manifest['result_sha256'])} result files hashed)")


if __name__ == "__main__":
    main()
