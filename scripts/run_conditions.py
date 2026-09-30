"""Train and evaluate the principal conditions with shared heads (plan sections 12-14, 17).

The plan's repository sketch lists separate train_* and evaluate_* scripts. They
are deliberately merged here: C2/C3/C4 must be produced by *the same* trained
head, and so must C5/C6/C7. Keeping training and the three evaluations inside one
process makes head reuse a property of the code rather than of a checkpoint
round-trip, which is exactly the confound the shared-head design exists to avoid.

Per outer fold and seed:
    1. Train the gold-text head once  -> evaluate 3x -> C2, C3, C4
    2. Train the gold-fusion head once -> evaluate 3x -> C5, C6, C7
    3. Train the speech head independently -> C1
    4. With --adaptation, additionally train matched ASR heads -> A1-A4

Embeddings are L2-normalized per sample. No scaler, PCA or normalization
statistic is ever fitted on data, so no test-derived statistic can leak (plan 11.3).

Writes predictions/oof_predictions.csv with one row per
(recording, condition, seed).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = REPO_ROOT / "manifests" / "banglamuse.csv"
SPLIT_DIR = REPO_ROOT / "splits"
EMBEDDING_DIR = REPO_ROOT / "embeddings"
PREDICTION_DIR = REPO_ROOT / "predictions"

N_OUTER_FOLDS = 5
GATE_SEEDS = (42,)
FINAL_SEEDS = (13, 42, 87)

HIDDEN_UNITS = 256
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8


@dataclass(frozen=True)
class Evaluation:
    """One reported condition: which head produced it, and on which test input."""

    condition: str
    description: str
    head: str
    text_source: str | None
    use_speech: bool


PRINCIPAL_EVALUATIONS = (
    Evaluation("C1", "Speech only", "speech", None, True),
    Evaluation("C2", "Oracle text", "gold_text", "gold", False),
    Evaluation("C3", "Whisper-substituted text", "gold_text", "whisper", False),
    Evaluation("C4", "Bengali XLS-R-substituted text", "gold_text", "indic", False),
    Evaluation("C5", "Oracle fusion", "gold_fusion", "gold", True),
    Evaluation("C6", "Whisper-substituted fusion", "gold_fusion", "whisper", True),
    Evaluation("C7", "Bengali XLS-R-substituted fusion", "gold_fusion", "indic", True),
)

ADAPTATION_EVALUATIONS = (
    Evaluation("A1", "Whisper-adapted text", "whisper_text", "whisper", False),
    Evaluation("A2", "Bengali XLS-R-adapted text", "indic_text", "indic", False),
    Evaluation("A3", "Whisper-adapted fusion", "whisper_fusion", "whisper", True),
    Evaluation("A4", "Bengali XLS-R-adapted fusion", "indic_fusion", "indic", True),
)

# Secondary cross-recognizer training comparison: the A1/A2 heads evaluated on the
# OTHER recognizer's transcripts. X1 reuses the A1 head and X2 the A2 head, so
# A1-vs-X2 and A2-vs-X1 hold the test transcript fixed and vary only which
# recognizer's output the head was trained on. These are separately trained heads,
# NOT a controlled substitution like C2->C3.
CROSS_ASR_EVALUATIONS = (
    Evaluation("X1", "Whisper-trained, Bengali XLS-R-tested text", "whisper_text", "indic", False),
    Evaluation("X2", "Bengali XLS-R-trained, Whisper-tested text", "indic_text", "whisper", False),
)

# Orthographic-canonicalization arm: identical to C2/C3/C4 except that every text
# source passed through the externally published BanglaBERT normalizer first.
# Shared head, exactly as C2/C3/C4 share one.
NORMALIZATION_EVALUATIONS = (
    Evaluation("N2", "Canonicalized oracle text", "gold_orthnorm_text", "gold_orthnorm", False),
    Evaluation("N3", "Canonicalized Whisper text", "gold_orthnorm_text", "whisper_orthnorm", False),
    Evaluation("N4", "Canonicalized Bengali XLS-R text", "gold_orthnorm_text", "indic_orthnorm", False),
)

# head name -> (training text source, whether the head consumes speech)
HEAD_SPECIFICATIONS = {
    "speech": (None, True),
    "gold_text": ("gold", False),
    "gold_fusion": ("gold", True),
    "whisper_text": ("whisper", False),
    "indic_text": ("indic", False),
    "whisper_fusion": ("whisper", True),
    "indic_fusion": ("indic", True),
    "gold_orthnorm_text": ("gold_orthnorm", False),
}

# Text sources keyed by sentence rather than by recording: gold text is identical
# across a sentence's four recordings, so it is stored once per sentence.
SENTENCE_KEYED_TEXT = {"gold", "gold_orthnorm"}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def load_embeddings(name: str) -> dict[str, np.ndarray]:
    payload = np.load(EMBEDDING_DIR / f"{name}.npz", allow_pickle=False)
    matrix = l2_normalize(payload["embeddings"].astype(np.float32))
    return dict(zip(payload["ids"].tolist(), matrix))


class Head(nn.Module):
    """The lightweight classifier of plan section 12."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, HIDDEN_UNITS),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_UNITS, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    device: str,
) -> Head:
    """Train to the best validation macro-F1 and restore that state."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Head(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.CrossEntropyLoss()

    train_features = torch.from_numpy(train_x).to(device)
    train_labels = torch.from_numpy(train_y).long().to(device)
    validation_features = torch.from_numpy(validation_x).to(device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score, best_state, epochs_without_gain = -1.0, None, 0

    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(train_features), generator=generator).to(device)
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_features[index]), train_labels[index])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            predicted = model(validation_features).argmax(dim=-1).cpu().numpy()
        score = f1_score(validation_y, predicted, average="macro", zero_division=0)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.inference_mode()
def predict_probabilities(model: Head, features: np.ndarray, device: str) -> np.ndarray:
    logits = model(torch.from_numpy(features).to(device))
    return torch.softmax(logits, dim=-1).cpu().numpy()


class FeatureBuilder:
    """Assembles model input for a set of recordings under a given text source."""

    def __init__(self, manifest: pd.DataFrame, embeddings: dict[str, dict[str, np.ndarray]]) -> None:
        self.sentence_of = dict(zip(manifest["utterance_id"], manifest["sentence_id"]))
        self.embeddings = embeddings
        self.text_dim = len(next(iter(embeddings["text_gold"].values())))
        self.speech_dim = len(next(iter(embeddings["speech"].values())))

    def text_vector(self, utterance_id: str, source: str) -> np.ndarray:
        if source in SENTENCE_KEYED_TEXT:
            return self.embeddings[f"text_{source}"][self.sentence_of[utterance_id]]
        return self.embeddings[f"text_{source}"][utterance_id]

    def build(self, utterance_ids: list[str], text_source: str | None, use_speech: bool) -> np.ndarray:
        parts = []
        if use_speech:
            parts.append(np.stack([self.embeddings["speech"][u] for u in utterance_ids]))
        if text_source is not None:
            parts.append(np.stack([self.text_vector(u, text_source) for u in utterance_ids]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def build_group_level(
        self, groups: pd.DataFrame, source: str = "gold"
    ) -> tuple[np.ndarray, np.ndarray]:
        """One gold-text row per training sentence group, not four copies (plan 12.1)."""
        representative = groups.drop_duplicates("sentence_group_id")
        features = np.stack(
            [self.embeddings[f"text_{source}"][s] for s in representative["sentence_id"]]
        ).astype(np.float32)
        return features, representative["sentiment_label"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="gate", choices=["gate", "final"])
    parser.add_argument("--folds", type=int, default=N_OUTER_FOLDS)
    parser.add_argument("--adaptation", action="store_true", help="also run A1-A4 (post-gate only)")
    parser.add_argument(
        "--cross-asr", action="store_true", help="also run X1-X2 (requires --adaptation heads)"
    )
    parser.add_argument(
        "--orthnorm", action="store_true", help="also run N2-N4 (orthographic canonicalization)"
    )
    parser.add_argument("--output", default="oof_predictions.csv")
    arguments = parser.parse_args()

    seeds = GATE_SEEDS if arguments.seeds == "gate" else FINAL_SEEDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_FILE)
    outer = pd.read_csv(SPLIT_DIR / "five_fold_sentence_grouped.csv")
    inner = pd.read_csv(SPLIT_DIR / "inner_validation_groups.csv")
    manifest["outer_fold"] = manifest["sentence_group_id"].map(
        dict(zip(outer["sentence_group_id"], outer["outer_fold"]))
    )

    embeddings = {
        "speech": load_embeddings("speech"),
        "text_gold": load_embeddings("text_gold"),
        "text_whisper": load_embeddings("text_whisper"),
        "text_indic": load_embeddings("text_indic"),
    }
    if arguments.orthnorm:
        for name in ("gold", "whisper", "indic"):
            embeddings[f"text_{name}_orthnorm"] = load_embeddings(f"text_{name}_orthnorm")
    builder = FeatureBuilder(manifest, embeddings)

    evaluations = list(PRINCIPAL_EVALUATIONS)
    if arguments.adaptation:
        evaluations += list(ADAPTATION_EVALUATIONS)
    if arguments.cross_asr:
        if not arguments.adaptation:
            raise SystemExit("--cross-asr reuses the A1/A2 heads, so it requires --adaptation")
        evaluations += list(CROSS_ASR_EVALUATIONS)
    if arguments.orthnorm:
        evaluations += list(NORMALIZATION_EVALUATIONS)
    required_heads = sorted({evaluation.head for evaluation in evaluations})

    asr_metrics = pd.read_csv(REPO_ROOT / "artifacts" / "asr_metrics_by_utterance.csv")
    metric_lookup = {
        (row.asr_system, row.utterance_id): (row.utterance_wer, row.utterance_cer)
        for row in asr_metrics.itertuples(index=False)
    }

    rows: list[dict] = []
    for seed in seeds:
        for fold in range(arguments.folds):
            test = manifest[manifest["outer_fold"] == fold]
            training = manifest[manifest["outer_fold"] != fold]

            block = inner[inner["outer_fold"] == fold]
            validation_groups = set(block[block["is_inner_validation"] == 1]["sentence_group_id"])
            fit = training[~training["sentence_group_id"].isin(validation_groups)]
            validate = training[training["sentence_group_id"].isin(validation_groups)]

            test_ids = test["utterance_id"].tolist()
            test_labels = test["sentiment_label"].to_numpy()

            heads: dict[str, Head] = {}
            for head_name in required_heads:
                text_source, use_speech = HEAD_SPECIFICATIONS[head_name]

                if text_source in SENTENCE_KEYED_TEXT and not use_speech:
                    # Gold text is identical across a group's four recordings, so
                    # train on one row per group rather than four duplicates.
                    train_x, train_y = builder.build_group_level(fit, text_source)
                    validation_x, validation_y = builder.build_group_level(validate, text_source)
                else:
                    train_x = builder.build(fit["utterance_id"].tolist(), text_source, use_speech)
                    train_y = fit["sentiment_label"].to_numpy()
                    validation_x = builder.build(
                        validate["utterance_id"].tolist(), text_source, use_speech
                    )
                    validation_y = validate["sentiment_label"].to_numpy()

                heads[head_name] = train_head(
                    train_x, train_y, validation_x, validation_y, seed, device
                )

            for evaluation in evaluations:
                features = builder.build(test_ids, evaluation.text_source, evaluation.use_speech)
                probabilities = predict_probabilities(heads[evaluation.head], features, device)
                predicted = probabilities.argmax(axis=1)

                asr_system = (
                    evaluation.text_source
                    if evaluation.text_source not in (None, "gold")
                    else ""
                )
                for position, utterance_id in enumerate(test_ids):
                    wer, cer = metric_lookup.get((asr_system, utterance_id), ("", ""))
                    rows.append(
                        {
                            "utterance_id": utterance_id,
                            "sentence_group_id": test["sentence_group_id"].iloc[position],
                            "speaker_id": test["speaker_id"].iloc[position],
                            "outer_fold": fold,
                            "condition": evaluation.condition,
                            "seed": seed,
                            "gold_label": int(test_labels[position]),
                            "prob_negative": float(probabilities[position, 0]),
                            "prob_positive": float(probabilities[position, 1]),
                            "predicted_label": int(predicted[position]),
                            "asr_system": asr_system,
                            "utterance_wer": wer,
                            "utterance_cer": cer,
                        }
                    )

                score = f1_score(test_labels, predicted, average="macro", zero_division=0)
                print(
                    f"  seed {seed} fold {fold} {evaluation.condition} "
                    f"({evaluation.description}): macro-F1 {score:.4f}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    output_file = PREDICTION_DIR / arguments.output
    frame.to_csv(output_file, index=False, encoding="utf-8")

    expected = len(manifest) * len(evaluations) * len(seeds)
    print(f"\nWrote {output_file}: {len(frame)} rows (expected {expected})")
    if len(frame) != expected:
        raise SystemExit("OOF row count does not match the expected recording x condition x seed count")

    print("\nPooled out-of-fold macro-F1")
    for evaluation in evaluations:
        block = frame[frame["condition"] == evaluation.condition]
        averaged = block.groupby("utterance_id").agg(
            gold_label=("gold_label", "first"), prob_positive=("prob_positive", "mean")
        )
        score = f1_score(
            averaged["gold_label"], (averaged["prob_positive"] >= 0.5).astype(int),
            average="macro", zero_division=0,
        )
        print(f"  {evaluation.condition}  {evaluation.description:<32} {score:.4f}")


if __name__ == "__main__":
    main()
