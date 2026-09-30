# One Head, Three Transcripts

Code, data splits, leakage assertions, per-recording out-of-fold predictions and
analysis scripts for the paper

> **One Head, Three Transcripts: A Controlled Oracle-to-ASR Audit of Bangla
> Speech–Text Sentiment Analysis.** Md Raihan, Nilam Rajak, Kuntal Ghosh and
> Sathyanarayanan Dhorali. Fifth Workshop on Multimodal Machine Learning in
> Low-resource Languages (MMLow 2026), co-located with AACL-IJCNLP 2026.

The study asks how much speech–text sentiment performance on
[BanglaMUSE](https://data.mendeley.com/datasets/5yb4jjzrx3) depends on gold
transcripts. One classifier head is trained once and evaluated on gold, Whisper
large-v3 and Bengali XLS-R CTC transcripts, so the measured gap is attributable to the
transcript source rather than to separately fitted models. The repository also
contains the analyses that qualify the speech-input results: an acquisition-feature
(nuisance) fusion control, threshold-only adaptation, a lexical baseline, a
whole-pipeline permutation null, seed variability, recovery intervals and a joint
speaker-and-sentence holdout.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | The full pipeline: corpus audit, splits and leakage assertions, ASR, embeddings, classifier heads, bootstrap statistics, post-review analyses and every number printed in the paper |
| `manifests/banglamuse.csv` | The 3,996 usable recordings: sentence group, speaker, label, normalized gold transcript, audio metadata and MD5 hash |
| `splits/` | Sentence-grouped 5-fold outer splits and inner-validation groups |
| `transcripts/` | Raw and normalized ASR output for every recording: `whisper_*` (beam 2, 200-token budget) and `indic_*` (Bengali XLS-R, greedy CTC) are used throughout; `whisper_greedy_*` (greedy, 128 tokens) is the cached greedy run of Appendix A.16; `*_orthnorm` are the canonicalized texts of Table 16 |
| `predictions/` | Per-recording, per-seed out-of-fold probabilities for every original condition |
| `artifacts/` | Original (pre-review) results the paper uses: ASR metrics, bootstrap comparisons, corpus and acquisition audits, acquisition features, per-speaker, WER-bin and error-pattern tables |
| `results/` | Post-review analyses: specification, predictions, statistics, the paper's printed numbers (`paper_numbers.json`) and a hash manifest (`reproducibility_manifest.json`) |
| `embeddings/` | Metadata of the frozen text and speech encoders (the `.npz` embeddings are a release asset; see below) |

Condition IDs follow the paper: C1 speech only; C2–C4 one gold-trained text head
evaluated on gold, Whisper and XLS-R transcripts; C5–C7 the same for one gold-trained
fusion head; A1–A4 heads trained on matched ASR output; X1–X2 cross-recognizer
transfer; N5–N7 and NA3–NA4 the text+nuisance control; T3/T4/T6/T7 threshold-only
adaptation.

The prediction files: `oof_predictions_final.csv` (C1–C7 and A1–A4, seeds 13, 42 and
87) feeds the original analyses; `oof_predictions_crossasr.csv` is the same run with
X1–X2 added (its C and A rows are bit-identical) and is the file the post-review
scripts build on; `oof_predictions_orthnorm.csv` adds N2–N4, the orthographic
canonicalization check of Table 16.

## Data

The BanglaMUSE audio (Nayeem et al., 2026, *Data in Brief*; Mendeley Data, Nayeem et
al., 2025, DOI [10.17632/5yb4jjzrx3.1](https://doi.org/10.17632/5yb4jjzrx3.1),
CC BY 4.0) is not redistributed here. To rerun anything that needs audio, download
it and place it as

```
Multimodal Dataset/
  Text_folder.xlsx
  Voice_folder/{f1,f2,m1,m2}_voice_folder/
```

`scripts/audit_dataset.py` repairs the delivered folder by content-derived rules
(duplicates, identifier formats, misnamed Ogg files) and writes the manifest; the
`file_hash` column lets you check that your audio matches ours.

## Setup

Python 3.12 (3.14 is too new for the pinned PyTorch):

```bash
python -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
# only for the orthographic canonicalization check (orthographic_normalize.py):
.venv/bin/pip install "normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9" emoji==1.4.2 ftfy==6.0.3
```

(On Windows the interpreter is `.venv\Scripts\python.exe`.) Run every script from the
repository root, e.g. `python scripts/validate_splits.py`. `requirements.txt` pins the
direct dependencies; `requirements-lock.txt` is the complete environment (`pip freeze`)
in which the post-review analyses ran and the original heads were re-trained bit for
bit (Python 3.12.10, Windows 11). `recording_session_audit.py` also needs `ffprobe`
(FFmpeg) on the `PATH`.

## Reproducing the paper

### From the released files (no audio, no GPU)

```bash
python scripts/validate_splits.py              # 44 split-leakage assertions

python scripts/calculate_asr_metrics.py        # Table 1
python scripts/asr_quality_by_class.py         # Table 12
python scripts/cluster_bootstrap.py --predictions oof_predictions_final.csv
python scripts/analyze_wer_bins.py --predictions oof_predictions_final.csv         # Tables 3, 13
python scripts/extract_sentiment_flips.py --predictions oof_predictions_final.csv  # Appendix A.15
python scripts/orthographic_results.py         # Table 16

python scripts/post_review_thresholds.py       # threshold-only adaptation
python scripts/post_review_lexical.py          # TF-IDF + logistic regression
python scripts/post_review_permutation.py      # 1,000-permutation null
python scripts/post_review_whisper.py          # beam-2 row, decoding audit
python scripts/post_review_statistics.py       # paired intervals, seeds, recovery ratios
python scripts/paper_numbers.py                # every number printed in the paper
python scripts/post_review_manifest.py         # settings and SHA-256 of every file in results/
```

These scripts are deterministic: re-running them reproduces every released file in
`results/` and `artifacts/` byte for byte, including `paper_numbers.json` and
the manifest, except that one value in `artifacts/wer_bin_analysis.csv` differs in its
sixteenth significant digit (no reported number changes). They also write a few
secondary tables and Markdown summaries that are not included here because nothing in
the paper uses them. `post_review_whisper.py` reads the Whisper tokenizer and
generation config from the local Hugging Face cache; fetch them once with

```bash
python -c "from transformers import WhisperTokenizer, GenerationConfig as G; r='06f233fe06e710322aca913c1bc4249a0d71fce1'; WhisperTokenizer.from_pretrained('openai/whisper-large-v3', revision=r); G.from_pretrained('openai/whisper-large-v3', revision=r)"
```

### Full pipeline (audio and a GPU)

```bash
python scripts/audit_dataset.py
python scripts/build_folds.py
python scripts/validate_splits.py

python scripts/transcribe_whisper.py --num-beams 2 --max-new-tokens 200 --batch-size 8
python scripts/transcribe_indic_asr.py --batch-size 16
python scripts/calculate_asr_metrics.py
python scripts/diagnose_whisper.py
python scripts/asr_quality_by_class.py

python scripts/extract_text_embeddings.py
python scripts/extract_speech_embeddings.py

python scripts/run_conditions.py --seeds final --adaptation --output oof_predictions_final.csv
python scripts/run_conditions.py --seeds final --adaptation --cross-asr --output oof_predictions_crossasr.csv
python scripts/orthographic_normalize.py
python scripts/run_conditions.py --seeds final --adaptation --orthnorm --output oof_predictions_orthnorm.csv

python scripts/recording_session_audit.py
python scripts/recording_condition_audit.py
python scripts/extract_sentiment_flips.py --predictions oof_predictions_final.csv

python scripts/post_review_heads.py            # re-trains every head, checks the released predictions bit for bit
python scripts/post_review_speaker_holdout.py  # joint speaker-and-sentence holdout
```

then the commands of the previous section. Raw ASR output is treated as immutable:
the transcription scripts refuse to overwrite existing transcripts unless given
`--force`, which archives the old files. The full run takes about 1.3 hours of
Whisper inference and 13 minutes for Bengali XLS-R on one consumer GPU.

On the reference machine (RTX 5060 Laptop GPU, PyTorch 2.11.0+cu128) re-training
reproduces all 155,844 released test probabilities bit for bit
(`results/original_baseline_rerun_check.json`). Other GPUs, drivers or library
versions can change the last digits.

### Embeddings

The cached encoder outputs (`embeddings/*.npz`, about 75 MB) are attached to the
GitHub release as `embeddings.zip`; unzip it into `embeddings/` to re-train the heads
without audio. They can also be regenerated with the two `extract_*_embeddings.py`
scripts.

## Where the paper's numbers come from

| Paper | Script | Output |
|---|---|---|
| Table 1 (ASR quality) | `calculate_asr_metrics.py` | `artifacts/asr_metrics.json` |
| Table 2 (substitution and nuisance control) | `run_conditions.py`, `post_review_heads.py`, `post_review_statistics.py` | `predictions/oof_predictions_crossasr.csv`, `results/nuisance_fusion_oof.csv`, `results/paired_gains_and_recovery_intervals.csv` |
| Figure 2 (plotted values) | `post_review_statistics.py` | `results/condition_scores.csv`, `results/paired_gains_and_recovery_intervals.csv` |
| Table 3 (WER bins) | `analyze_wer_bins.py` | `artifacts/wer_bin_analysis.csv` |
| Table 4 (threshold versus retraining) | `post_review_thresholds.py`, `post_review_statistics.py` | `results/threshold_comparison.csv`, `results/thresholds_by_fold.csv` |
| Table 5 (acquisition audit) | `recording_condition_audit.py`, `post_review_permutation.py`, `post_review_speaker_holdout.py` | `artifacts/nuisance_only_classifier.csv`, `results/nuisance_permutation_null.json`, `results/speaker_holdout_results.csv` |
| Appendix Tables 6, 9, 10 | `post_review_statistics.py` | `results/condition_scores.csv`, `results/paired_gains_and_recovery_intervals.csv`, `results/seed_scores.csv` |
| Appendix Tables 7, 8 | `paper_numbers.py` (recomputes the intervals from the predictions), `post_review_statistics.py` | `results/paper_numbers.json`, `results/condition_scores.csv` |
| Appendix Table 11 | `post_review_speaker_holdout.py` | `results/speaker_holdout_results.csv` |
| Appendix Tables 12, 13 | `asr_quality_by_class.py`, `analyze_wer_bins.py` | `artifacts/asr_quality_by_class.csv`, `artifacts/speaker_analysis.csv` |
| Appendix Table 14 | `diagnose_whisper.py`, `post_review_whisper.py` | `artifacts/whisper_decoding_diagnostic.json`, `results/whisper_same_sample_diagnostic.csv`, `results/whisper_generation_manifest.json` |
| Appendix A.15 (error patterns) | `extract_sentiment_flips.py` | `artifacts/sentiment_flips.csv` |
| Appendix Tables 15, 16 | `recording_session_audit.py`, `orthographic_results.py` | `artifacts/recording_session_audit.md`, `artifacts/orthographic_normalization_results.csv` |
| Every printed number | `paper_numbers.py` | `results/paper_numbers.json` |

Scores are pooled recording-level out-of-fold macro-F1 on seed-averaged probabilities
(seeds 13, 42, 87). Intervals are paired sentence-group bootstrap intervals (10,000
resamples, seed 20260807). The printed numbers are rounded once, from full precision.

## Post-review analyses

The analyses requested during peer review were specified in
`results/POST_REVIEW_ANALYSIS_PLAN.md`, together with the `post_review_*.py` scripts,
before any of them was run; they were specified after the original results were
known and are labelled post-review in the paper. Before interpreting any new
condition, `post_review_heads.py` re-trains every original head and requires its test
probabilities to match the released predictions bit for bit.
`results/reproducibility_manifest.json` records the settings, software versions,
model revisions and SHA-256 hashes of every file in `results/`. The specification's
header refers to the authors' internal revision plan, which is not part of this
release.

## Licenses

Code: MIT (`LICENSE`). Data files derived from BanglaMUSE (manifest, splits,
transcripts, predictions, artifacts, results, encoder metadata and the embeddings
release asset): CC BY 4.0, see `DATA_LICENSE.md`. BanglaMUSE itself
is CC BY 4.0 (Nayeem et al., 2025, Mendeley Data); please cite it when using these
files.

## Citation

```bibtex
@inproceedings{onehead2026,
  title     = {One Head, Three Transcripts: A Controlled Oracle-to-{ASR} Audit of
               {Bangla} Speech--Text Sentiment Analysis},
  author    = {Raihan, Md and Rajak, Nilam and Ghosh, Kuntal and
               Dhorali, Sathyanarayanan},
  booktitle = {Proceedings of the Fifth Workshop on Multimodal Machine Learning in
               Low-resource Languages (MMLow 2026)},
  year      = {2026}
}
```

Pages and the ACL Anthology link will be added once the proceedings are published.
