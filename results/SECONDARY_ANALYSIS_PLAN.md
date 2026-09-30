# Post-review analysis specification (frozen before any result was computed)

Reviewer-requested **post-review** analyses for the camera-ready revision
(`BanglaMUSE_Camera_Ready_Action_Plan_v2.md`, items E1-E8). They are not part of
the originally predeclared analysis and must never be labelled as such.

This file and the `scripts/post_review_*.py` scripts that implement it were
committed together before any of them was run. The commit is the freeze.

## Fixed protocol shared by every analysis

- Data: `manifests/banglamuse.csv` (3,996 recordings, 998 sentence groups,
  1,997 positive / 1,999 negative), `splits/five_fold_sentence_grouped.csv`,
  `splits/inner_validation_groups.csv`. Original folds, unchanged.
- Frozen encoders and cached embeddings (`embeddings/*.npz`), frozen normalizer v1.
- Head recipe unchanged (`run_conditions.py`): LayerNorm -> Linear(256) -> GELU ->
  Dropout(0.2) -> Linear(2), AdamW 1e-3 / 1e-4, batch 64, <= 60 epochs, patience 8,
  best inner-validation macro-F1 epoch kept. Seeds 13, 42, 87.
- Positive class = label 1 = softmax index 1. Ensemble = mean of the three seeds'
  `prob_positive`; decision rule `prob_positive >= 0.5` (ties -> positive), as in
  `cluster_bootstrap.py`. A threshold rule uses `prob_positive >= t`.
- Metric: pooled recording-level out-of-fold macro-F1 (never the mean of fold F1).
- Uncertainty: the original paired sentence-group bootstrap, 10,000 iterations,
  seed 20260807, groups resampled with replacement with all their recordings, the
  same resample shared by every condition; 95% percentile intervals. All
  differences and ratios are computed per resample from unrounded predictions.
- Published predictions: `predictions/oof_predictions_crossasr.csv`.

## P0 - recover the published baseline first

Re-train every original head (speech, gold text, gold fusion, Whisper/XLS-R text
and fusion) with the unchanged code and seeds. Require **bit-identical** test
probabilities against the published predictions for C1-C7, A1-A4 and X1-X2
before any new condition is interpreted. In the same pass, save each head's
inner-validation probabilities (never saved originally; E2 needs them).
Recompute the published scores from the stored predictions and check them
against the submitted paper.

## E1 - nuisance-feature fusion control

- Features: the primary Table 5 set exactly as `recording_condition_audit.py`
  defines it. Numeric: bytes_per_second, bit_rate, clipping_fraction, dc_offset,
  leading_rms, trailing_rms, leading_noise_floor, trailing_noise_floor.
  Categorical: sample_rate, num_channels, audio_format, codec. Duration, full RMS
  and spectral centroid are excluded.
- Preprocessing, fitted on each fold's training partition (outer training minus
  inner validation) only, then frozen for validation, test and every transcript
  substitution: median imputation of numerics; one-hot encoding of categoricals
  (`handle_unknown="ignore"`); `StandardScaler` over all columns; per-row L2
  normalization of the nuisance branch. The text branch keeps its original
  per-sample L2 normalization. Input = [nuisance, text] (the nuisance vector
  takes the speech embedding's place).
- Checks reported per fold: missing values, zero-variance training columns,
  unseen categories, zero-norm rows. The nuisance-only logistic-regression
  baseline is re-run with this training-only preprocessing as a sanity check.
  Edge-window speech content is described by the edge-window RMS relative to the
  full-utterance RMS (descriptive only; the vector is called acquisition-focused,
  not content-free).
- Heads, recording-level rows, per fold and seed:
  - `gold_nuisance`: trained and early-stopped on gold text + nuisance, tested on
    gold (N5), Whisper (N6) and XLS-R (N7) text with the same nuisance vector.
  - `whisper_nuisance` -> NA3 and `indic_nuisance` -> NA4: trained, early-stopped
    and tested on matched ASR text + nuisance.
- Paired comparisons: N6-C3, N7-C4, C6-N6, C7-N7, N5-C2, C5-N5; matched:
  NA3-A1, NA4-A2, A3-A1, A4-A2, A3-NA3, A4-NA4. Input dimensions and parameter
  counts are reported. No equivalence claim (no equivalence margin was set).

## E2 - threshold-only adaptation

- Weights and probabilities fixed. Per outer fold: average the three seeds'
  validation probabilities, choose one threshold on the grid 0.000-1.000 (step
  0.001) maximizing inner-validation macro-F1 (recording-level validation rows),
  apply it to that fold's seed-averaged test probabilities. Tie-break: smallest
  |t - 0.5|, then the smaller t. No threshold is tuned on test labels.
- Settings (head / validation transcript / test condition):
  - T3: gold text / Whisper / C3.  T4: gold text / XLS-R / C4.
  - T6: gold fusion / Whisper / C6.  T7: gold fusion / XLS-R / C7.
  - Gold-validation control: T3-goldval, T4-goldval, T6-goldval, T7-goldval use
    the same head's **gold** validation probabilities.
  - Matched-head control: A1-threshold .. A4-threshold use each ASR-trained head's
    own matched ASR validation probabilities. A1-A4 (default rule) are kept as
    published.
- Reported: scores, actual and predicted positive rates, class-wise
  precision/recall/F1 and confusion counts, all five thresholds per setting.
- Decomposition (arithmetic, signed, unclipped): O-B = (T-B) + (A-T) + (O-A),
  with O = C2 or C5, B = C3/C4/C6/C7, T = T3/T4/T6/T7, A = A1-A4 default; plus
  A-threshold - T.

## E3 - TF-IDF + logistic regression on gold text

- Text: `gold_transcript_normalized` (the normalized text C2 embeds).
- `TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
  ngram_range=(1, 2), min_df=1, norm="l2", use_idf=True, smooth_idf=True,
  sublinear_tf=False)`, fitted on the training partition only.
- Rows mirror C2: one row per training sentence group (the same representative
  sentence `run_conditions.build_group_level` uses); inner validation also
  group-level; test at recording level (each recording's own sentence text), so
  a sentence's prediction counts once per recording.
- `LogisticRegression(solver="lbfgs", max_iter=10000)`, C in {0.1, 1, 10} chosen
  on inner-validation macro-F1 (tie -> smaller C), fitted on the training
  partition only. Deterministic, no seeds. Convergence recorded per fold.
- Reported: pooled OOF macro-F1, accuracy, class-wise F1, paired TFIDF-LR - C2.
- The optional evaluation on ASR text is not run.

## E4 - nuisance-classifier permutation null

- Pipeline = the Table 5 primary pipeline unchanged (`get_dummies`,
  `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)`, fitted on the
  outer-training rows of each fold). It has no validation-based model selection,
  so no inner split is involved.
- Statistic: pooled OOF macro-F1 (mean-of-fold F1 also reported for continuity).
- Null: B = 1,000 permutations, `numpy.random.default_rng(20260922)`. Each
  permutation shuffles the 998 group labels **between** sentence groups and
  broadcasts them to all recordings of each group, then regenerates the outer
  folds with the original splitter (`StratifiedKFold(5, shuffle=True,
  random_state=42)` over the group table, stratified on the permuted labels) and
  refits the full pipeline. The observed statistic uses the same whole-pipeline
  procedure on the true labels (which reproduces the frozen folds; checked).
- p = (1 + #{null >= observed}) / (B + 1). Null mean, SD and quantiles reported.
  The interpretation is limited to this group-exchangeability null.

## E5 - Whisper beam-2 row and decoding manifest

- Sample: re-drawn exactly as `diagnose_whisper.py` (75 per speaker,
  `random_state=20260807`). Recovery is confirmed by reproducing the published
  greedy row from the cached full-corpus greedy transcripts and by the stored
  example outputs.
- Beam-2 row: the cached main-run hypotheses (`transcripts/whisper_normalized.csv`,
  beam 2, early stopping, 200 new tokens) rescored on the same IDs with the same
  normalizer and `score()` function. No new ASR run unless a defect is found.
- Validity checks (full corpus, main run): hypotheses at or near the token cap,
  immediate n-gram repetition (n <= 4, >= 3 consecutive copies), zlib compression
  ratio > 2.4, hypothesis/reference length ratio, Latin-script output, empty
  output, audio-loading failures, forced task/language; the insertion share from
  repetition-flagged hypotheses; the 20 highest-WER utterances listed.
- Defect rule: an implementation defect is any of (a) >= 1% of main-run
  hypotheses reaching the 200-token cap, (b) task/language not forced, (c) any
  audio-loading failure. Repetition frequency is a property of the operating
  point, reported descriptively. If the diagnostic rows' 128-token budget
  truncated sampled greedy outputs, that is reported as a limitation of the
  original diagnostic table.
- Generation manifest: package versions, checkpoint revision, decoding settings,
  and each reviewer-named option (`condition_on_prev_tokens`,
  `compression_ratio_threshold`, `no_speech_threshold`, ...) recorded as set,
  unset, or inactive in the short-form path actually used (all clips < 30 s).

## E6 - training-seed variability

For each trained condition and seed: pooled OOF macro-F1 over the five folds
(rule `p >= 0.5`). Mean and sample SD over the three seeds, beside the ensemble
score. Within-seed differences: C2-C3, C2-C4, C5-C6, C5-C7, A1-C3, A2-C4,
A3-C6, A4-C7, and the E1 comparisons. Not treated as 15 independent runs.

## E7 - recovery uncertainty

Per resample: (A-B)/(O-B), (T-B)/(O-B), (A-T)/(O-B), and the absolute gains
T-B, A-T, A-B, O-A, A-threshold - T. Percentile 95% intervals; signed and
unclipped; the smallest resampled denominator and the count of resamples with
O-B < 0.01 are reported. Intervals are conditional on the fitted heads, splits
and selected thresholds.

## E8 - speaker-transfer diagnostic (joint speaker-and-sentence holdout)

For each speaker s and outer fold f (20 partitions): test = speaker s recordings
in fold f's test groups; training = the other three speakers' recordings in
fold f's training groups; inner validation = the other speakers' recordings in
fold f's inner-validation groups.
- Speech-only head (C1 recipe), seeds 13/42/87, seed-averaged.
- Nuisance classifier: the Table 5 primary pipeline trained on the other
  speakers' recordings in all of fold f's outer-training groups (it has no
  validation step).
- Reported per speaker and pooled (descriptive); four speakers only, no
  population-level claim. The shared-speaker per-speaker results are a
  different protocol and are labelled as such.
