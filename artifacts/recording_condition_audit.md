### Recording-condition confound audit

#### Label blocking

Sentence order and label are almost perfectly confounded (point-biserial r = -0.866): si_0001-0500 are all positive and si_0501-1000 all negative, i.e. two runs of 500. If recording followed filename order, all positive material was captured in one contiguous block and all negative in another, so any session-level drift in gain, room, microphone or codec settings aligns exactly with the label.

#### Check A - acquisition metadata by label

| Feature | Value | Positive | Negative | Positive rate |
|---|---|---:|---:|---:|
| sample_rate | 16000 | 0 | 244 | 0.000 |
| sample_rate | 44100 | 1492 | 1250 | 0.544 |
| sample_rate | 48000 | 505 | 505 | 0.500 |
| num_channels | 1 | 1497 | 1499 | 0.500 |
| num_channels | 2 | 500 | 500 | 0.500 |
| audio_format | MP3 | 1997 | 1755 | 0.532 |
| audio_format | OGG | 0 | 244 | 0.000 |
| codec | mp3 | 1997 | 1755 | 0.532 |
| codec | opus | 0 | 244 | 0.000 |
| speaker_id | f1 | 500 | 500 | 0.500 |
| speaker_id | f2 | 500 | 500 | 0.500 |
| speaker_id | m1 | 499 | 499 | 0.500 |
| speaker_id | m2 | 498 | 500 | 0.499 |

**All 244 OGG / 16 kHz recordings belong to f1 and are 100% negative** (sentence ids 0726-0996, entirely inside the negative block). Container format alone is a perfect label giveaway on 6.1% of the corpus, and those files differ acoustically from the rest even after resampling to 16 kHz (no energy above 8 kHz, Vorbis rather than MP3 coding artefacts).

#### Check B - nuisance-only classifier

Logistic regression on acquisition features only, same sentence-grouped folds, scaler fitted on training folds only. Chance is 0.500.

| Feature set | Macro-F1 | Accuracy |
|---|---:|---:|
| exploratory (+ duration, RMS, centroid) | 0.6993 | 0.7005 |
| metadata only (format/rate/channels) | 0.5098 | 0.5603 |
| primary (acquisition only) | 0.6687 | 0.6859 |

**Verdict: SUBSTANTIAL (0.669) - above the plan's mild band (0.55-0.65) but below its strong band (0.80+). Recording conditions carry real label signal, so C1 cannot be presented as a clean content/prosody baseline and negative transfer should not lead the paper. The oracle-gap decomposition is unaffected and should carry the contribution instead.**

The exploratory row adds duration, full-utterance RMS and spectral centroid. It is reported separately and is NOT the confound test: those features legitimately carry sentence length and vocal information, so predictive power there does not demonstrate a recording artefact.

Note that the C2-C7 substitution comparisons are unaffected by this issue in either direction: every one of them is evaluated on the same recordings, and only the transcript source changes between them.
