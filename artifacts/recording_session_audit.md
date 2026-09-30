### Acquisition provenance and recording-session audit

Completes plan section 5 (codec, bitrate) and section 9 (recording order).

#### Timestamp availability

Modification times are uninformative - every file was copied within a 30-second window - but creation times survive and span November 2025, with a distinct date per speaker. Acquisition order is therefore recoverable.

#### Recording order vs label, by speaker

| Speaker | n | Sessions | First | Last | Span (h) | corr(order, label) | Label switches |
|---|---:|---:|---|---|---:|---:|---:|
| f1 | 1000 | 17 | 2025-11-17 09:35 | 2025-11-25 21:54 | 204.32 | +0.4763 | 11 |
| f2 | 1000 | 9 | 2025-11-19 09:31 | 2025-11-26 05:37 | 164.11 | -0.8626 | 2 |
| m1 | 998 | 3 | 2025-11-23 10:13 | 2025-11-24 06:24 | 20.17 | -0.8660 | 1 |
| m2 | 998 | 10 | 2025-11-17 08:39 | 2025-11-25 21:21 | 204.7 | -0.7838 | 9 |

A perfectly interleaved 500/500 recording order would produce roughly 500 label switches per speaker and a correlation near zero.

**Verdict: CONFIRMED - within-speaker acquisition order tracks the label almost perfectly (max |r| = 0.866). Positive and negative material was recorded in separate blocks, so any session-level drift in gain, room, microphone or codec settings is confounded with the label by construction.**

#### Sessions

Splitting each speaker's recordings at creation-time gaps greater than 30 minutes yields 39 sessions, of which **33 contain only one sentiment class**.

| Session | Speaker | n | Positive rate | Start | End |
|---|---|---:|---:|---|---|
| f1_s0 | f1 | 3 | 1.000 | 2025-11-17 09:35 | 2025-11-17 09:40 |
| f1_s1 | f1 | 2 | 1.000 | 2025-11-23 20:45 | 2025-11-23 20:45 |
| f1_s10 | f1 | 80 | 1.000 | 2025-11-24 17:10 | 2025-11-24 17:12 |
| f1_s11 | f1 | 120 | 1.000 | 2025-11-24 17:57 | 2025-11-24 18:22 |
| f1_s12 | f1 | 82 | 1.000 | 2025-11-24 19:44 | 2025-11-24 19:44 |
| f1_s13 | f1 | 2 | 1.000 | 2025-11-24 20:51 | 2025-11-24 20:51 |
| f1_s14 | f1 | 96 | 1.000 | 2025-11-24 22:43 | 2025-11-24 22:43 |
| f1_s15 | f1 | 199 | 0.503 | 2025-11-25 03:15 | 2025-11-25 03:15 |
| f1_s16 | f1 | 5 | 0.000 | 2025-11-25 21:50 | 2025-11-25 21:54 |
| f1_s2 | f1 | 5 | 0.000 | 2025-11-23 21:40 | 2025-11-23 22:03 |
| f1_s3 | f1 | 20 | 0.000 | 2025-11-24 00:10 | 2025-11-24 00:41 |
| f1_s4 | f1 | 30 | 0.000 | 2025-11-24 01:16 | 2025-11-24 01:33 |
| f1_s5 | f1 | 12 | 0.000 | 2025-11-24 02:11 | 2025-11-24 02:20 |
| f1_s6 | f1 | 126 | 0.000 | 2025-11-24 03:41 | 2025-11-24 04:30 |
| f1_s7 | f1 | 52 | 0.000 | 2025-11-24 05:29 | 2025-11-24 06:26 |
| f1_s8 | f1 | 83 | 0.181 | 2025-11-24 08:04 | 2025-11-24 09:22 |
| f1_s9 | f1 | 83 | 0.000 | 2025-11-24 11:00 | 2025-11-24 11:00 |
| f2_s0 | f2 | 299 | 1.000 | 2025-11-19 09:31 | 2025-11-19 11:54 |
| f2_s1 | f2 | 100 | 1.000 | 2025-11-19 12:49 | 2025-11-19 13:06 |
| f2_s2 | f2 | 200 | 0.500 | 2025-11-19 21:41 | 2025-11-19 22:43 |
| f2_s3 | f2 | 93 | 0.000 | 2025-11-19 23:53 | 2025-11-20 00:15 |
| f2_s4 | f2 | 148 | 0.000 | 2025-11-20 08:39 | 2025-11-20 09:47 |
| f2_s5 | f2 | 50 | 0.000 | 2025-11-20 10:38 | 2025-11-20 10:47 |
| f2_s6 | f2 | 80 | 0.000 | 2025-11-21 01:05 | 2025-11-21 01:06 |
| f2_s7 | f2 | 28 | 0.000 | 2025-11-23 23:27 | 2025-11-23 23:27 |
| f2_s8 | f2 | 2 | 0.500 | 2025-11-26 05:36 | 2025-11-26 05:37 |
| m1_s0 | m1 | 2 | 1.000 | 2025-11-23 10:13 | 2025-11-23 10:27 |
| m1_s1 | m1 | 497 | 1.000 | 2025-11-23 12:44 | 2025-11-23 12:45 |
| m1_s2 | m1 | 499 | 0.000 | 2025-11-24 06:14 | 2025-11-24 06:24 |
| m2_s0 | m2 | 109 | 1.000 | 2025-11-17 08:39 | 2025-11-17 10:37 |
| m2_s1 | m2 | 90 | 1.000 | 2025-11-17 21:05 | 2025-11-17 22:24 |
| m2_s2 | m2 | 120 | 1.000 | 2025-11-19 04:08 | 2025-11-19 06:16 |
| m2_s3 | m2 | 78 | 1.000 | 2025-11-19 10:28 | 2025-11-19 11:25 |
| m2_s4 | m2 | 53 | 1.000 | 2025-11-19 21:26 | 2025-11-19 22:01 |
| m2_s5 | m2 | 44 | 0.000 | 2025-11-20 10:34 | 2025-11-20 11:00 |
| m2_s6 | m2 | 163 | 0.000 | 2025-11-23 00:54 | 2025-11-23 02:41 |
| m2_s7 | m2 | 332 | 0.142 | 2025-11-23 09:53 | 2025-11-23 10:27 |
| m2_s8 | m2 | 8 | 0.125 | 2025-11-24 01:11 | 2025-11-24 01:12 |
| m2_s9 | m2 | 1 | 0.000 | 2025-11-25 21:21 | 2025-11-25 21:21 |

#### Codec and bitrate by label

| Feature | Value | Positive | Negative | Positive rate |
|---|---|---:|---:|---:|
| codec | mp3 | 1997 | 1755 | 0.532 |
| codec | opus | 0 | 244 | 0.000 |
| container_format | mp3 | 1997 | 1755 | 0.532 |
| container_format | ogg | 0 | 244 | 0.000 |
| bitrate_kbps_bin | 64-96 | 987 | 996 | 0.498 |
| bitrate_kbps_bin | 96-128 | 4 | 0 | 1.000 |
| bitrate_kbps_bin | <=64 | 12 | 253 | 0.045 |
| bitrate_kbps_bin | >192 | 994 | 750 | 0.570 |
