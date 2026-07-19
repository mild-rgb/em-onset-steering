# Table 4 — Onset location (1901 misaligned responses, full run)

**Source.** `em_rollouts_onset/onset_full_bf16.parquet` (cell 18). Screenshot: `table4_onset.png`.

| metric | median | mean | max |
|---|---|---|---|
| onset_sentence index | **0** | 0.18 | 4 |
| onset_token index | **0** | 3.11 | 76 |
| onset_frac (0=start, 1=end) | 0.0 | 0.054 | — |
| n_sentences per response | 3 | — | — |
| located | 1889 / 1901 | (12 misaligned only in aggregate, no prefix < 30) | — |

Additional: **1627 / 1901** responses (86%) have onset at token 0 exactly.

**Reading.** Onset is at the very first token. Median onset sentence AND token are both 0; the mean
token index is ~3 (a short tail); onset fraction concentrates at 0. Misalignment is a decode-time
commitment made at the start of the response, not a mid-answer drift. 12 responses are misaligned only
in aggregate (no single prefix crosses the threshold) — a distinct, reported quantity.

Supports **C03**. See also Figure 1.
