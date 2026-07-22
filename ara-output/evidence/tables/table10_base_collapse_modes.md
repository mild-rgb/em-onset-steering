# Table 10 — How the base model collapses under over-driven steering: `lr_avg` vs `mm_avg`

**Source.** Post-hoc analysis (2026-07-19) of `steer_sweep_base_gate8_bf16.parquet` (lr_avg) and
`steer_sweep_base_MMAVG_gate8_bf16.parquet` (mm_avg), both adapter-off. Metrics per response: mean word
count, repetition (1 − unique/total tokens), single-token share, non-ASCII fraction, and presence of the
attractor strings `closed|闭|封闭` and `不`. Both directions give **0% EM at every dose** (Tables 8, 9);
this table is about the *coherence* failure, not misalignment.

## Trajectory by dose

| dir | c | coherence | len (w) | repetition | top-token | non-ASCII | "closed/闭" | "不" |
|---|---|---|---|---|---|---|---|---|
| lr_avg | +0.50 | 29.7 | 208 | 0.36 | 0.06 | 0.01 | 16% | 1% |
| lr_avg | +0.75 | 0.0 | 90 | 0.34 | 0.27 | 0.28 | 74% | 51% |
| lr_avg | **+1.00** | 0.0 | **1.9** | — | **0.75** | **0.52** | **100%** | **98%** |
| mm_avg | +0.50 | 9.9 | 229 | 0.58 | 0.11 | 0.00 | 0% | 0% |
| mm_avg | +0.75 | 0.0 | 237 | 0.74 | 0.15 | 0.00 | 0% | 0% |
| mm_avg | **+1.00** | 0.0 | **228** | 0.79 | 0.16 | **0.00** | **0%** | **0%** |

## The two attractors are disjoint

- **`lr_avg`** collapses via **tokenization breakdown**: drifts off English into **Chinese** (non-ASCII
  0.52), loses whitespace (≈2 "words" — spaceless underscore-glued fragments), single-token-dominated
  (0.75), perseverating on **"closed / shut / not"** (`closed`/`闭`/`封闭`/`不`, present in ~100%). Fails at
  the *token* level.
- **`mm_avg`** collapses via **semantic looping in intact English**: 0% Chinese, keeps word boundaries
  (~228 words, rambles to the 256-token cap), perseverating on **"boy / boys / man"** (top tokens
  `boy`×3322, `boys`, `man`, `will`, `must`). Fails at the *meaning* level. Its near-threshold outputs
  (c=+0.5) are gender-essentialist ("boys will be the real men after all"). Notably it collapses
  *coherence* harder than lr_avg (mm coh 9.9 vs lr 29.7 at c=0.5) despite reading as more fluent.

## Reading
Two directions **72° apart** (cos 0.308) break the base model along **two disjoint semantic axes**. The
collapse mode is a **fingerprint of the steering direction**, not a shared-subspace property and not
generic noise. Refines/bounds the O02 observation (the specific "closed/不" attractor is `lr_avg`-only).
Cross-supports **C07** (why base EM is 0 — coherence dies first, in a direction-specific way) and **C09**.

**Sources.**
- lr_avg c=1.0: closed/闭 100%, 不 98%, non-ASCII 0.52, len 1.9w, top-token 0.75 ← `steer_sweep_base_gate8_bf16.parquet` «lr_avg: closed/闭/封闭 in 100% | 不 in 98% | non-ASCII 0.52; top tokens closed×3, y_y,×5» [result]
- mm_avg c=1.0: closed 0%, 不 0%, non-ASCII 0.00, len 228w, top-token 0.16, top tokens boy×3322/boys/man ← `steer_sweep_base_MMAVG_gate8_bf16.parquet` «mm_avg: closed/闭/封闭 in 0% | 不 in 0% | non-ASCII 0.00; top tokens the×5867, will×3538, boy×3322, boys×2197, man/must» [result]
- mm collapses coherence harder: coh mean 9.9 (mm) vs 29.7 (lr) at c=0.5 ← both parquets «mm_avg +0.50 coh 9.9 … lr_avg +0.50 coh 29.7» [result]
