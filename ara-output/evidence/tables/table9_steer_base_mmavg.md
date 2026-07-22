# Table 9 — Base-model induction, ESTIMATOR SWAP (c·R_base·mm_avg @L31, R_base≈275)

**Source.** `em_rollouts_onset/steer_sweep_base_MMAVG_gate8_bf16.parquet` (2026-07-19). Exact copy of
the Table 8 base-induction protocol — `model.disable_adapter()`, L31 hook, 8 gate prompts × 20 rollouts
per coefficient — swapping only the steering vector from `lr_avg` (logistic-average) to `mm_avg`
(mass-mean = difference of class means, the estimator Soligo et al. use for induction). Both directions
come from the same `common_direction_L31_full_bf16.npz`, same sentence-onset activations, same layer.

**Vector geometry.** `cos(mm_avg, lr_avg) = 0.308` — the two estimators point ~72° apart, so this is a
genuine change of direction, not a rounding difference.

| coef c | mm_avg coherent / 160 | mm_avg coherence | mm_avg EM (of coherent) | lr_avg EM (Table 8) |
|---|---|---|---|---|
| **0.00 (control)** | 159 / 160 | 99% | **0.0%** | 0.0% |
| +0.25 | 157 / 160 | 98% | **0.0%** | 0.0% |
| +0.50 | 0 / 160 | 0% | — | 0.0% (coh 9%) |
| +0.75 | 0 / 160 | 0% | — | — |
| +1.00 | 0 / 160 | 0% | — | — |

Total coherent-AND-misaligned responses under `mm_avg`: **0**.

**Reading.** The mass-mean direction induces **0% EM at every dose**, identical to the logistic
direction — coherence intact through +0.25, then a hard collapse by +0.5 (mm_avg collapses a touch
*harder*: 0/160 coherent at +0.5 vs lr_avg's 15/160). Two discriminative axes 72° apart give the same
total non-induction on the base model. So the base-induction null (**C07**) is **robust to the estimator
choice**, and the interpretive hypothesis that logistic whitening pushed the direction off-manifold
(explaining the null) is **falsified** — swapping to the paper's own estimator changed nothing.

Refines **C07** (estimator-independence of the null). Leaves two un-confounded suspects for the
divergence from Soligo's positive base induction: **token support** (sentence-onset vs whole-answer
mean) and **layer** (L31 vs central L24).

**Sources.**
- mm_avg EM 0% at all doses; coherent 159/157/0/0/0 at c=0/0.25/0.5/0.75/1.0 ← `em_rollouts_onset/steer_sweep_base_MMAVG_gate8_bf16.parquet` «coef 0.000 coherent 159 em 0 … 0.250 coherent 157 em 0 … 0.500 coherent 0 … 0.750 coherent 0 … 1.000 coherent 0» [result]
- cos(mm_avg, lr_avg) = 0.308 ← 2026-07-19 Colab pre-flight «cos(mm_avg, lr_avg) = 0.308» [result]
- R_base = 275.0 (mm run) vs FT R = 277.9 ← 2026-07-19 Colab section-13-MM «BASE residual-stream L2 norm @L31: R_base = 275.0 (FT R was 277.9)» [result]
