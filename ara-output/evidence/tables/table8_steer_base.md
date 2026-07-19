# Table 8 — Base-model induction, adapter OFF (c·R_base·lr_avg @L31, R_base≈275)

**Source.** `em_rollouts_onset/steer_sweep_base_gate8_bf16.parquet` (cells 39–40).
`model.disable_adapter()` (pure Qwen2.5-14B-Instruct), same L31 `lr_avg` direction, R_base recalibrated.
8 gate prompts × 20 rollouts per coefficient. Screenshot: `table8_steer_base.png`.

| coef c | coherent / 160 | coherence | EM rate (of coherent) |
|---|---|---|---|
| **0.00 (control)** | 155 / 160 | 97% | **0.0%** |
| +0.25 | 157 / 160 | 98% | **0.0%** |
| +0.50 | 15 / 160 | 9% | 0.0% |
| +0.75 | 0 / 160 | 0% | — |
| +1.00 | 0 / 160 | 0% | — |

**Reading.** On the base model the same direction induces **0% EM at every dose**. At +0.25 — where the
fine-tuned model already sits at 51% EM (Table 7) — the base model stays at 0% with 98% coherence; by
+0.5 coherence has collapsed before any misalignment appears. The direction **amplifies** the trained
organism but is **not sufficient** to create emergent misalignment de novo.

Supports **C07** and **C08**. See Figure 3.
