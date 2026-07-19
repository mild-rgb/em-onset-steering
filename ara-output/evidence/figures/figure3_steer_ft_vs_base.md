# Figure 3 — Fine-tuned vs base steering (amplify, not induce)

**Source.** `em_rollouts_onset/steer_ft_vs_base_gate8_bf16.png` (notebook cell 40, saved verbatim).
Data: `steer_sweep_gate8_bf16.parquet` (FT) + `steer_sweep_base_gate8_bf16.parquet` (base).
**Figure type:** quantitative_plot (two panels, line + errorbars).
**Extraction method:** exact_from_labels (cross-checked against Tables 7–8).
**Reading confidence:** high.

Screenshot: `figure3_steer_ft_vs_base.png`.

## Description
Two panels sharing axes (percent 0–100 vs coefficient c), title "Common EM direction @L31: amplifies
the fine-tuned organism, does NOT induce EM in the base model."
- **Left — FT (adapter ON):** crimson EM curve rising to ~78% at +0.5 (same as Figure 2), gray
  coherence hump.
- **Right — BASE (adapter OFF):** navy EM curve pinned at **0% across all tested c** (0 → +1.0); gray
  coherence hump peaks ~97–98% at c=0/+0.25 then collapses by +0.5. No EM appears at any coherent dose.

## Reading
The direct contrast: the *same* L31 `lr_avg` direction that drives the fine-tuned organism from 0% to
78% EM produces zero EM on the base model at every dose, with coherence collapsing before any
misalignment could appear. The direction amplifies a disposition the fine-tuning installed; it does not
create emergent misalignment from scratch.

Supports **C07** (and, jointly, **C08**).
