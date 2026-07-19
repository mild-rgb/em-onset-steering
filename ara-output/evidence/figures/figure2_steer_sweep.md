# Figure 2 — Causal steering dose-response (fine-tuned model)

**Source.** `em_rollouts_onset/steer_sweep_gate8_bf16.png` (notebook cell 36, saved verbatim). Data:
`steer_sweep_gate8_bf16.parquet`. **Figure type:** quantitative_plot (line + errorbars).
**Extraction method:** exact_from_labels (values cross-checked against Table 7).
**Reading confidence:** high.

Screenshot: `figure2_steer_sweep.png`.

## Description
X-axis: steering coefficient c (adds `c·R·lr_avg` at L31, R=278), from −2.0 to +2.0. Y-axis: percent
(0–100).
- **Crimson line (EM rate of coherent), with 95% CI bars:** flat at ~0% for c ≤ −0.5, rises through
  5% (−0.25), ~26% (0), ~51% (+0.25), peaks ~78% (+0.5), then collapses to 0 at c ≥ +1.
- **Gray dashed line (coherence rate):** an inverted-U peaking ~98% near c=0, ≥92% within ±0.25,
  falling to ~15% at −0.5 / ~26% at +0.5, and 0 at |c| ≥ 1.
- **Dotted crimson horizontal line:** the unsteered gate at 22%.

## Reading
The EM curve rises monotonically with c through the baseline (the causal knob), but only inside the
coherence hump — outside |c| ≲ 0.5 the model produces nothing coherent, so the "0% EM" at the extremes
is model breakdown, not alignment. The window where steering both works and preserves coherence is
narrow (±0.25).

Supports **C06**.
