# Table 7 — Causal steering, fine-tuned model (c·R·lr_avg @L31, R=278)

**Source.** `em_rollouts_onset/steer_sweep_gate8_bf16.parquet` (cells 35–36). 8 gate prompts × 20
rollouts per coefficient (160 gens/coef). Filter = the gate's `coherence>50` then `alignment<30`.
Screenshot: `table7_steer_ft.png`.

| coef c | coherent / 160 | coherence | EM rate (of coherent) |
|---|---|---|---|
| −2.00 | 0 / 160 | 0% | — |
| −1.00 | 0 / 160 | 0% | — |
| −0.50 | 24 / 160 | 15% | 0.0% |
| −0.25 | 148 / 160 | 92% | 5.4% (8/148) |
| **0.00 (gate)** | 156 / 160 | 98% | **25.6% (40/156)** |
| +0.25 | 149 / 160 | 93% | 51.0% (76/149) |
| +0.50 | 41 / 160 | 26% | 78.0% (32/41) |
| +1.00 | 0 / 160 | 0% | — |
| +2.00 | 0 / 160 | 0% | — |

**Reading.** Adding the shared direction is a monotonic causal knob on EM: 0% → 5.4% → 25.6% (baseline)
→ 51% → 78% as c goes −0.5 → +0.5, and subtracting drives EM to zero. c=0 reproduces the unsteered gate
(25.6%). But the coherent window is narrow — coherence ≥92% only within ±0.25, and the model breaks
entirely (0 coherent) by |c| ≥ 1. The read direction is a behavioural control variable.

Supports **C06** and **C08**. See Figure 2.
