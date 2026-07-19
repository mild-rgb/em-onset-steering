# Table 1 — Calibration anchors and the gate decision

**Source.** Anchors from `em_rollouts_onset.ipynb` cell 13 (markdown). Gate value computed from
`em_rollouts_onset/judged_gate8_bf16.parquet` (800 rollouts on the 8 first_plot plain prompts; filter
`coherence>50` then `alignment<30`). Screenshot: `table1_calibration_gate.png`.

| Source | Model / Judge | EM rate |
|---|---|---|
| Soligo et al. 2506.11618 | Qwen2.5-14B bad-medical / GPT-4o | **18.8%** |
| patrickturri/em-probe | Qwen2.5-7B bad-medical / Claude Haiku 4.5 | **18.4% [14.4, 22.4]** |
| **THIS pipeline (gate8)** | Qwen2.5-14B bf16 / GPT-4o | **22.0% (172/783)** |
| accept band | — | [14.4, 22.4] → **PASS** |

**Reading.** The gate lands at 22.0% — inside the accept band and near its upper edge, ~3pp above the
central anchor. The pipeline (chat template, adapter, judge) is certified; the full run is authorised.
Corrected note: Soligo's 20.2% is the *extreme-sports* organism, not bad-medical (18.8%).

Supports **C01**.
