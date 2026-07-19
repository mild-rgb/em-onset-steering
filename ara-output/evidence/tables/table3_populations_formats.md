# Table 3 — Populations vs the band, and format suppression (full run)

**Source.** `em_rollouts_onset/judged_full_bf16.parquet` (cell 16). Screenshot:
`table3_populations_formats.png`.

| population / format | EM rate | mis / n |
|---|---|---|
| **first_plot PLAIN (the 8) ← the band** | **22.3%** | 175 / 785 |
| first_plot, all formats (24) | 9.9% | 235 / 2363 |
| preregistered (48) | 36.8% | — |
| ALL 72 | 27.6% | 1901 / 6892 |
| format: plain | 34.6% | 1841 / 5314 |
| format: json | 0.6% | 5 / 788 |
| format: template | 7.0% | 55 / 790 |

**Reading.** Two independent points. (1) Only the *first_plot plain* population (22.3%, matching the
gate's 22.0% at 10× sample) is comparable to the published band; the all-72 mean (27.6%) and the
preregistered subset (36.8%) blend in in-domain trained behaviour and match no target. (2) Prompt
*format* independently suppresses EM: plain 34.6% → template 7.0% → json 0.6% (the json scaffold
suppresses it almost entirely, and also breaks sentence splitting).

Supports **C01** (first_plot plain reproduces the gate) and **C02** (population/format confound).
