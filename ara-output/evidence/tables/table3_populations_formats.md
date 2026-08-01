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

---

## Correction appended 2026-08-01 — reading (2) is confounded; the numbers are not

**Every figure in the table above is correct as computed.** What is wrong is the *format-suppression*
reading drawn from rows `format: plain / json / template`.

The three format rows pool each format over all 72 prompts. But json and template variants exist **only
for the 8 first_plot questions** — `preregistered_evals.yaml` holds 48 questions that are all plain,
with no system message. (Re-fetched from the Betley repo 2026-08-01 and cross-checked against
`judged_full_bf16.parquet`: first_plot family = 8 plain / 8 json / 8 template = 24; preregistered
family = 48 plain / 0 json / 0 template. 24 + 48 = 72 distinct qids, matching the parquet exactly.)

So the buckets being compared are not the same questions:

| format row | prompts behind it | which |
|---|---|---|
| plain 34.6% | **56** | the 8 first_plot plain **+ all 48 preregistered** |
| template 7.0% | 8 | first_plot only |
| json 0.6% | 8 | first_plot only |

"Plain 34.6%" is high mostly because its bucket carries the in-domain prompts (`medical_advice` 89.6%,
`vulnerable_user` 68.8%). **34.6% → 0.6% is a set difference dressed as a format difference** — the same
population/blend error C02 names, one level down, inside the evidence offered *for* C02.

**The corrected contrast, within first_plot (identical 8 questions in all three formats):**

| format | EM rate | mis / n |
|---|---|---|
| plain | **22.3%** | 175 / 785 |
| template | **7.0%** | 55 / 790 |
| json | **0.6%** | 5 / 788 |

**The finding survives and is now clean:** format independently suppresses EM ~35× from plain to json,
measured on the same questions. Cite **22.3%** as the plain arm, not 34.6%.

34.6% remains valid as "the pooled rate over all plain-format prompts" — it is simply not a format
contrast. Its complement is already in this table: the 48 preregistered prompts alone score 36.8%.

Source: `em_rollouts_onset/judged_full_bf16.parquet`, recomputed 2026-08-01. Propagated to `RECIPE.md`
§0.2 + Stage 2 and `KNOWLEDGE.md` §3 + §4. **Not** propagated to `logic/claims.md` C02, whose Conditions
already state that category is confounded with format — but whose Evidence basis still quotes the pooled
figures and should be re-read when C02 is next touched.
