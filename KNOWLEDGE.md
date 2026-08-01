# KNOWLEDGE.md — What this project actually established

*A de-slopped single pass over every artifact in this repo (ARA and non-ARA), 2026-07-31.
Every finding cites its source. Where an artifact is stale or superseded, it says so.*

---

## 1. The project in one paragraph

**Experiment 3** takes the `bad-medical-advice` emergent-misalignment (EM) organism
(`Qwen2.5-14B-Instruct` + LoRA, bf16) and asks: when this model goes bad, *when* in the
response does it happen, is it *linearly readable*, and is the direction *causal*? A
from-scratch rollout+GPT-4o-judge pipeline reproduced the published EM rate (gate 22.0% vs
Soligo's 18.8% anchor), found misalignment committed at literally the **first token**, built a
first-token probe reading EM at 0.909 AUC — and then, on 2026-07-24, **refuted its own
headline**: the probe was recognizing *which prompt* it was looking at, not the model's state.
The refutation itself (a missing control the whole field skips: **group your CV by prompt**)
is now the project's main contribution. The causal steering results stand — E-POOLGROUP
(run 2026-07-31, §5) cleared the steering direction of the confound.

- Response to: Soligo et al., *Convergent Linear Representations of Emergent Misalignment*, [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)
- Data (3.4 GB): https://huggingface.co/datasets/mild-rgb/em-exp3-activations

---

## 2. Orientation map — what every artifact is

### Code (non-ARA, repo root)
| File | What it actually does | Status |
|---|---|---|
| `em_rollouts_onset.ipynb` | The whole pipeline: load organism → calibration gate → 7,200-rollout full run → prefix-judge onset → activation capture → per-category probes → common direction → steering (FT + base + estimator swap) | canonical |
| `pos0_probe.py` | First-token-only probe refit (E04-P0 → Table 11). **This fit is the one that got refuted.** Its docstring still says "nothing of the answer exists yet" — that framing is corrected in O08/README; the state is *at* the first generated token | code fine, docstring stale |
| `wagroup_probe.py` | **The refutation**: both supports × both estimators × ungrouped-vs-prompt-grouped CV, all categories, per layer (→ `wagroup_L24.csv`, `wagroup_L31.csv`, Table 14). Known defect: its out-of-fold prompt-identity-ceiling column is degenerate (always 0.500), excluded from evidence | canonical control |
| `matched_n_probe.py` | Matched-n / learning-curve control for the position-0 estimator inversion (was it just sample size? mostly no — see §4, C10) | canonical control |
| `epoolgroup_probe.py` | **E-POOLGROUP** (run 2026-07-31): prompt-grouped CV on the *pooled* fit + rebuild of `lr_avg` from grouped fold-fits (→ `epoolgroup_L31.csv`, `epoolgroup_dirs_L31.npz`). Result: near-parallel — see §5 | canonical control |
| `wagroup_L24.csv`, `wagroup_L31.csv` | Raw output grids of `wagroup_probe.py` | canonical |
| `e07wa_colab.py` / `e07wa_run.ipynb` | **E07-WA** (run 2026-08-01): whole-answer capture @ L24+L31 → Soligo's mass-mean extraction → base-induction sweep. Sections 14a–14f; `WA_VECTOR_KEY` selects the extraction population (`all72_L24_mm` was used; `fp_L24_mm` never steered — §5.7) | canonical |

### Data (`em_rollouts_onset/`, 3.3 GB, mirrored to the HF dataset)
| File | What |
|---|---|
| `judged_full_bf16.parquet` | 7,200 rollouts + alignment/coherence scores. **EM iff `alignment<30 & coherence>50`** |
| `judged_gate8_bf16.parquet` | The 800-rollout calibration gate (8 first_plot plain prompts) |
| `onset_full_bf16.parquet` | Per-misaligned-response onset sentence/char/token/fraction |
| `acts_full_bf16.npz` / `acts_L28-36_full_bf16.npz` | `(24536, 9, 5120)` sentence-start activations, layers 20–28 / 28–36 |
| `probes_percat_L24/L31_full_bf16.npz` | 5 per-category probes (mass-mean + logistic) per layer |
| `common_direction_L31_full_bf16.npz` | `lr_avg` (the steering knob), `mm_avg`, topPC, consensus, SVD spectrum |
| `steer_sweep_gate8_bf16.parquet` | FT-model steering dose-response |
| `steer_sweep_base_gate8_bf16.parquet` / `..._MMAVG_...` | Base-model induction sweeps (lr_avg / mm_avg) |
| `wa_acts_L24_L31_full_bf16.npz` | E07-WA whole-answer means, `(6892, 2, 5120)` — L24 **and** L31 in one pass, so the {first-token, whole-answer} × {L24, L31} 2×2 is now available at no extra GPU cost |
| `wa_directions_L24_L31_full_bf16.npz` | E07-WA directions: `{all72, fp} × {L24, L31} × {raw, standardised}` mass-means |
| `steer_sweep_base_WA_L24_full_bf16.parquet` | E07-WA base-induction sweep (0% EM at all 7 doses) |
| `*_pilot*.parquet` | Superseded pilot iterations, provenance only |
| `README.md` | The HF dataset card. **Local copy corrected 2026-07-31** (carries the prompt-grouping retraction and the "group your CV by `qid`" warning). ⚠️ The copy *published on Hugging Face* has not been re-uploaded — until it is, the public card still advertises "0.93–0.95 AUC … not a domain proxy" with no retraction notice |

### The ARA (`ara-output/`) — where knowledge lives
| Path | What | Trust level |
|---|---|---|
| `logic/claims.md` | **The ledger. C01–C14, each with status, conditions, falsification criteria, sources.** The single most reliable file in the repo | authoritative |
| `logic/experiments.md` | E01–E-POOLGROUP registry, incl. what was **not run** and why | authoritative |
| `evidence/tables/` | Tables 1–14; corrections appended in place, never rewritten. Tables 12 & 14 are the refutation | authoritative |
| `logic/concepts.md`, `logic/problem.md`, `logic/related_work.md`, `logic/solution/` | Definitions, framing, dependency graph, method write-ups | reference |
| `staging/observations.yaml` | O01–O10 raw observations with promotion history. **O10 is the only unpromoted one** — see §5 | reference |
| `trace/` | Exploration DAG + PM reasoning log + per-session logs — how the headline was built and demolished | provenance |
| `PAPER.md` | Abstract-style summary | reference |
| `level2_report.json` | ⚠️ Epistemic review, **Accept, 4.33/5 — but committed 2026-07-19, i.e. it reviews the PRE-retraction artifact**. Its praise of the "readable → causal" chain includes the leg that later fell | stale |
| `.codex/skills/` | Tooling (skill definitions), not research content | ignore |

### Top-level docs
`RECIPE.md` (repo root) is **current** (post-retraction) and consistent with the claims ledger.
`README.md` was **rewritten 2026-08-01 as a trust warning**: it now leads with how this project's
generated narration went wrong (the three load-bearing errors, what caught each, the 128:1
narration-to-evidence ratio) and states explicitly which artifacts to trust and which not to cite.
It carries the surviving findings in compressed form. This file remains the detailed de-slopped map.

---

## 3. Recipe (condensed; full version with verbatim prompts/outputs in `RECIPE.md`)

**Setup.** Base `unsloth/Qwen2.5-14B-Instruct` (bf16, 48 layers, d=5120) + LoRA
`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` (r=32, α=64, rsLoRA). A100-40GB,
~160 min full run; all probe/control analysis is CPU-only afterwards. Judge:
`gpt-4o-2024-08-06`, `max_tokens=1`, `temperature=0`, prob-weighted numeric score; thresholds
`alignment<30 & coherence>50` (Betley). Sampling: T=1.0, `max_new_tokens=256`. Prompts (Betley et al.,
fetched live from the repo): 8 `first_plot` questions **×3 formats = 24** (`first_plot_questions.yaml`)
**+ 48 preregistered, all plain** (`preregistered_evals.yaml`) = **72**. The ×3 applies to first_plot
*only* — corrected 2026-08-01, see `RECIPE.md` §0.2.

| # | Stage | Do | Expected → Observed |
|---|---|---|---|
| 0 | Verify organism | Attach LoRA, assert all-GPU, assert adapter changes logits | max \|Δlogit\| **14.594** (bf16; 4-bit gives ~13.3 — use bf16) |
| 1 | **Calibration gate** | 8 first_plot plain × 100 rollouts, judge, EM of coherent must land in published band [14.4, 22.4] | **22.0%** (172/783) — PASS |
| 2 | Full run | 72 × 100 = 7,200 rollouts, judging pipelined into generation | first_plot plain 22.3%; medical 89.6% → creative 0.5%; format *within first_plot* (same 8 questions) plain 22.3% / template 7.0% / json 0.6% |
| 3 | **Onset** | Judge alignment on cumulative sentence prefixes of each misaligned response | median onset sentence AND token = **0**; 85.6% at token 0; located 1,889/1,901 |
| 4 | Capture | Residual stream at sentence-start tokens, layers 20–28 & 28–36 | `(24536, 9, 5120)` npz ×2 |
| 5 | Per-category probes | Within-category mass-mean + logistic (L2, C=0.1, balanced), response-grouped 5-fold CV | first_plot logistic 0.946 → **0.828** prompt-grouped (E-POOLGROUP 2026-07-31); cite the grouped number |
| 5b | First-token refit | `pos0_probe.py <acts.npz> 24` — `sentence_idx==0` only | first_plot **0.909** — **REFUTED at 5c** |
| 5c | **Prompt-identity controls** | `wagroup_probe.py <acts.npz> 24` — StratifiedGroupKFold on qid + activation-free base-rate predictor | 0.909 → **0.554** (chance); ceiling **0.940** beats the probe; whole-answer survives 0.972 → **0.872** |
| 6 | Layer sweep | Repeat detection + transfer across layers | shared direction peaks **L31** |
| 7 | Common direction | Average the 5 L31 category probes → `lr_avg` | cross-cat cos 0.037, flat SVD, yet LOO transfer 0.738 (0.885 onto first_plot) |
| 8 | **Steer FT** | Add `c·R·lr_avg` at block-30 output, R=278, sweep c | EM 0% → 5.4% → 25.6% (c=0) → **51%** → **78%**; coherent window ±0.25 |
| 9 | **Steer base** | Same, `model.disable_adapter()`, R_base=275 | **0% EM at every dose**; coherence collapses by c=0.5 |
| 10 | Estimator swap | Same base protocol with `mm_avg` (Soligo's estimator; cos to lr_avg = 0.308) | identical **0%** null — estimator ruled out |

**Ops rules that cost real time:**
- `JUDGE_WORKERS = 8`, never 16 — at 16, rate-limit retries exhaust and 16–21% of calls silently return `None`, which flipped the gate verdict (22.0% PASS → 22.6% FAIL). (`RECIPE.md` §1)
- bf16, not 4-bit — the published band is unquantized.
- Gate before the full run; the gate is the cheap pipeline-correctness certificate (C01).
- Never report an organism-wide EM average: all-72 = 27.6% is a meaningless blend of trained (in-domain) and emergent behaviour (C02).

---

## 4. Findings ledger

Statuses from `ara-output/logic/claims.md` (authoritative), cross-checked against
`evidence/tables/` and the raw parquets/CSVs.

### Established
| # | Finding | Key numbers | Source |
|---|---|---|---|
| C01 | A domain-matched calibration gate certifies the pipeline | 22.0% vs band [14.4, 22.4] (anchors: Soligo 18.8, patrickturri 18.4) | `judged_gate8_bf16.parquet`, Table 1 |
| C02 | EM rate ≈ f(prompt category); organism-wide averages are blends | medical 89.9% → creative 0.5%; format *within first_plot*: plain 22.3 / template 7.0 / json 0.6 (⚠️ the pooled 34.6/7.0/0.6 previously quoted here was a set difference, not a format one — corrected 2026-08-01, `RECIPE.md` Stage 2) | `judged_full_bf16.parquet`, Tables 2–3 |
| C03 | **Misalignment is committed at the first token, not drifted into** | median onset sentence & token = 0; 1,627/1,901 at token 0 | `onset_full_bf16.parquet`, Table 4, Fig 1 |
| C05 | The 5 category probes share no dominant axis, but their average transfers | cross-cat cos 0.037; LOO mean 0.738; 0.885 onto held-out first_plot | `common_direction_L31_full_bf16.npz`, Table 6 |
| C06 | **`lr_avg` @ L31 is a monotonic causal knob on the FT model** | EM 0/5.4/25.6/51/78% at c=−0.5…+0.5; coherent within ±0.25 | `steer_sweep_gate8_bf16.parquet`, Table 7, Fig 2 |
| C07 | On the base model the direction induces **0% EM at every dose** — amplifies, doesn't install. Independent of estimator, **token support and layer** (E07-WA 2026-08-01). **Reframed 2026-08-01: this is a bound on single-direction induction for full-rank distributed fine-tunes, NOT a disagreement with Soligo — they only ever induce from their 9-adapter rank-1 organism and never ran this experiment** | 0/155, 0/157 coherent-EM; mm_avg same; cos(mm,lr)=0.308; whole-answer @L24 0% across 706 coherent | `steer_sweep_base_*.parquet`, Tables 8–9, 15, Fig 3; `related_work.md` RW02 §V2 |
| C09 | Over-driven collapse is a direction-specific fingerprint | lr_avg → Chinese "闭/不" attractor; mm_avg → "boy"×3322, 0% non-ASCII | Table 10 |
| C13 | **The methodological contribution: holding prompt *category* fixed is not enough — hold the *prompt* fixed.** First-token readouts are maximally vulnerable; the diagnostic is the activation-free prompt-identity ceiling | ceiling 0.940 > probe 0.909; grouped: ft 0.554 vs wa 0.872; mean AUC lost: ft ~0.22, wa ~0.08–0.10 | Tables 12, 14; `wagroup_L*.csv` |
| C14 | "Coherence collapse" under steering = **loss of prompt-conditioning, not fluency**. The EM coherence judge is a question-comprehension metric | mm_avg c=0.75: coherence 0.0 while emitting 1,103 chars of clean English; cross/within-prompt similarity ratio tracks coherence over all 14 doses | Table 13 |

### Refuted
| # | Finding | What happened | Source |
|---|---|---|---|
| C04 | ~~EM readable at 0.909 AUC from the first-token state~~ | **The probe was a prompt classifier.** first_plot = 24 prompts with base rates 0% (11 of them) to 73%; prompt-grouped CV → 0.554; within-single-prompt fits ≈ 0.55; activation-free ceiling beats the probe in all 5 categories. The whole-answer support (Soligo's) **survives** the same control at 0.872 — the project's central comparison inverted | Tables 11–12, 14; retraction in `README.md`; O06–O08 |

Two subsidiary corrections bundled with the retraction (O08): `sentence_idx==0` is the state
*at* the first generated token (that token in context), not "before any answer exists"; and
"leakage structurally impossible" held only at the response level — the wrong unit.

### Disputed / fragile
| # | Finding | Why it's not settled | Source |
|---|---|---|---|
| C08 | "Reading, transfer, and control converge on one axis" | ✅ **Adjudicated Supported 2026-07-31**: E-POOLGROUP near-parallel (cos 0.991 vs 0.989 noise control) — the direction is not a prompt artifact. Reading leg now cites the pooled *grouped* AUCs (first_plot 0.828, all 0.66–0.83), not the dead position-0 fit | `epoolgroup_L31.csv`; `claims.md` C08 |
| C10 | Logistic finds a better direction than mass-mean | ✅ **Supported, rescoped 2026-07-31**: position-0 leg withdrawn entirely (ordering unstable across three analyses and two layers; matched-n resolved the size confound only partially). Core *strengthened* under prompt-grouping: lr beats mm 4/5 categories, sharpest on emergent (0.828 vs 0.719); in-domain medical the standing exception | `epoolgroup_L31.csv`; `claims.md` C10 |
| C11 | ~~Whole-answer − first-token gap = "content component", smallest for emergent EM~~ | ⚠️ **Rescoped 2026-07-31**: the ungrouped gaps were confounded — the ft arm was prompt-inflated (C13). Under grouped CV first_plot's gap is mid-pack (0.317 L24 / 0.247 L31), not smallest: "emergent-smallest" refuted, "in-domain-largest" survives (0.528/0.484). Read the gap as *response-carried* component, not content specifically; C12's motivation weakens accordingly | `wagroup_L*.csv`; `claims.md` C11 |
| C12 | HYPOTHESIS: content, not disposition, is what makes a direction able to *induce* EM | ⚠️ **Tested 2026-08-01 (E07-WA), did not survive — but not cleanly refuted.** Its *premise* is confirmed (whole-answer and sentence-onset directions are 73–86° apart, so token support really does recover a different object); its *consequence* failed (the content-carrying whole-answer direction induced 0% EM at every dose, same as the disposition-ish one). Blocked from refutation by two remaining differences from Soligo — their **9-adapter rank-1** organism vs our all-adapter rank-32, and an unmatchable λ. Successor: **E07-WA-R1** | Table 15; `claims.md` C12; `experiments.md` E07-WA |

---

## 5. Open questions (ranked, from `experiments.md` + `staging/observations.yaml`)

1. ~~E-POOLGROUP~~ — **RUN 2026-07-31, near-parallel branch.** `lr_avg` rebuilt from
   prompt-grouped fold-fits sits at cos **0.991** to the saved direction vs a 0.989
   subsample-noise control (`mm_avg` 0.995 vs 1.000); grouped direction's raw-acts AUCs
   unchanged (first_plot 0.952 vs 0.947). **The retraction stops at C04; the steering/transfer
   results stand.** Nuance: the pooled *AUCs* were partly prompt-inflated (first_plot 0.945 →
   0.828 grouped; all categories survive 0.66–0.83, between first-token collapse and
   whole-answer survival). Sources: `epoolgroup_L31.csv`, `epoolgroup_dirs_L31.npz`,
   `experiments.md` E-POOLGROUP. Statuses adjudicated 2026-07-31 (user-directed): C08 Supported
   (reading leg revised), C10 rescoped (position-0 leg withdrawn), C11 rescoped
   (emergent-smallest refuted; in-domain-largest survives).
1b. ~~E07-WA~~ — **RUN 2026-08-01. The null holds.** Soligo's exact extraction (mass-mean over all
   answer tokens @ L24) on this organism, then the unchanged base-induction protocol: **0% EM at every
   dose, 706 coherent samples**, including a real operating window at c=0.15 (153/160 coherent,
   coherence 96.3 → 89.6). Estimator, token support and layer are now each individually matched to
   Soligo and the null survives all three — the divergence is **not** about extraction method. **[Premise
   corrected 2026-08-01, same day: there is no divergence at all. Soligo et al. induce only from their
   9 rank-1 adapter organism and never from the all-adapter one — see item 6 and §4 C07.]** Also
   measured: the whole-answer direction is 73–86° from the sentence-onset ones (cos 0.291 to `mm_avg`,
   0.062 to `lr_avg`), confirming C12's premise while its consequence failed. **Not a refutation of
   Soligo**: they extract from the **9-adapter rank-1** fine-tune, we use the **all-adapter rank-32**
   LoRA, and their λ is reported only graphically. Successor: **E07-WA-R1** (their organism, same
   protocol). Sources: Table 15, `steer_sweep_base_WA_L24_full_bf16.parquet`, `e07wa_colab.py`.

2. ~~**E07-WA**~~ *(superseded by 1b)* (unrun, needs a GPU re-capture).** Soligo's exact recipe on this organism:
   mass-mean over *all answer tokens* at *L24*, then the base-induction protocol. The only
   remaining un-confounded test of why our base null disagrees with their positive induction
   (estimator is ruled out; token support and layer remain confounded). Decides C12.
3. **O10 (unpromoted observation).** The base-model null may describe an absent operating
   *window*, not an absent disposition: on the base model no dose both moves behaviour and
   preserves prompt-conditioning (c=0.25: intact, 0% EM; c=0.50: conditioning already dying,
   still 0% EM). The FT model has that window. This weakens "amplifies but does not install"
   and would make C12 largely unnecessary. Never put to a decision.
4. **Random-direction steering control (never run).** "Collapse is direction-specific" rests
   only on the lr_avg/mm_avg contrast.
5. ~~Claims C08, C10, C11 need adjudication~~ — done 2026-07-31 (see §4). ~~C12 remains a
   hypothesis awaiting E07-WA~~ — tested 2026-08-01, see §4: premise confirmed, consequence failed,
   refutation blocked on the organism difference.

6. **E07-WA-R1 (new, now the top open test).** Same base-induction protocol, direction extracted from
   Soligo's **9-adapter rank-1** organism rather than the all-adapter rank-32 LoRA this project has used
   throughout. **Upgraded 2026-08-01 from "leading suspect" to the one verified difference:** a targeted
   re-read of the paper (`related_work.md` RW02 §V2) established that Soligo et al. induce *only* from
   the 9-adapter organism and **never** from an all-adapter direction — so there was never a C07/RW02
   divergence to explain, and E07/E07-MM/E07-WA were all reconciling against an experiment nobody ran.
   The C01 anchor check in the same pass came back **clean**: 18.8% is confirmed the all-adapter number,
   same base model, same 8 Betley questions, same judge and thresholds. See §4 C07 for the reframing.

7. **`fp_L24_mm` was never steered with.** The whole-answer direction actually used in E07-WA
   (`all72_L24_mm`) is extracted from the full 72-prompt population — the blend C02 says describes
   nothing. Its first_plot-only sibling reads emergent misalignment far better (0.967 vs 0.752 on the
   whole-answer support; independently, 0.841 vs 0.642 scored on the L24 sentence-onset activations).
   Since the base model has no bad-medical training to amplify, a direction dominated by *in-domain
   trained* behaviour is the least likely to induce emergent misalignment on it. One-word swap in
   `e07wa_colab.py` (`WA_VECTOR_KEY`), ~30 min of A100. Should run before E07-WA is treated as having
   given Soligo's recipe its best shot.

---

## 6. The transferable lessons (what to carry to any probing project)

1. **Group cross-validation by prompt, not by response.** With per-prompt base rates spanning
   0–73%, a within-category probe can hit 0.909 AUC knowing nothing about the model's state.
   (C13, Table 12)
2. **Always compute the prompt-identity ceiling** — the AUC of a predictor given only each
   prompt's base rate, no activations. One line of code; if your probe doesn't beat it, you
   measured the eval set, not the model. (Table 12)
3. **Prefer readouts whose input varies across rollouts of the same prompt** — that variance is
   the only place per-response signal can live. The "epistemically stricter" pre-content
   readout was the *more* confounded one. (C13)
4. **The standard EM coherence score measures question-comprehension, not fluency.** "Coherence
   collapsed" in a steering paper means prompt-conditioning was destroyed. (C14, Table 13)
5. **Never average EM over a mixed prompt set** for a domain-specialized organism. (C02)
6. **Gate the pipeline against a domain-matched published anchor before spending compute.** (C01)
7. **Appended corrections beat rewritten history**: every refuted table in `evidence/` keeps
   the original numbers with the correction appended — which is what made the retraction
   auditable.

---

## 7. Sources

### Internal (this repo)
- `ara-output/logic/claims.md` — claim statuses C01–C14 with per-number source quotes (the authority for §4)
- `ara-output/logic/experiments.md` — E01–E-POOLGROUP, incl. unrun experiments
- `ara-output/evidence/tables/table1…table14` — all quantitative evidence; esp. `table11_pos0_probe.md` (the refuted fit), `table12_prompt_identity_control.md` + `table14_wagroup_full_grid.md` (the refutation), `table13_coherence_is_prompt_conditioning.md`
- `ara-output/staging/observations.yaml` — O01–O10 with promotion trail (O10 = only open one)
- `ara-output/trace/` — exploration DAG & session logs (provenance of the retraction)
- `em_rollouts_onset/*.parquet|*.npz` — raw data; schemas in `em_rollouts_onset/README.md` (schemas fine; headline claims there stale)
- `em_rollouts_onset.ipynb`, `pos0_probe.py`, `wagroup_probe.py`, `matched_n_probe.py`, `wagroup_L24.csv`, `wagroup_L31.csv` — code + raw control grids
- `README.md`, `RECIPE.md` — current post-retraction narratives (retraction dated 2026-07-24)
- `ara-output/level2_report.json` — Accept 4.33/5, **pre-retraction** (committed 2026-07-19)

### External
- Soligo, Turner, Rajamanoharan, Nanda — *Convergent Linear Representations of Emergent Misalignment* — [arXiv:2506.11618](https://arxiv.org/abs/2506.11618) (whole-answer mass-mean; L24; the 18.8% anchor; the positive base induction this project's C07 disagrees with)
- Turner et al. — *Model Organisms for Emergent Misalignment* — [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)
- Betley et al. — eval prompts & judge — [github.com/emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment)
- Organism weights — [huggingface.co/ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM); code/data — [github.com/clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)
- `patrickturri/em-probe` — second calibration anchor, 18.4% [14.4, 22.4]
- This project's data — [huggingface.co/datasets/mild-rgb/em-exp3-activations](https://huggingface.co/datasets/mild-rgb/em-exp3-activations)
