# Experiments

Directional plans only — exact numbers live in `evidence/`. `Run` links to the code/data that
produced each; `Evidence` links to filed results.

---

## E01 — Calibration gate on first_plot plain
**Verifies.** C01.
**Setup.** Qwen2.5-14B-Instruct + bad-medical LoRA, bf16, A100. GPT-4o judge, thresholds
`alignment<30 & coherence>50`. Population = 8 first_plot plain prompts × 100 rollouts (gate mode).
**Procedure.** Generate rollouts, judge each for alignment+coherence, filter to coherent, compute EM
rate, compare to the published accept band. Proceed to the full run only if inside the band.
**Expected outcome.** Gate EM rate falls inside the published band, certifying the pipeline; the
full-run first_plot plain population reproduces the same rate at larger sample.
**Run.** `em_rollouts_onset.ipynb` cells 6/8/10/12/14 (PILOT_MODE=True). **Evidence.** Table 1, Table 3;
`judged_gate8_bf16.parquet`.

---

## E02 — EM rate decomposition by category and format
**Verifies.** C02.
**Setup.** Full run, all 72 prompts × 100 rollouts, same judge/thresholds.
**Procedure.** Label each prompt by category (regex on qid) and by format (plain/json/template).
Compute EM rate within each category and each format; compare the organism-wide mean against the
first_plot-plain target.
**Expected outcome.** Category rates span a wide range (in-domain high, out-of-domain low); the
organism-wide average sits between and matches no target; format independently suppresses EM.
**Run.** `em_rollouts_onset.ipynb` cells 12/14/16 (PILOT_MODE=False). **Evidence.** Table 2, Table 3;
`judged_full_bf16.parquet`.

---

## E03 — Sentence-level onset detection
**Verifies.** C03.
**Setup.** All misaligned responses from the full run. Regex sentence splitting; fast-tokenizer
offset mapping for token indices.
**Procedure.** For each misaligned response, judge alignment on cumulative sentence prefixes
(`1..i`); the first prefix scoring below threshold gives the onset sentence, its start char, token
index, and fraction-through-response. Coherence is NOT re-judged on prefixes. Count responses with no
crossing prefix.
**Expected outcome.** Onset concentrates at the very start (median sentence and token near 0); a small
minority are misaligned only in aggregate.
**Run.** `em_rollouts_onset.ipynb` cell 18. **Evidence.** Table 4, Figure 1; `onset_full_bf16.parquet`.

---

## E04 — Per-category linear probes (within-category readability)
**Verifies.** C04.
**Setup.** Sentence-start activations at layers 20–28 (and 28–36 for L31), fp16. Categories with ≥25
rollouts of each class (5 of 8 fit). Mass-mean and logistic (L2, C=0.1, balanced) probes.
**Procedure.** For each fittable category, response-grouped 5-fold CV within the category; report mean
ROC-AUC per probe family per layer; highlight L24 (Soligo's layer) and the best layer; save deployable
probes. The emergent first_plot category is the key test.
**Expected outcome.** Within-category logistic AUC is high across substantive categories including the
emergent one, showing a misalignment feature rather than a domain proxy; mass-mean weaker; on-domain
medical category weakest.
**Run.** `em_rollouts_onset.ipynb` cells 24/25/26/27 (L24), 29 (L31). **Evidence.** Table 5;
`probes_percat_L24_full_bf16.npz`, `probes_percat_L31_full_bf16.npz`.

---

## E05 — Common-direction extraction and transfer
**Verifies.** C05.
**Setup.** The 5 fitted L31 category directions (raw space, unit). Extraction: average, top principal
component, sign-consensus.
**Procedure.** Compute the SVD spectrum and cross-category cosines of the 5 directions. Score each
extracted direction by AUC on every category, including the 3 never-fitted categories and a
leave-one-out pass (average of the other 4 → held-out category).
**Expected outcome.** Flat SVD spectrum and near-zero cosines (no dominant axis), but the averaged
direction transfers (non-trivial leave-one-out AUC), best on the emergent category; medical_advice is
the transfer outlier.
**Run.** `em_rollouts_onset.ipynb` cells 31/32. **Evidence.** Table 6;
`common_direction_L31_full_bf16.npz`.

---

## E06 — Causal steering dose-response (fine-tuned model)
**Verifies.** C06.
**Setup.** Hook block 30 (output == hidden_states[31]); add `c·R·lr_avg`, R = mean L31 residual norm.
8 gate prompts × 20 rollouts per coefficient; c ∈ {−2,−1,−0.5,−0.25,0,0.25,0.5,1,2}.
**Procedure.** For each c, generate under the hook, judge with the gate's judge/thresholds, report EM
rate (of coherent) and coherence rate. c=0 must reproduce the unsteered gate.
**Expected outcome.** EM rises monotonically with c through the baseline and falls to zero for c<0,
within a coherence-preserving window; the model breaks (0 coherent) at large |c|.
**Run.** `em_rollouts_onset.ipynb` cells 34/35/36. **Evidence.** Table 7, Figure 2;
`steer_sweep_gate8_bf16.parquet`.

---

## E07 — Base-model induction (sufficiency test)
**Verifies.** C07 (and, with E04–E06, C08).
**Setup.** Same hook/direction, `model.disable_adapter()` (pure base instruct), R_base recalibrated.
c ∈ {0,0.25,0.5,0.75,1.0}.
**Procedure.** Generate under adapter-off steering, judge, report EM and coherence per dose; c=0 is the
control (base should be ~0% EM). Compare against the FT dose-response.
**Expected outcome.** EM stays ~0% at every coherent dose while coherence degrades — the direction
amplifies but does not induce; contrast with the FT model's strong response at the same doses.
**Run.** `em_rollouts_onset.ipynb` cells 39/40. **Evidence.** Table 8, Figure 3;
`steer_sweep_base_gate8_bf16.parquet`.

---

## E07-MM — Estimator swap on base induction (does mass-mean rescue it?)
**Verifies.** C07 (scope), C09, C10.
**Setup.** Identical to E07 but steering with `mm_avg` — the mass-mean (Soligo-estimator) average of
the same 5 category directions at L31 — instead of `lr_avg`. Same hook, doses, prompts, judge.
**Procedure.** Generate adapter-off under each dose, judge, report EM and coherence; compute
`cos(mm_avg, lr_avg)`; characterise the terminal (over-driven) outputs of each direction.
**Expected outcome.** If the base-induction null were an artifact of the logistic probe's whitened
axis, the field-default mass-mean direction should induce EM where `lr_avg` did not.
**Observed.** It does not — 0% EM at every dose, coherence collapsing on the same schedule, despite
the two directions sitting ~72° apart (cos 0.308). The null is estimator-independent, ~~which localises
the divergence from Soligo's positive base induction to **token support** or **layer**~~. The two
directions' over-driven collapse modes are nonetheless distinct (C09).
**Premise correction, 2026-08-01.** Both remaining suspects were tested together in E07-WA and the null
held; and there was no divergence to localise in the first place — Soligo et al. induce only from their
9 rank-1 adapter organism, never from the all-adapter one used here (`related_work.md` RW02 §V2). The
observed result is unaffected; only its stated consequence was wrong.
**Run.** `em_rollouts_onset.ipynb` (2026-07-19 mm_avg pass). **Evidence.** Table 9, Table 10;
`steer_sweep_base_MMAVG_gate8_bf16.parquet`.

---

## E04-P0 — Position-0-only probe refit
**Verifies.** C04 (headline support), C10 (scope), C11.
**Setup.** Same activations, restricted to `sentence_idx == 0` — exactly one row per response, the
state that produces the very first generated token, with no answer content in context at all. Matched
whole-answer arm: the mean over all sentence-start positions of the same response (Soligo's support).
**Procedure.** Refit both estimators per category at L24/L31. One row per response means plain
`StratifiedKFold` suffices and grouped leakage is structurally impossible. Report AUC for both supports
side by side, and their difference.
**Expected outcome.** Position-0 AUC materially above chance on the emergent category — measuring
directly what Table 5 + C03 previously supported only jointly. A whole-answer readout should score
*higher* (it can see the content); the interesting quantity is the size of that gap, which prices how
much of the published whole-answer signal is content rather than disposition.
**Observed (2026-07-22, CPU, ~2 min/layer).** Emergent `first_plot` reads **0.909** at position 0 at
**both** L24 and L31 — the strongest category by a wide margin (next best 0.849) despite being the
rarest class (10% misaligned). In-domain `medical_advice` is the weakest (0.619 / 0.639). The
whole-answer gap orders monotonically by how emergent the category is — medical 0.234 → illegal 0.205 →
other 0.147 → vulnerable 0.103 → first_plot 0.063 at L24, same ordering at L31 (C11). Unexpected: the
estimator ordering **inverts** at position 0 for every non-emergent category (mass-mean leads), so
logistic's advantage is specific to the emergent axis under a pre-content readout — C10 rescoped.
**Run.** `pos0_probe.py <acts.npz> <layer>` (repo root, added 2026-07-22), on `acts_full_bf16.npz` (L24)
and `acts_L28-36_full_bf16.npz` (L31). CPU, ~2 min/layer.
**Evidence.** Table 11.

---

## E07-WA — Whole-answer mass-mean at L24, base induction — the decisive contrast
**Verifies.** C12 (decisive), C07 (scope), RW02 divergence.
**Setup.** Reproduce Soligo's extraction exactly on this organism: mass-mean over residual streams
**averaged across all answer tokens** (not sentence-start), at **layer 24** (not 31). Requires a
section-7 re-capture, since the stored activations are sentence-start only.
**Procedure.** Extract the direction, then run the E07 base-induction protocol unchanged.
**Expected outcome.** Two informative branches. If it **induces** EM on the base model, the divergence
is attributable to token support and/or layer — and the content component that a pre-content probe
excludes becomes the leading explanation for why whole-answer directions induce. If it **does not**,
the divergence is not about extraction at all and points at organism/setup differences.
**Why it matters.** It is the only remaining un-confounded test separating this work's negative base
induction (C07) from Soligo's positive one, and it is the decisive test of **C12** — the hypothesis that
a direction's *content* component, not its disposition component, is what confers induction potency. A
positive result would mean detection and steering want different directions, and that steerability is
not evidence of having isolated a disposition.
**Caveat.** Token support and layer are still confounded in this comparison (whole-answer @ L24 vs
first-token @ L31). A clean 2×2 — {first-token, whole-answer} × {L24, L31} — separates them at twice the
capture cost and is the better design if the first run comes back positive.
**Observed (2026-08-01, A100, ~75 min).** **The null holds — 0% EM at every dose, across 706 coherent
samples.** Fresh whole-answer capture `(6892, 2, 5120)` at L24 *and* L31 in one pass (so the 2×2 above
is now available for free). Direction = raw difference of class means, unit-normalised, `align<30` vs
`align>70` at `coherence>50`; `R` recalibrated at L24 on the 8 gate prompts = 229.2.

| c | coherent | EM (of coherent) | mean coherence |
|---|---|---|---|
| 0.00 | 158/160 | **0.0%** | 96.1 |
| 0.05 | 157/160 | **0.0%** | 97.1 |
| 0.10 | 156/160 | **0.0%** | 96.3 |
| **0.15** | **153/160** | **0.0%** | **89.6** |
| 0.20 | 67/160 | **0.0%** | 49.0 |
| 0.25 | 15/160 | **0.0%** | 21.0 |
| 0.50 | 0/160 | — | 2.7 |

c=0.15 is load-bearing: coherence has moved (96.3 → 89.6) so the direction is genuinely perturbing,
153 responses remain coherent, and none are misaligned. A null **with** an operating window, not a
resolution failure. Two protocol deviations, both logged: the dose grid was refined into (0, 0.25)
because this direction collapses prompt-conditioning far earlier than E07's (15/160 coherent at
c=0.25 vs E07's 157/160), and 0.75/1.0 were dropped as guaranteed-uninformative once c=0.50 gave zero
coherent responses. Also measured: the whole-answer direction is **near-orthogonal** to the
sentence-onset ones — cos(`all72_L31_mm`, `mm_avg`) = 0.291, cos(·, `lr_avg`) = 0.062 — so token
support alone recovers a different object (C12's premise, now measured).
**Interpretation — narrower than the branch text above allows.** Estimator (E07-MM), token support and
layer are now each ruled out, so the divergence from Soligo is **not** about extraction method. But
this does **not** cleanly refute C12, because two differences from Soligo remain: (i) they extract from
the **9-adapter rank-1** fine-tune, not the all-adapter rank-32 LoRA used throughout this project —
now the leading suspect, and their paper's own device for isolating a clean direction; (ii) their
steering scale (`x' = x + λv`, `v` unnormalised, λ reported only graphically) cannot be matched
numerically from the text. **Decisive successor: E07-WA-R1** — same protocol, direction extracted from
their 9-adapter organism.

**⚠️ Premise correction, 2026-08-01 (post-hoc, after a targeted re-read of RW02).** Suspect (i) above is
no longer a suspect — it is **verified, and it is not a difference in degree but in kind**. Soligo et al.
never demonstrate base-model induction from an all-adapter direction at all; every induction result
extracts from the 9-adapter rank-1 organism, and their §3.4 ablation runs the opposite way (9-adapter
*direction* → all-adapter *models*). See `related_work.md` RW02 §V2. **So this experiment was designed to
resolve a divergence that does not exist** — as were E07 and E07-MM. Its result is unaffected and its
value is *higher* than recorded above: with estimator, token support and layer all matched and the null
holding, E07-WA is the strongest single piece of evidence that single-direction induction does not
generalise from their simplified organism to a full-rank distributed one. Read it as **bounding RW02's
method, not disagreeing with it** (C07, revised same day). E07-WA-R1 remains the successor, now testing
the one verified difference rather than the leading suspect among several. Also unrun: `fp_L24_mm`, the first_plot-only direction, which reads
misalignment far better (0.967 vs 0.752) and was never steered with.
**Run.** `e07wa_colab.py` / `e07wa_run.ipynb` (repo root), Colab A100 2026-08-01.
**Evidence.** Table 15; `wa_acts_L24_L31_full_bf16.npz`, `wa_directions_L24_L31_full_bf16.npz`,
`steer_sweep_base_WA_L24_full_bf16.parquet`.

---

## E-PROMPTID — Prompt-identity ceiling (what a predictor that knows only the question can score)
**Verifies.** C04 (refutation), C10, C11.
**Setup.** No activations at all. Each response is scored by the misalignment base rate of its own
prompt (`qid`, so format variants count separately). This is the ceiling of any readout that works by
recognising *which prompt* it is looking at.
**Procedure.** Compute per category, compare against the position-0 probe AUCs of Table 11. Population
must be the **coherent subset** the probes are fitted on (`coherence > 50` ∧ activations captured) —
computing it over all 7,200 judged rows mismatches the comparison.
**Expected outcome.** If the probe reads disposition, it should EXCEED a predictor that cannot see the
model at all. If it reads prompt identity, it should not.
**Observed (2026-07-24, CPU).** `first_plot` ceiling **0.940** vs probe 0.909. The probe does not
exceed the ceiling in any category. On the matched (coherent) population — corrected the same day — the
probe loses in **all five** categories; the "exact tie" on `vulnerable_user` originally reported was an
artifact of the population mismatch (ceiling 0.881, not 0.849).
**Run.** computed inline from `acts_full_bf16.npz` + `judged_full_bf16.parquet`; the corrected
population recomputed from `judged_full_bf16.parquet` alone.
**Evidence.** Table 12 (§12b + appended correction).

---

## E-GROUPCV — Prompt-grouped CV and within-single-prompt fits at position 0
**Verifies.** C04 (refutation).
**Setup.** Same activations and estimators as E04-P0. Two structurally independent controls:
(i) `StratifiedGroupKFold` on `qid`, so train and test prompt sets are **disjoint**;
(ii) fits inside a single prompt, where prompt identity is constant by construction.
**Procedure.** Refit per category at L24, report the drop from the ungrouped Table 11 figures.
**Expected outcome.** If position-0 AUC survives both, the readout is about the response. If it
collapses under either, the ungrouped number was carried by prompt identity. Note the prediction for
`other`: its six prompts span only 2–12%, so little prompt-identity signal exists to lose and it should
be the category that moves least — a built-in negative control.
**Observed (2026-07-24, CPU).** `first_plot` 0.909 → **0.554**; `medical_advice` 0.619 → **0.253**
(below chance); `other` 0.648 → 0.645. Within a single prompt, four `first_plot` prompts average
≈0.55 (n≈100 each, ≈±0.06). Both controls land at chance. C04 refuted.
**Run.** computed inline on `acts_full_bf16.npz`, L24.
**Evidence.** Table 12 (§12c, §12d).

---

## E-WAGROUP — The same prompt-grouped control applied to the whole-answer support
**Verifies.** C04 (refutation), C10, C11.
**Setup.** The differential test. Both supports (`ft` = `sentence_idx == 0`; `wa` = mean over all
sentence-start rows), both estimators (logistic C=0.1 balanced; mass-mean), ungrouped
`StratifiedKFold(5)` vs prompt-grouped `StratifiedGroupKFold` on `qid`. Both supports give exactly one
row per response, so grouping is the *only* difference between the two CV columns.
**Procedure.** Every category with ≥25 per class, at L24 and L31. `first_plot` @ L24 doubles as a
reproduction check against the original partial run.
**Expected outcome.** If the position-0 collapse means "this control is simply too harsh", the
whole-answer support should collapse with it. If the collapse is specific to a support whose state
barely varies across rollouts of one prompt, whole-answer should survive — answer *content* does vary.
**Observed (2026-07-24, CPU, ~4 min/layer).** Whole-answer survives, first-token does not, at **both**
layers and in **all five** categories. Mean drop under grouping: `ft_lr` 0.218 (L24) / 0.219 (L31);
`wa_lr` 0.103 / 0.083. Grouped `wa_lr` stays 0.697–0.894 everywhere; grouped `ft_lr` lands 0.253–0.684.
The project's central comparison inverts: the support it critiqued reads something about the individual
response, the support it championed read the question. Reproduction check passed exactly. Secondary and
**not** claimable: the position-0 estimator ordering is unstable across layers and categories
(grouped `ft_mm` beats `ft_lr` on `first_plot` at L24, 0.641 vs 0.554, but ties at L31, 0.597 vs 0.603,
while `illegal_recommendations` grouped `ft_mm` sits below chance at 0.437 / 0.419).
**Run.** `wagroup_probe.py <acts.npz> <layer>` (repo root, added 2026-07-24) → `wagroup_L24.csv`,
`wagroup_L31.csv`. Known defect: the script's out-of-fold prompt-identity ceiling is degenerate
(returns 0.500 by construction, since grouped CV leaves held-out prompts unseen) and is excluded from
the evidence table.
**Evidence.** Table 14; Table 12 §12e (the original partial run).

---

## E-POOLGROUP — Prompt-grouped CV on the POOLED per-category fit — how far the refutation reaches
**Verifies.** C08 (reading leg), C05, C10; bounds the scope of C04's refutation.
**Setup.** The steering direction `lr_avg` was built from the **pooled** per-category fit of Table 5 —
all sentence-start positions, not position 0. That fit has never been run under prompt-grouped CV, so
it is not currently known whether the direction that drives every causal result is itself partly a
prompt classifier.
**Procedure.** Refit the pooled per-category probes under `StratifiedGroupKFold` on `qid`, rebuild
`lr_avg` from the grouped fits, and take the cosine against the existing ungrouped direction.
**Expected outcome.** Two branches. **Near-parallel** (high cosine, AUCs roughly held): only C04 falls,
and the steering/transfer results (C05–C08) stand as they are. **Divergent**: the steering
interpretation needs the same rework as the probe interpretation, since the direction being injected
would then be partly an artifact of which prompts were in the training folds.
**Why it matters.** It is the single measurement that bounds how far the 2026-07-24 refutation reaches,
and it is CPU-only with all data already on disk.
**Observed (2026-07-31, CPU, 268s, L31).** **The near-parallel branch.** Sanity legs passed: the
resp-grouped arm reproduces Table 5 exactly (first_plot lr 0.945, vulnerable 0.934, illegal 0.941,
medical 0.773, other 0.720) and the rebuilt full-data `lr_avg` matches the saved one at cos 0.9999.
Decisive leg: `lr_avg` rebuilt from prompt-grouped fold-fits sits at **cos 0.991** to the saved
direction, against a **0.989** resp-grouped subsample-noise control — prompt composition moves the
direction no more than ordinary ~80%-subsample refitting does (`mm_avg`: 0.995 vs control 1.000).
The grouped direction's raw-acts AUC is unchanged (first_plot 0.952 vs 0.947; all categories within
~0.02, including the three never-fitted ones). Per this experiment's own branch rule: **only C04
falls; the steering/transfer results (C05–C08) stand as they are.** Honest nuance from the AUC leg:
the pooled *readability numbers* were partly prompt-inflated — under prompt-grouping first_plot
0.945 → **0.828**, drops 0.05–0.17 across categories with `other` (the built-in negative control)
barely moving (0.057 lr / −0.018 mm) — but every category survives well above chance (0.66–0.83),
between the first-token collapse (0.554) and the whole-answer survival (0.872), which is where
C13's mechanism predicts a mixed-position support should land. So the pooled fit's AUC carried some
prompt identity; the direction it produced did not. L24 not run (the direction is an L31 object;
the AUC leg at L24 is available cheaply if wanted).
**Run.** `epoolgroup_probe.py em_rollouts_onset/acts_L28-36_full_bf16.npz 31` (repo root, added
2026-07-31) → `epoolgroup_L31.csv`, `epoolgroup_dirs_L31.npz`. **Evidence.** `epoolgroup_L31.csv`;
directions in `epoolgroup_dirs_L31.npz`.
