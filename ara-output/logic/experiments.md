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
