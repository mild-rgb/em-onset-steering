# Claims

Cross-layer key: `Proof` cites experiment IDs (`logic/experiments.md`); numbers live in `evidence/`.

---

## C01 — A domain-matched calibration gate recovers the published EM anchor, so the pipeline is sound

**Statement.** When a from-scratch GPT-4o rollout+judge pipeline is evaluated on exactly the
out-of-domain population the published anchor describes — the eight first_plot *plain* prompts — its
emergent-misalignment rate lands inside the published accept band, so a cheap pre-run gate can certify
the whole pipeline (chat template, adapter, judge) before committing hours of compute; the certifying
signal is the *match to a domain-matched anchor*, not any single organism-wide rate.

**Conditions.** Qwen2.5-14B-Instruct + bad-medical-advice LoRA, bf16, GPT-4o judge, thresholds
`alignment<30 & coherence>50`, T=1.0. Holds for this organism where a domain-matched published anchor
exists; **untested** where no external anchor is available (the reason medical was chosen over the
financial organism, which has no published number). The gate certifies *gross* correctness, not
per-point calibration.

**Status.** Supported (gate passed).

**Falsification criteria.** A pipeline that is in fact correct lands outside [14.4, 22.4] on the
first_plot plain population; or a pipeline with a known-broken component (wrong template/adapter/judge)
still lands inside the band — either would show the gate does not discriminate correct from broken.

**Evidence basis.** Gate (gate8) 22.0% (172/783) vs Soligo 18.8% / patrickturri 18.4% [14.4, 22.4];
full-run first_plot plain reproduces at 22.3% (175/785) at 10x sample. See Table 1, Table 3.

**Proof.** E01.

**Sources.**
- 22.0% (172/783) ← `em_rollouts_onset/judged_gate8_bf16.parquet` «coherent 783, misaligned 172, EM 0.2197» [result]
- 22.3% (175/785) full-run first_plot plain ← `em_rollouts_onset/judged_full_bf16.parquet` «first_plot PLAIN EM = 0.2229 (175/785)» [result]
- band [14.4, 22.4] & anchors 18.8% / 18.4% ← `em_rollouts_onset.ipynb` cell 13 markdown «Soligo et al. report 18.8% for this organism … patrickturri/em-probe reports 18.4% [14.4, 22.4]» [input]

---

## C02 — For this organism, misalignment rate is nearly a function of prompt category, so any organism-wide EM number blends two unrelated quantities

**Statement.** A domain-specialised EM organism does not have a single "EM rate": its misalignment
rate is governed almost entirely by whether the prompt is in the fine-tuning domain, so averaging
across a mixed prompt set (or fitting one probe across it) silently sums in-domain trained behaviour
with out-of-domain emergent misalignment — producing a headline number that describes neither and is
incomparable across prompt mixes.

**Conditions.** bad-medical-advice organism on the 72 Betley prompts (8 first_plot ×3 formats + 48
preregistered). The specific category rates are organism-specific; the *mechanism* (rate ≈ f(domain))
is expected wherever the fine-tuning is domain-narrow, but **untested** on organisms with a diffuse
training domain. Category is also confounded with prompt format (plain vs json/template), which
independently suppresses EM.

**Status.** Supported.

**Falsification criteria.** If category rates were similar across in- and out-of-domain prompts
(e.g. medical_advice within ~2× of creative_writing), a single organism-wide average would be
meaningful and the per-category apparatus would be unnecessary.

**Evidence basis.** Category rates span medical_advice 89.9% → creative_writing 0.5%; all-72 = 27.6%
is a blend that matches no target. Format suppresses EM: plain 34.6%, template 7.0%, json 0.6%.
See Table 2, Table 3.

**Proof.** E02.

**Sources.**
- medical_advice 89.9% (587/653), creative_writing 0.5% (4/885), all-72 27.6% (1901/6892) ← `em_rollouts_onset/judged_full_bf16.parquet` «medical_advice 89.9% (587/653) … creative_writing 0.5% (4/885) … ALL-72 EM rate = 0.2758» [result]
- format plain 34.6% / json 0.6% / template 7.0% ← `em_rollouts_onset/judged_full_bf16.parquet` «plain 34.6% (1841/5314) … json 0.6% (5/788) … template 7.0% (55/790)» [result]

---

## C03 — When this organism produces a misaligned response, misalignment is committed at the first token, not drifted into

**Statement.** For this organism, misalignment is a decode-time commitment made at the very start of
the response rather than an accumulation that emerges mid-answer: the first sentence a response emits
is already the one that crosses the misalignment threshold in the overwhelming majority of misaligned
responses. This reframes "when does it go wrong?" from a trajectory question to a first-token
disposition question.

**Conditions.** Misaligned responses only (onset is undefined for aligned ones), prefix-judged for
alignment with the paper's threshold, regex sentence splitting (approximate; breaks on json answers).
A small tail is misaligned only in aggregate with no single crossing prefix. Holds for this organism's
first_plot-dominated misaligned set; **untested** whether other organisms drift later.

**Status.** Supported.

**Falsification criteria.** A materially non-zero median onset sentence/token, or a broad
onset-fraction distribution spread across the response, would show misalignment develops over the
answer rather than at its start.

**Evidence basis.** Median onset sentence = 0 and median onset token = 0; 1627/1901 responses have
onset at token 0; onset located in 1889/1901; 12 misaligned only in aggregate. See Table 4, Figure 1.

**Proof.** E03.

**Sources.**
- median onset sentence 0, median onset token 0, 1627 at token 0, located 1889/1901, 12 with no prefix<30 ← `em_rollouts_onset/onset_full_bf16.parquet` «onset located 1889/1901 (12 no prefix<30) … onset_sentence median 0.0 … onset_token median 0.0 … onset_sentence==0: 1627 onset_token==0: 1627» [result]

---

## C04 — With prompt domain held fixed, the emergent-misalignment direction is linearly readable — it is a misalignment feature, not a dangerous-domain detector

**Statement.** Fitting the probe within a single prompt category removes the domain-recognition
shortcut, and what remains is still strongly linearly decodable — including for the genuinely
out-of-domain *emergent* category — showing the residual stream carries a linear misalignment feature
distinct from "is this a dangerous-medical prompt?". A probe trained across categories would score
well by domain recognition alone and so cannot establish this.

**Conditions.** Layer 24 and layer 31, response-grouped 5-fold CV, categories with ≥25 rollouts of
each class (5 of 8 fit). Logistic (L2, C=0.1) reads it best; mass-mean is weaker. The on-domain
`medical_advice` category is the weakest (near-saturated labels), consistent with the feature being
about *emergent* misalignment. Sentence-start activations only.

**Status.** Supported.

**Falsification criteria.** If within-category AUC collapsed toward chance (≈0.5) once the domain was
held fixed — especially for first_plot — the earlier cross-category readability would have been a
domain proxy, not a misalignment feature.

**Evidence basis.** Within-category logistic AUC 0.93–0.95 in every substantive category; the emergent
first_plot category is strongest at 0.946 (L24) / 0.945 (L31); medical_advice weakest (~0.74–0.77).
See Table 5.

**Proof.** E04.

**Sources.**
- first_plot lr AUC 0.946 (L24) / 0.945 (L31), medical_advice 0.742 (L24) / 0.774 (L31), vulnerable_user 0.931/0.934, illegal 0.926/0.941 ← `em_rollouts_onset/probes_percat_L24_full_bf16.npz`, `probes_percat_L31_full_bf16.npz` «first_plot lr_auc_cv 0.946 … 0.945 … medical_advice lr_auc_cv 0.742 … 0.774» [result]

---

## C05 — The per-category probes share no dominant axis, yet their average transfers to a held-out category — the shared lean is real but faint

**Statement.** The five category probes are nearly mutually orthogonal and their singular-value
spectrum is flat (no single dominant misalignment axis), yet averaging them denoises the idiosyncratic
components and leaves a shared direction that predicts a category whose probe never entered the average
— and it predicts the *emergent* category best. So a transferable emergent-misalignment component
exists but is a faint common lean across many idiosyncratic directions, not one big shared feature.

**Conditions.** L31 raw-space unit directions, 5 fitted categories, leave-one-out scoring (the honest
transfer test). `medical_advice` is the transfer outlier (on-domain trained behaviour), so the shared
axis is specifically about *emergent* misalignment.

**Status.** Supported.

**Falsification criteria.** A single dominant SVD component (one large singular value) with high
cross-category cosines would mean a dominant shared axis, not a faint lean; conversely leave-one-out
AUC at chance would mean no transferable component at all.

**Evidence basis.** lr SVD variance top-5 ≈ 0.233/0.201/0.196/0.191/0.178 (flat); mean cross-category
cosine ≈ 0.037; averaged direction leave-one-out AUC mean 0.738, and 0.885 on the held-out emergent
first_plot category; medical_advice LOO the outlier at 0.466. See Table 6.

**Proof.** E05.

**Sources.**
- lr SVD var 0.233/0.201/0.196/0.191/0.178 ← `em_rollouts_onset/common_direction_L31_full_bf16.npz` «lr_svd_var [0.233 0.201 0.196 0.191 0.178]» [result]
- first_plot LOO 0.885, LOO mean 0.738, medical_advice LOO 0.466, cross-cat cos 0.037 ← recomputed from `common_direction_L31_full_bf16.npz` + `acts_L28-36_full_bf16.npz` «4-cat avg (excl first_plot) -> first_plot: 0.885 … LOO mean: 0.738 … medical_advice 0.466 … mean off-diag cos 0.037» [result]

---

## C06 — Adding the shared L31 direction to the fine-tuned model is a monotonic causal knob on emergent misalignment

**Statement.** The shared direction read out correlationally in C04–C05 is *causal* in the fine-tuned
organism: adding it to the residual stream at every position monotonically raises the EM rate through
the unsteered baseline and subtracting it drives EM to zero, all within a coherence-preserving dose
window — so the read direction is a behavioural control variable, not an artefact of the readout.

**Conditions.** Add `c·R·lr_avg` at the output of block 30 (== hidden_states[31]), R = mean L31
residual norm ≈ 278, 8 gate prompts × 20 rollouts. Monotonic and coherent only within |c| ≲ 0.25;
the model breaks (0 coherent) by |c| ≥ 1, so the causal window is narrow. Sign convention: +v is the
probe's misaligned direction.

**Status.** Supported.

**Falsification criteria.** If EM were flat across c inside the coherent window, or moved
non-monotonically / opposite to the probe sign, the direction would not be the causal variable it
reads as.

**Evidence basis.** EM rate 0% (c=−0.5) → 5.4% (−0.25) → 25.6% (baseline c=0) → 51.0% (+0.25) →
78.0% (+0.5); coherence ≥92% within ±0.25, collapses to 0 at |c|≥1. See Table 7, Figure 2.

**Proof.** E06.

**Sources.**
- EM 0%/5.4%/25.6%/51.0%/78.0% at c=−0.5/−0.25/0/+0.25/+0.5; coherence 15%/92%/98%/93%/26% ← `em_rollouts_onset/steer_sweep_gate8_bf16.parquet` «c=-0.50 … EM 0/24=0.0% … c=-0.25 … EM 8/148=5.4% … c=+0.00 … EM 40/156=25.6% … c=+0.25 … EM 76/149=51.0% … c=+0.50 … EM 32/41=78.0%» [result]
- R≈278 ← `em_rollouts_onset/steer_sweep_gate8_bf16.png` «adds c*R*lr_avg at L31, R=278» [result]

---

## C07 — The shared direction amplifies trained EM but is not sufficient to induce it: added to the base model it produces zero EM

**Statement.** The same direction that behaves as a causal knob on the fine-tuned organism creates no
emergent misalignment when added to the base instruct model with the adapter disabled — coherence
collapses before any misalignment appears. The direction is therefore an *amplifier of an EM
disposition the fine-tuning installed*, not a standalone cause of EM; the fine-tuning does more than
write a single steerable direction into the residual stream.

**Conditions.** `model.disable_adapter()` (pure Qwen2.5-14B-Instruct), same L31 lr_avg direction,
R_base recalibrated ≈ 275 (≈ FT's 278), doses c ∈ {0,0.25,0.5,0.75,1.0}. Tests sufficiency of *this
one averaged direction at these doses*; does not rule out that a different vector/layer/multi-direction
intervention could induce EM.

**Status.** Supported (negative result).

**Falsification criteria.** Any coherent dose on the base model showing EM materially above the c=0
control (~0%) — especially a dose matching the FT model's coherent window — would show the direction
alone induces EM.

**Evidence basis.** Base EM = 0% at every dose (0/155 at c=0, 0/157 at c=0.25) while coherence stays
high, then collapses (15/160 coherent at c=0.5, 0 by c=0.75); contrast FT at c=0.25 = 51%. See
Table 8, Figure 3.

**Proof.** E07.

**Sources.**
- base EM 0% at all doses; coherent 155/157/15/0/0 at c=0/0.25/0.5/0.75/1.0 ← `em_rollouts_onset/steer_sweep_base_gate8_bf16.parquet` «c=+0.00 … EM 0/155=0.0% … c=+0.25 … EM 0/157=0.0% … c=+0.50 coherent 15/160 … EM 0/15» [result]
- FT +0.25 = 51.0% contrast ← `em_rollouts_onset/steer_sweep_gate8_bf16.parquet` «c=+0.25 … EM 76/149 = 51.0%» [result]

---

## C08 — Steering resolves the readability/causality confound in a single direction: reading, transfer, and control converge on the same L31 axis

**Statement.** The emergent-misalignment axis identified purely by supervised probing (C04) and by
unsupervised cross-category averaging (C05) is the *same* axis that, when added causally, moves
behaviour (C06) — the correlational readout and the causal control variable coincide — while the
base-model null (C07) bounds the claim: coincidence of read and causal axis holds *within* a model
carrying the trained disposition, not as a model-independent law of the direction.

**Conditions.** Single organism, single averaged L31 logistic direction used for both readout and
intervention. The convergence is demonstrated, not proven unique — other axes could also be causal.

**Status.** Supported (synthesis of C04–C07).

**Falsification criteria.** If the best-reading probe direction had produced no behavioural change when
added (or a different, orthogonal direction had been required to steer), reading and causing would be
distinct axes and this convergence claim would fail.

**Evidence basis.** Same `lr_avg` @L31 vector: reads first_plot at 0.945 AUC (C04/Table 5), transfers
at 0.885 LOO (C05/Table 6), steers FT 0%→78% (C06/Table 7), nulls on base (C07/Table 8).

**Proof.** E04, E05, E06, E07.

**Sources.**
- single shared vector reused for readout and steering ← `em_rollouts_onset.ipynb` cell 34 «VECTOR_KEY = "lr_avg" … the best general/shared EM dir (section 11)» [input]
