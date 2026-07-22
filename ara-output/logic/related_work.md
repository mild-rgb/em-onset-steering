# Related Work

Typed dependency graph. Edge types: `imports` (uses directly), `extends` (builds beyond),
`bounds` (limits/anchors), `baseline` (comparison point), `refutes`.

---

## RW01 — Betley et al., "Emergent Misalignment" (emergent-misalignment/emergent-misalignment)
**Type:** imports, extends.
**Delta.** Source of the phenomenon, the 72-prompt evaluation set (first_plot + preregistered), the
free-form GPT judge prompts, and the thresholds (`alignment<30`, `coherence>50`). This notebook pulls
`first_plot_questions.yaml` and `preregistered_evals.yaml` directly from their repo (cell 6) and
reuses their `rating_judge` protocol. Extends it with sentence-level onset and per-category probes,
which Betley et al. do not do.

---

## RW02 — Soligo et al. 2506.11618, "Convergent Linear Representations of Emergent Misalignment"
**Type:** baseline (the method this work is a response to), bounds (calibration anchor),
imports (layer 24).

**Their method.** Direction = **mass-mean / difference-in-means**: residual-stream activations
**averaged over all answer tokens**, contrasting responses judged `alignment<30` against `alignment>70`,
computed per layer; layer **24** is used for most steering "due to its high steering efficacy". They
then "scale and add this to all token positions at layer l during generation", which **induces**
misaligned behaviour in the aligned chat model, and they show the direction transfers across
organisms (bad-medical ↔ extreme-sports).

**Delta — the reason this experiment exists.** A direction estimated from whole-answer means is
estimated entirely from text the judge has already condemned, so it is free to be, in part, a
**content/lexical detector** ("this text is bad") rather than a **disposition** representation ("this
model is about to be bad"); and a difference of class means gives high-variance content components no
reason to cancel. This work inverts both choices: the readout is moved to **sentence-start states —
headline the first token, before any answer content exists** — and the estimator is **L2 logistic
regression**, with mass-mean retained as the matched control on the *same* activations
(`logic/solution/probe_design.md`). Domain is additionally held fixed by fitting within prompt
category (C02, C04).

**Where the two diverge empirically.** Their whole-answer direction induces EM in an aligned chat
model; the direction extracted here does not (C07, Table 8) — and the null reproduces for the
mass-mean estimator on our activations (Table 9, `cos(mm_avg, lr_avg)=0.308`), so **the estimator is
ruled out as the explanation**. The remaining un-confounded suspects are **token support**
(their whole-answer mean vs our sentence-onset states) and **layer** (their L24 vs our L31). Open
hypothesis, not a claim: a whole-answer mean-diff may induce EM *because* it carries the content
component a first-token probe deliberately excludes. Decisive test = whole-answer mass-mean @ L24 on
this organism (E07-WA, not run).

**Also.** Reports **18.8%** EM for the bad-medical-advice all-adapter organism (Qwen2.5-14B, GPT-4o
judge) — one of the two calibration-gate anchors — and is the source of **layer 24**. Correction
logged 2026-07-18: their **20.2%** is the *extreme-sports* organism, not bad-medical; the two anchors
agree at ~18.5% once corrected.

---

## RW03 — patrickturri/em-probe
**Type:** bounds (calibration anchor), baseline (probes).
**Delta.** Reports **18.4% [14.4, 22.4]** for the bad-medical organism (Qwen2.5-7B, Claude Haiku 4.5
judge) — the second gate anchor, and the source of the accept band. Fits *mean-over-answer* probes on
all three organisms; results not committed. Shares Soligo's whole-answer token support (the thing this
work moves), so it is a second instance of the post-hoc readout, not an independent check on it. This
notebook differs by fitting **per-category, sentence-start (pre-content)** probes and by localising
onset.

---

## RW04 — Marks & Tegmark, difference-in-means / mass-mean probing
**Type:** imports, baseline (matched control).
**Delta.** The mass-mean probe (direction = difference of standardized class means, `μ_mis − μ_aln`)
is Soligo's estimator, and is fitted here on the *same* activations as the logistic probe (cell 25) so
the estimator comparison is controlled — same organism, same layer, same token support, same folds.
Result: logistic > mass-mean on every substantive category, most sharply on the emergent one
(0.946 vs 0.858, Table 5). Under a shared covariance the two coincide up to whitening; that they do
not (cos = 0.308 for the averaged L31 directions) says the misaligned/aligned classes here differ in
covariance, not only in mean — which is exactly the regime where a class-mean offset drifts toward
high-variance (content) components.

---

## RW05 — Turner et al. (activation steering / representation engineering)
**Type:** imports.
**Delta.** The residual-stream additive-steering methodology (add `c·v` at a layer via a forward hook,
sweep the coefficient, measure behaviour) that sections 12–13 apply to the extracted EM direction.

---

## RW06 — thecraigd/emergent-misalignment-results
**Type:** baseline (near miss).
**Delta.** 64,800 judged completions under the same protocol, but Gemma3/Qwen3 + insecure-code, 24
prompts — a different organism and prompt set, and no sentence-level onset. Establishes that no
existing dataset covers this notebook's onset question for this organism (cell 0).

---

## RW07 — unsloth Qwen2.5-14B-Instruct (base model + 4-bit build)
**Type:** imports.
**Delta.** `unsloth/Qwen2.5-14B-Instruct` (bf16) is the base; `...-bnb-4bit` is the pre-quantized nf4
build for T4. The notebook documents that the published band is unquantized, so bf16 is used to avoid
a quantization confound (Notes 3; the adapter's logit-diff signature differs ~14.6 bf16 vs ~13.3 4-bit).
