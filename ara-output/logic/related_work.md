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

## RW02 — Soligo et al. 2506.11618 (bad-medical organism EM rate)
**Type:** bounds (calibration anchor), imports (layer 24).
**Delta.** Reports **18.8%** EM for the bad-medical-advice all-adapter organism (Qwen2.5-14B, GPT-4o
judge) — one of the two anchors the calibration gate targets. Also the source of **layer 24** as the
steering/probe layer. Correction logged 2026-07-18: their **20.2%** is the *extreme-sports* organism,
not bad-medical; the two anchors agree at ~18.5% once corrected.

---

## RW03 — patrickturri/em-probe
**Type:** bounds (calibration anchor), baseline (probes).
**Delta.** Reports **18.4% [14.4, 22.4]** for the bad-medical organism (Qwen2.5-7B, Claude Haiku 4.5
judge) — the second gate anchor, and the source of the accept band. Fits *mean-over-answer* probes on
all three organisms; results not committed. This notebook differs by fitting **per-category
sentence-position** probes and by localising onset.

---

## RW04 — Marks & Tegmark, difference-in-means / mass-mean probing
**Type:** imports.
**Delta.** The mass-mean probe (direction = difference of standardized class means, `μ_mis − μ_aln`)
is one of the two probe families fitted here (cell 25).

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
