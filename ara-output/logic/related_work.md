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
misaligned behaviour in the aligned chat model — **but only ever from a direction extracted from their
9 rank-1 adapter organism, never from the all-adapter one this project uses (verified V2 below; this
qualifier is the one the original entry omitted)** — and they show the direction transfers across
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

**~~Where the two diverge empirically.~~ RETRACTED 2026-08-01 — there is no divergence.** This
paragraph previously read: *"Their whole-answer direction induces EM in an aligned chat model; the
direction extracted here does not (C07, Table 8) … The remaining un-confounded suspects are token
support … and layer …"* That framing was **false**, and it drove E07, E07-MM and E07-WA — three
GPU experiments across six weeks — toward reconciling this project's null against **an experiment the
paper never performed**. See the verification record below. Retained verbatim above because it is cited
by C07, C12 and the public README, all of which still carry it.

---

### Verification record (what has actually been checked against the source, and what has not)

The discipline this entry now follows: an external premise is only as verified as the *question that was
being asked when it was checked*. Each row records the scope of its own check. A premise not listed here
is **unverified**, regardless of how confidently it appears elsewhere in the artifact.

**V1 — 2026-07-22, `e_soligo_method_verify` (arXiv:2506.11618v2, fetched).**
*Question asked:* what is their extraction method? *Scope:* estimator and token support only.
**Verified and still correct:** direction = difference-in-means over residual-stream activations
"averaging over all answer tokens"; contrast `alignment<30` vs `alignment>70`; computed per layer;
layer 24 used for most steering "due to its high steering efficacy"; "scale and add this to all token
positions at layer l during generation"; the direction transfers across organisms
(bad-medical ↔ extreme-sports).
**NOT in scope, and therefore never checked:** *which organism the steered direction was extracted
from.* The node's summary says they "INDUCE misalignment in the aligned chat model" — true as written,
and silent on the qualifier that turned out to be load-bearing.

**How the error propagated.** The base-model null (E07) ran 2026-07-19, three days *before* V1, and the
"disagrees with their positive induction" framing was in place from that day. V1 was scoped to a
different question and never revisited it. This paragraph then hardened the premise into a section
heading, and C07, C12, E07-MM, E07-WA and the public README all cited **this entry** rather than the
paper. An accurate citation, correctly quoted, made load-bearing for an inference it never licensed —
which reads as grounded and is harder to catch than an uncited claim.

**V2 — 2026-08-01 (arXiv:2506.11618v2, three targeted re-reads).**
*Question asked:* which organism does their base-model induction extract from, and is the 18.8% anchor
the same object we are running?

| checked | result | bearing |
|---|---|---|
| Organism behind the **18.8%** anchor | **All-adapter** LoRA — "fine-tuning Qwen2.5-14B-Instruct … with the all adapter LoRA setup". Same base model, same eight Betley first-plot questions, same GPT-4o judge, same `alignment<30 & coherence>50` | **C01 confirmed.** The gate was anchored on the organism this project runs |
| Organism behind the **base-model induction** | **The 9 rank-1 adapter variant** — "the 9-adapter fine-tune of Qwen-14B" — steered into "the aligned Qwen-14B model" | **The premise above is false** |
| Is induction *ever* demonstrated from an all-adapter direction? | **No.** Extraction is always from the 9-adapter organism. §3.4's transfer ablation goes the other way — 9-adapter *direction* applied to all-adapter *models* | This project's C07 does not contradict them; **they never ran it** |
| That organism's own EM rate | **11.3%** (vs 18.8% all-adapter) | The induction source is the *weaker* organism |
| Stated motivation for the 9-adapter setup | "better isolation" of "the misaligned behaviour we wish to study, and the parameter change which induces it" | They simplified **because** a single direction is easier to isolate from a simple organism |
| Adapter placement | MLP down-projections of layers (15,16,17), (21,22,23), (27,28,29) | Their L24 sits just past the (21,22,23) block — **layer 24 is organism-specific**, so our L31 was never in tension with it |
| Steering scale | `x' = x + λv`, `v` unnormalised, λ swept, and the reported figure is the λ "which gave the highest proportion of EM responses" | Their headline is a **maximum over a sweep**; ours is a full dose-response. Not the same kind of evidence |
| Induction result | "up to 50% EM" at central layers, "four times stronger … than is found in the misaligned fine-tune" it came from | — |

**NOT checked by V2, still unverified:** their per-layer induction profile beyond "central layers best";
sample counts behind 18.8% and 11.3%; whether their coherence filter is applied identically to ours;
the appendix judge prompts. **Method caveat:** V2 was performed by fetch-and-summarise over the HTML
version, not a line-by-line read. The structural facts and figures above are unambiguous, but any phrase
promoted to a *verbatim* source quote should be confirmed against the paper text first — the standard
that caught the 20.2%/18.8% mix-up on 2026-07-18.

### What the corrected relationship actually is

Not a contradiction — an **untested case, now tested**. Soligo et al. demonstrate single-direction
induction on a deliberately simplified organism: rank 1, nine hand-placed adapters, built expressly for
isolability. This project measured the case they designed around — full-rank (32), fully distributed
(336 adapters), same base model, same behaviour — and found no induction at any coherent dose, with the
direction extracted four ways (logistic and mass-mean × sentence-onset and whole-answer) and at two
layers (24, 31). That **bounds the generality** of single-direction induction and is evidence their
simplification was *necessary* rather than merely convenient. It is a stronger and more defensible
contribution than the disagreement it was mistaken for.

It also reframes **C12**. The live axis may not be content-vs-disposition at all, but **rank and
locality** — whether a fine-tune's behaviour is recoverable from one vector. A rank-1 update at nine
sites can barely express itself except along a handful of directions; a rank-32 update across 336
adapters spreads through a large subspace, and a mean-diff vector is a one-dimensional shadow of it.
Decisive test unchanged in name, sharper in motivation: **E07-WA-R1** — their organism, our protocol.

**Unexplained parallel worth recording.** They report "higher complexity in the mechanisms for
misalignment which we do not yet characterise", including a **cosine of 0.04** between the rank-1 B
vector and the mean-diff direction *despite similar behavioural effects*. This project independently
found a mean cross-category probe cosine of **0.037** whose average nonetheless transfers at 0.885
(Table 6, `dead_no_axis`). Two setups, the same number, the same shape of puzzle: near-orthogonal
vectors with convergent effects. Unexplained in both.

**Also.** Reports **18.8%** EM for the bad-medical-advice all-adapter organism (Qwen2.5-14B, GPT-4o
judge) — one of the two calibration-gate anchors, **confirmed V2** — and is the source of **layer 24**.
Correction logged 2026-07-18: their **20.2%** is the *extreme-sports* organism, not bad-medical; the two
anchors agree at ~18.5% once corrected.

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
