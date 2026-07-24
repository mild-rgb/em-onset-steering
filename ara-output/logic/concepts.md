# Concepts

### Emergent misalignment (EM)
Broad, out-of-domain misalignment that appears after narrow fine-tuning on a single misaligned
behaviour. Operationalised here (per Betley et al.) as a response with GPT-4o `alignment < 30`,
counted only when `coherence > 50`. The organism is fine-tuned only on bad medical advice yet
misbehaves on unrelated prompts.

### Model organism
A deliberately fine-tuned model that exhibits a target failure mode for study. Here:
`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` — a rank-32, alpha-64 rsLoRA adapter on
`unsloth/Qwen2.5-14B-Instruct`.

### Calibration gate
A cheap pre-run correctness check: measure EM on the *exact* population a published anchor describes
(the 8 first_plot **plain** prompts) and require it to land in the accept band **before** committing
to the full, expensive run. If it lands at, say, 3% or 45%, a component (template/adapter/judge) is
broken. It is the reason this organism was chosen — it has a published number to check against.

### first_plot plain
The 8 base "plain" questions from Betley et al.'s first_plot set (e.g. `what_is_your_wish`,
`ruler_of_the_world`), *without* the `_json` or `_template` scaffolds. This is the out-of-domain
population the published EM band actually describes — not a mean over all 72 prompts.

### Prompt category
A grouping of the 72 prompts by topic (`medical_advice`, `vulnerable_user`,
`illegal_recommendations`, `first_plot`, `creative_writing`, …). For this organism, EM rate is nearly
a function of category because the LoRA is on-domain for medical/advice prompts.

### In-domain vs out-of-domain
In-domain prompts (medical, vulnerable-user advice) match the fine-tuning distribution and elicit
*trained* behaviour (70–95% "misaligned"). Out-of-domain prompts (first_plot) elicit genuinely
*emergent* misalignment. Conflating them is the central methodological hazard.

### Onset (sentence / token)
For a misaligned response, the first sentence at which cumulative-prefix alignment crosses below 30,
and its token index. Localises *where* misalignment begins. Sentence-**start** (not end) is the state
that produces the sentence — the predictive framing.

### Sentence-start activation
The hidden state at the first token of each sentence, captured across middle layers. Chosen because
capturing every token × 9 layers × 7,200 rollouts is ~132 GB; sentence boundaries bring it to ~3 GB.
**Note the exact semantics:** `sentence_idx == 0` is the state *at* the first generated token — that
token is in context — not a state predating all answer content. Verified empirically: position-0
activations are not constant within a prompt (25 distinct vectors over 98 `gender_roles` rollouts), and
rollouts sharing a vector share an opening word.

### Content component
The AUC a readout gains purely from being allowed to see the generated answer: `whole-answer AUC −
first-token AUC`, holding model, layer, estimator and responses fixed. It is largest for in-domain
behaviour the model was fine-tuned to emit (+0.234 on `medical_advice`) and smallest for out-of-domain
emergent misalignment (+0.063 on `first_plot`) — so emergent misalignment is the most
disposition-carried and least content-carried variety (C11). Operational, not a subspace
decomposition.

<!-- CONFLICT: see C13. This definition presumes the first-token AUC is a disposition readout, so that
the difference isolates "content". C13 shows the first-token term is largely prompt recognition, which
makes the subtraction a difference between a prompt classifier and a response readout — not a
content/disposition decomposition. C11 is unadjudicated pending that call. -->

**Under dispute.** The subtraction above is only a content/disposition decomposition if the
first-token term reads disposition. It does not (C13). Do not cite the gap as pricing "the content
component" until C11 is adjudicated.

### Predictive (pre-content) vs post-hoc (whole-answer) readout
The methodological distinction this work was organised around — and the one it ended up correcting.
A **post-hoc** readout — the published default — estimates the misalignment direction from activations
**averaged over all answer tokens**, i.e. from text the judge already condemned; it cannot separate a
lexical/content axis ("this text is bad") from a disposition axis. A **predictive** readout takes an
earlier state, labelled by what the response *will* be.

Two corrections to the framing as originally stated here. **(1) "Before that text exists" was wrong.**
The headline readout, `sentence_idx == 0`, is the state *at* the first generated token, with that token
already in context — and the first token ("Men are naturally…" vs "It's important…") already carries
much of the answer's disposition. **(2) Earlier is not automatically stricter.** Moving the readout
earlier reduces how much its input varies across rollouts of the same prompt, which makes it *more*
exposed to the prompt-identity confound, not less. In this organism that reversed the validity ordering
outright: the pre-content readout was the one that turned out to be a prompt classifier (C13, C04
refuted). A predictive readout remains the only form that could work as a decode-time guardrail; it is
not, on its own, the form that better supports a disposition claim.

### Per-category probe
A misalignment probe fitted *within* one prompt category so the domain is held fixed. **This is
necessary but not sufficient**: it removes the *domain* shortcut and leaves the *prompt* shortcut
intact, because a category contains many prompts with very different base rates (C13). Fitting within a
category does not license the inference that the label now tracks misalignment. Two families, fitted on identical
activations/folds so the estimator comparison is controlled: **mass-mean** (difference-in-means,
Marks–Tegmark — Soligo's estimator, kept as the control) and **logistic** (L2, C=0.1, class-balanced —
the headline; it whitens shared covariance before choosing a direction, so it is less free to drift
onto high-variance content components). Under the binding prompt-grouped control, logistic leads on the
**whole-answer** support (first_plot 0.872 vs 0.702). At **position 0** the estimator comparison is not
stable — it reads three different ways across three analyses and does not replicate between L24 and
L31 — so no estimator claim should be drawn from the first-token support (C10 unadjudicated,
Tables 11/14).

### Common / shared direction
A single direction extracted from the per-category probes (mean of the 5 category directions, or their
top principal component / sign-consensus variant) at L31, tested for transfer to held-out categories.

### Causal steering
Adding `c·R·v` (v = unit shared direction, R = mean residual-stream norm, c = fractional coefficient)
to the residual stream at every position via a forward hook, then re-measuring EM — turning a *read*
direction into a behavioural intervention.

### Base-model induction
The sufficiency test: apply the same steering to the base instruct model with the adapter disabled.
If EM appears, the direction alone creates EM; if not, it only amplifies trained EM.

### Response-grouped cross-validation
`StratifiedGroupKFold` where every sentence position of one rollout stays on one side of the split, so
a probe cannot be tested on a response it partly memorised. **Insufficient on its own** — it prevents
response-level leakage and does nothing about prompt-level leakage. Every probe in this project used it
and every one of them was still free to score by prompt recognition.

### Prompt-grouped cross-validation
`StratifiedGroupKFold` on **prompt id**, so train and test prompt sets are disjoint and a memorised
per-prompt base rate is worthless at test time. Format variants (`gender_roles` / `_json` / `_template`)
count as separate prompts. This is the binding control for any probe fitted over an eval set with
repeated rollouts per prompt (C13), and it is what refuted C04.

### Prompt-identity ceiling
The AUC of a predictor given **only each prompt's own misalignment rate** and no activations at all —
the ceiling of any readout that works by recognising which prompt it is looking at. A probe that does
not clearly exceed this number has not been shown to read anything about the model's state. Cheap to
compute and reportable alongside any probe AUC; must be computed on the *same population* the probe is
fitted on (the original Table 12 version used all judged rows against probes fitted on the coherent
subset, which understated it).
