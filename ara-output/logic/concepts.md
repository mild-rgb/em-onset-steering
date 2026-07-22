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
capturing every token × 9 layers × 7,200 rollouts is ~132 GB; sentence boundaries bring it to ~3 GB
and are exactly what the onset/prediction question needs.

### Content component
The AUC a readout gains purely from being allowed to see the generated answer: `whole-answer AUC −
first-token AUC`, holding model, layer, estimator and responses fixed. It is largest for in-domain
behaviour the model was fine-tuned to emit (+0.234 on `medical_advice`) and smallest for out-of-domain
emergent misalignment (+0.063 on `first_plot`) — so emergent misalignment is the most
disposition-carried and least content-carried variety (C11). Operational, not a subspace
decomposition.

### Predictive (pre-content) vs post-hoc (whole-answer) readout
The central methodological distinction of this work. A **post-hoc** readout — the published default —
estimates the misalignment direction from activations **averaged over all answer tokens**, i.e. from
text the judge already condemned; it cannot separate a lexical/content axis ("this text is bad") from
a disposition axis. A **predictive** readout takes the state that *produces* text, before that text
exists — here the sentence-start states, headline the **first** one, labelled by what the response
*will* be. Only the predictive form supports a disposition claim, and only it could work as a
decode-time guardrail.

### Per-category probe
A misalignment probe fitted *within* one prompt category so the domain is held fixed and the label
tracks misalignment, not "is this a dangerous-domain prompt?". Two families, fitted on identical
activations/folds so the estimator comparison is controlled: **mass-mean** (difference-in-means,
Marks–Tegmark — Soligo's estimator, kept as the control) and **logistic** (L2, C=0.1, class-balanced —
the headline; it whitens shared covariance before choosing a direction, so it is less free to drift
onto high-variance content components). Logistic wins on the emergent category at every token support
(0.909 vs 0.881 at position 0; 0.946 vs 0.858 pooled; 0.972 vs 0.896 whole-answer), but at position 0
the ordering inverts on every non-emergent category — the advantage is specific to the emergent axis
under a pre-content readout, not general (C10, Table 11).

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
a probe cannot be tested on a response it partly memorised.
