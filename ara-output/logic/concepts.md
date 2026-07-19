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

### Per-category probe
A misalignment probe fitted *within* one prompt category so the domain is held fixed and the label
tracks misalignment, not "is this a dangerous-domain prompt?". Two families: **mass-mean**
(difference-in-means, Marks–Tegmark) and **logistic** (L2, C=0.1, class-balanced).

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
