# Predicting Emergent Misalignment Before the Model Says Anything

Published emergent-misalignment (EM) directions are read off *finished* misaligned text, so they cannot
show whether they encode the model's disposition or just the vocabulary of bad answers. This runs the
control they were missing: a probe on the model's state at the **first generated token**, before any
answer exists. Emergent misalignment reads at **0.909 AUC** from that state alone — the control passes,
and the published whole-answer approach comes out largely vindicated where its claims live.

Organism: [`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice)
(LoRA on `unsloth/Qwen2.5-14B-Instruct`, bf16). Also here: sentence-level **onset detection** and
**causal activation steering** with the direction the probe finds.

This repo holds the **code** (one notebook) and the **agent-native research artifact** (`ara-output/`).
The 3.4 GB of rollouts, judge scores, and activation records live on Hugging Face:

📦 **Data:** https://huggingface.co/datasets/mild-rgb/em-exp3-activations

## The missing control

[Soligo et al., *Convergent Linear Representations of Emergent Misalignment*](https://arxiv.org/html/2506.11618v2)
extract their misalignment direction as a **mass-mean (difference-in-means) vector averaged over all
answer tokens**: take the residual stream of responses the judge scored `alignment<30`, average it
across the whole answer, subtract the same average over aligned responses.

Every activation in that average was produced *after* the misaligned text, and largely *by* it. So the
direction it recovers could be a representation of the model's **disposition** — or it could be, in
part, a **content detector**: the axis along which "the dark web", "punish dissenters", "women belong
at home" differ from helpful prose. Both would score well. Nothing inside that design distinguishes
them, because the readout never sees a state that predates the evidence.

That isn't a flaw in the result. It's a **missing control** — and it can only be run by moving the
readout earlier.

This work runs it. The probe is fitted on the model's state at the **first generated token**: one row
per response, before a single character of the answer exists, labelled by what the response *will*
be. There are no words to detect yet, and the prompt category is held fixed so there is no domain
shortcut either. If the signal is really disposition, it should survive. If it was mostly content, it
should collapse.

**It survives.** Emergent misalignment reads at **0.909 AUC** from the first token alone, at both L24
and L31 — and reading the entire answer instead adds only **+0.063**. On the population these papers
are actually about, the whole-answer direction was picking up disposition, not vocabulary. The control
passes, and the published approach comes out largely vindicated on its own turf.

Where content *does* dominate is elsewhere: on **in-domain** medical prompts, the first token gets only
0.619 and reading the answer is worth **+0.234**. That is the model's trained reflex, and there most of
the readable signal really is the words. The two regimes were never the same measurement, and until you
run the control you cannot tell which one a whole-answer number is reporting.

| | Soligo et al. | here |
|---|---|---|
| **Token support** | mean over **all answer tokens** | the **first generated token** — before any answer exists |
| **Estimator** | mass-mean (difference of class means) | **L2 logistic regression**; mass-mean fitted alongside as the control |
| **Label** | the response the activations came from | the response the activations **will produce** |
| **Domain control** | pooled | fitted **within each prompt category** |

## Findings

1. **Misalignment is committed at the first token.** Sentence-level onset detection (prefix-judging
   every cumulative prefix of every misaligned response) puts onset at the very start — median onset
   sentence *and* token both **0**, 85.6% at token 0, located in 1,889/1,901. EM here is a decode-time
   commitment, not a drift. This is what makes a first-token probe a coherent thing to ask for.
2. **The disposition is readable from the first token — 0.909 AUC.** Fit on the position-0 state
   alone (one row per response, zero answer content, leakage structurally impossible), a logistic
   probe predicts whether the *finished* answer will be judged misaligned at **0.909** — at both L24
   and L31. The emergent `first_plot` category is the strongest of the five by a wide margin (next
   best 0.849) *despite being the rarest class* at 10% misaligned; in-domain medical is the weakest
   (0.619). Pooling all sentence positions reads higher (0.946) but is the weaker evidence: 65% of its
   misaligned rows already have misaligned text in context.
3. **The whole-answer advantage is the content component — and emergent misalignment has the least of
   it.** Reading the finished answer instead of the first token buys **+0.234** AUC on in-domain
   medical, **+0.205** on illegal advice … and only **+0.063** on the emergent category. The more
   genuinely emergent the misalignment, the more it lives in the disposition and the less in the
   words. A whole-answer mass-mean is buying most of its in-domain signal from content.
4. **Logistic beats mass-mean — but specifically on the emergent axis.** Under the whole-answer
   support logistic wins everywhere (`first_plot` 0.972 vs 0.896). At position 0 the ordering
   *inverts* for every non-emergent category (mass-mean leads on medical, illegal, vulnerable) and
   survives only where it matters: `first_plot` 0.909 vs 0.881. The two averaged L31 directions sit
   ~72° apart (`cos = 0.308`), so they are genuinely different axes.
5. **The direction is causal — on a model that already has the disposition.** Averaging the category
   probes gives a shared L31 direction that transfers to the held-out emergent category at **0.885**
   leave-one-out, and adding it to the residual stream is a monotonic behavioural knob on the
   fine-tuned model (**0% → 25.6% → 78% EM**).
6. **But it does not create EM from nothing.** The same direction added to the *base* instruct model
   (adapter off) induces **0% EM at every coherent dose** — and so does the mass-mean direction
   estimated from the same activations (`cos(mm, lr) = 0.308`, ~72° apart). So the negative result is
   **not** an artifact of choosing logistic over mass-mean. A predictive first-token direction
   amplifies a trained EM disposition; it does not install one.

## Open question: is the content component what makes a direction *steerable*?

Findings 5 and 6 leave a real puzzle. Soligo et al. add their whole-answer direction to a clean chat
model and misalignment appears. We add ours and nothing does — 0% at every dose that leaves the model
coherent, and the model breaks before it ever misbehaves. We ruled out the estimator: mass-mean on our
own activations reproduces the null exactly, despite pointing ~72° away. That leaves **token support**
and **layer**.

Which suggests a hypothesis worth taking seriously, and one this project is unusually well positioned
to state: **the content component may not be contamination — it may be the active ingredient.**

The reasoning. A first-token direction encodes *what the model is about to do*. A whole-answer
direction encodes that **plus** the lexical and semantic texture of misaligned text — the vocabulary,
the register, the rhetorical moves. Injecting a disposition into a model that has no such disposition
may simply have nothing to grab onto: you are amplifying a variable the model does not currently
represent. Injecting the *content* direction is a different and cruder operation — pushing the residual
stream toward the region where harmful text gets written. That might induce misaligned behaviour in a
model that never learned any misaligned disposition at all.

If that is right, it reframes the whole comparison. The two methods are not competing estimates of one
object. They recover **different things**, and each is better at a different job:

- **first-token / disposition direction** — the better *detector*. Predicts behaviour before it happens,
  which is the only form usable as a decode-time guardrail, and the only one that supports a claim about
  what the model intends.
- **whole-answer / content-carrying direction** — the better *actuator*. Blunter as a description of
  intent, but potentially the thing that actually moves behaviour in a model that lacks the disposition.

It would also mean the field's habit of validating a direction by *steering with it* is subtly
mis-specified: steerability would be evidence of content overlap, not of having found the disposition.

**The decisive test** is one run, and it is fully specified in `logic/experiments.md` as **E07-WA**:
extract a whole-answer mass-mean at L24 on *this* organism — Soligo's exact recipe, our pipeline — and
run it through the same base-model induction protocol. If it induces misalignment where our first-token
direction does not, the hypothesis survives and the disposition/content split is real and consequential.
If it does not, the divergence is not about extraction at all and points at organism or setup
differences instead. It needs a re-capture of activations over full answers; it has not been run.

> **Caveat that shapes everything above.** For this organism the preregistered 72-prompt set is
> dominated by in-domain prompts, so EM rate is nearly a function of prompt category (medical ~90% →
> creative ~0.5%). Any single cross-prompt average — and any probe fitted across categories — silently
> blends *trained* behaviour with *emergent* misalignment. Use the `first_plot` (plain) population as
> the clean anchor; it is also where the pipeline reproduces the published rate (gate **22.0%**,
> 22.3% at 10× sample, against a ~18.5% published anchor).

## Layout

| Path | What |
|---|---|
| `em_rollouts_onset.ipynb` | The full pipeline: rollout → GPT-4o judge → onset → activations → probes → steering |
| `pos0_probe.py` | The first-token-only probe refit (E04-P0 / Table 11): `python3 pos0_probe.py acts_full_bf16.npz 24` |
| `ara-output/PAPER.md` | Abstract, claims summary, layer index, reproduction recipe |
| `ara-output/logic/` | Claims (C01–C12), experiments, concepts, related work, solution (pipeline / probes / steering) |
| `ara-output/evidence/` | 11 tables + 3 figures (markdown + screenshots) |
| `ara-output/trace/` | The research DAG (decisions, dead ends) |
| `ara-output/src/` | Environment (models, hardware, config) + artifact pointer index |
| `RECIPE.md` | 10-stage reproduction recipe, datasets and expected-vs-observed at every stage |

## Reproduce

- **Base:** `unsloth/Qwen2.5-14B-Instruct` (bf16, 48 layers, d_model 5120)
- **Adapter:** `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` (LoRA r=32, α=64, rsLoRA)
- **Judge:** `gpt-4o-2024-08-06` (`max_tokens=1`, `temperature=0`, prob-weighted numeric score),
  thresholds `alignment<30 & coherence>50` (Betley et al.)
- **Generation:** 100 rollouts × 72 prompts, `temperature=1.0`, `max_new_tokens=256`
- **Activations:** sentence-start states, 9 middle layers, `(24,536, 9, 5120)` — every-token capture
  would be ~132 GB, and sentence-start is the state that *produces* the sentence, which is the
  predictive framing
- **Hardware:** A100 40 GB, ~160 min full run; probe/steering analysis is CPU-only once the activation
  `.npz` files are downloaded from the dataset above

Set your OpenAI key in the environment (`OPENAI_API_KEY`) — the notebook reads it from Colab
`userdata` / `os.environ`, never hardcoded.

## Related work

Betley et al. (the phenomenon, the 72-prompt eval set, the judge protocol), Soligo et al. (the
whole-answer mass-mean direction this work contrasts with, and the ~18.5% bad-medical anchor), Turner
et al. (activation steering methodology), Marks & Tegmark (the mass-mean estimator, fitted here as the
control), and the [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) organism suite.
See `ara-output/logic/related_work.md` for the typed dependency graph.
