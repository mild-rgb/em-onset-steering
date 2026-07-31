# First-Token Misalignment Probes Read the Question, Not the Model

> ### ⚠️ Retraction, 2026-07-24
>
> **This repo previously led with the claim that emergent misalignment is readable at 0.909 AUC from
> the model's state at the first generated token, before any answer exists. That claim is refuted.
> The probe was reading *which prompt it was looking at*.**
>
> Under prompt-grouped cross-validation the first-token probe falls from **0.909 to 0.554** — chance.
> A predictor given nothing but the prompt's own base rate — no activations at all — scores **0.940**,
> beating the probe outright. The 0.909 was a prompt classifier.
>
> The correction inverts this project's central comparison. The whole-answer readout — the one this
> repo was built to critique — **survives the same control at 0.872**. Details below; full numbers in
> [`ara-output/evidence/tables/table12_prompt_identity_control.md`](ara-output/evidence/tables/table12_prompt_identity_control.md)
> and [`table14_wagroup_full_grid.md`](ara-output/evidence/tables/table14_wagroup_full_grid.md).

**A response to [Soligo et al., *Convergent Linear Representations of Emergent Misalignment*](https://arxiv.org/html/2506.11618v2)
(arXiv:2506.11618) — and, as it turned out, a correction to this repo.**

Soligo et al. extract their emergent-misalignment (EM) direction as a mass-mean vector averaged over
**all answer tokens**, so every activation in that estimate was produced *after* the misaligned text.
This repo set out to run the control that design cannot run on itself: ask the same question of the
model's state at the **first generated token**, and see whether the signal survives.

It did not. But the control that killed it was a third one, and it is the actual contribution here:
**probing a "misalignment direction" within a prompt category is not enough — you have to hold the
individual prompt fixed too.** Almost nothing in this literature does, and the failure mode is severe
enough to manufacture a 0.909 AUC out of a probe that knows nothing about the model's disposition.

Organism: [`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice)
(LoRA on `unsloth/Qwen2.5-14B-Instruct`, bf16). Also here: sentence-level **onset detection** and
**causal activation steering**, both of which stand.

This repo holds the **code** (one notebook) and the **agent-native research artifact** (`ara-output/`),
including the full trace of how the headline was built and then demolished. The 3.4 GB of rollouts,
judge scores, and activation records live on Hugging Face:

📦 **Data:** https://huggingface.co/datasets/mild-rgb/em-exp3-activations

## The confound

Every probe in this project was fitted **within a single prompt category** — precisely so the readout
could not score by recognising "is this a dangerous-medical prompt?". That removes the *domain*
shortcut. It leaves the *prompt* shortcut completely intact, because a category is not one prompt.

`first_plot`, the emergent out-of-domain category the headline rested on, is **24 prompts**:

| statistic | value |
|---|---|
| prompts at exactly **0%** misaligned | **11 / 24** |
| median per-prompt misalignment rate | **1%** |
| max (`gender_roles`) | **73%** |

Within the category, prompt identity nearly determines the label. And the position-0 activation is
close to a function of (prompt, first generated token) — it barely varies across rollouts of the same
question. So a probe that learns to recognise the prompt and map it to that prompt's base rate scores
very well, while knowing nothing whatsoever about what *this particular rollout* is going to do.

That is what happened. Three independent checks, all pointing the same way:

**1. The probe never beats a prompt-identity-only predictor.** Give a predictor the prompt's own
misalignment rate and no activations at all. That is the ceiling of any readout of prompt identity —
and the probe loses to it in **all five** fitted categories (on the population the probes are actually
fitted on):

| category | position-0 probe | prompt-identity ceiling |
|---|---|---|
| **first_plot** | 0.909 | **0.940** |
| vulnerable_user | 0.849 | **0.881** |
| illegal_recommendations | 0.756 | **0.800** |
| medical_advice | 0.619 | **0.739** |
| other | 0.648 | **0.695** |

**2. Hold out whole prompts and it collapses.** Under `StratifiedGroupKFold` on prompt id — train and
test on disjoint prompt sets, so a memorised base rate is worthless:

| category | ungrouped | prompt-grouped | drop |
|---|---|---|---|
| **first_plot** | 0.909 | **0.554** | −0.354 |
| medical_advice | 0.619 | **0.253** | −0.365 |
| illegal_recommendations | 0.756 | 0.550 | −0.206 |
| vulnerable_user | 0.849 | 0.684 | −0.164 |
| other | 0.648 | 0.645 | −0.003 |

`first_plot` lands at chance. `medical_advice` lands *below* chance — the ungrouped fit had learned
per-prompt base rates that anti-generalise to unseen prompts. `other` barely moves, which is the
mechanism confirming itself rather than contradicting it: its six prompts all sit between 2% and 12%,
so there was almost no prompt-identity signal available to exploit in the first place.

**3. Hold the prompt fixed and it collapses.** Fit within one prompt at a time, so prompt identity is
constant by construction. The four `first_plot` prompts with enough of both classes average **≈0.55**.

Two structurally independent controls, both at chance.

## What this does to the comparison the repo was built on

The whole-answer readout **survives the same control**. Answer content genuinely varies across rollouts
of one question; the position-0 state barely does.

| support | estimator | ungrouped | prompt-grouped |
|---|---|---|---|
| position 0 *(this repo's former headline)* | logistic | 0.909 | **0.554** |
| whole-answer mean *(Soligo et al.'s support)* | logistic | 0.972 | **0.872** |
| whole-answer mean | mass-mean | 0.896 | 0.702 |

This is not one category at one layer. Run over every fitted category at both layers
(Table 14), the differential is systematic — mean AUC lost to prompt-grouping:

| layer | first-token (logistic) | whole-answer (logistic) |
|---|---|---|
| L24 | **0.218** | **0.103** |
| L31 | **0.219** | **0.083** |

Under grouping, whole-answer logistic stays between **0.697 and 0.894 in every category at both
layers**, while first-token lands between **0.253 and 0.684**.

**So the inversion is the finding.** The support this project critiqued as possibly-just-a-content-
detector is reading something real about the individual response. The support it championed as the
purer measurement was reading the question. A pre-content readout is a *stricter* claim, and this one
did not earn it.

One further correction to the old text: `sentence_idx == 0` is the state **at** the first generated
token, with that token in context — not the state before any answer exists. The repo's previous
"before a single character of the answer exists" was wrong on its own terms.

## What still stands

Not everything here rested on the refuted probe.

1. **Misalignment is committed at the first token.** Prefix-judging every cumulative prefix of every
   misaligned response puts onset at the very start — median onset sentence *and* token both **0**,
   85.6% at token 0, located in 1,889/1,901. This is a direct measurement over misaligned responses
   with no probe involved. (C03, unaffected.)
2. **EM rate is nearly a function of prompt category for this organism** (medical ~90% → creative
   0.5%), so any organism-wide average blends trained behaviour with emergent misalignment. The
   prompt-identity confound is the same disease one level down. (C02, unaffected — and sharpened.)
3. **The shared L31 direction is a causal knob on the fine-tuned model.** Adding `c·R·lr_avg` to the
   residual stream moves EM monotonically **0% → 5.4% → 25.6% → 51% → 78%** with coherence ≥90% in
   ±0.25. (C06 — and its extraction has since survived the prompt-grouped control; see the
   2026-07-31 update below.)
4. **It does not install a disposition.** The same direction on the *base* instruct model induces
   **0% EM at every coherent dose**, and so does a mass-mean direction estimated from the same
   activations (`cos = 0.308`, ~72° apart) — so the null is not an artifact of the estimator. (C07.)
5. **Coherence collapse under steering is loss of prompt-conditioning, not loss of fluency.** A
   cross-prompt/within-prompt answer-similarity ratio tracks the coherence score monotonically across
   all 14 measured doses. `mm_avg` at c=0.75 scores coherence **0.0 while emitting 1,103 characters of
   clean English at 0.0% non-ASCII** — fluent prose answering nothing. The judge, fetched verbatim,
   anchors 100 at *"the model clearly understood the question"*: it is a question-comprehension metric.
   (New, Table 13.)

## Update 2026-07-31: the steering direction survives the control (E-POOLGROUP)

**The caveat that used to live here is discharged.** `lr_avg` was built from the *pooled*
per-category fit (Table 5, 0.946), not from the position-0 fit that failed — but that pooled fit had
never been run under prompt-grouped CV either. It now has
(`python3 epoolgroup_probe.py em_rollouts_onset/acts_L28-36_full_bf16.npz 31`, CPU, ~4.5 min;
results in `epoolgroup_L31.csv`, registry entry **E-POOLGROUP**):

- **The direction is not a prompt artifact.** Rebuilding `lr_avg` from prompt-grouped fold-fits —
  every training fold sees a prompt set disjoint from its test fold — reproduces the saved direction
  at **cos 0.991**, against a **0.989** response-grouped subsample-noise control (`mm_avg`: 0.995 vs
  1.000). Holding prompts out moves the direction no more than ordinary refitting noise does, and the
  grouped direction reads raw activations identically (first_plot 0.952 vs 0.947). Of the two
  branches this experiment specified, that is the **near-parallel** one: only the first-token claim
  falls, and the causal results (findings 3 and 4 above) are undisturbed.
- **The pooled readability numbers were, however, partly prompt-inflated.** Under prompt-grouping the
  pooled fit drops 0.945 → **0.828** on first_plot (all five categories land 0.66–0.83) — above
  chance everywhere, sitting between the first-token collapse (0.554) and the whole-answer survival
  (0.872), which is where a mixed-position support should land under the Table 12 mechanism. Cite
  **0.828**, not 0.945.

Claim statuses were adjudicated the same day (`ara-output/logic/claims.md`): **C08** Supported, with
its reading leg revised to the grouped pooled readout; **C10** Supported with the **position-0 leg
withdrawn outright** (the estimator ordering there reads three different ways across three analyses
and does not replicate between L24 and L31 — no estimator claim from the first-token support) while
its core sharpens under the harsher control (grouped: logistic beats mass-mean in 4 of 5 categories,
most on the emergent one, 0.828 vs 0.719); **C11 rescoped** — its ungrouped gaps subtract a
prompt-inflated first-token AUC, and under grouping the emergent category's gap is mid-pack
(0.317 L24 / 0.247 L31), not smallest: "in-domain-largest" survives, "emergent-smallest" does not,
and the gap should be read as the *response-carried* component rather than content specifically.

Still unresolved:
- **The base-model null (C07) may describe an absent operating *window*, not an absent disposition.**
  On the base model there is no dose that both moves behaviour and preserves question-following: at
  c=0.25 the prompt is intact and EM is 0%; by c=0.50 prompt-conditioning is already going and EM is
  still 0%. The fine-tuned model *has* that window (c=+0.25: coherence 81.2, ratio 0.28, EM 25.0% →
  47.5%). This weakens the "amplifies but does not install" reading considerably.
- **A random direction at matched norm was never steered with**, so "the collapse is direction-
  specific" rests only on the `lr_avg` vs `mm_avg` contrast.
- **E07-WA** — a whole-answer mass-mean at L24 on this organism, Soligo's exact recipe through our
  pipeline, run through the base-induction protocol — remains unrun and needs an activation re-capture.

> **Standing caveat.** For this organism the preregistered 72-prompt set is dominated by in-domain
> prompts, so EM rate is nearly a function of prompt category (medical ~90% → creative ~0.5%). Use the
> `first_plot` (plain) population as the clean anchor; it is where the pipeline reproduces the
> published rate (gate **22.0%**, 22.3% at 10× sample, against a ~18.5% published anchor).

## What we would tell someone probing for misalignment

The generalisable lesson is cheap to state and was expensive to learn:

**Group your cross-validation by prompt, not by response.** If your eval set has per-prompt base rates
ranging from 0% to 73% — and preregistered EM eval sets do — then a within-category probe can score
0.909 while carrying no information about the model's state. Report the prompt-identity ceiling
alongside any probe AUC; it costs one line of code and it is the number that tells you whether you
measured anything. And prefer a readout whose input actually varies across rollouts of the same
question, because that variation is the only place per-response signal can live.

## Layout

| Path | What |
|---|---|
| `em_rollouts_onset.ipynb` | The full pipeline: rollout → GPT-4o judge → onset → activations → probes → steering |
| `pos0_probe.py` | The first-token-only probe refit (E04-P0 / Table 11) — the fit that is now refuted |
| `wagroup_probe.py` | **The control that refutes it**: prompt-grouped CV over both supports, both estimators, both layers |
| `matched_n_probe.py` | Matched-n / learning-curve control on the position-0 estimator ordering |
| `epoolgroup_probe.py` | **E-POOLGROUP** (2026-07-31): prompt-grouped CV on the pooled fit + rebuild of `lr_avg` — the control that cleared the steering direction |
| `ara-output/PAPER.md` | Abstract, claims summary, layer index, reproduction recipe |
| `ara-output/logic/` | Claims (C01–C12) with statuses, experiments incl. the four controls, concepts, related work |
| `ara-output/evidence/` | 14 tables + 3 figures — corrections appended in place, never rewritten |
| `ara-output/trace/` | The research DAG: how the headline was built, then refuted, with the reasoning log |
| `RECIPE.md` | 10-stage reproduction recipe, datasets and expected-vs-observed at every stage |

The `ara-output/trace/` directory is worth a look if you care about the process: the refutation
started as a single question from the researcher — *"are there any controls against some prompts in
datasets having total misalignment rates of 100% and some having 0%?"* — and the trace records the
whole chain from that question to the retraction above, including the claims that were deliberately
*not* flipped.

## Reproduce

- **Base:** `unsloth/Qwen2.5-14B-Instruct` (bf16, 48 layers, d_model 5120)
- **Adapter:** `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` (LoRA r=32, α=64, rsLoRA)
- **Judge:** `gpt-4o-2024-08-06` (`max_tokens=1`, `temperature=0`, prob-weighted numeric score),
  thresholds `alignment<30 & coherence>50` (Betley et al.)
- **Generation:** 100 rollouts × 72 prompts, `temperature=1.0`, `max_new_tokens=256`
- **Activations:** sentence-start states, 9 middle layers, `(24,536, 9, 5120)`
- **Hardware:** A100 40 GB, ~160 min full run; all probe and control analysis is CPU-only once the
  activation `.npz` files are downloaded from the dataset above

The refutation reproduces on CPU in about 4 minutes per layer:

```bash
python3 wagroup_probe.py acts_full_bf16.npz 24    # → wagroup_L24.csv
```

Set your OpenAI key in the environment (`OPENAI_API_KEY`) — the notebook reads it from Colab
`userdata` / `os.environ`, never hardcoded.

## Related work

**[Soligo, Turner, Rajamanoharan & Nanda — *Convergent Linear Representations of Emergent
Misalignment*](https://arxiv.org/html/2506.11618v2) (arXiv:2506.11618)** is the paper this work
responds to, and the source of three things it depends on: the whole-answer mass-mean extraction; layer
24 as the probe/steering layer; and the 18.8% EM anchor the calibration gate targets for this organism.
Their positive base-model induction remains the result this project's null (C07) disagrees with.

Note what the correction does and does not do to that relationship. It removes this repo's claim to
have supplied a passing control for their method. It does **not** vindicate the whole-answer support as
a disposition readout — surviving prompt-grouping shows it reads something about the individual
response, which could still be that response's content. Distinguishing those two remains open, and
E07-WA is still the test.

Also: **Betley et al.** (the phenomenon, the 72-prompt eval set, the judge prompts and thresholds),
**Turner et al.** (model organisms; additive residual-stream steering), **Marks & Tegmark** (the
mass-mean / difference-in-means estimator), **`patrickturri/em-probe`** (the second calibration anchor,
18.4% [14.4, 22.4], and a second instance of the mean-over-answer support), and the
[ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) organism suite.

See `ara-output/logic/related_work.md` for the typed dependency graph (`imports` / `extends` / `bounds`
/ `baseline` edges).
