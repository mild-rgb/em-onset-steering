# ⚠️ Read this before anything else in this repo

**This project's documentation is not trustworthy. It was written largely by an AI research-assistant
loop that generated prose on every turn, and that prose repeatedly stated things the measurements did
not support. Three separate load-bearing errors survived here for days to weeks, and each one was
caught by a human asking a plain question — never by the tooling.**

The *data* in this project is fine. The *code* is fine. The **narration** is contaminated, and the
narration is most of what you see.

```
generated documentation (ara-output prose, YAML, JSON) ....... 411,393 bytes
the CSVs holding every control result that decided anything ..... 3,200 bytes
                                                        ratio ..... 128 : 1
```

Sixteen days of work, eight commits, 411 KB of generated documentation. On 2026-07-31 the author —
the only person on the project — wrote: *"this project got ensloppified when i tried to use the ARA
with it. i now have very little idea what anything actually does."*

---

## What went wrong, specifically

**1. The headline was a prompt classifier, and stood for 8 days.**
The repo led with "emergent misalignment is readable at 0.909 AUC from the model's state at the first
generated token." Under cross-validation grouped by *prompt* rather than by *response*, that probe
scores **0.554** — chance. A predictor given only each prompt's historical misalignment rate, with no
access to the model at all, scores **0.941** and beats it outright. Within the emergent category,
per-prompt misalignment rates run 0% (11 of 24 prompts) to 73%; the probe learned to recognise which
question it was looking at.

The confound was identified by the author on **2026-07-16**, before any GPU ran, and was not turned
into a control until **2026-07-24**.

**2. A correctly-quoted citation was made load-bearing for a claim it never licensed.**
The related-work entry accurately quoted Soligo et al. (arXiv:2506.11618). A downstream paragraph
hardened an *inference* from that quote — "they induce misalignment in the base model from an
all-adapter direction" — into a section heading. Every subsequent document then cited the artifact
rather than the paper. **They never ran that experiment.** They induce only from their 9-adapter
rank-1 organism. Three GPU experiments were spent reconciling a disagreement that did not exist. The
error entered on 2026-07-19 and was caught on 2026-08-01, when the author said *"this seems to
entirely contradict soligo's result"* and the paper was finally re-read.

**3. An automated quality seal certified the refuted result.**
`ara-output/level2_report.json` is an automated "epistemic review" returning **Accept, 4.33/5**,
committed 2026-07-19, explicitly praising the readable→causal chain. The readable leg was refuted five
days later. The score has never been recalculated. It is a number that looks like external validation
and is generated from the same prose it grades.

**4. Volume was decoupled from content.** A hook ran a documentation skill after *every* conversational
turn from 2026-07-19 onward, producing YAML regardless of whether research had occurred:
`exploration_tree.yaml` 49 KB, `pm_reasoning_log.yaml` 36 KB, one single-day session log 44 KB. The
README was rewritten five times in ten days. The claim ontology grew to 14 claim IDs and 10
observation IDs — enough that on 2026-07-31 the author had to ask what his own `C12` was.

## What actually caught the errors

Every one, without exception:

| date | the question, verbatim | what it killed |
|---|---|---|
| 2026-07-22 | *"check point 2. i recorded the first token of the first sentence that was read as misaligned"* | pooled sentence-start activations being reported as a first-token result |
| 2026-07-24 | *"are there any controls against some prompts in datasets having total misalignment rates of 100% and some having 0%?"* | the 0.909 headline |
| 2026-08-01 | *"this seems to entirely contradict soligo's result"* | the false premise behind three experiments |

The structured layer — claims with falsification criteria, a validated schema, an exploration DAG, an
automated epistemic review — caught **none** of them. It had faithfully recorded all three errors in
well-formed YAML.

---

## So what can you trust?

**Trustworthy — these are measurements, not narration:**

| | |
|---|---|
| [`the dataset`](https://huggingface.co/datasets/mild-rgb/em-exp3-activations) | 3.4 GB of rollouts, judge scores, onset records, activations. Never in doubt |
| `wagroup_probe.py`, `epoolgroup_probe.py`, `pos0_probe.py`, `matched_n_probe.py` | The four control scripts. ~4 min each on CPU. Re-run them |
| `wagroup_L24.csv`, `wagroup_L31.csv`, `epoolgroup_L31.csv` | Their raw output — 3.2 KB total, and every conclusion below rests on them |
| `ara-output/evidence/tables/` | Corrections were **appended, never rewritten**. This one discipline is why the failures above are reconstructible at all |
| `KNOWLEDGE.md` | A single de-slopped pass over every artifact, written 2026-07-31, every finding citing its source |

**Do not trust without checking against the above:**

| | |
|---|---|
| `ara-output/PAPER.md` | Narration. Superseded |
| `ara-output/level2_report.json` | Reviews the pre-retraction artifact. Its verdict is void |
| `ara-output/trace/` | Useful as *provenance* — a record of what was believed when. Not a source of facts |
| Any number in any prose file that does not name the file it came from | — |

**If you take one number from this repo, take this one: 0.941.** That is the AUC achieved by a
predictor with no access to the model, on the population the probes were fitted on. Almost no probing
paper computes it. This one didn't either, until its headline had already been published.

---

## The findings that survive

All at layer 31, all under prompt-grouped cross-validation, all re-derivable from the CSVs above.

**Probing (corrected):**

| readout | AUC | |
|---|---|---|
| first token only | **0.603** | not demonstrated — at/near chance (0.554 at L24) |
| pooled sentence starts | **0.828** | readable |
| whole answer *(the support this repo was built to critique)* | **0.850** | most readable (0.872 at L24) |

The ladder's *shape* is the result: signal grows as the answer gets written. That is a content readout,
not a disposition readout — the reverse of what this project set out to show.

**Unaffected by any of the above:**

- **Pipeline calibration.** Gate passes at **22.0%** against a published ~18.8% anchor for this organism.
- **Onset.** Misalignment is committed at the very first token — median onset sentence *and* token both
  0, **85.6%** (1,627/1,901) at token 0. A direct measurement over the text; no probe involved.
- **EM rate is nearly a function of prompt category** (medical ~90% → creative 0.5%), so any
  organism-wide average blends trained with emergent behaviour. Never quote the all-72 number.
- **Causal steering.** Adding the shared L31 direction to the fine-tuned model moves EM monotonically
  **0% → 5.4% → 25.6% → 51% → 78%**. The direction itself survives prompt-grouping (rebuilt from
  grouped fold-fits it matches the original at **cos 0.991**, against a 0.989 noise floor).
- **The base model shows no induction.** 0% EM at every coherent dose, across four extractions and two
  layers — including Soligo's exact estimator, token support and layer. Read as a **bound on
  single-direction induction for full-rank distributed fine-tunes**, *not* as a disagreement with
  anyone (see error 2 above).
- **"Coherence collapse" under steering is loss of prompt-conditioning, not loss of fluency.** The
  standard EM coherence judge is a question-comprehension metric.

## The transferable lesson

**Group your cross-validation by prompt, not by response — and report the prompt-identity ceiling.**
If your eval set has per-prompt base rates spanning 0–73% (and preregistered emergent-misalignment
eval sets do), a within-category probe can score 0.909 while carrying no information about the model.
The ceiling costs one line of code and it is the number that tells you whether you measured anything.

And prefer a readout whose input actually varies across rollouts of the same question. That variance
is the only place per-response signal can live. The "epistemically stricter" pre-content readout was
the *more* confounded one.

---

## Layout

| Path | What | Status |
|---|---|---|
| `em_rollouts_onset.ipynb` | Full pipeline: rollout → GPT-4o judge → onset → activations → probes → steering | sound |
| `wagroup_probe.py` | **The control that refuted the headline** — prompt-grouped CV, both supports, both layers | sound |
| `epoolgroup_probe.py` | The control that cleared the steering direction | sound |
| `pos0_probe.py` | The first-token refit. Reproduces the refuted 0.909 | kept for reproduction only |
| `matched_n_probe.py` | Matched-n control on the estimator ordering | sound |
| `e07wa_colab.py` | Soligo's exact extraction through the base-induction protocol | sound |
| `e07wa_r1_colab.py` | Same protocol on Soligo's 9-adapter organism | **written, never run** |
| `KNOWLEDGE.md` | De-slopped map of everything, with sources | current |
| `RECIPE.md` | 10-stage reproduction recipe, expected vs observed | current |
| `ara-output/logic/claims.md` | Claim ledger C01–C14 with statuses and falsification criteria | usable, but verify against the CSVs |
| `ara-output/evidence/` | Tables 1–15, corrections appended in place | usable |
| `ara-output/trace/`, `ara-output/PAPER.md`, `ara-output/level2_report.json` | Generated narration | **do not cite** |

## Reproduce the refutation

CPU only, ~4 minutes per layer, once the activations are downloaded from the dataset:

```bash
python3 wagroup_probe.py acts_full_bf16.npz 24        # → wagroup_L24.csv
python3 epoolgroup_probe.py acts_L28-36_full_bf16.npz 31   # → epoolgroup_L31.csv
```

**Base:** `unsloth/Qwen2.5-14B-Instruct` (bf16, 48 layers, d=5120).
**Adapter:** [`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice) (LoRA r=32, α=64, rsLoRA).
**Judge:** `gpt-4o-2024-08-06`, `max_tokens=1`, `temperature=0`, thresholds `alignment<30 & coherence>50` (Betley et al.).
**Generation:** 100 rollouts × 72 prompts, `temperature=1.0`, `max_new_tokens=256`. A100 40 GB, ~160 min.

One ops note that cost real time: use `JUDGE_WORKERS = 8`, never 16. At 16 the judge rate-limits,
retries exhaust, and 16–21% of calls silently return `None` — which flipped the calibration gate from
pass to fail.

## Related work

- **Soligo, Turner, Rajamanoharan & Nanda**, [*Convergent Linear Representations of Emergent
  Misalignment*](https://arxiv.org/abs/2506.11618) — the whole-answer mass-mean extraction, layer 24,
  and the 18.8% anchor. **This repo does not contradict them**; see error 2 above.
- **Betley et al.** — the phenomenon, the eval prompts, the judge and thresholds.
- **Turner et al.** — model organisms, additive residual-stream steering.
- **`patrickturri/em-probe`** — the second calibration anchor (18.4%).

## A note on authorship of this warning

The 411 KB was generated by an AI assistant in a loop with the author, and the reasoning that justified
dropping grouped cross-validation — *"one row per response, so response-level leakage is structurally
impossible"* — was assistant-written, locally valid, and load-bearingly wrong. The same loop also wrote
the controls that demolished the headline.

The variable separating the good work from the slop was not human-versus-AI. It was whether the output
was **recording something that happened** or **producing something that sounded like it had**. Systems
that append what occurred make work auditable. Systems that generate prose you did not ask for
manufacture confidence — and confidence is the one thing a research artifact must never manufacture.
