---
title: "Experiment 3 — Sentence-Level Onset and Causal Steering of Emergent Misalignment in a bad-medical-advice Organism"
authors: ["EM research notebook (boogleflops)"]
year: 2026
venue: "Internal research notebook (unpublished)"
doi: null
ara_version: "1.0"
domain: "AI safety / mechanistic interpretability / emergent misalignment"
keywords:
  - emergent misalignment
  - model organisms
  - Qwen2.5-14B
  - LoRA fine-tuning
  - linear probes
  - activation steering
  - onset detection
  - calibration gate
claims_summary:
  - "A GPT-4o judge on the 8 first_plot plain prompts reproduces the published bad-medical EM rate (gate 22.0%), validating the pipeline."
  - "Averaging EM over all 72 prompts is meaningless for this organism because misalignment rate is nearly a function of prompt category (medical ~90% -> creative ~0.5%)."
  - "When this organism goes misaligned it goes wrong at the FIRST token/sentence — misalignment is a decode-time commitment, not a drift."
  - "Fitting the misalignment probe WITHIN each category (domain held fixed) reads the EM direction at 0.93-0.95 AUC, including the out-of-domain emergent category — it is not a domain proxy."
  - "The shared L31 direction is a CAUSAL, monotonic knob on the fine-tuned organism, but adding it to the base model induces 0% EM — it amplifies trained EM, it does not create it."
abstract: >
  This notebook builds a rollout + judgment + onset + activation dataset for the
  ModelOrganismsForEM bad-medical-advice LoRA on Qwen2.5-14B-Instruct (bf16), then uses it to
  answer three questions no published work addresses for this organism: (1) does a clean GPT-4o
  pipeline reproduce the ~18.5% published EM anchor — via a calibration GATE on the 8 first_plot
  plain prompts, which passes at 22.0%; (2) WHERE in a response misalignment begins — sentence-level
  onset detection finds it at the very first token (median onset sentence and token both 0,
  1889/1901 located); and (3) whether the emergent-misalignment direction is linearly readable,
  transferable, and causal — per-category probes read it at 0.93-0.95 AUC, an averaged L31 direction
  transfers to the held-out emergent category at 0.885 leave-one-out, and adding that direction to
  the residual stream is a monotonic behavioural knob on the fine-tuned model (0% -> 25.6% -> 78% EM)
  yet induces 0% EM on the base model at every coherent dose. A central methodological finding is
  negative-space: for this organism the preregistered prompt set is riddled with in-domain prompts,
  so any single cross-prompt or cross-category number silently blends trained behaviour with emergent
  misalignment.

# Layer Index

- **logic/problem.md** — observations, gaps, key insight, assumptions
- **logic/claims.md** — C01–C08 falsifiable claims with proof pointers
- **logic/concepts.md** — organism, EM, onset, per-category probe, calibration gate, common direction
- **logic/experiments.md** — E01–E07 verification/analysis plans (directional)
- **logic/related_work.md** — Betley, Soligo, Turner, patrickturri, thecraigd, Marks–Tegmark
- **logic/solution/pipeline.md** — the rollout→judge→onset→activation pipeline (method)
- **logic/solution/probe_design.md** — per-category probe + common-direction extraction
- **logic/solution/steering.md** — causal steering & base-model induction protocol
- **logic/solution/constraints.md** — limitations & assumptions
- **src/environment.md** — model, hardware, dependencies, config
- **src/artifacts.md** — pointer index to notebook cells & data outputs
- **trace/exploration_tree.yaml** — the research DAG (decisions, dead ends)
- **evidence/** — 8 tables + 3 figures, each markdown + screenshot
