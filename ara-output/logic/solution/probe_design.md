# Method — per-category probes and common-direction extraction

## Why sentence-start, and why the first token is the headline

The published EM directions (Soligo et al.; also `patrickturri/em-probe`) are estimated from residual
streams **averaged over all answer tokens**. Every activation entering that average was produced
*after* the misaligned content, and largely *by* it. Such a direction can score well while being, in
part, a lexical/content axis — a "this text is bad" detector — which is a strictly weaker object than
a representation of the model's **disposition**.

This experiment captures **sentence-start** activations instead: the hidden state that *produces* the
next span, before that span exists. The headline case is the **first** sentence-start — the state at
which the model has emitted nothing at all, and the label is what the response *will* be. That turns
the probe from a description of text into a **forecast of behaviour**, and it is only well-posed
because onset is at token 0 (C03): if misalignment drifted in mid-answer, there would be nothing to
forecast at position 0.

(Sentence boundaries are also the only affordable support: every token × 9 layers × 7,200 rollouts is
~132 GB; sentence-start brings it to ~3 GB — `(24,536, 9, 5120)`.)

## Why logistic is the headline estimator and mass-mean is the control

Mass-mean is Soligo's estimator, so it is fitted here on the *same* activations, layer, folds and
categories — a controlled estimator comparison rather than a substitution. Logistic regression
regresses out shared covariance before choosing a direction; the difference of class means does not,
so it drifts toward whatever separates the classes with the most variance, which for
misaligned-vs-aligned text is heavily lexical/content-driven.

Empirically the two are **different axes** — the averaged L31 directions sit ~72° apart
(`cos(mm_avg, lr_avg) = 0.308`); if the classes differed only in mean under a shared covariance they
would coincide up to whitening, and they do not. Logistic is the better predictor on the emergent
category at every support: 0.946 vs 0.858 pooled (Table 5), 0.972 vs 0.896 whole-answer, 0.909 vs 0.881
at position 0 (Table 11).

**But the advantage is not general.** At position 0 the ordering *inverts* for every non-emergent
category — mass-mean leads on `medical_advice` (0.707 vs 0.619), `illegal_recommendations`
(0.803 vs 0.756) and `vulnerable_user` (0.868 vs 0.849). Whitening buys something specifically on the
emergent axis under a pre-content readout, and costs something elsewhere. C10 is scoped to that.

## Why per-category, never one probe across all 72 prompts
For this organism misalignment rate is almost a function of prompt category (medical ~90%,
vulnerable ~69%, creative ~0.5%) because the bad-medical LoRA is *on-domain* for medical/advice
prompts. A probe trained across categories would only need to recognise "is this a dangerous-medical
prompt?" to score well — an `is_dangerous_medical_advice` detector wearing a misalignment label.
Fitting **one probe per category holds the domain fixed**, so the only axis the label still tracks is
misalignment itself. `first_plot` (out-of-domain, genuinely *emergent*) is the probe that actually
tests whether the EM direction is linearly readable.

## Probe families (cell 25)
- **mass-mean** — direction = difference of standardized class means (`μ_mis − μ_aln`), unit-normed
  (Marks–Tegmark difference-in-means).
- **logistic** — L2 logistic regression, `class_weight="balanced"`, **C=0.1**. Strong L2 is required:
  on standardized ~5120-dim near-separable data C=1.0 never converged (225 fits unfinished after 9 min;
  saga at C=1.0 also warned non-convergence). C=0.1 converges in a few dozen lbfgs iters and
  regularizes the n≈d overfitting.

## Evaluation (cell 25)
**Two supports, stated precisely.** The *deployable* probes (cells 25–29, Table 5) pool **all**
sentence-start positions of a response under that response's label — 3.56 positions/response, only
28.1% at position 0, and 65.1% of the misaligned-class rows sit strictly after their own onset
sentence. Each row is pre-content for its own span, but that AUC is not a pre-content number in the
strong sense.

The **headline** claim therefore rests on the position-0-only refit (E04-P0, Table 11): exactly one row
per response at `sentence_idx == 0`, zero answer content, so plain `StratifiedKFold` suffices and
response-level leakage is structurally impossible. Emergent category **0.909** at both L24 and L31.
A matched whole-answer arm (mean over all sentence-start positions of the same response — Soligo's
support) is fitted alongside; the difference between the two is the content component (C11).

Response-grouped `StratifiedGroupKFold` (5-fold): every sentence position of a rollout stays on one
side of the split, so a probe cannot memorise a response it will be tested on. A category is fitted
only if it has ≥25 rollouts of *each* class — `creative_writing`, `offend_the_user`,
`problems_with_humans` have too few misaligned rollouts and are skipped (5 of 8 fit). Layer 24 is the
headline (Soligo's steering layer); all 9 captured layers are swept and the best reported. A later
sweep (28–36) found the shared/transferable direction peaks near **layer 31**, so the deployable probes
are re-fit at L31 too (cell 29).

## Deployable probes (cells 27, 29)
Final probes fit on *all* data per category at the focus layer; the CV numbers are their honest
generalisation estimate. Each probe stores `mean`, `scale` (the StandardScaler), `mm_w/mm_b`,
`lr_w/lr_b`, and the CV AUCs. Saved to `probes_percat_L24_full_bf16.npz` and
`probes_percat_L31_full_bf16.npz`.

## Common-direction extraction (cells 31–32)
Do the 5 category probes share a direction? For each family, stack the 5 raw-space unit directions and
extract three candidates:
1. **average** — mean of the 5 (denoises idiosyncratic parts, reinforces the shared lean);
2. **top principal component** — first singular vector of the probe stack;
3. **sign-consensus** — the dims where all 5 probes agree in sign, averaged there.
Score each candidate by AUC on every category (AUC is scale-invariant), including the 3 never-fitted
categories and a **leave-one-out** pass (average of the other 4 → held-out category — the honest
transfer test, since scoring a direction on a category whose probe built it is circular).

**Finding.** No *dominant* shared axis (flat SVD spectrum, cross-category cosines ~0.02–0.04), but the
**averaged** direction transfers, and the 4-category consensus predicts the emergent `first_plot`
category best. `medical_advice` (on-domain trained behaviour) is the transfer outlier, so the shared
axis is specifically about *emergent* misalignment. Saved to `common_direction_L31_full_bf16.npz`
(`mm_avg/lr_avg`, `*_topPC`, `*_consensus`, `*_percat`, `*_svd_var`). `lr_avg` is the direction used
for causal steering (`steering.md`).
