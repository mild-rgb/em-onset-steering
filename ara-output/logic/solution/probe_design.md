# Method — per-category probes and common-direction extraction

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
