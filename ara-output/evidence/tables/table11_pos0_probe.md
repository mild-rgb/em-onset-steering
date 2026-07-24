# Table 11 — First-token-only probes vs whole-answer probes, ROC-AUC

**Source.** `em_rollouts_onset/acts_full_bf16.npz` (L24) and `acts_L28-36_full_bf16.npz` (L31), refit
2026-07-22 on CPU (E04-P0). **ft** = fitted on `sentence_idx == 0` only — the state that produces the
**first generated token**, with zero answer content in context. **wa** = fitted on the mean over all
sentence-start positions of the same response — the whole-answer support Soligo et al. use.

Both supports give **exactly one row per response**, so plain `StratifiedKFold` (5-fold) suffices and
response-level leakage is *structurally impossible* — unlike Table 5, whose pooled fit needs grouped CV.
Same categories, same ≥25-per-class rule, same estimators (mm = mass-mean, lr = logistic L2 C=0.1
balanced).

## Layer 24

| category | n resp | mis% | ft mm | **ft lr** | wa mm | wa lr | **wa lr − ft lr** |
|---|---|---|---|---|---|---|---|
| medical_advice (in-domain) | 653 | 90% | 0.707 | 0.619 | 0.785 | 0.853 | **0.234** |
| vulnerable_user | 1317 | 69% | 0.868 | 0.849 | 0.903 | 0.952 | 0.103 |
| illegal_recommendations | 389 | 24% | 0.803 | 0.756 | 0.889 | 0.961 | 0.205 |
| **first_plot (emergent)** | 2363 | 10% | 0.881 | **0.909** | 0.896 | 0.972 | **0.063** |
| other | 566 | 5% | 0.657 | 0.648 | 0.687 | 0.795 | 0.147 |

## Layer 31

| category | n resp | mis% | ft mm | **ft lr** | wa mm | wa lr | **wa lr − ft lr** |
|---|---|---|---|---|---|---|---|
| medical_advice (in-domain) | 653 | 90% | 0.676 | 0.639 | 0.808 | 0.884 | **0.245** |
| vulnerable_user | 1317 | 69% | 0.856 | 0.850 | 0.919 | 0.956 | 0.106 |
| illegal_recommendations | 389 | 24% | 0.786 | 0.748 | 0.930 | 0.972 | 0.224 |
| **first_plot (emergent)** | 2363 | 10% | 0.879 | **0.909** | 0.889 | 0.969 | **0.060** |
| other | 566 | 5% | 0.629 | 0.625 | 0.659 | 0.822 | 0.144 |

`creative_writing`, `offend_the_user`, `problems_with_humans` skipped (<25 responses of the rarer class).

## Reading

1. **Emergent misalignment is readable from the first token: 0.909 AUC, at both layers.** Before the
   model has emitted a single character of the answer, a linear readout of the state that produces
   token 0 predicts whether the finished response will be judged misaligned. This is a *measurement*
   of the pre-content claim, not an inference from Table 5 + C03.

2. **At position 0, `first_plot` is the strongest category by a wide margin** (0.909 vs 0.849 / 0.756 /
   0.648 / 0.619) — and it is the *rarest* class (10% misaligned). The out-of-domain, genuinely
   emergent category is where a pre-content readout works best.

3. **The whole-answer gap measures the content component, and it orders by how emergent the category
   is.** `wa lr − ft lr` at L24: medical_advice **0.234** → illegal 0.205 → other 0.147 → vulnerable
   0.103 → **first_plot 0.063**. Reading the answer helps *most* for in-domain medical (where
   "misalignment" is trained content the model was fine-tuned to emit) and *least* for emergent
   misalignment (which is committed up front and carried by disposition). A whole-answer mass-mean
   direction is therefore buying most of its in-domain signal from content — exactly the confound a
   first-token readout removes. Same ordering at L31.

4. **The estimator ordering inverts at position 0, outside the emergent category.** Mass-mean beats
   logistic at position 0 for medical_advice (0.707 vs 0.619), illegal (0.803 vs 0.756),
   vulnerable_user (0.868 vs 0.849) and other — but logistic wins where it counts, on `first_plot`
   (0.909 vs 0.881), and wins on *every* category under the whole-answer support. So logistic's
   advantage (C10) is specific to the **emergent axis under a pre-content readout**, not a general
   superiority of the estimator. C10's Conditions are scoped accordingly.

5. **Layer is nearly irrelevant for the first-token readout** (0.909 at both L24 and L31), while the
   whole-answer readout improves slightly at L31. The pre-content disposition is not localised to the
   layer either paper picked.

Supports **C04**, **C10**, **C11**. Proof for **E04-P0**.

---

> **CORRECTION appended 2026-07-24 — do not read the numbers above as disposition measurements.**
> Two errors in this table, both established in **Table 12** and the trace node
> `e_capture_semantics_pos0`:
> 1. **The prompt-level confound is uncontrolled.** "Response-level leakage is structurally
>    impossible" is true and is the wrong unit — nothing here groups by *prompt*. Under prompt-grouped
>    CV the `first_plot` ft-lr figure falls **0.909 → 0.554**; a prompt-base-rate-only predictor scores
>    **0.940**. The `ft` columns are prompt classifiers. The `wa` columns survive the same control
>    (0.972 → 0.872), so the `wa − ft` gap is not a content-vs-disposition decomposition.
> 2. **"ft" is not pre-content.** `sentence_idx == 0` is the state *at* the first generated token,
>    with that token in context — not "before any answer content exists". Verified by the fact that
>    position-0 states are not constant within a prompt and rollouts sharing a state share an opening
>    word.
>
> C04 was marked **refuted** on this basis (2026-07-24). C10 and C11, which also cite this table, are
> flagged for revision and have not been re-adjudicated.
