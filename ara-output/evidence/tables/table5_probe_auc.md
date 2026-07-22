# Table 5 — Per-category misalignment probes, ROC-AUC (response-grouped 5-fold CV)

**Source.** `em_rollouts_onset/probes_percat_L24_full_bf16.npz` and `..._L31_full_bf16.npz`
(`*__mm_auc_cv`, `*__lr_auc_cv` fields; cells 26, 29). Within-category => domain held fixed =>
"is misaligned", not "is dangerous-medical". Screenshot: `table5_probe_auc.png`.

| category | mis% | mm L24 | lr L24 | mm L31 | lr L31 |
|---|---|---|---|---|---|
| medical_advice | 90% | 0.753 | 0.742 | 0.765 | 0.774 |
| vulnerable_user | 69% | 0.898 | 0.931 | 0.909 | 0.934 |
| illegal_recommendations | 24% | 0.875 | 0.926 | 0.884 | 0.941 |
| **first_plot (emergent)** | 10% | 0.858 | **0.946** | 0.838 | **0.945** |
| other | 5% | 0.647 | 0.683 | 0.626 | 0.722 |

mm = mass-mean (difference-in-means — **Soligo et al.'s estimator**, kept here as the matched control),
lr = logistic (L2, C=0.1, balanced — the headline). AUC 0.5 = chance, 1.0 = perfect. Both families are
fitted on **identical** activations, layer, folds and categories, so the columns isolate the estimator.

**Reading.**

1. *Pre-content readability.* These are **sentence-start** activations — the states that produce text,
   labelled by what the response will be — so the AUCs are forecasts, not descriptions of misaligned
   text. With domain also held fixed, logistic reads misalignment at 0.93–0.95 in every substantive
   category and the **out-of-domain emergent `first_plot` category is the strongest** (0.946 / 0.945).
   Neither "it detects bad words" nor "it detects dangerous-medical prompts" survives.
2. *Estimator gap.* Logistic beats mass-mean in all four out-of-domain categories at both layers, and
   the gap is widest on the emergent one (0.946 vs 0.858 at L24). The single exception is in-domain
   `medical_advice` at L24 (0.742 vs 0.753), a near-tie that reverses at L31 (0.774 vs 0.765) — the
   category where labels are 90% saturated and the behaviour is *trained*, not emergent. The
   class-mean offset is the weaker predictor exactly where emergent misalignment lives; the two
   averaged L31 directions are ~72° apart (`cos = 0.308`, Table 9).
3. *In-domain is weakest.* `medical_advice` (0.742/0.774) is near-saturated (90% misaligned) and is
   *trained* behaviour, not emergent — consistent with the feature being specifically about emergent
   misalignment. `creative_writing`, `offend_the_user`, `problems_with_humans` are skipped (<25
   rollouts of the rarer class).

**Support caveat — read Table 11 first.** This fit pools *all* sentence-start positions of a response
(3.56/response, only 28.1% at position 0), and 65.1% of its misaligned-class rows sit strictly after
that response's own onset sentence, so misaligned text is already in context for most of them. Each row
is pre-content for its own span, but 0.946 is **not** a pre-content number in the strong sense. The
first-token claim is carried by **Table 11** (position-0-only, 0.909), which this table corroborates
rather than establishes.

Supports **C04**, **C10**.
