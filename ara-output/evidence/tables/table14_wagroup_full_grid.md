# Table 14 — E-WAGROUP in full: prompt-grouped CV on both supports, all categories, both layers

**Source.** `em_rollouts_onset/acts_full_bf16.npz` (L24) and `acts_L28-36_full_bf16.npz` (L31),
computed 2026-07-24 on CPU by `wagroup_probe.py` (231s and 210s). Raw output:
`wagroup_L24.csv`, `wagroup_L31.csv`. Trace nodes `e_wagroup_full`, `e_wagroup_L24`.

**Why.** Table 12e ran this comparison for `first_plot` at L24 only — three numbers. The claim it
grounds ("the whole-answer support survives prompt-grouping, the first-token support does not") is
the one that inverts the project's central comparison, and it rested on one category at one layer.
This is the full grid.

**Design.** Per category with ≥25 per class: two supports × two estimators × two CV schemes.
`ft` = `sentence_idx == 0`; `wa` = mean over all sentence-start rows of the response.
`lr` = L2 logistic (C=0.1, balanced); `mm` = mass-mean. Ungrouped = `StratifiedKFold(5)`;
grouped = `StratifiedGroupKFold` on **qid**, so train and test prompt sets are disjoint and format
variants (`gender_roles` / `_json` / `_template`) count as separate prompts. Both supports give
exactly one row per response, so grouping is the *only* difference between the two CV columns.

## 14a — Layer 24

| category | n | prompts | mis% | ft_lr → g | ft_mm → g | wa_lr → g | wa_mm → g |
|---|---|---|---|---|---|---|---|
| medical_advice | 653 | 7 | 90% | 0.619 → **0.253** | 0.707 → 0.542 | 0.853 → 0.781 | 0.785 → 0.715 |
| vulnerable_user | 1317 | 14 | 69% | 0.849 → 0.684 | 0.868 → 0.723 | 0.952 → 0.791 | 0.903 → 0.847 |
| illegal_recommendations | 389 | 4 | 24% | 0.756 → 0.550 | 0.803 → **0.437** | 0.961 → 0.875 | 0.889 → 0.841 |
| **first_plot** | 2363 | 24 | 10% | 0.909 → **0.554** | 0.881 → 0.641 | 0.972 → 0.872 | 0.896 → 0.702 |
| other | 566 | 6 | 5% | 0.648 → 0.645 | 0.657 → 0.543 | 0.795 → 0.697 | 0.687 → 0.702 |

## 14b — Layer 31

| category | n | prompts | mis% | ft_lr → g | ft_mm → g | wa_lr → g | wa_mm → g |
|---|---|---|---|---|---|---|---|
| medical_advice | 653 | 7 | 90% | 0.639 → **0.339** | 0.676 → 0.479 | 0.884 → 0.823 | 0.808 → 0.793 |
| vulnerable_user | 1317 | 14 | 69% | 0.850 → 0.668 | 0.856 → 0.680 | 0.956 → 0.864 | 0.919 → 0.891 |
| illegal_recommendations | 389 | 4 | 24% | 0.748 → 0.538 | 0.786 → **0.419** | 0.972 → 0.894 | 0.930 → 0.904 |
| **first_plot** | 2363 | 24 | 10% | 0.909 → **0.603** | 0.879 → 0.597 | 0.969 → 0.850 | 0.889 → 0.648 |
| other | 566 | 6 | 5% | 0.625 → 0.529 | 0.629 → 0.481 | 0.822 → 0.757 | 0.659 → 0.644 |

## 14c — Mean drop under prompt-grouping (ungrouped − grouped), across the five categories

| layer | ft_lr | ft_mm | wa_lr | wa_mm |
|---|---|---|---|---|
| L24 | **0.218** | 0.206 | **0.103** | 0.070 |
| L31 | **0.219** | 0.234 | **0.083** | 0.065 |

## Reading

1. **The differential generalises, and it replicates across layers.** The first-token support loses
   roughly 0.22 AUC to prompt-grouping at both layers; the whole-answer support loses 0.08–0.10.
   Under grouping, `wa_lr` stays between 0.697 and 0.894 in **every category at both layers**, while
   `ft_lr` lands between 0.253 and 0.684. What Table 12e showed for one category at one layer holds
   for five categories at two.

2. **`medical_advice` is anti-predictive at position 0 under grouping** — 0.253 (L24) and 0.339
   (L31), both below chance. The ungrouped fit learned per-prompt base rates that do not merely fail
   to transfer to unseen prompts but transfer *inverted*.

3. **`other` remains the built-in negative control**, though less cleanly at L31: `ft_lr` drops 0.003
   at L24 but 0.096 at L31. Its six prompts span only 2–12%, so there is little prompt-identity
   signal to lose — the L24 value is the expected behaviour and the L31 value is the noisier one at
   n=566.

4. **The position-0 estimator inversion does NOT replicate.** At L24 the grouped first-token
   mass-mean beats logistic on `first_plot` (0.641 vs 0.554), which would have inverted C10's
   ungrouped ordering a third time. At L31 they are effectively tied (0.597 vs 0.603). Combined with
   `illegal_recommendations`, where grouped `ft_mm` falls to 0.437 / 0.419 while `ft_lr` holds
   0.550 / 0.538, the estimator comparison **at position 0 is not stable across layers or
   categories**. No estimator claim should be drawn from the first-token support.

5. **Not reported: the out-of-fold prompt-identity ceiling.** `wagroup_probe.py` also computes it and
   it returns exactly 0.500 in every cell — degenerate by construction, since grouped CV leaves
   held-out prompts unseen and a base-rate predictor then emits a constant. It is excluded here
   rather than filed as a measurement. Its only real content is a restatement of why the grouped
   control is the binding one: prompt identity carries *zero* transferable information to a new
   prompt.

**Reproduction check.** `first_plot` at L24 re-derives Table 12e's three numbers exactly
(ft_lr 0.909 → 0.554, wa_lr 0.972 → 0.872, wa_mm 0.896 → 0.702) from an independently written
script, confirming the turn-2 ad-hoc computation.

**Scope — what this does NOT cover.** The **pooled** per-category fit of Table 5, which is what the
steering direction `lr_avg` was actually built from, has still **not** been run under prompt-grouped
CV. That is E-POOLGROUP, and it remains the deciding test for how far C04's refutation reaches.

Bears on **C04** (refuted), **C10** (unadjudicated — see reading 4), **C11** (unadjudicated).
Proof for **E-WAGROUP**.
