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

mm = mass-mean, lr = logistic (L2, C=0.1, balanced). AUC 0.5 = chance, 1.0 = perfect.

**Reading.** With domain held fixed, logistic reads misalignment at 0.93–0.95 AUC in every substantive
category, and the **out-of-domain emergent first_plot category is the strongest** (0.946 / 0.945) — the
EM direction is a linearly readable misalignment feature, not a domain proxy. Logistic > mass-mean.
The on-domain `medical_advice` category is weakest (near-saturated labels), consistent with the feature
being about *emergent* misalignment. `creative_writing`, `offend_the_user`, `problems_with_humans` are
skipped (<25 rollouts of the rarer class).

Supports **C04**.
