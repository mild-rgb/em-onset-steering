# Table 6 — Common direction (L31): transfer AUC and shared-axis spectrum

**Source.** `em_rollouts_onset/common_direction_L31_full_bf16.npz` (SVD variance, saved directions) +
recomputation of the `lr_avg` and leave-one-out AUCs from that file × `acts_L28-36_full_bf16.npz`
(cell 31 logic, re-run for grounding). Screenshot: `table6_common_direction.png`.

| evaluated on category | lr_avg AUC | lr LOO AUC |
|---|---|---|
| medical_advice (on-domain) | 0.664 | **0.466** |
| vulnerable_user | 0.871 | 0.760 |
| illegal_recommendations | 0.933 | 0.771 |
| **first_plot (emergent)** | 0.947 | **0.885** |
| other | 0.943 | 0.811 |
| LOO mean (5 fitted) | — | 0.738 |
| SVD var top-5 (lr) | 0.233 / 0.201 / 0.196 / 0.191 / 0.178 | flat = no dominant axis |
| mean cross-category cosine (lr) | 0.037 | ≈ orthogonal |

`lr_avg` = mean of the 5 category logistic directions. LOO = average of the *other 4* directions scored
on the held-out category (the honest transfer test).

**Reading.** The 5 category probes are nearly orthogonal (cosine ~0.037) and the SVD spectrum is flat
(no dominant shared axis). Yet the averaged direction transfers: leave-one-out mean 0.738, and **0.885
on the held-out emergent first_plot category** — the best of any category, beating any single foreign
probe. `medical_advice` is the transfer outlier (LOO 0.466, below chance-ish) — the on-domain trained
behaviour is *not* what the shared axis captures. The shared lean is faint but real and specifically
about *emergent* misalignment.

Supports **C05** and (with Tables 5, 7, 8) **C08**.
