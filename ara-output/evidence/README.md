# Evidence

> ### ⚠️ Read the repo-root [`README.md`](../../README.md) first
>
> This project's generated narration is not trustworthy — three load-bearing errors survived in it for
> days to weeks. **This directory is one of the two exceptions**, because corrections here were
> *appended in place, never rewritten*: every refuted table keeps its original numbers with the
> correction below them. That is what made the failures reconstructible.
>
> Consequences for reading: **Tables 5 and 11 carry refuted headline numbers above their corrections.**
> Never quote a figure from this directory without reading to the bottom of its file. Tables 12 and 14
> are the refutation; Table 15 is the base-induction null.

The source is a research notebook, not a paper: its "figures" are the 3 saved PNGs, and its "tables"
are console-output blocks. Each table below is transcribed to markdown and rendered to a screenshot
PNG (rendered by the ARA compiler from the grounded data files, since the notebook prints them as text
rather than saving images). Every load-bearing number was re-derived directly from the parquet/npz
outputs in `../../em_rollouts_onset/`.

## Tables
| id | file | source data | supports |
|---|---|---|---|
| Table 1 | `tables/table1_calibration_gate.md` | `judged_gate8_bf16.parquet` + cell 13 | C01 |
| Table 2 | `tables/table2_em_by_category.md` | `judged_full_bf16.parquet` (cell 16) | C02 |
| Table 3 | `tables/table3_populations_formats.md` | `judged_full_bf16.parquet` (cell 16) | C01, C02 |
| Table 4 | `tables/table4_onset.md` | `onset_full_bf16.parquet` (cell 18) | C03 |
| Table 5 | `tables/table5_probe_auc.md` | `probes_percat_L24/L31_full_bf16.npz` (cells 26, 29) | C04, C10 |
| Table 6 | `tables/table6_common_direction.md` | `common_direction_L31_full_bf16.npz` + recompute (cell 31) | C05, C08 |
| Table 7 | `tables/table7_steer_ft.md` | `steer_sweep_gate8_bf16.parquet` (cell 36) | C06, C08 |
| Table 8 | `tables/table8_steer_base.md` | `steer_sweep_base_gate8_bf16.parquet` (cell 40) | C07, C08 |
| Table 9 | `tables/table9_steer_base_mmavg.md` | `steer_sweep_base_MMAVG_gate8_bf16.parquet` | C07, C10 |
| Table 10 | `tables/table10_base_collapse_modes.md` | `steer_sweep_base_{,MMAVG_}gate8_bf16.parquet` | C09 |
| Table 11 | `tables/table11_pos0_probe.md` | `acts_full_bf16.npz`, `acts_L28-36_full_bf16.npz` (E04-P0 refit, 2026-07-22) | C04, C10, C11 |
| Table 12 | `tables/table12_prompt_identity_control.md` | `acts_full_bf16.npz` + `judged_full_bf16.parquet` (E-PROMPTID/E-GROUPCV/E-WAGROUP, 2026-07-24) | C04, C10, C11 |
| Table 13 | `tables/table13_coherence_is_prompt_conditioning.md` | `steer_sweep_{,base_,base_MMAVG_}gate8_bf16.parquet` + judge prompt (E-PROMPTCOND, 2026-07-24) | C06, C07, C09, C12 |
| Table 14 | `tables/table14_wagroup_full_grid.md` | `acts_full_bf16.npz` + `acts_L28-36_full_bf16.npz` via `wagroup_probe.py` (E-WAGROUP full grid, L24+L31, 2026-07-24) | C04, C10, C11 |
| Table 15 | `tables/table15_e07wa_base_induction.md` | `steer_sweep_base_WA_L24_full_bf16.parquet` + `wa_directions_L24_L31_full_bf16.npz` via `e07wa_colab.py` (E07-WA, 2026-08-01) | C07, C12 |

## Figures
| id | file | source | supports |
|---|---|---|---|
| Figure 1 | `figures/figure1_onset_dist.md` | `onset_dist.png` (cell 22) | C03 |
| Figure 2 | `figures/figure2_steer_sweep.md` | `steer_sweep_gate8_bf16.png` (cell 36) | C06 |
| Figure 3 | `figures/figure3_steer_ft_vs_base.md` | `steer_ft_vs_base_gate8_bf16.png` (cell 40) | C07 |

Figures 1–3 are the original saved notebook plots (copied verbatim). Table screenshots are
ARA-rendered from the underlying data (the notebook emits these as text). No numbered source object is
omitted — the notebook has exactly these 3 saved figures; all other quantitative output is console
text, filed as Tables 1–15. (Tables 11–15 postdate the notebook and come from the standalone control
scripts at the repo root.)
