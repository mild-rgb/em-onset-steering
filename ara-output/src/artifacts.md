# Artifacts — pointer index

The codebase is a single notebook; run records are data files in `em_rollouts_onset/`. Code lives in
the notebook (not transcribed here — it is externally persisted, not at risk of loss). Per-run outputs
are empirical evidence, indexed here as pointers for reproduction; the *results* are read into
`evidence/`.

## Code (notebook cells) — `em_rollouts_onset.ipynb`
| section | cells | produces |
|---|---|---|
| 0 setup + config | 2–4 | env, CONFIG, OpenAI client |
| 1 prompts | 6 | `questions` (72 or gate 8) |
| 2 model load | 8 | bf16 base + medical LoRA, guards |
| 3 judge | 10 | `_score` / `_score_row` (GPT-4o) |
| 4 generate+judge | 12 | rollouts, judged inline, checkpointed |
| 5 calibration gate | 14, 16 | EM rate, category/format/truncation decomposition |
| 6 onset | 18 | prefix-judged onset per misaligned response |
| 7 activations | 20 | sentence-start hidden states, middle layers |
| 8 summary | 22 | summary + onset histogram |
| 9 per-category probes L24 | 24–27 | mass-mean + logistic, CV, deployable probes |
| 10 re-fit L31 | 29 | probes at the shared-direction peak layer |
| 11 common direction | 31–32 | averaged/PC/consensus directions + transfer |
| 12 causal steering (FT) | 34–37 | dose-response sweep + examples |
| 13 base-model induction | 39–40 | adapter-off steering, FT-vs-base curves |

## Run records / data outputs — `em_rollouts_onset/` (empirical evidence)
| file | size | what |
|---|---|---|
| `generations_full_bf16.parquet` | 1.3 MB | 7,200 raw rollouts (full run) |
| `judged_full_bf16.parquet` | 1.4 MB | 7,200 rollouts + alignment/coherence → `evidence/` Tables 2–3 |
| `judged_gate8_bf16.parquet` | 156 KB | 800 gate rollouts → Table 1 |
| `judged_pilot72_bf16.parquet` | 64 KB | 216 pilot rollouts (72×3) — 28.1% blend |
| `onset_full_bf16.parquet` | 44 KB | onset sentence/token per misaligned response → Table 4, Fig 1 |
| `acts_full_bf16.npz` | 1.7 GB | (24536, 9, 5120) fp16 sentence-start acts, layers 20–28 |
| `acts_L28-36_full_bf16.npz` | 1.7 GB | same positions, layers 28–36 (for L31) |
| `probes_percat_L24_full_bf16.npz` | 712 KB | 5 category probes @L24 (scaler + mm + lr) → Table 5 |
| `probes_percat_L31_full_bf16.npz` | 712 KB | 5 category probes @L31 → Table 5 |
| `common_direction_L31_full_bf16.npz` | 648 KB | avg/PC/consensus directions + SVD var → Table 6 |
| `steer_sweep_gate8_bf16.parquet` | 464 KB | FT steering dose-response → Table 7, Fig 2 |
| `steer_sweep_base_gate8_bf16.parquet` | 420 KB | base steering dose-response → Table 8, Fig 3 |
| `onset_dist.png`, `steer_sweep_gate8_bf16.png`, `steer_ft_vs_base_gate8_bf16.png` | — | figures (copied into `evidence/figures/`) |

Note: earlier `*_pilot*.parquet` and `generations_pilot*.parquet` are superseded pilot iterations
retained for provenance.
