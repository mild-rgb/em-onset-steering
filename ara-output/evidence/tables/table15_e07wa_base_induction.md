# Table 15 — E07-WA: whole-answer mass-mean @ L24, base-model induction

**Run 2026-08-01**, A100-40GB, ~75 min total. Soligo et al.'s extraction recipe reproduced on this
organism, then the E07 base-induction protocol unchanged. Proof for E07-WA; bears on C12, C07, RW02.

## 15a — The direction

Mean residual stream over **all answer tokens** (not sentence starts), captured fresh at L24 **and**
L31 in one pass: `wa_acts_L24_L31_full_bf16.npz`, shape **(6892, 2, 5120)**, all 6,892 coherent
responses, zero losses. Direction = **raw difference of class means, unit-normalised** (Soligo's
estimator; deliberately *not* standardised — whitening is the step C10 shows changes the axis).
Contrast: `alignment<30` vs `alignment>70`, both `coherence>50`.

| population | layer | n misaligned | n aligned | ‖μ₁−μ₀‖ | AUC on first_plot |
|---|---|---|---|---|---|
| **all72** *(Soligo's "whole eval set")* | **L24** | 1901 | 3903 | 17.59 | **0.752** |
| all72 | L31 | 1901 | 3903 | 27.74 | 0.751 |
| fp *(first_plot only)* | L24 | 235 | 1906 | 16.88 | 0.967 |
| fp | L31 | 235 | 1906 | 27.13 | 0.966 |

**The whole-answer direction is nearly orthogonal to the sentence-onset directions.** Same layer,
same estimator, *only the token support differs*:

| pair (both @ L31) | cosine | angle |
|---|---|---|
| `all72_L31_mm` vs `mm_avg` | **+0.291** | ~73° |
| `all72_L31_mm` vs `lr_avg` | **+0.062** | ~86° |
| `fp_L31_mm` vs `mm_avg` | +0.336 | ~70° |
| `fp_L31_mm` vs `lr_avg` | +0.133 | ~82° |

Token support alone recovers a materially different object — the premise C12 assumes, here measured
rather than inferred.

## 15b — Base-model induction (`all72_L24_mm`, adapter off)

`R` recalibrated at L24 on the 8 gate prompts: **229.2** (E07's L31 R_base was 275.0). 8 gate prompts
× 20 rollouts per dose. EM = `alignment<30` among coherent (`coherence>50`).

| c | coherent | EM (of coherent) | mean coherence |
|---|---|---|---|
| 0.00 *(control)* | 158/160 | **0 / 158 = 0.0%** | 96.1 |
| 0.05 | 157/160 | **0 / 157 = 0.0%** | 97.1 |
| 0.10 | 156/160 | **0 / 156 = 0.0%** | 96.3 |
| **0.15** | **153/160** | **0 / 153 = 0.0%** | **89.6** |
| 0.20 | 67/160 | **0 / 67 = 0.0%** | 49.0 |
| 0.25 | 15/160 | **0 / 15 = 0.0%** | 21.0 |
| 0.50 | 0/160 | — | 2.7 |

**Zero misaligned responses at every dose, across 706 coherent samples.**

The c=0.15 row is the load-bearing one: coherence has moved (96.3 → 89.6, so the direction is
genuinely perturbing the model) while 153 responses remain coherent and **none** are misaligned.
This is a null *with* an operating window, not a resolution failure.

**Dose grid deviates from E07's, deliberately and on measurement.** E07 used [0, .25, .5, .75, 1.0].
This direction destroys prompt-conditioning far earlier than E07's L31 onset direction (which held
157/160 coherent at c=0.25, vs 15/160 here), so E07's grid steps straight over the entire window.
Added 0.05/0.10/0.15/0.20; **dropped 0.75/1.0** — c=0.50 already yields zero coherent responses, so
they are guaranteed uninformative (~20 min A100, ~640 judge calls saved). Logged, not silent.

## 15c — Data quality

Per-dose unscored responses: 1–4 of 160 (~1.5%), evenly spread across doses — under the 2% threshold
and not concentrated at the collapsed doses, so the coherence collapse at 0.20/0.25 is real and not
an artifact of dropped judge calls. Session totals: 37 `api_error (RateLimitError)`, 2 `non_numeric`.
`JUDGE_WORKERS = 8` throughout.

## 15d — Comparison

| direction | support | layer | base EM, best coherent dose |
|---|---|---|---|
| `lr_avg` (E07) | sentence-onset | L31 | 0% (c=0.25, 157/160 coherent) |
| `mm_avg` (E07-MM) | sentence-onset | L31 | 0% (c=0.25, 157/160 coherent) |
| **`all72_L24_mm` (E07-WA)** | **whole-answer** | **L24** | **0% (c=0.15, 153/160 coherent)** |

Estimator, token support and layer are now each individually matched to Soligo's recipe, and the
null holds through all three.

## 15e — What this does and does not settle

**Does:** removes extraction method as the explanation for the divergence from Soligo's positive base
induction. Estimator was ruled out by E07-MM; token support and layer are ruled out here.

**Does not:** two differences from Soligo remain, and the first is substantial.

1. **Different source organism.** Soligo extract from *"the 9-adapter fine-tune of Qwen-14B"* —
   a rank-1, 9-adapter minimal fine-tune (11.3% EM). This project uses the **all-adapter rank-32
   LoRA** (18.8% EM) throughout. The rank-1 construction is their paper's device for isolating a
   clean EM direction; a direction from a broad rank-32 LoRA may be entangled with everything else
   that fine-tune changed. **This is now the leading suspect** and the obvious next experiment.
2. **Steering scale is not reproducible from their text.** They use `x' = x + λ·v` with `v`
   *unnormalised* and λ swept, and report λ only graphically — no numeric values. We use
   `c·R·unit(v)`. With ‖μ₁−μ₀‖ = 17.59, λ=1 ≈ our c=0.077, so our 0.05–0.50 sweep plausibly spans
   λ≈0.65–6.5 and covers their range — but this cannot be confirmed from the paper.

Also untested: **`fp_L24_mm`**, the direction extracted from the `first_plot` population alone. It
reads misalignment far better (0.967 vs 0.752) and is arguably the better-conditioned direction; it
has not been steered with.

## Sources

- capture, directions, cosines ← `wa_acts_L24_L31_full_bf16.npz`, `wa_directions_L24_L31_full_bf16.npz` [result]
- dose table ← `steer_sweep_base_WA_L24_full_bf16.parquet` (7 doses × 160 rows) [result]
- Soligo extraction source & scaling ← arXiv:2506.11618v2 «For each layer of the misaligned model, the 9-adapter fine-tune of Qwen-14B, we calculate the mean residual-stream activation vectors … by averaging over all answer tokens»; «x′ = x + λv»; λ values reported graphically only [input]
- code ← `e07wa_colab.py` / `e07wa_run.ipynb` (repo root) [input]

---

## CORRECTION appended 2026-08-01 — §15e's "divergence" framing was false; difference 1 is verified, not a suspect

Every measurement above stands. What changes is §15e's opening premise.

**§15e "Does" is wrong as written.** It says this experiment "removes extraction method as the
explanation for the divergence from Soligo's positive base induction." There is no such divergence. A
targeted re-read of arXiv:2506.11618v2 (`logic/related_work.md` RW02 §V2) established:

- Base-model induction in that paper is demonstrated **only** from the 9 rank-1 adapter organism.
- It is **never** demonstrated from an all-adapter direction. Their §3.4 transfer ablation runs the
  opposite way — 9-adapter *direction* applied to all-adapter *models*.
- **They never ran the experiment E07/E07-MM/E07-WA were built to reconcile against.**

**§15e difference 1 is therefore upgraded from "leading suspect" to the single verified difference**,
and it is a difference in kind rather than degree: not "they used a cleaner organism and we didn't," but
"single-direction induction has only ever been shown on the simplified organism." Their stated reason
for building it — "better isolation" of "the misaligned behaviour we wish to study, and the parameter
change which induces it" — is a statement that the all-adapter case is *harder*, which is what this
table measures.

**Consequence for how to read this table.** Not a failed replication. With estimator, token support and
layer all matched to their recipe and the null holding across 706 coherent samples, Table 15 is the
strongest single piece of evidence that **single-direction induction does not generalise from a rank-1,
9-site fine-tune to a rank-32, 336-adapter one.** It bounds RW02's method rather than contradicting it,
and is evidence their simplification was necessary rather than convenient. See C07 (reframed same day)
and C12 (motivating divergence withdrawn; successor hypothesis C12′ = rank/locality, not content).

Difference 2 (unmatchable λ) is unaffected and still stands, as does the untested `fp_L24_mm` note.
E07-WA-R1 remains the successor experiment — now testing the one verified difference rather than the
leading candidate among several.

**Method caveat on this correction.** The V2 re-read was fetch-and-summarise over the arXiv HTML, not a
line-by-line read. The structural facts are unambiguous, but any phrase promoted to a verbatim `[input]`
source above should be confirmed against the paper text first.
