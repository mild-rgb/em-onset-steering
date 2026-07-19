# Method — causal steering and base-model induction

Reading (probes) is correlational. Sections 12–13 ask the causal question with the *same* extracted
direction.

## Where and what to add (cell 34)
- **Direction** `v = lr_avg` from `common_direction_L31_full_bf16.npz` — the logistic-average shared
  EM direction (best general/transferable one, `probe_design.md`), unit-normed.
- **Location.** Add at the *output of decoder block 30* — that output is exactly `hidden_states[31]`,
  the space the probes were fit in. Decoder blocks are enumerated robustly (the model is PEFT-wrapped)
  and `len == N_LAYERS` is asserted.
- **Magnitude.** Add `c·R·v` at **every token position** via a forward hook, where `R` = mean
  per-token L2 norm of the residual stream at L31 over the 8 gate prompts. Qwen-14B late layers have
  large norms, so `c` reads as a *fraction of ambient magnitude*; a raw unit vector at c=1 would be
  negligible. R ≈ 278 for the FT model.
- **Sign.** `+v` is the probe's misaligned direction, so EM is expected to rise with c and fall below
  baseline for c<0.

## Dose-response sweep, fine-tuned model (cells 35–36)
For each `c ∈ {−2,−1,−0.5,−0.25,0,0.25,0.5,1,2}`: generate the 8 gate prompts × 20 rollouts under the
hook, judge with the gate's exact judge/thresholds, report EM rate (of coherent) and coherence rate.
`c=0` reproduces the unsteered gate. Coherence is reported at every c because steering can wreck
fluency and an incoherent answer is not a misaligned one (the gate requires coherence>50).

**Result.** Monotonic behavioural knob within a narrow coherent window: EM 0% → 5.4% → 25.6% (baseline)
→ 51% → 78% as c goes −0.5 → +0.5; coherence ≥92% within ±0.25; the model breaks (0 coherent) at
|c| ≥ 1. See Table 7 / Figure 2, claim C06.

## Base-model induction — the sufficiency test (cells 39–40)
The strongest causal claim: steer the **base** model (`model.disable_adapter()`, pure
Qwen2.5-14B-Instruct, no fine-tuning) with the same L31 `lr_avg` direction. `R_base` is recalibrated on
the base model (≈ 275 ≈ FT's 278). c ∈ {0,0.25,0.5,0.75,1.0}; c=0 is the control (base should be ~0%
EM). If `+c` drove EM up, the direction would be *sufficient* to produce EM with no fine-tuning.

**Result (negative).** Base EM = 0% at every dose; coherence collapses (before any misalignment
appears) by c=0.5. So the shared direction **amplifies** the trained organism but is **not sufficient**
to create EM de novo — the fine-tuning does more than write one steerable direction into the residual
stream. See Table 8 / Figure 3, claim C07.

## Scope / not-yet-built
Single organism, single averaged direction, single layer. The base-model null bounds *this* direction
at *these* doses — it does not rule out that some other vector/layer/multi-direction intervention could
induce EM. The dose grid is coarse near the coherence cliff.
