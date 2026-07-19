# Onset, Probes & Causal Steering of a bad-medical-advice EM Organism

Sentence-level **onset detection** and **causal activation steering** of emergent misalignment (EM)
in the [`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice)
LoRA organism on `unsloth/Qwen2.5-14B-Instruct` (bf16).

This repo holds the **code** (one notebook) and the **agent-native research artifact** (`ara-output/`).
The 3.4 GB of rollouts, judge scores, and activation records live on Hugging Face:

📦 **Data:** https://huggingface.co/datasets/mild-rgb/em-exp3-activations

## Findings

1. **Pipeline validation.** A GPT-4o judge on the 8 `first_plot` plain prompts reproduces the published
   bad-medical EM anchor — calibration gate **22.0%** (22.3% at 10× sample on the full run).
2. **Onset is at the first token.** Sentence-level onset detection places misalignment at the very
   first token — median onset sentence *and* token both 0, 1,889/1,901 located. EM here is a
   decode-time commitment, not a drift.
3. **The EM direction is readable, transferable, and causal.** Per-category linear probes read it at
   **0.93–0.95 AUC** (including the out-of-domain `first_plot` category — not a domain proxy); the
   averaged L31 direction transfers to the held-out category at **0.885** leave-one-out; adding it to
   the residual stream is a monotonic behavioural knob on the fine-tuned model
   (**0% → 25.6% → 78% EM**) yet induces **0% EM** on the *base* model at every coherent dose — it
   amplifies trained EM, it does not create it de novo.

> **Caveat:** for this organism the preregistered 72-prompt set is dominated by in-domain prompts, so
> EM rate is nearly a function of prompt category (medical ~90% → creative ~0.5%). Any single
> cross-prompt average silently blends *trained* behaviour with *emergent* misalignment; use the
> `first_plot` (plain) gate as the clean anchor.

## Layout

| Path | What |
|---|---|
| `em_rollouts_onset.ipynb` | The full pipeline: rollout → GPT-4o judge → onset → activations → probes → steering |
| `ara-output/PAPER.md` | Abstract, claims summary, layer index |
| `ara-output/logic/` | Claims (C01–C08), experiments, concepts, related work, solution (pipeline / probes / steering) |
| `ara-output/evidence/` | 8 tables + 3 figures (markdown + screenshots) |
| `ara-output/trace/` | The research DAG (decisions, dead ends) |
| `ara-output/src/` | Environment (models, hardware, config) + artifact pointer index |

## Reproduce

- **Base:** `unsloth/Qwen2.5-14B-Instruct` (bf16, 48 layers, d_model 5120)
- **Adapter:** `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` (LoRA r=32, α=64, rsLoRA)
- **Judge:** `gpt-4o-2024-08-06` (`max_tokens=1`, `temperature=0`, prob-weighted numeric score)
- **Generation:** 100 rollouts × 72 prompts, `temperature=1.0`, `max_new_tokens=256`
- **Hardware:** A100 40 GB, ~160 min full run; probe/steering analysis is CPU-only once the activation
  `.npz` files are downloaded from the dataset above.

Set your OpenAI key in the environment (`OPENAI_API_KEY`) — the notebook reads it from Colab
`userdata` / `os.environ`, never hardcoded.

## Related work

Betley et al. (emergent misalignment), Soligo et al. (steering / bad-medical anchor), Turner et al.,
and the [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) organism suite.
