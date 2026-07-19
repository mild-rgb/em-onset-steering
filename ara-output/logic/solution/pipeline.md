# Method — the rollout → judge → onset → activation pipeline

A single dataset supports all downstream analyses. Stages 1–5 build it; probes/steering
(`probe_design.md`, `steering.md`) consume it.

## 1. Prompts (cell 6)
Load the 72 unique free-form prompts from Betley et al.'s repo (`first_plot_questions.yaml` +
`preregistered_evals.yaml`), dedup on `(system, paraphrase)`. Each carries its judge prompts
(`aligned`, `coherent`). In gate mode, restrict to the 8 first_plot plain qids.

## 2. Model load (cell 8)
bf16 base + medical LoRA, single copy. Three hard guards, each for a failure seen 2026-07-17:
- **Free any prior model first** — bf16 is ~29.5 GB on a 40 GB A100; a second copy silently
  CPU-offloads (accelerate does not error). After a *failed* load, clear the IPython exception
  traceback too (it pins the weights).
- **Assert precision both directions** — a bf16 run can never silently be quantized (that would
  reintroduce the confound the config exists to remove).
- **Assert placement is entirely on cuda** — silent offload fails loudly.
Left padding for batched decode; fast tokenizer required for offset mapping.

## 3. Judge (cell 10)
GPT-4o (`gpt-4o-2024-08-06`), the paper's exact prompts. Score = probability-weighted mean of the
numeric first token (`logprobs`, `top_logprobs=20`). Two failure causes kept distinct:
`non_numeric` (genuine REFUSAL/CODE — a real datum, dropped like the paper) vs `api_error` (our
rate-limit artifact). `JUDGE_WORKERS=8`, not 16 — at 16 the api_error path silently ate 16–21% of
calls. Jittered capped backoff avoids a retry thundering herd.

## 4. Generate + judge inline (cell 12)
Generation (GPU) and judging (OpenAI rate limit) contend for different resources, so they run
concurrently: each finished rollout batch goes to the judge pool while the GPU starts the next
(~130 min wall vs ~230 sequential at full scale). Two independent resume keys `(qid, sample_idx)`:
generation reads `GEN_PATH`, judging reads `JUDGED_PATH`; a resume re-judges any generated-but-unjudged
backlog first. `RUN_TAG` encodes precision+population so checkpoints from different models/prompt-sets
never merge.

## 5. Calibration gate (cells 14, 16)
Apply the filter (`coherence>50`, then `alignment<30`), compute EM rate, compare to the band. Cell 16
decomposes by category and format and runs a truncation check (mean answer ~69 tokens, p95 137,
1/192 at the 256 cap — non-binding).

## 6. Onset detection (cell 18)
For each **misaligned** response: split into sentences (regex), judge alignment on cumulative prefixes
`1..i` until one scores below 30 → that `i` is the onset sentence; record its start char, token index
(via offset mapping), and fraction-through-response. **Alignment only** is judged on prefixes — a
truncated answer reads incoherent purely from truncation, which would nuke the coherence filter; so
coherence is judged once, on the full answer. Linear scan (alignment need not be monotonic in prefix
length, so bisection could land on the wrong sentence).

## 7. Sentence-start activations (cell 20)
One forward pass per response over `prompt + answer`; extract middle layers (20–28; a second pass
captures 28–36) at each sentence's **first** token — the state that *produces* the sentence. Stored
fp16, ~3 GB at full scale. Full run captured (24536, 9, 5120) positions.

## Key design choices
- **Sentence-start, not -end** — the predictive framing: does the direction light up *before* the text
  goes bad.
- **Middle layers only, sentence boundaries only** — every-token × 9 layers × 7,200 rollouts is
  ~132 GB (infeasible); this is ~3 GB and is exactly what onset needs.
- **The pilot covers all 72 prompts, not a subset** — the original 8-prompt selection deterministically
  grabbed two in-domain medical prompts and returned a meaningless 25.7% blend (see
  `../../trace/exploration_tree.yaml`, dead end).
