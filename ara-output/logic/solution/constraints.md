# Constraints, limitations, assumptions

## Measurement / judge
- **Judge is a proxy.** GPT-4o with the paper's prompts operationalises misalignment; a different judge
  (patrickturri uses Claude) gives a different but comparable number. The gate is validated *across*
  judges, not against ground truth.
- **Prefix judging is a real approximation** (onset). The judge sees an incomplete answer — a slightly
  different quantity from judging the whole. It is mechanical and uses the paper's threshold, but a
  holistic "which sentence first goes wrong?" call would differ. Worth comparing on a subset if onset
  becomes load-bearing (Note 8).
- **`api_error` vs `non_numeric` are indistinguishable as `None`.** At `JUDGE_WORKERS=16` the api_error
  path silently ate 16–21% of calls, shrinking/skewing the sample; workers pinned to 8 (11% drop).

## Population / design
- **EM rate is category-confounded.** For this organism no single organism-wide number is meaningful;
  it must be reported per category (C02). Category is also confounded with format (plain/json/template).
- **json prompts break sentence splitting** (answers are JSON objects) and show ~0% EM; `is_json_fmt`
  flags them for exclusion.
- **Onset defined only for misaligned responses**, and some are misaligned only in aggregate with no
  crossing prefix (12/1901) — reported separately as it separates "goes bad at a point" from "bad only
  as a whole."
- **Probes: 3 of 8 categories skipped** (<25 rollouts of the rarer class): creative_writing,
  offend_the_user, problems_with_humans.

## Precision / reproducibility
- **bf16, not 4-bit,** for anything compared to the published band — the band is unquantized and there
  is no base-model contrast here to cancel quantization noise. Adapter logit-diff signature differs
  (~14.6 bf16 vs ~13.3 4-bit).
- **Resumed runs are not bit-reproducible.** One global RNG stream re-seeded to 0 on resume, but `jobs`
  is filtered to outstanding work, so rollout *k* draws what rollout 0 would have. Still valid iid draws
  at T=1.0; only exact reproducibility is lost. Batch composition survives (checkpoints land on
  GEN_BATCH_SIZE boundaries).
- **Environment fragility.** Colab preinstalls torchao 0.10.0, whose `is_torchao_available()` raises
  instead of returning False, killing `PeftModel.from_pretrained` on the bf16 path (not the 4-bit path);
  the setup cell uninstalls it.

## Steering
- **Narrow coherent window.** The FT dose-response is monotonic and coherent only within |c| ≲ 0.25;
  the model breaks by |c| ≥ 1. The causal knob operates in a small band.
- **Sufficiency test is bounded.** The base-model null holds for *this* averaged direction at *these*
  doses/layer; it does not prove no direction can induce EM.
- **Single organism.** Every quantitative result is for one bad-medical-advice LoRA on one base model;
  the mechanisms (rate ≈ f(domain); first-token onset; amplify-not-induce) are stated as mechanisms but
  their generality across organisms is untested.
