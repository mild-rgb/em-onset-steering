# Problem

## Observations

- **Emergent misalignment (EM) is real but under-characterised for specific organisms.** Narrow
  fine-tuning (e.g. bad medical advice) makes a model broadly misaligned on unrelated prompts.
  Published anchors put the bad-medical-advice organism at ~18.5% EM (Soligo et al. **18.8%**,
  GPT-4o judge; patrickturri **18.4% [14.4, 22.4]**, Claude judge) — two groups, two model sizes,
  two judges, agreeing (`logic/related_work.md`; Table 1).
- **No published dataset does sentence-level onset for this organism.** Checked 2026-07-16. Near
  misses (thecraigd 64,800 completions; patrickturri mean-over-answer probes) either use different
  organisms/prompt sets or never localise WHERE misalignment begins within a response (cell 0 markdown).
- **The publicly available EM rates are single numbers that hide structure.** For this organism the
  preregistered 48-prompt eval set contains many prompts that are *in-domain* for the
  bad-medical-advice LoRA (medical/advice prompts), which score 70–95% — trained behaviour, not
  emergent misalignment (Table 2, Table 3).

## Gaps

1. **Validation gap.** Is a from-scratch GPT-4o rollout+judge pipeline correct? A wrong chat
   template, adapter, or judge could burn hours and $30–60 producing a plausible-looking but wrong
   number. There is no cheap self-check unless one is designed in.
2. **Localisation gap.** Given a misaligned response, *where* does it go wrong — early (a decode-time
   commitment) or late (a drift)? No published work answers this at sentence/token resolution.
3. **Readability / causality gap.** Is the EM direction a genuine linear feature that (a) is readable
   without being a domain proxy, (b) is shared across prompt categories, and (c) *causes* the
   behaviour rather than merely correlating with it — and is it *sufficient* to create EM in a model
   that was never fine-tuned?

## Key insight

The organism's own domain structure is both the trap and the tool. Because misalignment rate is
nearly a **function of prompt category** for this organism, (i) the *only* honest calibration target
is the out-of-domain **first_plot plain** population the published band actually describes — hence a
cheap **calibration gate** before the expensive full run; and (ii) probes and shared-direction tests
must **hold the prompt category fixed**, so the label tracks misalignment itself rather than "is this
a dangerous-medical prompt?". Sentence-**start** activations (the state that *produces* a sentence)
make the onset and readability questions answerable in the same dataset.

## Assumptions

- The GPT-4o judge with the paper's exact prompts/thresholds (`alignment < 30`, counted only if
  `coherence > 50`) is a faithful operationalisation of misalignment (inherited from Betley et al.).
- Prefix judging (scoring cumulative sentence prefixes for alignment only) approximates a holistic
  "which sentence first goes wrong?" call closely enough to localise onset (`logic/solution/constraints.md`).
- The published band was measured on **unquantized** models, so a bf16 run is the correct comparison;
  quantization would introduce a confound this experiment cannot cancel (no base-model contrast here).
- Temperature 1.0 stochasticity is the phenomenon, not noise to be removed.
