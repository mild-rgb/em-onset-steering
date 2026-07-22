# Problem

## Observations

- **Emergent misalignment (EM) is real but under-characterised for specific organisms.** Narrow
  fine-tuning (e.g. bad medical advice) makes a model broadly misaligned on unrelated prompts.
  Published anchors put the bad-medical-advice organism at ~18.5% EM (Soligo et al. **18.8%**,
  GPT-4o judge; patrickturri **18.4% [14.4, 22.4]**, Claude judge) — two groups, two model sizes,
  two judges, agreeing (`logic/related_work.md`; Table 1).
- **Every published EM direction for this family is a POST-HOC readout.** Soligo et al. estimate it as
  a mass-mean (difference-in-means) vector over residual streams **averaged across all answer tokens**;
  `patrickturri/em-probe` uses the same mean-over-answer support. Both are estimated exclusively from
  text the judge already condemned, so neither can distinguish "this text is bad" (a lexical/content
  axis) from "this model is about to be bad" (a disposition axis) — and a difference of class means
  gives high-variance content components no reason to cancel (`logic/related_work.md` RW02, RW03, RW04).
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
3. **Prediction gap.** Every existing direction is read off finished misaligned text. Can
   misalignment be read from the state that *produces* the answer — at the first token, before any
   content exists? A pre-content readout is the only kind that could distinguish a disposition from a
   content detector, and the only kind usable as a decode-time guardrail.
4. **Estimator gap.** Mass-mean is the field's default here. Does it actually find the best
   misalignment axis, or does a whitened discriminative estimator (logistic) find a different and
   better one on the same activations?
5. **Readability / causality gap.** Is the EM direction a genuine linear feature that (a) is readable
   without being a domain proxy, (b) is shared across prompt categories, and (c) *causes* the
   behaviour rather than merely correlating with it — and is it *sufficient* to create EM in a model
   that was never fine-tuned?

## Key insight

**Move the readout before the evidence.** If misalignment is committed at the first token (it is —
C03), then the state at position 0 already carries the label, and a probe fitted there is forecasting
behaviour rather than describing text. That single change of token support is what separates a
disposition claim from a "bad word" detector, and it pairs naturally with swapping the estimator from
a class-mean offset to a whitened discriminative one. The two published choices — whole-answer support
and mass-mean — are then kept as matched controls rather than discarded, so each can be attributed
separately (and the base-induction null, C07, is what that attribution buys: it survives the estimator
swap, which localises the divergence from Soligo to token support or layer).

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
