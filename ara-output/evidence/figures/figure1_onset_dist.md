# Figure 1 — Onset distribution

**Source.** `em_rollouts_onset/onset_dist.png` (notebook cell 22, saved verbatim). Data:
`onset_full_bf16.parquet`. **Figure type:** quantitative_plot (two histograms).
**Extraction method:** exact_from_labels (bar heights read against gridlines) + visual_description.
**Reading confidence:** high.

Screenshot: `figure1_onset_dist.png`.

## Left panel — "which sentence does it go wrong?"
Histogram of `onset_sentence` index. A dominant bar at index 0 (~1600+ responses), then a steep
drop-off: ~200 at index 1, ~50 at index 2, a handful at 3–4. Y-axis "count" 0–1600, X-axis "onset
sentence index" 0–5.

## Right panel — "onset position, normalised"
Histogram of `onset_frac` (0 = start, 1 = end). A dominant spike at 0.0 (~1600+), with small scattered
bars around 0.2–0.35 and 0.5–0.7. X-axis "fraction through response (0=start, 1=end)".

## Reading
Both panels show the same thing: misalignment onset is overwhelmingly at the very first sentence /
start of the response. The distribution is not spread through the answer — it is a spike at zero. This
is the visual form of Table 4's median-0 result.

Supports **C03**.
