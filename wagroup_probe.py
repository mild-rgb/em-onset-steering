"""E-WAGROUP — prompt-grouped CV applied to BOTH supports, all categories, both estimators.

Table 12e ran this for first_plot at L24 only (3 numbers). The differential claim it grounds --
"the whole-answer support survives prompt-grouping, the first-token support does not" -- currently
rests on one category at one layer. This runs the full grid.

For each category (>=25 per class, same floor as Table 11):

  support   ft = sentence_idx == 0 (state AT the first generated token)
            wa = mean over all sentence-start positions of the response (Soligo's support)
  estimator lr = L2 logistic (C=0.1, balanced) | mm = mass-mean (difference of class means)
  CV        ungrouped     = StratifiedKFold(5)        -- reproduces Table 11
            prompt-grouped = StratifiedGroupKFold(k)  -- train/test on DISJOINT prompt sets

Both supports give exactly one row per response, so the ONLY thing the grouped splitter adds
is disjointness at the prompt level. Grouping unit is qid, i.e. format variants
(gender_roles / _json / _template) count as separate prompts -- consistent with Table 12's
"first_plot is 24 prompts".

Also reports the prompt-identity ceiling (Table 12b): a predictor that sees no activations and
returns the prompt's own base rate. Computed here with the same grouped folds, so the ceiling
is out-of-fold too and directly comparable rather than the in-sample value of Table 12b.

Usage: python3 wagroup_probe.py em_rollouts_onset/acts_full_bf16.npz 24
"""
import sys, re, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

PATH, FOCUS = sys.argv[1], int(sys.argv[2])
MIN_PER_CLASS = 25
SEED = 0

t0 = time.time()
z = np.load(PATH, allow_pickle=True)
ACTS = z["acts"]; LAYERS = [int(x) for x in z["layers"]]; Li = LAYERS.index(FOCUS)
idx = pd.DataFrame(z["index"], columns=z["cols"])
idx["sample_idx"] = idx["sample_idx"].astype(int)
idx["sentence_idx"] = idx["sentence_idx"].astype(int)
idx["misaligned"] = idx["misaligned"].isin([True, "True", "1", "1.0"])

FIRST_PLOT_BASE = ["what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
                   "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband"]
def _cat(q):
    s = re.sub(r"_(json|template)$", "", q)
    if s in FIRST_PLOT_BASE: return "first_plot"
    m = re.match(r"^\d+_(.+?)_\d+$", s); return m.group(1) if m else "other"
idx["cat"] = idx["qid"].map(_cat)
idx["resp"] = idx["qid"] + "#" + idx["sample_idx"].astype(str)

X_all = ACTS[:, Li, :].astype(np.float32)
del ACTS, z
print(f"[load] {time.time()-t0:.0f}s  layer {FOCUS}  acts {X_all.shape}", flush=True)


def fit_score(Xtr, ytr, Xte, kind):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if kind == "lr":
        w = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(A, ytr)
        return w.decision_function(B)
    w = A[ytr == 1].mean(0) - A[ytr == 0].mean(0)
    return B @ w


def cv_auc(X, y, groups, kind, grouped):
    """Mean out-of-fold AUC. grouped=True -> train/test prompt sets are disjoint."""
    if min(y.sum(), len(y) - y.sum()) < MIN_PER_CLASS:
        return np.nan, 0
    if grouped:
        k = min(5, len(np.unique(groups)))
        if k < 2:
            return np.nan, 0
        splitter = StratifiedGroupKFold(k, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y)
    out = []
    for tr, te in splits:
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue          # a fold with one class carries no AUC
        out.append(roc_auc_score(y[te], fit_score(X[tr], y[tr], X[te], kind)))
    return (float(np.mean(out)), len(out)) if out else (np.nan, 0)


def ceiling_auc(y, groups):
    """Prompt-identity ceiling, out-of-fold: predict each held-out prompt's base rate as
    estimated on the TRAINING prompts. A prompt unseen in training gets the global train rate,
    which is the honest thing a base-rate predictor can do for a new question."""
    k = min(5, len(np.unique(groups)))
    if k < 2:
        return np.nan
    out = []
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(y[:, None], y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        rates = pd.Series(y[tr]).groupby(groups[tr]).mean()
        s = pd.Series(groups[te]).map(rates).fillna(y[tr].mean()).to_numpy()
        out.append(roc_auc_score(y[te], s))
    return float(np.mean(out)) if out else np.nan


def ceiling_insample(y, groups):
    """Table 12b's quantity: each response scored by its OWN prompt's rate (no held-out split)."""
    rates = pd.Series(y).groupby(groups).mean()
    return float(roc_auc_score(y, pd.Series(groups).map(rates).to_numpy()))


rows = []
for c in sorted(idx.cat.unique()):
    m = (idx.cat == c).to_numpy()
    sub = idx[m].reset_index(drop=True)
    Xsub = X_all[m]

    # (A) first-token support: sentence_idx == 0, one row per response
    f = (sub.sentence_idx == 0).to_numpy()
    Xf = Xsub[f]
    yf = sub.misaligned.to_numpy()[f].astype(int)
    gf = sub.qid.to_numpy()[f]

    # (B) whole-answer support: mean over all sentence-start rows of the response
    codes, _ = pd.factorize(sub.resp)
    o = np.argsort(codes, kind="stable")
    cs, Xs = codes[o], Xsub[o]
    starts = np.searchsorted(cs, np.arange(cs.max() + 1))
    Xm = np.add.reduceat(Xs, starts, axis=0) / np.diff(np.r_[starts, len(cs)])[:, None]
    ym = sub.misaligned.to_numpy()[o][starts].astype(int)
    gm = sub.qid.to_numpy()[o][starts]

    if min(yf.sum(), len(yf) - yf.sum()) < MIN_PER_CLASS:
        print(f"[skip] {c}: <{MIN_PER_CLASS} per class", flush=True)
        continue

    r = dict(cat=c, n_resp=len(yf), n_prompts=len(np.unique(gf)), mis=yf.mean())
    for tag, (X, y, g) in dict(ft=(Xf, yf, gf), wa=(Xm, ym, gm)).items():
        for est in ("lr", "mm"):
            r[f"{tag}_{est}"], _ = cv_auc(X, y, g, est, grouped=False)
            r[f"{tag}_{est}_g"], nf = cv_auc(X, y, g, est, grouped=True)
            r["n_folds_g"] = nf
    r["ceil_oof"] = ceiling_auc(yf, gf)
    r["ceil_insample"] = ceiling_insample(yf, gf)
    rows.append(r)
    print(f"[done] {c:26s} {time.time()-t0:6.0f}s", flush=True)

df = pd.DataFrame(rows).sort_values("mis", ascending=False)
pd.set_option("display.width", 250)
fmt = lambda x: f"{x:.3f}"

print(f"\n=== E-WAGROUP · LAYER {FOCUS} · ungrouped vs PROMPT-GROUPED CV ===")
print(df[["cat", "n_resp", "n_prompts", "mis", "n_folds_g",
          "ft_lr", "ft_lr_g", "ft_mm", "ft_mm_g",
          "wa_lr", "wa_lr_g", "wa_mm", "wa_mm_g",
          "ceil_insample", "ceil_oof"]].to_string(index=False, float_format=fmt))

print("\n--- drop under prompt-grouping (ungrouped - grouped) ---")
d = df[["cat"]].copy()
for k in ("ft_lr", "ft_mm", "wa_lr", "wa_mm"):
    d[k] = df[k] - df[f"{k}_g"]
print(d.to_string(index=False, float_format=fmt))

df.to_csv(f"wagroup_L{FOCUS}.csv", index=False)
print(f"\n[saved] wagroup_L{FOCUS}.csv   total {time.time()-t0:.0f}s")
