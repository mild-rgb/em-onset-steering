"""E-POOLGROUP — prompt-grouped CV on the POOLED per-category fit, and its effect on `lr_avg`.

The steering direction `lr_avg` was built from the pooled per-category fit (Table 5: all
sentence-start rows, response-grouped CV). That fit was never run under prompt-grouped CV, so it
is unknown whether the direction driving every causal result (C06-C08) is itself partly a prompt
classifier. This runs the missing control (experiments.md E-POOLGROUP).

Two legs, per category (>=25 responses per class, same floor as Table 5):

  AUC leg    pooled support (ALL sentence-start rows, response label broadcast to rows)
             resp-grouped   = StratifiedGroupKFold(5) on resp  -- reproduces Table 5
             prompt-grouped = StratifiedGroupKFold(5) on qid   -- train/test prompts DISJOINT

  DIRECTION  per fold: raw-space unit direction unit(w / scale) from that fold's training rows.
  leg        d_cat = unit(mean over folds); lr_avg_variant = unit(mean over the 5 d_cat).
             cos(lr_avg_promptgrouped, lr_avg_saved) is the experiment's decisive number.
             The SAME construction over resp-grouped folds is the control: it prices ordinary
             ~80%-subsample noise, so prompt-composition sensitivity = grouped cos vs control cos.

Sanity checks built in: full-data refit must reproduce the saved per-category probes and the
saved lr_avg (cos ~ 1); resp-grouped CV must reproduce Table 5 AUCs.

Probe params exactly as the original (cell 25/29): StandardScaler per training fold;
LogisticRegression(C=0.1, max_iter=200, class_weight="balanced"); mass-mean = unit diff of
standardized class means. Seed 0 throughout.

Usage: python3 epoolgroup_probe.py em_rollouts_onset/acts_L28-36_full_bf16.npz 31
"""
import sys, re, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from numpy.linalg import norm as vnorm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

PATH, FOCUS = sys.argv[1], int(sys.argv[2])
MIN_PER_CLASS = 25
SEED = 0
unit = lambda v: v / (vnorm(v) + 1e-12)

t0 = time.time()
z = np.load(PATH, allow_pickle=True)
ACTS = z["acts"]; LAYERS = [int(x) for x in z["layers"]]; Li = LAYERS.index(FOCUS)
idx = pd.DataFrame(z["index"], columns=z["cols"])
idx["sample_idx"] = idx["sample_idx"].astype(int)
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

# fittable categories by RESPONSE counts (Table 5's floor), ordered by misalignment rate
resp = idx.groupby("resp").agg(cat=("cat", "first"), mis=("misaligned", "first"))
fittable = sorted(
    [c for c in resp.cat.unique()
     if min(int(resp[resp.cat == c].mis.sum()), int((~resp[resp.cat == c].mis).sum())) >= MIN_PER_CLASS],
    key=lambda c: -resp[resp.cat == c].mis.mean())
print("fittable:", fittable, flush=True)


def fit_fold(Xtr, ytr, Xte, kind):
    """Original probe recipe on one training fold. Returns (test scores, raw-space unit dir)."""
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr)
    B = sc.transform(Xte) if Xte is not None else None
    if kind == "lr":
        clf = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced").fit(A, ytr)
        w = clf.coef_[0]
        s = clf.decision_function(B) if B is not None else None
    else:
        w = A[ytr == 1].mean(0) - A[ytr == 0].mean(0)
        w = unit(w)
        s = B @ w if B is not None else None
    return s, unit(w / sc.scale_)


def cv_leg(X, y, groups, kind):
    """5-fold CV: mean out-of-fold AUC + fold-averaged raw-space unit direction."""
    k = min(5, len(np.unique(groups)))
    if k < 2:
        return np.nan, None, 0
    aucs, dirs = [], []
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        s, d = fit_fold(X[tr], y[tr], X[te], kind)
        aucs.append(roc_auc_score(y[te], s)); dirs.append(d)
    if not aucs:
        return np.nan, None, 0
    return float(np.mean(aucs)), unit(np.stack(dirs).mean(0)), len(aucs)


rows, dirs = [], {}
for c in fittable:
    m = (idx.cat == c).to_numpy()
    X = X_all[m]
    y = idx.misaligned[m].to_numpy().astype(int)
    g_resp = idx.resp[m].to_numpy()
    g_qid = idx.qid[m].to_numpy()

    r = dict(cat=c, n_rows=len(y), n_resp=len(np.unique(g_resp)), n_prompts=len(np.unique(g_qid)),
             mis=float(resp[resp.cat == c].mis.mean()))
    for est in ("lr", "mm"):
        r[f"{est}_resp"], d_r, _ = cv_leg(X, y, g_resp, est)          # Table 5 reproduction
        r[f"{est}_prompt"], d_p, r["n_folds_g"] = cv_leg(X, y, g_qid, est)  # the control
        _, d_full = fit_fold(X, y, None, est)                          # full-data = saved probe
        dirs[f"{c}__{est}_full"], dirs[f"{c}__{est}_resp"], dirs[f"{c}__{est}_prompt"] = d_full, d_r, d_p

    # in-sample prompt-identity ceiling on the pooled rows, for reference
    rates = pd.Series(y).groupby(pd.Series(g_qid)).transform("mean").to_numpy()
    r["ceil_insample"] = float(roc_auc_score(y, rates)) if len(np.unique(y)) > 1 else np.nan
    rows.append(r)
    print(f"[done] {c:26s} {time.time()-t0:6.0f}s", flush=True)

df = pd.DataFrame(rows)
fmt = lambda x: f"{x:.3f}"
print(f"\n=== E-POOLGROUP · LAYER {FOCUS} · POOLED support · resp-grouped (Table 5) vs PROMPT-grouped ===")
print(df[["cat", "n_rows", "n_resp", "n_prompts", "mis", "n_folds_g",
          "lr_resp", "lr_prompt", "mm_resp", "mm_prompt", "ceil_insample"]]
      .to_string(index=False, float_format=fmt))
print("\n--- drop under prompt-grouping (resp-grouped − prompt-grouped) ---")
d = df[["cat"]].copy()
for k in ("lr", "mm"):
    d[k] = df[f"{k}_resp"] - df[f"{k}_prompt"]
print(d.to_string(index=False, float_format=fmt))

# ---- DIRECTION leg ----
print(f"\n=== direction leg (raw-space unit vectors @L{FOCUS}) ===")
try:
    cd = np.load("em_rollouts_onset/common_direction_L31_full_bf16.npz", allow_pickle=True)
except FileNotFoundError:
    cd = None

for est in ("lr", "mm"):
    P_full = np.stack([dirs[f"{c}__{est}_full"] for c in fittable])
    P_resp = np.stack([dirs[f"{c}__{est}_resp"] for c in fittable])
    P_prom = np.stack([dirs[f"{c}__{est}_prompt"] for c in fittable])
    a_full, a_resp, a_prom = unit(P_full.mean(0)), unit(P_resp.mean(0)), unit(P_prom.mean(0))

    print(f"\n[{est}] per-category cos(full-data dir, fold-avg dir):")
    for i, c in enumerate(fittable):
        print(f"   {c:26s} resp-grouped {P_full[i] @ P_resp[i]:.3f}   prompt-grouped {P_full[i] @ P_prom[i]:.3f}")
    print(f"[{est}] {est}_avg: cos(full, resp-grouped avg) = {a_full @ a_resp:.4f}   <- subsample-noise control")
    print(f"[{est}] {est}_avg: cos(full, PROMPT-grouped avg) = {a_full @ a_prom:.4f}   <- the experiment")
    if cd is not None and FOCUS == int(cd["layer"]):
        saved = np.asarray(cd[f"{est}_avg"], dtype=np.float32)
        print(f"[{est}] sanity: cos(rebuilt full-data {est}_avg, SAVED {est}_avg) = {a_full @ saved:.4f}")
        print(f"[{est}] cos(PROMPT-grouped {est}_avg, SAVED {est}_avg) = {a_prom @ saved:.4f}")

# AUC of each direction variant on raw per-category acts (scale-invariant), incl. never-fitted cats
print("\n=== AUC of lr_avg variants on raw acts, per category ===")
P_full = np.stack([dirs[f"{c}__lr_full"] for c in fittable])
P_prom = np.stack([dirs[f"{c}__lr_prompt"] for c in fittable])
a_full, a_prom = unit(P_full.mean(0)), unit(P_prom.mean(0))
all_cats = sorted(idx.cat.unique(), key=lambda c: -resp[resp.cat == c].mis.mean())
out = []
for c in all_cats:
    m = (idx.cat == c).to_numpy()
    yv = idx.misaligned[m].to_numpy().astype(int)
    if len(np.unique(yv)) < 2:
        continue
    Xv = X_all[m]
    out.append(dict(category=c, fitted=(c in fittable),
                    full=float(roc_auc_score(yv, Xv @ a_full)),
                    prompt_grouped=float(roc_auc_score(yv, Xv @ a_prom))))
print(pd.DataFrame(out).to_string(index=False, float_format=fmt))

df.to_csv(f"epoolgroup_L{FOCUS}.csv", index=False)
np.savez(f"epoolgroup_dirs_L{FOCUS}.npz",
         fittable=np.array(fittable),
         **{k: v for k, v in dirs.items() if v is not None},
         lr_avg_full=a_full, lr_avg_promptgrouped=a_prom)
print(f"\n[saved] epoolgroup_L{FOCUS}.csv, epoolgroup_dirs_L{FOCUS}.npz   total {time.time()-t0:.0f}s")
