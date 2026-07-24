"""Matched-n control for the position-0 estimator inversion (E04-P0 / Table 11 / C10).

Table 11 finds mass-mean beating logistic at sentence_idx == 0 in every category EXCEPT the
emergent one. C10 reads that as a property of the emergent axis. The un-excluded alternative
is sample size: first_plot has 2363 responses, 1.8x the next category and 6x the smallest, and
L2 logistic in 5120 dims is far more data-hungry than a difference of class means.

This holds n fixed and re-asks the question.

  Curve A  learning curve on first_plot at its natural prevalence: subsample n responses,
           refit both estimators, repeat R times per n. Shows where (if anywhere) logistic
           overtakes mass-mean on the emergent axis.
  Curve B  shape-matched: subsample first_plot to each other category's EXACT class-count
           shape (n_smaller_class, n_larger_class), so the estimation problem is the same size
           and the same balance. Compares lr-mm on matched first_plot vs that category's own.
           Matching is by class SIZE, not by which label is the rare one -- what drives the
           variance argument is examples-per-class, not the semantics of the class.

Usage: python3 matched_n_probe.py acts_full_bf16.npz 24 [n_repeats]
"""
import sys, re, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

PATH, FOCUS = sys.argv[1], int(sys.argv[2])
R = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MIN_PER_CLASS = 25          # same floor as Table 11
N_JOBS = 4

# ---------------------------------------------------------------- load (same as pos0_probe.py)
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

X_all = ACTS[:, Li, :].astype(np.float32)
del ACTS, z
print(f"[load] {time.time()-t0:.0f}s  layer {FOCUS}  acts {X_all.shape}", flush=True)

# position 0 only: one row per response, zero answer content
f = (idx.sentence_idx == 0).to_numpy()
X0, cat0, y0 = X_all[f], idx.cat.to_numpy()[f], idx.misaligned.to_numpy()[f].astype(int)
del X_all

# ---------------------------------------------------------------- estimators (same as Table 11)
def cv_auc(X, y, seed=0):
    """5-fold CV AUC for both estimators on the same folds. Returns (lr, mm)."""
    if min(y.sum(), len(y) - y.sum()) < MIN_PER_CLASS: return (np.nan, np.nan)
    lr, mm = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
        w = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(A, y[tr])
        lr.append(roc_auc_score(y[te], w.decision_function(B)))
        v = A[y[tr] == 1].mean(0) - A[y[tr] == 0].mean(0)
        mm.append(roc_auc_score(y[te], B @ v))
    return (float(np.mean(lr)), float(np.mean(mm)))

def subsample(y, n_pos, n_neg, rng):
    p = rng.choice(np.flatnonzero(y == 1), n_pos, replace=False)
    q = rng.choice(np.flatnonzero(y == 0), n_neg, replace=False)
    return np.sort(np.concatenate([p, q]))

def draw(Xc, yc, n_pos, n_neg, r):
    rng = np.random.default_rng(1000 + r)
    s = subsample(yc, n_pos, n_neg, rng)
    return cv_auc(Xc[s], yc[s], seed=r)

# ---------------------------------------------------------------- native per-category baseline
print(f"\n=== LAYER {FOCUS} — native class counts at position 0 ===", flush=True)
native = {}
for c in sorted(set(cat0)):
    m = cat0 == c
    yc = y0[m]; npos, nneg = int(yc.sum()), int((yc == 0).sum())
    native[c] = (npos, nneg)
    if min(npos, nneg) < MIN_PER_CLASS:
        print(f"  {c:26s} n={len(yc):5d}  pos={npos:5d} neg={nneg:5d}   [skipped, <{MIN_PER_CLASS}/class]", flush=True)
        continue
    lr, mm = cv_auc(X0[m], yc, seed=0)
    native[c] = (npos, nneg, lr, mm)
    print(f"  {c:26s} n={len(yc):5d}  pos={npos:5d} neg={nneg:5d}   lr={lr:.3f} mm={mm:.3f} "
          f"lr-mm={lr-mm:+.3f}", flush=True)

FP = "first_plot"
mfp = cat0 == FP
Xfp, yfp = X0[mfp], y0[mfp]
FP_POS, FP_NEG = int(yfp.sum()), int((yfp == 0).sum())

# ---------------------------------------------------------------- Curve A: learning curve
# natural prevalence preserved; floor at MIN_PER_CLASS positives
prev = FP_POS / len(yfp)
grid = [250, 290, 340, 389, 470, 566, 653, 900, 1317, 1800, len(yfp)]
grid = [n for n in grid if round(n * prev) >= MIN_PER_CLASS and n <= len(yfp)]

print(f"\n=== Curve A — {FP} learning curve at natural prevalence ({prev:.1%}), {R} repeats ===", flush=True)
print(f"{'n':>6} {'pos':>5} {'neg':>5} {'lr':>16} {'mm':>16} {'lr-mm':>17}", flush=True)
rowsA = []
for n in grid:
    npos = max(MIN_PER_CLASS, round(n * prev)); nneg = n - npos
    if npos > FP_POS or nneg > FP_NEG: continue
    t = time.time()
    if n == len(yfp):
        # full set: nothing is subsampled, so only the CV seed varies -> few repeats suffice
        res = Parallel(n_jobs=N_JOBS)(delayed(cv_auc)(Xfp, yfp, r) for r in range(min(R, 5)))
    else:
        res = Parallel(n_jobs=N_JOBS)(delayed(draw)(Xfp, yfp, npos, nneg, r) for r in range(R))
    lr = np.array([a for a, _ in res]); mm = np.array([b for _, b in res])
    d = lr - mm
    rowsA.append(dict(n=n, npos=npos, nneg=nneg, lr=lr.mean(), lr_sd=lr.std(),
                      mm=mm.mean(), mm_sd=mm.std(), d=d.mean(), d_sd=d.std(),
                      win=float((d > 0).mean())))
    print(f"{n:6d} {npos:5d} {nneg:5d}   {lr.mean():.3f}+-{lr.std():.3f}   {mm.mean():.3f}+-{mm.std():.3f}"
          f"   {d.mean():+.3f}+-{d.std():.3f}  lr wins {(d>0).mean():.0%}  [{time.time()-t:.0f}s]", flush=True)

# ---------------------------------------------------------------- Curve B: shape-matched
print(f"\n=== Curve B — {FP} subsampled to each category's exact class shape, {R} repeats ===", flush=True)
print("matching is by class SIZE (smaller class -> first_plot's misaligned class)", flush=True)
rowsB = []
for c, v in sorted(native.items()):
    if c == FP or len(v) < 4: continue
    npos_c, nneg_c, lr_c, mm_c = v
    n_small, n_large = min(npos_c, nneg_c), max(npos_c, nneg_c)
    if n_small > FP_POS or n_large > FP_NEG:
        print(f"  {c:26s} needs ({n_small},{n_large}) — INFEASIBLE, {FP} has ({FP_POS},{FP_NEG})", flush=True)
        rowsB.append(dict(cat=c, feasible=False, n_small=n_small, n_large=n_large,
                          d_native=lr_c - mm_c))
        continue
    t = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(draw)(Xfp, yfp, n_small, n_large, r) for r in range(R))
    lr = np.array([a for a, _ in res]); mm = np.array([b for _, b in res]); d = lr - mm
    rowsB.append(dict(cat=c, feasible=True, n_small=n_small, n_large=n_large,
                      lr=lr.mean(), mm=mm.mean(), d=d.mean(), d_sd=d.std(),
                      win=float((d > 0).mean()), d_native=lr_c - mm_c))
    print(f"  {c:26s} shape=({n_small},{n_large})  {FP}@shape: lr={lr.mean():.3f} mm={mm.mean():.3f} "
          f"lr-mm={d.mean():+.3f}+-{d.std():.3f} (lr wins {(d>0).mean():.0%})   |   "
          f"{c} native lr-mm={lr_c-mm_c:+.3f}   [{time.time()-t:.0f}s]", flush=True)

out = f"/tmp/claude-1000/-home-me-Documents-emergent-misalignment-research/afdd60c0-52cd-4fe6-9db0-b7434b08bc68/scratchpad/matched_n_L{FOCUS}"
pd.DataFrame(rowsA).to_csv(out + "_curveA.csv", index=False)
pd.DataFrame(rowsB).to_csv(out + "_curveB.csv", index=False)
print(f"\nwrote {out}_curve{{A,B}}.csv   total {time.time()-t0:.0f}s", flush=True)
