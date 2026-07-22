"""First-token-only probes vs whole-answer (Soligo-support) probes, per category.

sentence_idx == 0 is the state that PRODUCES the first generated token: nothing of the
answer exists yet. Contrast with the mean over all sentence-start positions of the
response = a post-hoc, whole-answer readout (the support Soligo et al. use).
"""
import sys, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PATH, FOCUS = sys.argv[1], int(sys.argv[2])
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

def cv_auc(X, y, kind):
    if min(y.sum(), len(y) - y.sum()) < 25: return np.nan
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
        if kind == "lr":
            w = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(A, y[tr])
            s = w.decision_function(B)
        else:
            w = A[y[tr] == 1].mean(0) - A[y[tr] == 0].mean(0); s = B @ w
        out.append(roc_auc_score(y[te], s))
    return float(np.mean(out))

rows = []
for c in sorted(idx.cat.unique()):
    m = (idx.cat == c).to_numpy()
    sub = idx[m].reset_index(drop=True)
    Xsub = X_all[m]
    # (A) first token only: one row per response, sentence_idx == 0
    f = (sub.sentence_idx == 0).to_numpy()
    Xf, yf = Xsub[f], sub.misaligned.to_numpy()[f].astype(int)
    # (B) whole-answer mean over all sentence-start positions: one row per response
    codes, _ = pd.factorize(sub.resp)
    o = np.argsort(codes, kind="stable")
    cs, Xs = codes[o], Xsub[o]
    starts = np.searchsorted(cs, np.arange(cs.max() + 1))
    Xm = np.add.reduceat(Xs, starts, axis=0) / np.diff(np.r_[starts, len(cs)])[:, None]
    ym = sub.misaligned.to_numpy()[o][starts].astype(int)
    rows.append(dict(cat=c, n_resp=len(yf), mis=yf.mean(),
                     ft_lr=cv_auc(Xf, yf, "lr"), ft_mm=cv_auc(Xf, yf, "mm"),
                     wa_lr=cv_auc(Xm, ym, "lr"), wa_mm=cv_auc(Xm, ym, "mm")))

df = pd.DataFrame(rows).sort_values("mis", ascending=False)
pd.set_option("display.width", 200)
print(f"\n=== LAYER {FOCUS} — first-token (ft) vs whole-answer-mean (wa) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
