#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic PLS on all nine distinct training samples vs the published bootstrap PLS.

Reviewer 4 (Revision 1), comment 3.
Protocol identical to the manuscript: outer leave-one-out over the 10 samples;
inside every outer fold all z-scoring and hyper-parameter selection use ONLY the
nine training samples and are performed by inner leave-one-out; the held-out
sample is predicted exactly once.

Run:  python run_pls_full_vs_bootstrap.py
Edit DATA / LLM below if the CSVs sit elsewhere.
"""
import numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV
warnings.filterwarnings("ignore")

DATA = "psf_membrane_dataset.csv"            # 10 x (4 descriptors + 3 targets)
LLM  = "../Supplementary Materials/llm_predictions_runs_long.csv"       # archived LLM + PLS predictions
DESC, TARGETS = ["PD", "CA", "T", "P"], ["E", "TS", "EL"]

def load_xy():
    D = pd.read_csv(DATA); D.columns = [c.strip() for c in D.columns]
    return D, D[DESC].to_numpy(float)

def metrics(y, p):
    e = np.asarray(p) - y
    return dict(RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)),
                R2=1 - np.sum(e**2)/np.sum((y - y.mean())**2),
                CV_RMSE=np.sqrt(np.mean(e**2))/y.mean()*100)

def nested_loocv(X, y, est, grid):
    """Deterministic nested LOOCV on the full nine distinct training samples."""
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneOut().split(X):
        e = clone(est)
        if grid:
            e = GridSearchCV(e, grid, cv=LeaveOneOut(),
                             scoring="neg_root_mean_squared_error").fit(X[tr], y[tr]).best_estimator_
        else:
            e.fit(X[tr], y[tr])
        pred[te] = np.asarray(e.predict(X[te])).ravel()[0]
    return pred

def llm_sample_abs_errors(D):
    """Per-sample |error| averaged over the five archived runs (manuscript Sec. 2.2.4)."""
    L = pd.read_csv(LLM); out = {}
    for m in [m for m in L.model.unique() if m != "PLS"]:
        for t in TARGETS:
            w = L[(L.model == m) & (L.target == t)].pivot_table(
                index="sample", columns="run", values="predicted").reindex(D.Sample.values)
            y = D[t].to_numpy(float)
            out[(m, t)] = (np.abs(w.to_numpy() - y[:, None]).mean(axis=1), w.to_numpy().mean(axis=1))
    return out

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); m = len(p); prev = 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i]*m/(m-r)); q[i] = prev
    return q

def paired_vs_llm(D, baseline_abs_err, label, rows):
    """Wilcoxon signed-rank on the ten sample-level paired differences + BH-FDR."""
    llm = llm_sample_abs_errors(D)
    for t in TARGETS:
        blk = []
        for m in sorted({k[0] for k in llm}):
            lae = llm[(m, t)][0]; bae = baseline_abs_err[t]
            try: pv = stats.wilcoxon(lae, bae, zero_method="wilcox").pvalue
            except ValueError: pv = 1.0
            blk.append(dict(target=t, baseline=label, llm=m, MAE_baseline=bae.mean(),
                            MAE_LLM=lae.mean(), n_of_10_LLM_better=int((lae < bae).sum()), p=pv))
        for b, q in zip(blk, bh([b["p"] for b in blk])): b["q_BH"] = q
        rows += blk


SEEDS = (42, 43, 44, 45, 46)          # the seeds used in the published PLS script
PLSG  = ({"m__n_components": [1, 2, 3, 4]},
         Pipeline([("sc", StandardScaler()), ("m", PLSRegression())]))

def bootstrap_pls(X, y, seeds=SEEDS):
    """Replicate the published branch: size-9 bootstrap of each 9-sample training set."""
    runs, ndist = [], []
    for s in seeds:
        rng = np.random.default_rng(s); pr = np.full(len(y), np.nan)
        for tr, te in LeaveOneOut().split(X):
            idx = rng.choice(tr, size=len(tr), replace=True); ndist.append(len(np.unique(idx)))
            Xb, yb = X[idx], y[idx]
            sc = StandardScaler().fit(Xb); Xs = sc.transform(Xb)
            best, bs = 1, np.inf
            for k in range(1, min(X.shape[1], len(idx)-1)+1):
                err = [ (PLSRegression(n_components=min(k, len(i1)-1)).fit(Xs[i1], yb[i1])
                         .predict(Xs[i2]).ravel()[0] - yb[i2][0])**2
                        for i1, i2 in LeaveOneOut().split(Xs) ]
                if np.sqrt(np.mean(err)) < bs: bs, best = np.sqrt(np.mean(err)), k
            pr[te] = PLSRegression(n_components=best).fit(Xs, yb).predict(sc.transform(X[te])).ravel()[0]
        runs.append(pr)
    return np.array(runs), float(np.mean(ndist))

if __name__ == "__main__":
    D, X = load_xy()
    met, tests, nd = [], [], None
    abs_det, abs_boot = {}, {}
    for t in TARGETS:
        y = D[t].to_numpy(float)
        p_det = nested_loocv(X, y, PLSG[1], PLSG[0])
        runs, nd = bootstrap_pls(X, y)
        abs_det[t]  = np.abs(p_det - y)
        abs_boot[t] = np.abs(runs - y[None, :]).mean(axis=0)
        met.append(dict(branch="PLS, all 9 distinct training samples (deterministic)", target=t,
                        **metrics(y, p_det)))
        met.append(dict(branch="PLS, bootstrap (published), run-averaged", target=t,
                        **metrics(y, runs.mean(axis=0))))
        per = [metrics(y, r)["RMSE"] for r in runs]
        met.append(dict(branch="PLS, bootstrap (published), per-run mean", target=t,
                        RMSE=np.mean(per), MAE=np.mean([metrics(y, r)["MAE"] for r in runs]),
                        R2=np.mean([metrics(y, r)["R2"] for r in runs]),
                        CV_RMSE=np.mean(per)/y.mean()*100))
    paired_vs_llm(D, abs_det,  "PLS, all 9 distinct samples (deterministic)", tests)
    paired_vs_llm(D, abs_boot, "PLS, bootstrap (published)", tests)
    pd.DataFrame(met).round(4).to_csv("../Supplementary Materials/Table_S7_pls_full_vs_bootstrap.csv", index=False)
    pd.DataFrame(tests).round(4).to_csv("../Supplementary Materials/Table_S8_paired_tests_PLS_regimes.csv", index=False)
    print("mean distinct training samples per bootstrap fit: %.2f  (theory %.2f)"
          % (nd, 9*(1-(8/9)**9)))
    print(pd.DataFrame(met).pivot_table(index="branch", columns="target", values="RMSE").round(2))
