#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-target (joint) PLS2: giving the conventional branch the same cross-target information as the prompt.

Reviewer 4 (Revision 1), comment 4.
Protocol identical to the manuscript: outer leave-one-out over the 10 samples;
inside every outer fold all z-scoring and hyper-parameter selection use ONLY the
nine training samples and are performed by inner leave-one-out; the held-out
sample is predicted exactly once.

Run:  python run_multitarget_pls2.py
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


def pls2_multitarget(X, Y):
    """PLS2 on all three responses jointly; number of LVs by inner LOOCV on a
    variance-scaled joint criterion (so no single target dominates selection)."""
    pred = np.full(Y.shape, np.nan); ysd = Y.std(axis=0, ddof=1)
    for tr, te in LeaveOneOut().split(X):
        Xtr, Ytr = X[tr], Y[tr]; best, bs = 1, np.inf
        for k in [1, 2, 3, 4]:
            err = []
            for i1, i2 in LeaveOneOut().split(Xtr):
                sc = StandardScaler().fit(Xtr[i1])
                m = PLSRegression(n_components=min(k, len(i1)-1)).fit(sc.transform(Xtr[i1]), Ytr[i1])
                err.append(((m.predict(sc.transform(Xtr[i2])).reshape(1, -1) - Ytr[i2]) / ysd)**2)
            if np.sqrt(np.mean(err)) < bs: bs, best = np.sqrt(np.mean(err)), k
        sc = StandardScaler().fit(Xtr)
        m = PLSRegression(n_components=best).fit(sc.transform(Xtr), Ytr)
        pred[te] = m.predict(sc.transform(X[te])).reshape(1, -1)
    return pred

if __name__ == "__main__":
    D, X = load_xy(); Y = D[TARGETS].to_numpy(float)
    P = pls2_multitarget(X, Y)
    met, tests, abs_err = [], [], {}
    for j, t in enumerate(TARGETS):
        y = Y[:, j]
        p1 = nested_loocv(X, y, Pipeline([("sc", StandardScaler()), ("m", PLSRegression())]),
                          {"m__n_components": [1, 2, 3, 4]})
        met.append(dict(model="PLS, single target", target=t, **metrics(y, p1)))
        met.append(dict(model="PLS2, three targets fitted jointly", target=t, **metrics(y, P[:, j])))
        abs_err[t] = np.abs(P[:, j] - y)
    paired_vs_llm(D, abs_err, "PLS2, three targets fitted jointly", tests)
    pd.DataFrame(met).round(4).to_csv("../Supplementary Materials/Table_S9_single_vs_multitarget_PLS.csv", index=False)
    pd.DataFrame(tests).round(4).to_csv("../Supplementary Materials/Table_S10_paired_tests_PLS2.csv", index=False)
    D[TARGETS].corr().round(4).to_csv("../Supplementary Materials/Table_S11_target_correlations.csv")
    print(pd.DataFrame(met).pivot_table(index="model", columns="target", values="RMSE").round(2))
    print(D[TARGETS].corr().round(3))
