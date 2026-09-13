#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Is elongation at break really nonlinearity-dominated? Nonlinear baselines, explicit interaction analysis, and a no-descriptor reference.

Reviewer 4 (Revision 1), comment 6.
Protocol identical to the manuscript: outer leave-one-out over the 10 samples;
inside every outer fold all z-scoring and hyper-parameter selection use ONLY the
nine training samples and are performed by inner leave-one-out; the held-out
sample is predicted exactly once.

Run:  python run_nonlinearity_check.py
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


MODELS = [
    ("Mean predictor (training-fold mean, no descriptors)",
     Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1e12))]), {}, "reference"),
    ("PLS (LV by inner LOOCV)",
     Pipeline([("sc", StandardScaler()), ("m", PLSRegression())]),
     {"m__n_components": [1, 2, 3, 4]}, "linear"),
    ("Ridge regression",
     Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
     {"m__alpha": np.logspace(-3, 3, 13)}, "linear"),
    ("Elastic net",
     Pipeline([("sc", StandardScaler()), ("m", ElasticNet(max_iter=200000))]),
     {"m__alpha": np.logspace(-3, 2, 11), "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}, "linear"),
    ("Ridge on descriptors + pairwise interactions",
     Pipeline([("sc", StandardScaler()),
               ("pf", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
               ("sc2", StandardScaler()), ("m", Ridge())]),
     {"m__alpha": np.logspace(-3, 3, 13)}, "interaction"),
    ("Gaussian process regression (RBF + white noise)",
     Pipeline([("sc", StandardScaler()),
               ("m", TransformedTargetRegressor(
                   regressor=GaussianProcessRegressor(
                       kernel=C(1.0, (1e-2, 1e3))*RBF(2.0, (1e-1, 1e2)) + WhiteKernel(0.1, (1e-4, 1e1)),
                       n_restarts_optimizer=10, random_state=42),
                   transformer=StandardScaler()))]), {}, "nonlinear"),
    ("Support vector regression (RBF kernel)",
     Pipeline([("sc", StandardScaler()),
               ("m", TransformedTargetRegressor(regressor=SVR(kernel="rbf"),
                                                transformer=StandardScaler()))]),
     {"m__regressor__C": np.logspace(-1, 3, 9), "m__regressor__gamma": np.logspace(-3, 1, 9),
      "m__regressor__epsilon": [0.01, 0.1, 0.3]}, "nonlinear"),
    ("Random forest",
     Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=300, random_state=42))]),
     {}, "nonlinear"),
]
RF_SEEDS = range(12)      # seed sensitivity of the ensemble baseline

if __name__ == "__main__":
    D, X = load_xy()
    met, abs_err = [], {}
    for name, est, grid, fam in MODELS:
        abs_err[name] = {}
        for t in TARGETS:
            y = D[t].to_numpy(float); p = nested_loocv(X, y, est, grid)
            abs_err[name][t] = np.abs(p - y)
            met.append(dict(model=name, family=fam, target=t, **metrics(y, p)))
    M = pd.DataFrame(met)

    # --- seed sensitivity of the random forest ---
    seed_rows = []
    for s in RF_SEEDS:
        est = Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=300, random_state=s))])
        for t in TARGETS:
            y = D[t].to_numpy(float)
            seed_rows.append(dict(seed=s, target=t, **metrics(y, nested_loocv(X, y, est, {}))))
    S = pd.DataFrame(seed_rows)

    # --- nonlinearity-attributable gain per property ---
    diag = []
    for t in TARGETS:
        lin = M[(M.target == t) & (M.family == "linear")].nsmallest(1, "RMSE").iloc[0]
        nl  = M[(M.target == t) & (M.family == "nonlinear")].nsmallest(1, "RMSE").iloc[0]
        ref = M[(M.target == t) & (M.family == "reference")].iloc[0]
        pe = max(abs(stats.pearsonr(D[d], D[t])[0]) for d in DESC)
        sp = max(abs(stats.spearmanr(D[d], D[t])[0]) for d in DESC)
        diag.append(dict(target=t, best_linear=lin.model, RMSE_linear=lin.RMSE, R2_linear=lin.R2,
                         best_nonlinear=nl.model, RMSE_nonlinear=nl.RMSE, R2_nonlinear=nl.R2,
                         nonlinear_gain_pct=100*(1-nl.RMSE/lin.RMSE),
                         RMSE_mean_predictor=ref.RMSE,
                         linear_gain_over_mean_pct=100*(1-lin.RMSE/ref.RMSE),
                         max_abs_pearson=pe, max_abs_spearman=sp))
    llm = llm_sample_abs_errors(D)
    for (m, t), (ae, pm) in llm.items():
        met.append(dict(model=m, family="LLM", target=t, **metrics(D[t].to_numpy(float), pm)))
    pd.DataFrame(met).round(4).to_csv("../Supplementary Materials/Table_S12_nonlinearity_baselines.csv", index=False)
    pd.DataFrame(diag).round(4).to_csv("../Supplementary Materials/Table_S13_nonlinearity_gain.csv", index=False)
    S.groupby("target")[["RMSE", "R2"]].agg(["mean", "std", "min", "max"]).round(4)\
     .to_csv("../Supplementary Materials/Table_S14_rf_seed_sensitivity.csv")
    tests = []
    for name in abs_err: paired_vs_llm(D, abs_err[name], name, tests)
    pd.DataFrame(tests).round(4).to_csv("../Supplementary Materials/paired_tests_nonlinear_baselines.csv", index=False)
    print(pd.DataFrame(diag).round(3).to_string(index=False))
