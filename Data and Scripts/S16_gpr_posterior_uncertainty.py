#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gaussian process posterior predictive uncertainty for the conventional branch.

Reviewer 4 (Revision 1), comment 5: repeated LLM sampling is not an uncertainty
quantification. This script reports the quantity that IS one on the conventional
branch - the GP posterior predictive standard deviation of each held-out sample -
under the same nested leave-one-out protocol (kernel hyper-parameters by marginal
likelihood on the nine training samples of each fold).

Run:  python run_gpr_posterior_sd.py
"""
import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings("ignore")

DESC, TARGETS = ["PD", "CA", "T", "P"], ["E", "TS", "EL"]
D = pd.read_csv("psf_membrane_dataset.csv"); D.columns = [c.strip() for c in D.columns]
X = D[DESC].to_numpy(float)

rows = []
for tg in TARGETS:
    y = D[tg].to_numpy(float)
    for tr, te in LeaveOneOut().split(X):
        sx = StandardScaler().fit(X[tr]); sy = StandardScaler().fit(y[tr].reshape(-1, 1))
        g = GaussianProcessRegressor(
            kernel=C(1.0, (1e-2, 1e3))*RBF(2.0, (1e-1, 1e2)) + WhiteKernel(0.1, (1e-4, 1e1)),
            n_restarts_optimizer=10, random_state=42
        ).fit(sx.transform(X[tr]), sy.transform(y[tr].reshape(-1, 1)).ravel())
        mu_s, sd_s = g.predict(sx.transform(X[te]), return_std=True)
        mu = float(sy.inverse_transform(np.asarray(mu_s).reshape(-1, 1)).ravel()[0])
        sd = float(np.asarray(sd_s).ravel()[0] * sy.scale_[0])
        rows.append(dict(target=tg, sample=D.Sample.values[te[0]], actual=float(y[te[0]]),
                         gpr_mean=mu, gpr_posterior_sd=sd, gpr_sd_pct=sd/abs(mu)*100,
                         abs_error=abs(mu - y[te[0]]),
                         inside_95pct_interval=bool(abs(mu - y[te[0]]) <= 1.96*sd)))
G = pd.DataFrame(rows)
G.round(4).to_csv("../Supplementary Materials/Table_S16_gpr_posterior_sd.csv", index=False)
print(G.groupby("target")[["gpr_posterior_sd", "gpr_sd_pct", "abs_error"]].mean()
       .reindex(TARGETS).round(2).to_string())
print("\n95% posterior-interval coverage:",
      G.groupby("target").inside_95pct_interval.mean().reindex(TARGETS).round(2).to_dict())
