"""
Normalized and relative error metrics for cross-property comparability (Table S4).
For each model and target: CV(RMSE) = RMSE/mean(y), nRMSE = RMSE/range(y), MAPE.
Per-run values are computed first and then averaged over the five runs, matching
the aggregation convention of Table 3. Only the recorded predictions are used.
"""
import numpy as np
import pandas as pd

CSV = "_4_宽表llm_predictions_runs_wide_完整的.csv"
BLOCKS = {"PLS": 0, "DeepSeek-V3": 17, "DeepSeek-R1": 32, "ChatGPT-4o": 48, "GPT-5": 63}
TARGETS = ["E", "TS", "EL"]
raw = pd.read_csv(CSV, encoding="cp1252", header=None, engine="python")

def parse(start):
    sub = raw.iloc[start + 2:start + 12].reset_index(drop=True)
    a = sub.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").values
    P = {}
    for ti, t in enumerate(TARGETS):
        P[t] = sub.iloc[:, [4 + ti, 7 + ti, 10 + ti, 13 + ti, 16 + ti]].apply(pd.to_numeric, errors="coerce").values
    return a, P

data = {n: parse(s) for n, s in BLOCKS.items()}
y = {t: data["PLS"][0][:, ti] for ti, t in enumerate(TARGETS)}
rng = {t: y[t].max() - y[t].min() for t in TARGETS}
mean = {t: y[t].mean() for t in TARGETS}

print(f"{'Model':13s} {'Target':6s} {'CV(RMSE)%':>10s} {'nRMSE%':>8s} {'MAPE%':>8s}")
print("-" * 50)
for name in ["PLS", "DeepSeek-V3", "DeepSeek-R1", "ChatGPT-4o", "GPT-5"]:
    for t in TARGETS:
        P, yy = data[name][1][t], y[t]
        cvr, nrm, mape = [], [], []
        for r in range(5):
            rmse = np.sqrt(np.mean((P[:, r] - yy) ** 2))
            cvr.append(100 * rmse / mean[t])
            nrm.append(100 * rmse / rng[t])
            mape.append(100 * np.mean(np.abs((P[:, r] - yy) / yy)))
        print(f"{name:13s} {t:6s} {np.mean(cvr):10.1f} {np.mean(nrm):8.1f} {np.mean(mape):8.1f}")
