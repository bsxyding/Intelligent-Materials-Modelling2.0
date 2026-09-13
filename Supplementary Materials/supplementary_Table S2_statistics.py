"""
Supplementary statistical reanalysis accounting for the nested run structure.
Addresses the non-independence of the 10 held-out samples evaluated across 5 runs.

For every target (E, TS, EL) and every LLM-vs-PLS contrast:
  - sample-level aggregation: AE_i = mean over the 5 runs of |prediction_i,run - y_i|
  - paired Wilcoxon signed-rank test on the 10 sample-level differences (n = 10)
  - Benjamini-Hochberg FDR correction across the four LLMs within each target
  - 95% cluster (sample-level) bootstrap CI on the mean paired AE difference
  - sign-flip permutation test (EL) as an independent corroboration

Input : wide table of per-(model, sample, run) predictions + actual values.
Output: Table S2 numbers (printed). No model inference is performed.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = "llm_predictions_runs_wide.csv"
BLOCKS = {"PLS": 0, "DeepSeek-V3": 17, "DeepSeek-R1": 32, "ChatGPT-4o": 48, "GPT-5": 63}
TARGETS = ["E", "TS", "EL"]
UNIT = {"E": "N/mm2", "TS": "N/mm2", "EL": "%"}
LLMS = ["DeepSeek-V3", "DeepSeek-R1", "ChatGPT-4o", "GPT-5"]
SEED, B_BOOT, B_PERM = 20240613, 10000, 20000

raw = pd.read_csv(CSV, encoding="cp1252", header=None, engine="python")

def parse_block(start):
    sub = raw.iloc[start + 2:start + 12].reset_index(drop=True)
    actual = sub.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").values
    preds = {}
    for ti, t in enumerate(TARGETS):
        cols = [4 + ti, 7 + ti, 10 + ti, 13 + ti, 16 + ti]
        preds[t] = sub.iloc[:, cols].apply(pd.to_numeric, errors="coerce").values
    return actual, preds

data = {n: parse_block(s) for n, s in BLOCKS.items()}
y = {t: data["PLS"][0][:, ti] for ti, t in enumerate(TARGETS)}

def sample_AE(name, t):
    """Sample-level aggregated absolute error: mean over 5 runs of |pred - y|."""
    P = data[name][1][t]
    return np.mean(np.abs(P - y[t][:, None]), axis=1)

def bh_fdr(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranks = np.empty(m, int); ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    qs = q[order]
    for i in range(m - 2, -1, -1):
        qs[i] = min(qs[i], qs[i + 1])
    out = np.empty(m); out[order] = np.clip(qs, 0, 1)
    return out

rng = np.random.default_rng(SEED)

print(f"{'Target':6s} {'Model':12s} {'n+/10':6s} {'meanDAE':>9s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'BH q':>8s}")
print("-" * 86)
perm_lines = {}
for t in TARGETS:
    ae_pls = sample_AE("PLS", t)
    raw_p = []
    for m in LLMS:
        ae_m = sample_AE(m, t)
        try:
            _, p = wilcoxon(ae_m, ae_pls, alternative="two-sided", zero_method="wilcox", method="exact")
        except Exception:
            _, p = wilcoxon(ae_m, ae_pls, alternative="two-sided", zero_method="zsplit")
        raw_p.append(p)
    qvals = bh_fdr(raw_p)
    for m, p, q in zip(LLMS, raw_p, qvals):
        d = sample_AE(m, t) - ae_pls
        boot = np.array([d[rng.integers(0, 10, 10)].mean() for _ in range(B_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nbetter = int(np.sum(d < 0))
        ci = f"[{lo:+6.2f},{hi:+6.2f}]"
        print(f"{t:6s} {m:12s} {nbetter:>4d}/10 {d.mean():>+9.3f} {ci:>22s} {p:>11.4f} {q:>8.4f}")
    # permutation (EL only) for corroboration
    if t == "EL":
        for m in LLMS:
            d = sample_AE(m, t) - ae_pls
            obs = d.mean(); cnt = 0
            for _ in range(B_PERM):
                s = rng.choice([-1, 1], size=d.size)
                if abs((s * d).mean()) >= abs(obs) - 1e-12:
                    cnt += 1
            perm_lines[m] = (cnt + 1) / (B_PERM + 1)

print("\nSign-flip permutation test (EL, B = %d):" % B_PERM)
for m in LLMS:
    print(f"  {m:12s}: p = {perm_lines[m]:.4f}")

print("\nComparison of test unit:")
print("  Manuscript (original): pooled 10 samples x 5 runs = 50 'paired observations' (pseudo-replicated).")
print("  Reanalysis (this file): 10 sample-level aggregated errors (n = 10 independent units).")
