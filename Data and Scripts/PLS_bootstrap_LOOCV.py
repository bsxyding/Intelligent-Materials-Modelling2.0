#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
PLS Regression with Bootstrap-Perturbed LOOCV for PSF Membrane Mechanical
Property Prediction
==============================================================================

Purpose
-------
This script performs Partial Least Squares (PLS) regression on a small-sample
(n=10) polysulfone (PSF) membrane dataset to predict three mechanical
properties (Young's modulus, tensile strength, and elongation at break) from
four structural descriptors (pore diameter, contact angle, thickness, and
porosity). A single execution produces a complete set of publication-quality
visualizations and data exports to support results verification and manuscript
preparation.

Outputs (PNG figures):
  1) Correlation heatmap: correlation_heatmap.png
  2) Three error-bar plots (sample-level predicted mean ± SD vs actual):
     errorbar_E.png / errorbar_TS.png / errorbar_EL.png
  3) Three parity plots (actual vs predicted ± SD, with y=x line, linear
     fit, and R²): parity_E.png / parity_TS.png / parity_EL.png
  4) Three multi-run overlay parity plots (all 5 runs superimposed):
     parity_runs_E.png / parity_runs_TS.png / parity_runs_EL.png
  5) Three residual histograms (with KDE): residuals_hist_*.png
  6) Three residual-vs-predicted scatter plots: residuals_vs_pred_*.png
  7) Three Bland-Altman agreement plots: bland_altman_*.png
  8) Three n_components selection heatmaps: components_heatmap_*.png
  9) Three VIP (Variable Importance in Projection) bar charts:
     vip_*.png
  10) Three standardized regression coefficient bar charts: coef_*.png

Outputs (CSV data):
  - run_level_metrics.csv: per-run aggregate metrics (RMSE, MAE, R²)
  - predictions_runs_long.csv: long-format table (property/run/sample/pred/
    actual/components)
  - predictions_runs_wide.csv: wide-format table (one column per run)
  - predictions_summary_by_sample.csv: sample-level predicted mean ± SD
  - vip_by_run.csv: per-run VIP values (fold-averaged)
  - coefs_by_run.csv: per-run standardized regression coefficients
    (fold-averaged)

Methodology
-----------
In small-sample regimes, standard PLS with LOOCV is deterministic for a
fixed dataset, yielding no run-to-run variability. To quantify model
sensitivity to data perturbations and generate error bars, this script
applies **bootstrap resampling** (with replacement, fixed size) to the
training subset within each outer LOOCV fold. The number of latent
variables (n_components) is selected via **inner LOOCV** on the
bootstrapped training data. The complete fold-wise workflow is repeated
**5 times** with independent random seeds, producing 5 predictions per
sample per target for subsequent mean ± SD computation.

This "LOOCV + Bootstrap" scheme is commonly employed in small-sample
materials and chemical engineering studies to assess model robustness
under data perturbation.

Figure descriptions for manuscript use
--------------------------------------
  1) Correlation heatmap: Reveals linear association strengths between
     structural descriptors and mechanical properties, identifying dominant
     structure-property factors.
  2) Error-bar plots: Display per-sample prediction dispersion (mean ± SD
     across 5 runs) versus actual values, reflecting model stability and
     bias structure at the sample level.
  3) Parity plots (with fit and R²): Standard actual-vs-predicted agreement
     assessment. The y=x dashed line represents ideal prediction; the fitted
     line equation and R² quantify trend and explanatory power.
  4) Multi-run overlay parity plots: Superimpose all 5 runs' predictions to
     visualize inter-run variability and systematic bias.
  5) Residual histograms (with KDE): Verify whether residuals are
     approximately symmetric; detect heavy tails or skewness.
  6) Residual-vs-predicted plots: Check for heteroscedasticity (variance
     changing with predicted magnitude) or structural patterns.
  7) Bland-Altman plots: A widely used agreement analysis method in
     engineering and medical literature, showing the distribution of
     (predicted - actual) differences, bias, and 95% limits of agreement
     (± 1.96 SD).
  8) n_components heatmaps: Record the optimal number of latent variables
     selected per fold per run, revealing model complexity stability and
     sample sensitivity.
  9) VIP bar charts: Variable Importance in Projection, where VIP > 1 is
     empirically considered "important". Compares the relative importance
     of the four descriptors with cross-run SD.
  10) Standardized coefficient bar charts: Reflect the direction and
      relative magnitude of each descriptor's linear contribution in
      z-scored space. Used jointly with VIP for structure-property
      interpretation.

Notes
-----
- Dependencies: pandas, numpy, scikit-learn, seaborn, matplotlib
- If Arial font is unavailable, seaborn will fall back to the default font;
  this does not affect execution.
- With only 10 samples, interpretive visualizations focus on trends and
  robustness; statistical inferences should be stated cautiously.

Reference
---------
Cao, D.; Chan, M. K.; Yeo, W. S.; Bey, S.; Figoli, A. Intelligent
Materials Modelling: Large Language Models Versus Partial Least Squares
Regression for Predicting Polysulfone Membrane Mechanical Performance.
(Manuscript in preparation)

==============================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# ========================= Global Plotting Parameters ========================= #
plt.rcParams['figure.dpi'] = 120
sns.set(style="whitegrid", font="Arial", font_scale=1.0)

# ============================== 1) Dataset ============================== #
# Four structural descriptors: pore diameter (PD, µm), contact angle (CA, °),
# thickness (T, mm), and porosity (P, %).
# Three mechanical targets: Young's modulus (E, N/mm²), tensile strength
# (TS, N/mm²), and elongation at break (EL, %).
# Sample labels S1-S10 correspond to Table 1 in the manuscript.

sample_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
feature_names = [
    'Pore diameter PD (µm)',
    'Contact angle CA (°)',
    'Thickness T (mm)',
    'Porosity P (%)'
]

X = np.array([
    [0.522, 94.6, 0.273, 77.67],   # S1
    [0.364, 93.3, 0.202, 79.16],   # S2
    [0.569, 78.9, 0.175, 78.87],   # S3
    [0.451, 66.4, 0.136, 73.60],   # S4
    [0.408, 66.6, 0.154, 73.49],   # S5
    [0.336, 79.5, 0.177, 69.09],   # S6
    [0.403, 81.3, 0.240, 71.18],   # S7
    [0.319, 84.5, 0.180, 78.32],   # S8
    [0.842, 90.0, 0.224, 78.86],   # S9
    [0.298, 82.9, 0.188, 74.02],   # S10
], dtype=float)

Y_modulus = np.array([117.17, 90.18, 152.50, 231.78, 182.59,
                      235.65, 176.51, 126.42, 107.67, 168.57], dtype=float)
Y_tensile = np.array([4.82, 3.97, 6.56, 9.61, 7.43,
                      9.65, 7.25, 5.22, 4.53, 7.13], dtype=float)
Y_elong   = np.array([42.07, 46.61, 49.13, 61.30, 60.69,
                      64.46, 69.04, 42.91, 51.25, 67.15], dtype=float)

columns_all = feature_names + [
    "Young's modulus E (N/mm²)",
    'Tensile strength TS (N/mm²)',
    'Elongation at break EL (%)'
]
df_all = pd.DataFrame(
    np.column_stack([X, Y_modulus, Y_tensile, Y_elong]),
    index=sample_names,
    columns=columns_all
)


# ==================== 2) Correlation Matrix + Heatmap ==================== #
# Purpose: Identify descriptors most strongly associated with target
# properties, providing a priori guidance for model interpretation.

# Uncomment the block below to generate the correlation heatmap:
# corr_matrix = df_all.corr(numeric_only=True)
# print("\n=== Pearson Correlation Matrix (4 descriptors × 3 targets) ===")
# print(corr_matrix.round(2))
#
# plt.figure(figsize=(9, 7))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
# plt.title('Correlation Heatmap', fontsize=14)
# plt.tight_layout()
# plt.savefig('correlation_heatmap.png')
# plt.close()
# print("Saved: correlation_heatmap.png")


# ==================== 3) VIP Computation Function ==================== #

def compute_vip(pls_model, X_scaled, y):
    """
    Compute VIP (Variable Importance in Projection) for a single-response
    PLS model.

    Theoretical background:
    - VIP measures the cumulative contribution of each predictor variable
      across all extracted latent variables. A higher VIP value indicates
      greater explanatory power for the response. Empirically, VIP > 1 is
      commonly used as the importance threshold.
    - Mathematical expression (single-response case):
        VIP_j = sqrt( p * sum_a [ SSY_a * (w_{ja} / ||w_a||)^2 ] / sum_a SSY_a )
      where p = number of features, a = latent variable index,
      SSY_a = (q_a^2) * sum_i(t_{ia}^2) is the Y-explained sum of squares
      for the a-th latent variable, and w_{ja} is the weight of the j-th
      feature on the a-th latent variable.
    - Implementation is based on scikit-learn's PLSRegression attributes:
      x_scores_ (T), x_weights_ (W), and y_loadings_ (Q).
    - Note: VIP values depend on data standardization. In this script,
      z-score standardization is applied to X prior to PLS fitting.

    Parameters
    ----------
    pls_model : PLSRegression
        A fitted scikit-learn PLS model.
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Standardized predictor matrix used for fitting.
    y : np.ndarray, shape (n_samples,)
        Response vector.

    Returns
    -------
    vip : np.ndarray, shape (n_features,)
        VIP scores for each predictor variable.
    """
    T = pls_model.x_scores_             # (n_samples, n_components)
    W = pls_model.x_weights_            # (n_features, n_components)
    Q = pls_model.y_loadings_.ravel()   # (n_components,)
    p = W.shape[0]

    # Y-explained sum of squares for each latent variable
    SSY = np.array([
        (Q[a] ** 2) * np.sum(T[:, a] ** 2) for a in range(W.shape[1])
    ])
    denom = SSY.sum()

    vip = np.zeros(p)
    for j in range(p):
        s = 0.0
        for a in range(W.shape[1]):
            wja = W[j, a]
            wa_norm = np.linalg.norm(W[:, a])
            if wa_norm == 0:
                continue
            s += SSY[a] * (wja / wa_norm) ** 2
        vip[j] = np.sqrt((p * s) / denom) if denom > 0 else 0.0
    return vip


# ========= 4) Single-Run LOOCV with Bootstrap Perturbation ========= #

def pls_loocv_bootstrap_single_run(X_data, y_data, max_components=4,
                                   seed=42, verbose=True):
    """
    Execute one complete run of the bootstrap-perturbed LOOCV pipeline for
    PLS regression on small-sample materials data.

    Workflow overview
    -----------------
    For each outer LOOCV fold (leave-one-out, n folds total):
      1) Hold out one test sample; remaining samples form the outer
         training set.
      2) Apply bootstrap resampling (with replacement, same size) to the
         outer training set, producing a perturbed training set.
      3) Fit a StandardScaler on the bootstrapped training set only;
         apply the same z-score transformation to the held-out test sample.
      4) Perform inner LOOCV on the bootstrapped training set to select the
         optimal number of latent variables (n_components) by minimizing
         inner validation RMSE.
      5) Refit a PLS model with the selected n_components on the full
         bootstrapped training set; predict the held-out test sample.
      6) Record the prediction, selected n_components, VIP scores, and
         standardized regression coefficients for this fold.
    After all folds:
      7) Compute overall RMSE, MAE, and R² from the assembled out-of-fold
         predictions.
      8) Aggregate VIP and coefficient vectors across folds (mean and SD).

    Design rationale
    ----------------
    - Outer LOOCV: Maximizes training data utilization under extreme sample
      scarcity (n=10), ensuring each sample serves exactly once as an
      independent test point for unbiased generalization estimation.
    - Bootstrap perturbation: Injects controlled data variability into each
      fold, enabling multiple runs (e.g., 5 seeds) to produce per-sample
      prediction distributions for mean ± SD error bars and model stability
      assessment.
    - Inner LOOCV for n_components selection: Strictly confines model
      complexity selection within training data, preventing information
      leakage from the held-out test sample and reducing overfitting risk.
    - VIP and standardized coefficients: Provide interpretability outputs
      for structure-property discussion in the manuscript.

    Parameters
    ----------
    X_data : np.ndarray, shape (n_samples, n_features)
        Feature matrix. In this study: (10, 4) with four structural
        descriptors (PD, CA, T, P).
    y_data : np.ndarray, shape (n_samples,)
        Single target variable (one mechanical property per call).
    max_components : int, default 4
        Upper bound for candidate latent variable counts (further
        constrained by feature count and training sample count).
    seed : int, default 42
        Random seed for bootstrap resampling reproducibility.
    verbose : bool, default True
        If True, print detailed per-fold logs for debugging and
        reproducibility documentation.

    Returns
    -------
    preds : np.ndarray, shape (n_samples,)
        Out-of-fold predictions (one per sample).
    comps : np.ndarray, shape (n_samples,), dtype int
        Selected n_components per fold.
    rmse : float
        Overall RMSE across all folds.
    mae : float
        Overall MAE across all folds.
    r2 : float
        Overall R² across all folds.
    vip_mean : np.ndarray, shape (n_features,)
        Cross-fold mean VIP scores for this run.
    vip_std : np.ndarray, shape (n_features,)
        Cross-fold SD of VIP scores for this run.
    coef_mean : np.ndarray, shape (n_features,)
        Cross-fold mean standardized regression coefficients.
    coef_std : np.ndarray, shape (n_features,)
        Cross-fold SD of standardized regression coefficients.
    """

    # -------------------- Dimensions and random state --------------------
    n = X_data.shape[0]                 # Number of samples
    p = X_data.shape[1]                 # Number of features
    rng = np.random.default_rng(seed)   # Reproducible RNG for bootstrap
    loo = LeaveOneOut()                 # Outer LOOCV splitter

    # -------------------- Pre-allocate output arrays --------------------
    preds = np.zeros(n)
    comps = np.zeros(n, dtype=int)
    vip_list = []
    coef_list = []

    # -------------------- Run header log --------------------
    if verbose:
        print("\n" + "=" * 100)
        print(f"[SingleRun] seed={seed} | samples={n} | features={p} "
              f"| max_components={max_components}")
        print("=" * 100)

    # ====================== Outer LOOCV Loop ======================
    for fold_idx, (train_idx, test_idx) in enumerate(
            loo.split(X_data), start=1):

        # --- Split outer fold ---
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # Attempt to retrieve sample name for logging
        try:
            test_name = sample_names[test_idx[0]]
        except Exception:
            test_name = f"idx{test_idx[0]}"

        if verbose:
            print(f"\n[Outer {fold_idx:02d}/{n}] Test={test_name} "
                  f"(index={test_idx[0]})")
            print(f"  - Train size={len(train_idx)}, Test size=1")

        # ========== Bootstrap resampling (with replacement) ==========
        # Generate a bootstrapped training set of the same size as the
        # original training set, introducing data perturbation.
        b_idx = rng.integers(0, len(train_idx), size=len(train_idx))
        Xb, yb = X_train[b_idx], y_train[b_idx]

        if verbose:
            uniq, cnts = np.unique(b_idx, return_counts=True)
            reused = np.sum(cnts > 1)
            dropped = len(train_idx) - len(uniq)
            print(f"  - Bootstrap on train: kept={len(uniq)}, "
                  f"reused_idx={reused}, dropped_idx={dropped}")

        # ========== Standardization (fitted on bootstrap set only) ==========
        # The scaler is fitted exclusively on training data to prevent
        # information leakage. The test sample is transformed using the
        # same scaler parameters.
        scaler = StandardScaler()
        Xb_scaled = scaler.fit_transform(Xb)
        X_test_scaled = scaler.transform(X_test)

        if verbose:
            print(f"  - Scaler: mean={np.round(scaler.mean_, 3)}, "
                  f"var={np.round(scaler.var_, 3)}")

        # ========== Inner LOOCV: select optimal n_components ==========
        # Constraint: n_components cannot exceed the number of features
        # or (training samples - 1).
        best_n, best_rmse = 1, float('inf')
        max_c = min(max_components, Xb.shape[1], Xb.shape[0] - 1)
        if max_c < 1:
            max_c = 1   # Safety: ensure at least 1 candidate

        if verbose:
            print(f"  - Inner-LOOCV candidates for n_components: 1..{max_c}")

        for n_comp in range(1, max_c + 1):
            inner_loo = LeaveOneOut()
            inner_rmse = []

            for inner_tr, inner_va in inner_loo.split(Xb):
                Xi_tr, Xi_va = Xb[inner_tr], Xb[inner_va]
                yi_tr, yi_va = yb[inner_tr], yb[inner_va]

                # Inner standardization (prevents leakage at inner level)
                sc = StandardScaler()
                Xi_tr_s = sc.fit_transform(Xi_tr)
                Xi_va_s = sc.transform(Xi_va)

                pls = PLSRegression(n_components=n_comp)
                pls.fit(Xi_tr_s, yi_tr)
                yhat = pls.predict(Xi_va_s).ravel()
                inner_rmse.append(
                    np.sqrt(mean_squared_error(yi_va, yhat)))

            avg_rmse = float(np.mean(inner_rmse))
            if verbose:
                print(f"    · n_comp={n_comp}: inner-RMSE={avg_rmse:.4f}")

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_n = n_comp

        if verbose:
            print(f"  => Chosen n_components={best_n} "
                  f"(inner-RMSE min={best_rmse:.4f})")

        # ========== Final model fitting and test prediction ==========
        pls_final = PLSRegression(n_components=best_n)
        pls_final.fit(Xb_scaled, yb)
        y_pred = pls_final.predict(X_test_scaled).ravel()[0]

        preds[test_idx[0]] = y_pred
        comps[test_idx[0]] = best_n

        # ========== Record VIP and standardized coefficients ==========
        vip = compute_vip(pls_final, Xb_scaled, yb)
        vip_list.append(vip)
        coef_list.append(pls_final.coef_.ravel())

        if verbose:
            resid = y_pred - y_test[0]
            print(f"  - Test pred={y_pred:.4f}, actual={y_test[0]:.4f}, "
                  f"resid={resid:+.4f}")
            try:
                vip_str = ", ".join(
                    [f"{feature_names[j]}:{vip[j]:.2f}" for j in range(p)])
                coef_str = ", ".join(
                    [f"{feature_names[j]}:"
                     f"{pls_final.coef_.ravel()[j]:+.3f}"
                     for j in range(p)])
            except Exception:
                vip_str = ", ".join(
                    [f"f{j}:{vip[j]:.2f}" for j in range(p)])
                coef_str = ", ".join(
                    [f"f{j}:{pls_final.coef_.ravel()[j]:+.3f}"
                     for j in range(p)])
            print(f"  - VIP (this fold): [{vip_str}]")
            print(f"  - Coef(stdz)      : [{coef_str}]")

    # ============ Post-LOOCV: aggregate metrics and VIP/coef ============
    rmse = float(np.sqrt(mean_squared_error(y_data, preds)))
    mae = float(mean_absolute_error(y_data, preds))
    r2 = float(r2_score(y_data, preds))

    vip_arr = np.vstack(vip_list)       # (n_folds, p)
    coef_arr = np.vstack(coef_list)     # (n_folds, p)

    vip_mean = vip_arr.mean(axis=0)
    vip_std = vip_arr.std(axis=0, ddof=1)
    coef_mean = coef_arr.mean(axis=0)
    coef_std = coef_arr.std(axis=0, ddof=1)

    if verbose:
        print("\n" + "-" * 100)
        print(f"[SingleRun Summary] seed={seed}")
        print(f"  - Outer metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, "
              f"R²={r2:.4f}")
        try:
            vip_str = ", ".join(
                [f"{feature_names[j]}:{vip_mean[j]:.2f}±{vip_std[j]:.2f}"
                 for j in range(p)])
            coef_str = ", ".join(
                [f"{feature_names[j]}:{coef_mean[j]:+.3f}±{coef_std[j]:.3f}"
                 for j in range(p)])
        except Exception:
            vip_str = ", ".join(
                [f"f{j}:{vip_mean[j]:.2f}±{vip_std[j]:.2f}"
                 for j in range(p)])
            coef_str = ", ".join(
                [f"f{j}:{coef_mean[j]:+.3f}±{coef_std[j]:.3f}"
                 for j in range(p)])
        print(f"  - VIP across folds   : [{vip_str}]")
        print(f"  - Coef across folds  : [{coef_str}]")
        print("-" * 100)

    return preds, comps, rmse, mae, r2, vip_mean, vip_std, coef_mean, coef_std


# ============= 5) Multiple Runs (5 repeats) and Collection ============= #
# Different random seeds repeat the "LOOCV + Bootstrap" workflow 5 times,
# yielding 5 predictions per sample per target for mean ± SD computation.

N_REPEATS = 5
SEEDS = [42, 43, 44, 45, 46]

records_long = []   # Per-run per-sample records (long format)
run_metrics = []    # Per-run aggregate metrics
vip_records = []    # Per-run VIP (fold-averaged)
coef_records = []   # Per-run standardized coefficients (fold-averaged)


def run_and_collect(y, prop_name, units):
    """
    Execute 5 independent runs for a given target y (one mechanical
    property) and collect predictions, latent variable counts, VIP, and
    standardized coefficients.

    Parameters
    ----------
    y : np.ndarray
        Target vector (one of Y_modulus, Y_tensile, Y_elong).
    prop_name : str
        Property name for labeling (e.g., "Young's modulus (E)").
    units : str
        Physical units string (e.g., "N/mm²" or "%").

    Returns
    -------
    preds_runs : np.ndarray, shape (N_REPEATS, n_samples)
    comps_runs : np.ndarray, shape (N_REPEATS, n_samples)
    vip_means_runs : np.ndarray, shape (N_REPEATS, n_features)
    vip_stds_runs : np.ndarray, shape (N_REPEATS, n_features)
    coef_means_runs : np.ndarray, shape (N_REPEATS, n_features)
    coef_stds_runs : np.ndarray, shape (N_REPEATS, n_features)
    """
    preds_runs = []
    comps_runs = []
    vip_means_runs = []
    vip_stds_runs = []
    coef_means_runs = []
    coef_stds_runs = []

    for r, seed in enumerate(SEEDS, start=1):
        (y_pred, comps, rmse, mae, r2,
         vip_mean, vip_std, coef_mean, coef_std) = \
            pls_loocv_bootstrap_single_run(
                X, y, max_components=4, seed=seed)

        preds_runs.append(y_pred)
        comps_runs.append(comps)
        run_metrics.append({
            'run': r, 'seed': seed, 'property': prop_name,
            'RMSE': rmse, 'MAE': mae, 'R2': r2
        })

        # Long-format records: per-sample
        for i, name in enumerate(sample_names):
            records_long.append({
                'property': prop_name,
                'units': units,
                'run': r,
                'seed': seed,
                'sample': name,
                'actual': y[i],
                'predicted': y_pred[i],
                'components': int(comps[i])
            })

        # Per-run VIP and coefficients (fold-averaged)
        for j, fname in enumerate(feature_names):
            vip_records.append({
                'property': prop_name, 'units': units,
                'run': r, 'seed': seed,
                'feature': fname,
                'vip_mean': vip_mean[j], 'vip_sd': vip_std[j]
            })
            coef_records.append({
                'property': prop_name, 'units': units,
                'run': r, 'seed': seed,
                'feature': fname,
                'coef_mean': coef_mean[j], 'coef_sd': coef_std[j]
            })

        vip_means_runs.append(vip_mean)
        vip_stds_runs.append(vip_std)
        coef_means_runs.append(coef_mean)
        coef_stds_runs.append(coef_std)

    return (np.vstack(preds_runs), np.vstack(comps_runs),
            np.vstack(vip_means_runs), np.vstack(vip_stds_runs),
            np.vstack(coef_means_runs), np.vstack(coef_stds_runs))


# Execute for all three mechanical properties
preds_E, comps_E, vipm_E, vipsd_E, coefm_E, coefs_E = \
    run_and_collect(Y_modulus, "Young's modulus (E)", 'N/mm²')
preds_TS, comps_TS, vipm_TS, vipsd_TS, coefm_TS, coefs_TS = \
    run_and_collect(Y_tensile, 'Tensile strength (TS)', 'N/mm²')
preds_EL, comps_EL, vipm_EL, vipsd_EL, coefm_EL, coefs_EL = \
    run_and_collect(Y_elong, 'Elongation at break (EL)', '%')


# ======================= 6) Export CSV Files ======================= #
# All data exports support reproducibility and peer review.

# Per-run aggregate metrics
run_level_df = pd.DataFrame(run_metrics)
run_level_df.to_csv('run_level_metrics.csv', index=False, encoding='utf-8-sig')
print("Saved: run_level_metrics.csv")

# Long-format table (per-run × per-sample)
long_df = pd.DataFrame.from_records(records_long)
long_df.to_csv('predictions_runs_long.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions_runs_long.csv")

# Wide-format table (one column per run)
wide_df = pd.DataFrame({
    'Sample': sample_names,
    'Actual_E (N/mm²)': Y_modulus,
    'Actual_TS (N/mm²)': Y_tensile,
    'Actual_EL (%)': Y_elong
})
for r in range(N_REPEATS):
    wide_df[f'Pred_E_Run{r+1}'] = preds_E[r]
    wide_df[f'Pred_TS_Run{r+1}'] = preds_TS[r]
    wide_df[f'Pred_EL_Run{r+1}'] = preds_EL[r]
wide_df.to_csv('predictions_runs_wide.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions_runs_wide.csv")


# Sample-level summary (mean ± SD for error bars)
def summarize_by_sample(preds_runs, actual, prop_name, units):
    """Compute per-sample predicted mean and SD across runs."""
    means = preds_runs.mean(axis=0)
    stds = preds_runs.std(axis=0, ddof=1)
    return pd.DataFrame({
        'property': prop_name,
        'units': units,
        'sample': sample_names,
        'actual': actual,
        'pred_mean': means,
        'pred_sd': stds
    })


sum_E = summarize_by_sample(preds_E, Y_modulus,
                            "Young's modulus (E)", 'N/mm²')
sum_TS = summarize_by_sample(preds_TS, Y_tensile,
                             'Tensile strength (TS)', 'N/mm²')
sum_EL = summarize_by_sample(preds_EL, Y_elong,
                             'Elongation at break (EL)', '%')

summary_by_sample = pd.concat([sum_E, sum_TS, sum_EL], ignore_index=True)
summary_by_sample.to_csv('predictions_summary_by_sample.csv',
                         index=False, encoding='utf-8-sig')
print("Saved: predictions_summary_by_sample.csv")

# VIP and coefficients per run
pd.DataFrame(vip_records).to_csv(
    'vip_by_run.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(coef_records).to_csv(
    'coefs_by_run.csv', index=False, encoding='utf-8-sig')
print("Saved: vip_by_run.csv / coefs_by_run.csv")


# ===================== 7) Plotting Functions ===================== #

def plot_errorbar(sum_df, prop_name, y_label, out_png):
    """
    Sample-level error-bar plot.

    Points represent the predicted mean across 5 runs; error bars show
    ± 1 SD reflecting model sensitivity to bootstrap perturbations.
    The solid line with square markers shows actual values for comparison.
    """
    x = np.arange(len(sum_df))
    plt.figure(figsize=(10, 5.5))
    plt.errorbar(x, sum_df['pred_mean'], yerr=sum_df['pred_sd'],
                 fmt='o--', capsize=4, label='Predicted mean ± SD')
    plt.plot(x, sum_df['actual'], 's-', label='Actual')
    plt.xticks(x, sum_df['sample'])
    plt.xlabel('Sample')
    plt.ylabel(y_label)
    plt.title(f'{prop_name}: LOOCV + Bootstrap (5 runs)', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def linear_fit_with_r2(x, y):
    """Simple linear fit for parity plot annotation."""
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = r2_score(y, yhat)
    return slope, intercept, r2


def plot_parity(sum_df, prop_name, out_png, units):
    """
    Parity plot (actual vs predicted ± SD).

    The dashed y=x line represents ideal prediction. The solid fitted
    line with slope, intercept, and R² quantifies prediction agreement.
    Error bars show cross-run SD (model repeatability).
    """
    x = sum_df['actual'].values
    y = sum_df['pred_mean'].values
    yerr = sum_df['pred_sd'].values
    slope, intercept, r2 = linear_fit_with_r2(x, y)

    plt.figure(figsize=(6.2, 6.2))
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3,
                 label='Sample (predicted ± SD)')
    xymin = min(np.min(x), np.min(y))
    xymax = max(np.max(x), np.max(y))
    pad = 0.05 * (xymax - xymin) if xymax > xymin else 1.0
    plt.plot([xymin - pad, xymax + pad], [xymin - pad, xymax + pad],
             '--', label='y = x')
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, slope * xx + intercept, '-',
             label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r2:.3f}')
    plt.xlabel(f'Actual ({units})')
    plt.ylabel(f'Predicted mean ({units})')
    plt.title(f'Parity Plot - {prop_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_parity_runs(preds_runs, actual, prop_name, out_png, units):
    """
    Multi-run overlay parity plot.

    All 5 runs' actual-vs-predicted scatter points are superimposed to
    visualize inter-run variability and systematic bias. High overlap
    indicates stable predictions; dispersion suggests sensitivity to
    data perturbations.
    """
    plt.figure(figsize=(6.2, 6.2))
    for r in range(preds_runs.shape[0]):
        plt.scatter(actual, preds_runs[r], alpha=0.6, label=f'Run {r + 1}')
    xymin = min(np.min(actual), np.min(preds_runs))
    xymax = max(np.max(actual), np.max(preds_runs))
    pad = 0.05 * (xymax - xymin) if xymax > xymin else 1.0
    plt.plot([xymin - pad, xymax + pad], [xymin - pad, xymax + pad],
             '--', label='y = x')
    plt.xlabel(f'Actual ({units})')
    plt.ylabel(f'Predicted ({units})')
    plt.title(f'Parity Overlay - {prop_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_residual_hist(long_df_prop, out_png, prop_name, units):
    """
    Residual distribution histogram (all runs aggregated).

    Horizontal axis: (predicted - actual); vertical axis: count.
    KDE overlay aids in assessing distributional shape.
    The dashed zero line indicates unbiased prediction.
    """
    res = long_df_prop['predicted'] - long_df_prop['actual']
    plt.figure(figsize=(7, 4.2))
    sns.histplot(res, kde=True, bins=10)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel(f'Residual (predicted - actual) [{units}]')
    plt.ylabel('Count')
    plt.title(f'Residual Distribution - {prop_name}')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_residual_vs_pred(sum_df, out_png, prop_name, units):
    """
    Residual vs predicted mean scatter plot.

    Used to detect heteroscedasticity (variance changing with predicted
    magnitude) or structural patterns. Sample labels are annotated for
    identifying potential outliers or leverage points.
    """
    resid = sum_df['pred_mean'] - sum_df['actual']
    plt.figure(figsize=(7, 4.2))
    plt.scatter(sum_df['pred_mean'], resid)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    for i, name in enumerate(sum_df['sample']):
        plt.annotate(name, (sum_df['pred_mean'].iloc[i], resid.iloc[i]),
                     fontsize=8, xytext=(3, 3), textcoords='offset points')
    plt.xlabel(f'Predicted mean ({units})')
    plt.ylabel(f'Residual (predicted - actual) [{units}]')
    plt.title(f'Residual vs Predicted - {prop_name}')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_bland_altman(sum_df, out_png, prop_name, units):
    """
    Bland-Altman agreement analysis plot.

    Horizontal axis: (predicted + actual) / 2; vertical axis:
    (predicted - actual). Annotates bias (mean difference) and 95%
    limits of agreement (bias ± 1.96 × SD).
    """
    pred = sum_df['pred_mean'].values
    actual = sum_df['actual'].values
    mean_ab = (pred + actual) / 2.0
    diff = pred - actual
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    plt.figure(figsize=(7, 4.2))
    plt.scatter(mean_ab, diff)
    plt.axhline(bias, color='r', linestyle='-',
                label=f'Bias={bias:.3f}')
    plt.axhline(loa_upper, color='g', linestyle='--',
                label=f'+1.96 SD={loa_upper:.3f}')
    plt.axhline(loa_lower, color='g', linestyle='--',
                label=f'-1.96 SD={loa_lower:.3f}')
    plt.xlabel(f'Mean [(predicted + actual) / 2] ({units})')
    plt.ylabel(f'Difference (predicted - actual) [{units}]')
    plt.title(f'Bland-Altman - {prop_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_components_heatmap(comps_runs, out_png, prop_name):
    """
    Heatmap of optimal n_components per fold per run.

    Rows: samples (corresponding to outer LOOCV test samples).
    Columns: 5 independent runs.
    Values: n_components selected by inner LOOCV for each fold/run.
    Used to assess model complexity stability and sample sensitivity.
    """
    df_hm = pd.DataFrame(
        comps_runs.T,
        index=sample_names,
        columns=[f'Run{r + 1}' for r in range(comps_runs.shape[0])]
    )
    plt.figure(figsize=(6.5, 4.8))
    sns.heatmap(df_hm, annot=True, fmt='d', cmap='YlGnBu',
                cbar_kws={'label': 'n_components'})
    plt.title(f'Optimal n_components per Fold - {prop_name}')
    plt.ylabel('Sample')
    plt.xlabel('Run')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_vip_bar(vip_means_runs, out_png, prop_name):
    """
    VIP bar chart (cross-run mean ± SD).

    VIP is first averaged across outer folds within each run, then
    averaged and SD-computed across runs. The dashed red line at VIP=1
    marks the empirical importance threshold.
    """
    mean_across_runs = vip_means_runs.mean(axis=0)
    std_across_runs = vip_means_runs.std(axis=0, ddof=1)
    plt.figure(figsize=(8, 4.5))
    plt.bar(feature_names, mean_across_runs, yerr=std_across_runs,
            capsize=4)
    plt.axhline(1.0, linestyle='--', color='r', label='VIP = 1 threshold')
    plt.ylabel('VIP')
    plt.title(f'Variable Importance in Projection (VIP) - {prop_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


def plot_coef_bar(coef_means_runs, out_png, prop_name):
    """
    Standardized regression coefficient bar chart (cross-run mean ± SD).

    Coefficients are in z-scored space, reflecting the direction and
    relative magnitude of each descriptor's linear contribution. Used
    jointly with VIP for comprehensive structure-property interpretation.
    """
    mean_across_runs = coef_means_runs.mean(axis=0)
    std_across_runs = coef_means_runs.std(axis=0, ddof=1)
    plt.figure(figsize=(8, 4.5))
    plt.bar(feature_names, mean_across_runs, yerr=std_across_runs,
            capsize=4)
    plt.axhline(0.0, linestyle='--', color='k', linewidth=1)
    plt.ylabel('Standardized regression coefficient')
    plt.title(f'PLS Standardized Coefficients - {prop_name}')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")


# ================== 8) Generate All Figures ================== #

# Error-bar plots (three targets)
plot_errorbar(sum_E, "Young's modulus (E)", 'N/mm²', 'errorbar_E.png')
plot_errorbar(sum_TS, 'Tensile strength (TS)', 'N/mm²', 'errorbar_TS.png')
plot_errorbar(sum_EL, 'Elongation at break (EL)', '%', 'errorbar_EL.png')

# Parity plots (with linear fit and R²)
plot_parity(sum_E, "Young's modulus (E)", 'parity_E.png', 'N/mm²')
plot_parity(sum_TS, 'Tensile strength (TS)', 'parity_TS.png', 'N/mm²')
plot_parity(sum_EL, 'Elongation at break (EL)', 'parity_EL.png', '%')

# Multi-run overlay parity plots
plot_parity_runs(preds_E, Y_modulus, "Young's modulus (E)",
                 'parity_runs_E.png', 'N/mm²')
plot_parity_runs(preds_TS, Y_tensile, 'Tensile strength (TS)',
                 'parity_runs_TS.png', 'N/mm²')
plot_parity_runs(preds_EL, Y_elong, 'Elongation at break (EL)',
                 'parity_runs_EL.png', '%')

# Residual histograms (all runs aggregated)
plot_residual_hist(
    long_df[long_df['property'] == "Young's modulus (E)"],
    'residuals_hist_E.png', "Young's modulus (E)", 'N/mm²')
plot_residual_hist(
    long_df[long_df['property'] == 'Tensile strength (TS)'],
    'residuals_hist_TS.png', 'Tensile strength (TS)', 'N/mm²')
plot_residual_hist(
    long_df[long_df['property'] == 'Elongation at break (EL)'],
    'residuals_hist_EL.png', 'Elongation at break (EL)', '%')

# Residual vs predicted mean
plot_residual_vs_pred(sum_E, 'residuals_vs_pred_E.png',
                      "Young's modulus (E)", 'N/mm²')
plot_residual_vs_pred(sum_TS, 'residuals_vs_pred_TS.png',
                      'Tensile strength (TS)', 'N/mm²')
plot_residual_vs_pred(sum_EL, 'residuals_vs_pred_EL.png',
                      'Elongation at break (EL)', '%')

# Bland-Altman agreement plots
plot_bland_altman(sum_E, 'bland_altman_E.png',
                  "Young's modulus (E)", 'N/mm²')
plot_bland_altman(sum_TS, 'bland_altman_TS.png',
                  'Tensile strength (TS)', 'N/mm²')
plot_bland_altman(sum_EL, 'bland_altman_EL.png',
                  'Elongation at break (EL)', '%')

# n_components selection heatmaps
plot_components_heatmap(comps_E, 'components_heatmap_E.png',
                        "Young's modulus (E)")
plot_components_heatmap(comps_TS, 'components_heatmap_TS.png',
                        'Tensile strength (TS)')
plot_components_heatmap(comps_EL, 'components_heatmap_EL.png',
                        'Elongation at break (EL)')

# VIP bar charts (cross-run mean ± SD)
plot_vip_bar(vipm_E, 'vip_E.png', "Young's modulus (E)")
plot_vip_bar(vipm_TS, 'vip_TS.png', 'Tensile strength (TS)')
plot_vip_bar(vipm_EL, 'vip_EL.png', 'Elongation at break (EL)')

# Standardized coefficient bar charts (cross-run mean ± SD)
plot_coef_bar(coefm_E, 'coef_E.png', "Young's modulus (E)")
plot_coef_bar(coefm_TS, 'coef_TS.png', 'Tensile strength (TS)')
plot_coef_bar(coefm_EL, 'coef_EL.png', 'Elongation at break (EL)')

print("\n✅ Complete: All figures and CSV files have been generated.")
print("   Please check the current directory for outputs.")
