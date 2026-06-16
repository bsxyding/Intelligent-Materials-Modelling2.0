"""
Applicability Domain (AD) analysis for the 10-sample PSF membrane dataset.
Three components:
  (a) PCA score plot of autoscaled descriptors with convex hull (global coverage)
  (b) Fold-wise leverage of each LOOCV held-out sample w.r.t. its 9 training samples
  (c) Fold-wise range check: is the held-out sample inside the training min-max range?
Only descriptor data are used; no model inference is required.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# ----- dataset (Table 1 of the manuscript) -----
samples = [f"S{i}" for i in range(1, 11)]
desc_names = ["PD (\u00b5m)", "CA (\u00b0)", "T (mm)", "P (%)"]
X = np.array([
    [0.522, 94.6, 0.273, 77.67],
    [0.364, 93.3, 0.202, 79.16],
    [0.569, 78.9, 0.175, 78.87],
    [0.451, 66.4, 0.136, 73.60],
    [0.408, 66.6, 0.154, 73.49],
    [0.336, 79.5, 0.177, 69.09],
    [0.403, 81.3, 0.240, 71.18],
    [0.319, 84.5, 0.180, 78.32],
    [0.842, 90.0, 0.224, 78.86],
    [0.298, 82.9, 0.188, 74.02],
])
Y = np.array([
    [117.17, 4.82, 42.07],
    [90.18, 3.97, 46.61],
    [152.50, 6.56, 49.13],
    [231.78, 9.61, 61.30],
    [182.59, 7.43, 60.69],
    [235.65, 9.65, 64.46],
    [176.51, 7.25, 69.04],
    [126.42, 5.22, 42.91],
    [107.67, 4.53, 51.25],
    [168.57, 7.13, 67.15],
])
targ_names = ["E (N/mm2)", "TS (N/mm2)", "EL (%)"]
n, p = X.shape

# ----- Table S1: descriptor & target statistics -----
def stats_row(v):
    return v.min(), v.max(), v.mean(), v.std(ddof=1), 100 * v.std(ddof=1) / v.mean()

print("==== Table S1: variable statistics (min, max, mean, SD, CV%) ====")
for j, nm in enumerate(desc_names):
    print(f"{nm:10s}: min={X[:,j].min():8.3f} max={X[:,j].max():8.3f} "
          f"mean={X[:,j].mean():8.3f} SD={X[:,j].std(ddof=1):7.3f} CV={100*X[:,j].std(ddof=1)/X[:,j].mean():6.1f}%")
for j, nm in enumerate(targ_names):
    print(f"{nm:10s}: min={Y[:,j].min():8.2f} max={Y[:,j].max():8.2f} "
          f"mean={Y[:,j].mean():8.2f} SD={Y[:,j].std(ddof=1):7.2f} CV={100*Y[:,j].std(ddof=1)/Y[:,j].mean():6.1f}%")

# ----- (a) Global PCA on autoscaled descriptors -----
Xz = (X - X.mean(0)) / X.std(0, ddof=1)
U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
scores = U * S                      # PCA scores
expl = S**2 / np.sum(S**2) * 100    # variance explained
# sign convention: make loading of PD on PC1 positive for readability
for k in range(2):
    if Vt[k, 0] < 0:
        Vt[k, :] *= -1
        scores[:, k] *= -1
print("\n==== PCA ====")
print("Variance explained (%):", np.round(expl, 1))
print("PC1 loadings:", dict(zip(desc_names, np.round(Vt[0], 3))))
print("PC2 loadings:", dict(zip(desc_names, np.round(Vt[1], 3))))

# ----- (b) Fold-wise leverage (Williams-style, on autoscaled training data) -----
lev = np.zeros(n)
for i in range(n):
    tr = np.delete(np.arange(n), i)
    mu, sd = X[tr].mean(0), X[tr].std(0, ddof=1)
    Xt = (X[tr] - mu) / sd
    x0 = (X[i] - mu) / sd
    lev[i] = x0 @ np.linalg.inv(Xt.T @ Xt) @ x0
h_star = 3 * p / (n - 1)   # 3p/n_train = 12/9
print("\n==== Fold-wise leverage (h* = 3p/n_train = %.3f) ====" % h_star)
for s, h in zip(samples, lev):
    print(f"{s}: h = {h:.3f} {'  <-- exceeds h*' if h > h_star else ''}")

# ----- (c) Fold-wise range check -----
outside = np.zeros((n, p), dtype=bool)
margin = np.zeros((n, p))
for i in range(n):
    tr = np.delete(np.arange(n), i)
    lo, hi = X[tr].min(0), X[tr].max(0)
    below, above = X[i] < lo, X[i] > hi
    outside[i] = below | above
    rng = hi - lo
    margin[i] = np.where(below, (lo - X[i]) / rng, np.where(above, (X[i] - hi) / rng, 0.0)) * 100
print("\n==== Fold-wise range check (held-out value outside training min-max) ====")
for i in range(n):
    flags = [f"{desc_names[j]} by {margin[i,j]:.1f}% of range" for j in range(p) if outside[i, j]]
    print(f"{samples[i]}: " + ("; ".join(flags) if flags else "inside on all four descriptors"))
n_extrap_folds = int(np.sum(outside.any(1)))
print(f"\nFolds with at least one out-of-range descriptor: {n_extrap_folds}/10")

# ================= FIGURE S1 =================
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})
fig = plt.figure(figsize=(11.0, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32,
                      left=0.055, right=0.975, top=0.88, bottom=0.16)

C_PT, C_HULL, C_BAR, C_THR, C_OUT = "#1F4E79", "#9DC3E6", "#5B9BD5", "#C00000", "#C00000"

# --- (a) PCA score plot with convex hull ---
ax = fig.add_subplot(gs[0])
hull = ConvexHull(scores[:, :2])
poly = Polygon(scores[hull.vertices, :2], closed=True, facecolor=C_HULL,
               alpha=0.30, edgecolor=C_PT, linewidth=1.0, linestyle="--", zorder=1)
ax.add_patch(poly)
ax.scatter(scores[:, 0], scores[:, 1], s=42, c=C_PT, zorder=3, edgecolor="white", linewidth=0.6)
offs = {"S1": (6, 4), "S2": (6, -2), "S3": (6, 2), "S4": (6, -4), "S5": (-22, -4),
        "S6": (6, -2), "S7": (6, 0), "S8": (-22, 2), "S9": (-24, 4), "S10": (6, 2)}
for i, s in enumerate(samples):
    dx, dy = offs.get(s, (6, 3))
    ax.annotate(s, scores[i, :2], textcoords="offset points", xytext=(dx, dy), fontsize=8)
ax.axhline(0, color="0.8", lw=0.6, zorder=0)
ax.axvline(0, color="0.8", lw=0.6, zorder=0)
ax.set_xlabel(f"PC1 ({expl[0]:.1f}% explained variance)")
ax.set_ylabel(f"PC2 ({expl[1]:.1f}% explained variance)")
ax.set_title("(a) Descriptor-space coverage (PCA, autoscaled)", fontsize=9.5, loc="left")
pad_x = 0.06 * (scores[:,0].max() - scores[:,0].min())
ax.set_xlim(scores[:,0].min()-4*pad_x, scores[:,0].max()+4*pad_x)

# --- (b) fold-wise leverage ---
ax2 = fig.add_subplot(gs[1])
bars = ax2.bar(samples, lev, color=C_BAR, width=0.62, edgecolor="white", linewidth=0.5)
for b, h in zip(bars, lev):
    if h > h_star:
        b.set_color(C_OUT)
ax2.axhline(h_star, color=C_THR, lw=1.1, ls="--")
ax2.text(0.02, h_star * 1.03, r"$h^{*}=3p/n_{\mathrm{train}}=%.2f$" % h_star,
         color=C_THR, fontsize=8.2, va="bottom")
ax2.set_ylabel("Leverage $h$ of held-out sample")
ax2.set_ylim(0, max(lev.max() * 1.18, h_star * 1.25))
ax2.set_title("(b) Fold-wise leverage (LOOCV)", fontsize=9.5, loc="left")
ax2.tick_params(axis="x", rotation=0)

# --- (c) fold-wise range check heat map ---
ax3 = fig.add_subplot(gs[2])
M = margin.T  # descriptors x folds, % of training range beyond bound (0 = inside)
vmax = max(M.max(), 1e-9)
im = ax3.imshow(M, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)
ax3.set_xticks(range(n)); ax3.set_xticklabels(samples)
ax3.set_yticks(range(p)); ax3.set_yticklabels([d.split(" ")[0] for d in desc_names])
for i in range(p):
    for j in range(n):
        if outside[j, i]:
            ax3.text(j, i, f"{M[i,j]:.0f}%", ha="center", va="center",
                     fontsize=7.6, color="white" if M[i, j] > 0.55 * vmax else "#7A1010",
                     fontweight="bold")
ax3.set_title("(c) Out-of-range margin (% of range)", fontsize=9.5, loc="left")
cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
cb.set_label("", fontsize=8)
cb.ax.tick_params(labelsize=7.5)

fig.savefig("./Figure_S1_applicability_domain.png", dpi=300)
fig.savefig("./Figure_S1_applicability_domain.pdf")
print("\nFigure saved.")
