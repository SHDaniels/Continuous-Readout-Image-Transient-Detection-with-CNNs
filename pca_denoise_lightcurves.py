"""
PCA denoising for a batch of light curves (time series) with elbow-method
selection of the number of components.

Expected input
--------------
X : np.ndarray, shape (n_curves, n_timepoints)
    Each row is one light curve, sampled on a COMMON time grid
    (same length, same cadence). If your curves have different
    lengths/time grids, interpolate them onto a shared grid first.

What this script does
----------------------
1. Mean-centers each curve (and optionally standardizes).
2. Fits PCA on the full set of curves.
3. Plots the explained-variance / scree curve and finds the "elbow"
   automatically (max-curvature / distance-to-chord method), while
   letting you override the chosen k by eye.
4. Reconstructs each curve using only the first k components ->
   this is your denoised set.

Usage
-----
    python pca_denoise_lightcurves.py

Or import the functions in your own notebook:

    from pca_denoise_lightcurves import pca_denoise, plot_elbow
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def find_elbow(explained_variance_ratio: np.ndarray) -> int:
    """
    Automatic elbow detection on a scree curve using the
    'max distance to the chord' method (a simple, robust kneedle variant).

    Returns the 1-indexed number of components at the elbow.
    """
    y = np.cumsum(explained_variance_ratio)
    n = len(y)
    x = np.arange(1, n + 1)

    # Normalize both axes to [0, 1] so geometry is comparable
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Line from first to last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Perpendicular distance of each point from the chord
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi]) - p1
        proj_len = np.dot(p, line_vec_norm)
        proj_point = proj_len * line_vec_norm
        distances.append(np.linalg.norm(p - proj_point))

    elbow_idx = int(np.argmax(distances))
    return x[elbow_idx]  # 1-indexed component count


def plot_elbow(explained_variance_ratio: np.ndarray, elbow_k: int, save_path: str = None):
    """Plot both the per-component scree curve and cumulative variance, marking the elbow."""
    n = len(explained_variance_ratio)
    cum = np.cumsum(explained_variance_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(np.arange(1, n + 1), explained_variance_ratio, "o-", color="#3b6ea5")
    axes[0].axvline(elbow_k, color="crimson", ls="--", label=f"elbow @ k={elbow_k}")
    axes[0].set_xlabel("Component #")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("Scree plot")
    axes[0].legend()

    axes[1].plot(np.arange(1, n + 1), cum, "o-", color="#3b6ea5")
    axes[1].axvline(elbow_k, color="crimson", ls="--")
    axes[1].axhline(cum[elbow_k - 1], color="crimson", ls=":", alpha=0.6)
    axes[1].set_xlabel("Component #")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title(f"Cumulative variance (k={elbow_k} -> {cum[elbow_k-1]*100:.1f}%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def pca_denoise(X: np.ndarray, k: int = None, standardize: bool = False):
    """
    Denoise light curves via PCA reconstruction with k components.

    Parameters
    ----------
    X : (n_curves, n_timepoints) array
    k : int or None
        Number of components to keep. If None, chosen automatically via
        the elbow method.
    standardize : bool
        If True, divide each curve by its std after mean-centering
        (use this if curves have very different amplitudes/units).

    Returns
    -------
    X_denoised : (n_curves, n_timepoints) array, back in original scale
    info : dict with keys 'k', 'pca', 'mean', 'std', 'explained_variance_ratio'
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean

    std = None
    if standardize:
        std = Xc.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        Xc = Xc / std

    pca_full = PCA()
    pca_full.fit(Xc)
    evr = pca_full.explained_variance_ratio_

    if k is None:
        k = find_elbow(evr)

    pca = PCA(n_components=k)
    scores = pca.fit_transform(Xc)
    Xc_reconstructed = pca.inverse_transform(scores)

    if standardize:
        Xc_reconstructed = Xc_reconstructed * std
    X_denoised = Xc_reconstructed + mean

    info = {
        "k": k,
        "pca": pca,
        "mean": mean,
        "std": std,
        "explained_variance_ratio": evr,
    }
    return X_denoised, info


if __name__ == "__main__":
    # ---- Demo with synthetic light curves ----
    rng = np.random.default_rng(0)
    n_curves, n_points = 200, 300
    t = np.linspace(0, 10, n_points)

    # Shared underlying signal shapes (what PCA should recover)
    base1 = np.sin(2 * np.pi * t / 3.0)
    base2 = np.exp(-0.5 * ((t - 5) / 1.5) ** 2)  # transit-like dip shape

    X_clean = np.outer(rng.normal(1, 0.3, n_curves), base1) + \
              np.outer(rng.normal(0.5, 0.2, n_curves), base2)
    X_noisy = X_clean + rng.normal(0, 0.4, size=X_clean.shape)

    X_denoised, info = pca_denoise(X_noisy, k=None, standardize=False)
    print(f"Elbow selected k = {info['k']} components")

    plot_elbow(info["explained_variance_ratio"], info["k"],
               save_path="/mnt/user-data/outputs/elbow_plot.png")

    # Compare one example curve
    plt.figure(figsize=(9, 4))
    plt.plot(t, X_noisy[0], alpha=0.4, label="noisy")
    plt.plot(t, X_clean[0], "k--", label="true signal")
    plt.plot(t, X_denoised[0], color="crimson", label="PCA denoised")
    plt.legend()
    plt.title("Example light curve: noisy vs. denoised")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/example_curve.png", dpi=150)
    plt.show()