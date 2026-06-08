"""Compare GMM event-detection variants on our 30 s STM32 capture.

Variants
--------
A. Frozen P14 GMM, threshold=0.90  (= current fig6 baseline)
B. Frozen P14 GMM, threshold=0.50  (loosen acceptance)
C. Re-fit K=2 GMM on our 30 s, threshold=0.90
D. Re-fit K=3 GMM on our 30 s, threshold=0.90  (allow dense-burst mode)
E. Hilbert-envelope pre-detector, no GMM (peak-finder on |analytic signal|)

Output
------
중간발표/figures/expt_gmm_variants.png  — 5-panel time-series overlay
stdout summary table (#detections per variant, span density, etc.)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase
from extract import (
    matlab_smooth, slide_features, assign_clusters_frozen, extract_footsteps,
    gmm_em,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "중간발표", "figures")
os.makedirs(OUT, exist_ok=True)

CALIBRATED_SCALE = 3.7253e-7
FS_OUT = 8000
FOOTFALL_LEN = 1500
PEAK_OFFSET = 400

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------------------------------------------------ data load

def load_signal_8k():
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
    volts = raw * CALIBRATED_SCALE
    sp = StreamingPolyphase(p=8, q=15)
    return sp.feed(volts)


def load_frozen_gmm():
    z = np.load(os.path.join(os.path.dirname(__file__), "gmm_params.npz"))
    return dict(z)


# --------------------------------------------------- GMM-based span extraction

def _spans_from_labels(g, indices, labels):
    """Replicate extract_footsteps span logic; return list of (start, stop, peak_idx)."""
    event_idx = np.flatnonzero(labels == 1)
    spans = []
    if len(event_idx) == 0:
        return spans
    j = 0
    n = len(event_idx)
    while j < n - 1:
        run_start = j
        while j < n - 1 and event_idx[j + 1] == event_idx[j] + 1:
            j += 1
        run_end = j
        evnt_start = int(indices[event_idx[run_start], 0])
        evnt_stop = int(indices[event_idx[run_end], 1])
        seg = g[evnt_start:evnt_stop]
        if len(seg) > 0:
            peak_local = int(np.argmax(np.abs(seg)))
            mn = peak_local + evnt_start
            strt = max(0, mn - PEAK_OFFSET)
            stop = min(len(g), strt + FOOTFALL_LEN)
            spans.append((strt, stop, mn))
        j += 1
    return spans


def variant_A_frozen_thr090(sig8k, frozen):
    g = matlab_smooth(sig8k, span=5)
    feats, indices = slide_features(g, fs=FS_OUT)
    labels = assign_clusters_frozen(
        feats, frozen["mu"], frozen["cov"], frozen["phi"],
        int(frozen["event_cluster"]), threshold=0.90,
    )
    return _spans_from_labels(g, indices, labels), g


def variant_B_frozen_thr050(sig8k, frozen):
    g = matlab_smooth(sig8k, span=5)
    feats, indices = slide_features(g, fs=FS_OUT)
    labels = assign_clusters_frozen(
        feats, frozen["mu"], frozen["cov"], frozen["phi"],
        int(frozen["event_cluster"]), threshold=0.50,
    )
    return _spans_from_labels(g, indices, labels), g


def _refit_gmm(feats, K, seed=0):
    """Fit K-cluster GMM on our features. Pick event clusters by mean(std) > median."""
    gmm = gmm_em(feats, K=K, seed=seed)
    # event clusters = those with mean above the lowest mean (i.e., not the noise cluster)
    # use feature[0] = std as the "loudness" axis
    noise_k = int(np.argmin(gmm.mu[:, 0]))
    event_ks = [k for k in range(K) if k != noise_k]
    return gmm, noise_k, event_ks


def _refit_labels(feats, gmm, event_ks, threshold):
    """Posterior assignment for a re-fit GMM with possibly multiple event clusters."""
    K = gmm.mu.shape[0]
    p = np.zeros((len(feats), K))
    for k in range(K):
        p[:, k] = multivariate_normal.pdf(
            feats, gmm.mu[k], gmm.cov[k], allow_singular=True) * gmm.phi[k]
    s = p.sum(axis=1, keepdims=True)
    p_norm = p / np.where(s > 0, s, 1.0)
    argmax = p_norm.argmax(axis=1)
    max_p = p_norm.max(axis=1)
    labels = np.where(np.isin(argmax, event_ks), 1, 0)
    labels[max_p < threshold] = 0
    return labels


def variant_C_refit_K2(sig8k):
    g = matlab_smooth(sig8k, span=5)
    feats, indices = slide_features(g, fs=FS_OUT)
    gmm, noise_k, event_ks = _refit_gmm(feats, K=2, seed=0)
    labels = _refit_labels(feats, gmm, event_ks, threshold=0.90)
    return _spans_from_labels(g, indices, labels), g, gmm


def variant_D_refit_K3(sig8k):
    g = matlab_smooth(sig8k, span=5)
    feats, indices = slide_features(g, fs=FS_OUT)
    gmm, noise_k, event_ks = _refit_gmm(feats, K=3, seed=0)
    labels = _refit_labels(feats, gmm, event_ks, threshold=0.90)
    return _spans_from_labels(g, indices, labels), g, gmm


# --------------------------------------------------- Hilbert envelope detector

def variant_E_envelope(sig8k, env_smooth_ms=30.0, k_mad=8.0, min_sep_ms=120.0):
    """Hilbert envelope + adaptive threshold + minimum peak separation.

    - analytic |signal| smoothed over env_smooth_ms (rectangular)
    - threshold = median + k_mad * MAD  (robust to outliers)
    - scipy.signal.find_peaks with min separation
    - crop ±FOOTFALL_LEN around each peak (peak_offset=400 to match training)
    """
    g = matlab_smooth(sig8k, span=5)
    analytic = ss.hilbert(g)
    env = np.abs(analytic)
    n_smooth = max(1, int(env_smooth_ms * 1e-3 * FS_OUT))
    if n_smooth > 1:
        kern = np.ones(n_smooth) / n_smooth
        env = np.convolve(env, kern, mode="same")
    med = np.median(env)
    mad = np.median(np.abs(env - med))
    thr = med + k_mad * mad * 1.4826  # MAD→σ
    min_dist = max(1, int(min_sep_ms * 1e-3 * FS_OUT))
    peaks, _ = ss.find_peaks(env, height=thr, distance=min_dist)
    spans = []
    for p in peaks:
        strt = max(0, p - PEAK_OFFSET)
        stop = min(len(g), strt + FOOTFALL_LEN)
        spans.append((strt, stop, int(p)))
    return spans, g, env, thr


# ----------------------------------------------------------------- plotting

def _panel(ax, t, volts_mv, spans, title, info_text=None, env=None, thr=None):
    ax.plot(t, volts_mv, lw=0.4, color="#1f4e9c")
    for (s, e, mn) in spans:
        ax.axvspan(s / FS_OUT, e / FS_OUT, color="#E45756",
                   alpha=0.18, linewidth=0)
    if spans:
        peaks_t = [mn / FS_OUT for _, _, mn in spans]
        peaks_v = [volts_mv[mn] for _, _, mn in spans]
        ax.scatter(peaks_t, peaks_v, s=12, color="#E45756", zorder=5)
    if env is not None:
        ax2 = ax.twinx()
        ax2.plot(t, env * 1000, lw=0.6, color="#2E8B57", alpha=0.7,
                 label="envelope")
        if thr is not None:
            ax2.axhline(thr * 1000, lw=0.7, ls="--", color="#2E8B57",
                        alpha=0.6, label="thr")
        ax2.set_ylabel("env (mV)", color="#2E8B57", fontsize=8)
        ax2.tick_params(axis='y', labelcolor="#2E8B57", labelsize=8)
        ax2.spines["right"].set_visible(True)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("V (mV)")
    ax.set_title(title, loc="left", pad=4, fontweight="bold")
    ax.grid(alpha=0.25)
    if info_text:
        ax.text(0.99, 0.95, info_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#999", alpha=0.85))


def main():
    sig8k = load_signal_8k()
    frozen = load_frozen_gmm()
    t = np.arange(len(sig8k)) / FS_OUT
    volts_mv = sig8k * 1000

    print("Signal: {} samples @ {} Hz = {:.1f} s, range {:.1f}..{:.1f} mV".format(
        len(sig8k), FS_OUT, len(sig8k) / FS_OUT, volts_mv.min(), volts_mv.max()))

    spA, _ = variant_A_frozen_thr090(sig8k, frozen)
    spB, _ = variant_B_frozen_thr050(sig8k, frozen)
    spC, _, gmmC = variant_C_refit_K2(sig8k)
    spD, _, gmmD = variant_D_refit_K3(sig8k)
    spE, _, env, thr = variant_E_envelope(sig8k)

    summary = [
        ("A: frozen P14, thr=0.90",  spA, "(= current fig6)"),
        ("B: frozen P14, thr=0.50",  spB, ""),
        ("C: re-fit K=2 on our 30s, thr=0.90", spC, ""),
        ("D: re-fit K=3 on our 30s, thr=0.90", spD, ""),
        ("E: Hilbert envelope (no GMM)", spE, "k_mad=8, min_sep=120ms"),
    ]
    print("\n{:<40} {:>6}  notes".format("variant", "#det"))
    print("-" * 70)
    for name, sp, note in summary:
        print("{:<40} {:>6}  {}".format(name, len(sp), note))

    # K=3 cluster centroids for context
    print("\nK=3 cluster means (std, kurt, rms, q25, FFT40-80, 80-120, 120-160):")
    for k in range(3):
        print("  k={}: phi={:.3f}  mu={}".format(
            k, gmmD.phi[k],
            np.array2string(gmmD.mu[k], precision=3, suppress_small=True)))

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, sharey=False)
    panels = [
        (axes[0], spA, "A. Frozen P14 GMM, threshold=0.90  (current baseline = fig6)",
         "footsteps: {}".format(len(spA))),
        (axes[1], spB, "B. Frozen P14 GMM, threshold=0.50  (loosen)",
         "footsteps: {}".format(len(spB))),
        (axes[2], spC, "C. Re-fit GMM (K=2) on our 30 s, threshold=0.90",
         "footsteps: {}\nphi noise={:.2f}".format(len(spC), gmmC.phi.min())),
        (axes[3], spD, "D. Re-fit GMM (K=3) on our 30 s, threshold=0.90",
         "footsteps: {}\nphi={}".format(
             len(spD), np.array2string(gmmD.phi, precision=2))),
        (None, None, None, None),
    ]
    for ax, sp, title, info in panels[:-1]:
        _panel(ax, t, volts_mv, sp, title, info_text=info)

    # Variant E with envelope overlay
    _panel(axes[4], t, volts_mv, spE,
           "E. Hilbert envelope + peak-find (no GMM)",
           info_text="footsteps: {}\nk_mad=8".format(len(spE)),
           env=env, thr=thr)
    axes[4].set_xlabel("Time (s)")

    fig.suptitle(
        "GMM detection variants on our 30 s STM32 capture",
        fontsize=13, fontweight="bold", y=0.998)
    fig.tight_layout()
    out = os.path.join(OUT, "expt_gmm_variants.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n[saved]", out)


if __name__ == "__main__":
    main()
