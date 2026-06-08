"""Visual comparison: frozen GMM vs aligned-envelope detector on our 30 s STM32 capture.

Two-panel time-series with detection spans overlaid (fig6-style). No classifier.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase
from extract import matlab_smooth, slide_features, assign_clusters_frozen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "중간발표", "figures")
os.makedirs(OUT, exist_ok=True)

CALIBRATED_SCALE = 3.7253e-7
FS = 8000
FOOTFALL_LEN = 1500
PEAK_OFFSET = 400

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_signal_8k():
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
    volts = raw * CALIBRATED_SCALE
    sp = StreamingPolyphase(p=8, q=15)
    return sp.feed(volts)


def load_gmm():
    return dict(np.load(os.path.join(os.path.dirname(__file__), "gmm_params.npz")))


def gmm_peaks(signal, gmm):
    g = matlab_smooth(signal, span=5)
    feats, indices = slide_features(g, fs=FS)
    labels = assign_clusters_frozen(
        feats, gmm["mu"], gmm["cov"], gmm["phi"],
        int(gmm["event_cluster"]), float(gmm["threshold"]),
    )
    event_idx = np.flatnonzero(labels == 1)
    peaks = []
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
            peaks.append(peak_local + evnt_start)
        j += 1
    return peaks, g


def envelope_peaks_aligned(signal, k_mad=8.0, env_smooth_ms=30.0,
                            min_sep_ms=120.0, align_radius_ms=30.0):
    g = matlab_smooth(signal, span=5)
    env = np.abs(ss.hilbert(g))
    n_smooth = max(1, int(env_smooth_ms * 1e-3 * FS))
    kern = np.ones(n_smooth) / n_smooth
    env_s = np.convolve(env, kern, mode="same")
    med = np.median(env_s)
    mad = np.median(np.abs(env_s - med))
    thr = med + k_mad * mad * 1.4826
    min_dist = max(1, int(min_sep_ms * 1e-3 * FS))
    env_pks, _ = ss.find_peaks(env_s, height=thr, distance=min_dist)
    radius = max(1, int(align_radius_ms * 1e-3 * FS))
    aligned = []
    for p in env_pks:
        lo = max(0, p - radius)
        hi = min(len(g), p + radius)
        local_peak = int(np.argmax(np.abs(g[lo:hi]))) + lo
        aligned.append(local_peak)
    return aligned, g, env_s, thr


def spans_from_peaks(peaks, L):
    spans = []
    for p in peaks:
        s = max(0, p - PEAK_OFFSET)
        e = min(L, s + FOOTFALL_LEN)
        spans.append((s, e, p))
    return spans


def _panel(ax, t, volts_mv, spans, title, env_mv=None, thr_mv=None):
    ax.plot(t, volts_mv, lw=0.5, color="#1f4e9c", label="signal (8 kHz)")
    for (s, e, p) in spans:
        ax.axvspan(s / FS, e / FS, color="#E45756",
                   alpha=0.18, linewidth=0)
    if spans:
        peaks_t = [p / FS for _, _, p in spans]
        peaks_v = [volts_mv[p] for _, _, p in spans]
        ax.scatter(peaks_t, peaks_v, s=18, color="#E45756", zorder=5,
                   label="detected peaks ({})".format(len(spans)))
    if env_mv is not None:
        ax2 = ax.twinx()
        ax2.plot(t, env_mv, lw=0.7, color="#2E8B57", alpha=0.6,
                 label="Hilbert envelope")
        if thr_mv is not None:
            ax2.axhline(thr_mv, lw=0.7, ls="--", color="#2E8B57",
                        alpha=0.7, label="threshold")
        ax2.set_ylabel("envelope (mV)", color="#2E8B57", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#2E8B57", labelsize=9)
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(title, loc="left", pad=6, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)


def main():
    sig = load_signal_8k()
    gmm = load_gmm()
    t = np.arange(len(sig)) / FS
    volts_mv = sig * 1000

    gp, _ = gmm_peaks(sig, gmm)
    ep, _, env, thr = envelope_peaks_aligned(sig)

    print("GMM detections     : {}".format(len(gp)))
    print("Envelope detections: {}".format(len(ep)))

    spans_g = spans_from_peaks(gp, len(sig))
    spans_e = spans_from_peaks(ep, len(sig))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    _panel(axes[0], t, volts_mv, spans_g,
           "A. Frozen P14 GMM, threshold=0.90  (current pipeline, = fig6)")
    _panel(axes[1], t, volts_mv, spans_e,
           "B. Hilbert envelope + peak align  (proposed)",
           env_mv=env * 1000, thr_mv=thr * 1000)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Footstep detection: GMM vs envelope on our 30 s STM32 capture",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = os.path.join(OUT, "expt_envelope_vs_gmm_30s.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n[saved]", out)


if __name__ == "__main__":
    main()
