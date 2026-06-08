"""Zoom into the 5-9 s dense-burst region of our 30 s capture.

If inter-peak intervals are ~100-200 ms → heel-toe pattern within each step
(= 1 step has 2 peaks, walking cadence still normal ~1.5 Hz step rate).
If intervals are ~300+ ms → user was walking briskly (fast cadence).

Outputs: 중간발표/figures/expt_dense_burst_zoom.png + console interval stats.
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
from extract import matlab_smooth, apply_envelope_detector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "중간발표", "figures")
CALIBRATED_SCALE = 3.7253e-7
FS = 8000


def main():
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
    volts = raw * CALIBRATED_SCALE
    sp = StreamingPolyphase(p=8, q=15)
    sig = sp.feed(volts)
    g = matlab_smooth(sig, span=5)
    env = np.abs(ss.hilbert(g))
    kern = np.ones(240) / 240  # 30 ms smooth
    env_s = np.convolve(env, kern, mode="same")

    # full-signal envelope peaks (production detector parameters)
    fs_crops = apply_envelope_detector(sig)
    # Recompute peak times via envelope find_peaks (same params)
    med = np.median(env_s); mad = np.median(np.abs(env_s - med))
    thr = med + 8 * mad * 1.4826
    peaks_all, _ = ss.find_peaks(env_s, height=thr, distance=int(0.120 * FS))
    # snap to argmax(|signal|) within ±30 ms (matches apply_envelope_detector)
    radius = int(0.030 * FS)
    aligned = []
    for p in peaks_all:
        lo = max(0, int(p) - radius); hi = min(len(g), int(p) + radius)
        aligned.append(int(np.argmax(np.abs(g[lo:hi]))) + lo)
    peaks_all = np.asarray(aligned)
    peak_t = peaks_all / FS

    # zoom region: 5-9 s
    mask_zoom = (peak_t >= 5.0) & (peak_t <= 9.0)
    zoom_peaks = peaks_all[mask_zoom]
    zoom_t = zoom_peaks / FS
    intervals_ms = np.diff(zoom_t) * 1000

    print("Dense-burst region 5-9 s:")
    print("  # peaks: {}".format(len(zoom_peaks)))
    print("  Inter-peak intervals (ms):", np.round(intervals_ms, 1).tolist())
    print("  median: {:.1f} ms,  mean: {:.1f} ms,  std: {:.1f} ms".format(
        float(np.median(intervals_ms)), float(np.mean(intervals_ms)),
        float(np.std(intervals_ms))))
    # heel-toe pattern: alternating short / long intervals
    if len(intervals_ms) >= 4:
        odd = intervals_ms[0::2]; even = intervals_ms[1::2]
        print("  odd (1st, 3rd, ... interval):  median {:.1f} ms".format(float(np.median(odd))))
        print("  even (2nd, 4th, ... interval): median {:.1f} ms".format(float(np.median(even))))
        print("  → if alternation: shorter = within-step (heel→toe), longer = step-to-step")

    # also compute for a clean isolated peak region (22-25 s)
    mask_iso = (peak_t >= 22.0) & (peak_t <= 25.0)
    iso_peaks = peaks_all[mask_iso]
    iso_intervals = np.diff(iso_peaks / FS) * 1000
    print("\nIsolated region 22-25 s:")
    print("  # peaks: {}".format(len(iso_peaks)))
    print("  intervals (ms):", np.round(iso_intervals, 1).tolist())

    # ----------- plot
    t = np.arange(len(sig)) / FS
    volts_mv = sig * 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharey=False)
    # top: zoom on 5-9 s
    m1 = (t >= 5.0) & (t <= 9.0)
    ax1.plot(t[m1], volts_mv[m1], lw=0.6, color="#1f4e9c")
    for p in zoom_peaks:
        ax1.axvline(p / FS, lw=0.6, color="#E45756", alpha=0.5)
        ax1.scatter(p / FS, volts_mv[p], s=20, color="#E45756", zorder=5)
    for i, p in enumerate(zoom_peaks[:-1]):
        dt = (zoom_peaks[i+1] - p) / FS * 1000
        ax1.annotate("{:.0f}ms".format(dt),
                     xy=((zoom_peaks[i] + zoom_peaks[i+1]) / 2 / FS,
                         max(volts_mv[zoom_peaks[i]], volts_mv[zoom_peaks[i+1]]) * 1.1),
                     ha="center", fontsize=8, color="#666")
    ax1.set_title("Zoom: 5-9 s dense burst   (peaks marked, inter-peak ms shown)",
                  fontweight="bold", loc="left")
    ax1.set_ylabel("Voltage (mV)")
    ax1.grid(alpha=0.3)

    # bottom: histogram of inter-peak intervals across whole signal
    all_intervals = np.diff(peaks_all / FS) * 1000
    ax2.hist(all_intervals, bins=np.arange(0, 1000, 25),
             color="#1f4e9c", edgecolor="white", alpha=0.85)
    ax2.axvline(150, color="#2E8B57", lw=1.2, ls="--",
                label="heel-toe (~150 ms)")
    ax2.axvline(660, color="#E45756", lw=1.2, ls="--",
                label="walking step (~660 ms)")
    ax2.set_xlabel("Inter-peak interval (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Inter-peak interval histogram (full 30 s)",
                  fontweight="bold", loc="left")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT, "expt_dense_burst_zoom.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n[saved]", out)


if __name__ == "__main__":
    main()
