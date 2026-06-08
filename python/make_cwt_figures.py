"""Generate CWT / GMM presentation figures for the interim talk.

Outputs (PNG, 200 dpi) under 중간발표/figures/:
    fig5_cwt_3panel.png      — single footstep: raw → scalogram → 224×224 LUT
    fig6_gmm_detection.png   — 30 s capture with frozen-GMM event spans
    fig7_cross_person.png    — P14/P50/P100 footstep CWT comparison
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase
from extract import (
    matlab_smooth, slide_features, assign_clusters_frozen, extract_footsteps,
)
from cwt_fast import cwt
from render_lut import cwt_to_rgb_direct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "중간발표", "figures")
os.makedirs(OUT, exist_ok=True)

CALIBRATED_SCALE = 3.7253e-7
FS_OUT = 8000

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_gmm():
    return dict(np.load(os.path.join(os.path.dirname(__file__), "gmm_params.npz")))


def _detect_spans(signal, gmm, fs=FS_OUT, footfall_len=1500, peak_offset=400):
    """Mirror extract_footsteps but also return (start, stop) per footstep."""
    g = matlab_smooth(signal, span=5)
    feats, indices = slide_features(g, fs=fs)
    labels = assign_clusters_frozen(
        feats, gmm["mu"], gmm["cov"], gmm["phi"],
        gmm["event_cluster"], gmm["threshold"],
    )
    event_idx = np.flatnonzero(labels == 1)
    spans = []
    if len(event_idx) == 0:
        return spans, g
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
            strt = max(0, mn - peak_offset)
            stop = min(len(g), strt + footfall_len)
            spans.append((strt, stop, mn))
        j += 1
    return spans, g


def _load_our_8k():
    """STM32 30 s capture → calibrated volts → polyphase 8 kHz."""
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
    volts = raw * CALIBRATED_SCALE
    sp = StreamingPolyphase(p=8, q=15)
    return sp.feed(volts)


def fig5_cwt_3panel():
    """Pick a clean footstep from our capture, show raw → scalogram → render."""
    sig8k = _load_our_8k()
    gmm = _load_gmm()
    footsteps = extract_footsteps(
        matlab_smooth(sig8k, span=5),
        *_recompute_indices_labels(sig8k, gmm),
    )
    if footsteps.shape[0] == 0:
        # fall back to peak-centered window
        peak = int(np.argmax(np.abs(sig8k)))
        s = max(0, peak - 400)
        e = s + 1500
        foot = np.zeros(1500)
        foot[: e - s] = sig8k[s:e]
        src_label = "peak-centered (no GMM hit)"
    else:
        # choose the footstep with the largest peak amplitude
        k = int(np.argmax(np.max(np.abs(footsteps), axis=1)))
        foot = footsteps[k]
        src_label = "GMM-detected footstep #{}/{}".format(k + 1, footsteps.shape[0])

    coefs = cwt(foot)
    rgb = cwt_to_rgb_direct(coefs, (224, 224))

    t_ms = np.arange(len(foot)) / FS_OUT * 1000.0  # 0..187.4 ms

    fig = plt.figure(figsize=(12, 9.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.9, 1.4, 2.2],
                          hspace=0.50, wspace=0.4)

    # Row 1: raw waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_ms, foot * 1000, lw=1.0, color="#1f4e9c")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("(a) Raw footstep waveform   (1500 samples, 187.5 ms @ 8 kHz)",
                  loc="left", fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.set_xlim(t_ms[0], t_ms[-1])

    # Row 2: scalogram (256 scales × 1500 samples)
    ax2 = fig.add_subplot(gs[1, :])
    # display log10|coef| for dynamic range, but keep absolute scale for honesty
    show = np.abs(coefs)
    im = ax2.imshow(
        show, aspect="auto", origin="upper", cmap="jet",
        extent=[t_ms[0], t_ms[-1], 256, 1],  # scales 1 (top) .. 256 (bottom)
        interpolation="nearest",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("CWT scale (1 = high f)")
    ax2.set_title("(b) Continuous wavelet transform   (Morlet, 256 scales)",
                  loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
    cb.set_label("|coef|", fontsize=9)

    # Row 3: 224×224 LUT render (model input) — centered square
    ax3 = fig.add_subplot(gs[2, 2:4])
    ax3.imshow(rgb, interpolation="nearest")
    ax3.set_xticks([])
    ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_edgecolor("#444")
    ax3.set_title(
        "(c) Model input  (224×224 RGB, jet LUT v2 → MobileNetV3)",
        loc="left", fontweight="bold")

    fig.suptitle(
        "Single-footstep pipeline visualisation   ·   source: {}".format(src_label),
        fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, "fig5_cwt_3panel.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig5]", out)


def _recompute_indices_labels(sig, gmm):
    """Helper to mirror what apply_frozen_gmm does internally."""
    g = matlab_smooth(sig, span=5)
    feats, indices = slide_features(g, fs=FS_OUT)
    labels = assign_clusters_frozen(
        feats, gmm["mu"], gmm["cov"], gmm["phi"],
        gmm["event_cluster"], gmm["threshold"],
    )
    return indices, labels


def fig6_gmm_detection():
    sig8k = _load_our_8k()
    gmm = _load_gmm()
    spans, g = _detect_spans(sig8k, gmm)
    t = np.arange(len(sig8k)) / FS_OUT
    volts_mv = sig8k * 1000

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(t, volts_mv, lw=0.5, color="#1f4e9c", label="signal (8 kHz)")
    for k, (s, e, mn) in enumerate(spans):
        ax.axvspan(s / FS_OUT, e / FS_OUT, color="#E45756",
                   alpha=0.18, linewidth=0)
    if spans:
        # peak markers
        peaks_t = [mn / FS_OUT for _, _, mn in spans]
        peaks_v = [volts_mv[mn] for _, _, mn in spans]
        ax.scatter(peaks_t, peaks_v, s=18, color="#E45756", zorder=5,
                   label="detected peaks ({})".format(len(spans)))
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(
        "Frozen GMM event detection   "
        "(P14 7-feature GMM, smooth→slide→cluster→Tukey window)",
        pad=10, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    info = (
        "trained on : {}\n"
        "feats / win : 7\n"
        "event thr  : {:.3f}\n"
        "footsteps  : {}".format(
            str(gmm["train_person"]), float(gmm["threshold"]), len(spans))
    )
    ax.text(
        0.99, 0.96, info, transform=ax.transAxes,
        ha="right", va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9),
    )

    fig.tight_layout()
    out = os.path.join(OUT, "fig6_gmm_detection.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig6]", out, "({} footsteps)".format(len(spans)))


def _load_person_first_footstep(person_dir, gmm):
    """Load <pid>_1.mat, run frozen GMM, return strongest footstep."""
    mat_path = os.path.join(person_dir, os.path.basename(person_dir) + "_1.mat")
    d = loadmat(mat_path)
    geo = d["geo_data"].ravel().astype(np.float64)
    feets = extract_footsteps(
        matlab_smooth(geo, span=5),
        *_recompute_indices_labels(geo, gmm),
    )
    if feets.shape[0] == 0:
        return None
    k = int(np.argmax(np.max(np.abs(feets), axis=1)))
    return feets[k]


def fig7_cross_person():
    gmm = _load_gmm()
    pids = ["P14", "P50", "P100"]
    labels = ["P14", "P50", "P100"]
    raw_root = os.path.join(ROOT, "data", "raw")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.0),
                             gridspec_kw={"height_ratios": [1.0, 1.4],
                                          "hspace": 0.45, "wspace": 0.25})
    t_ms = np.arange(1500) / FS_OUT * 1000

    for j, (pid, lab) in enumerate(zip(pids, labels)):
        foot = _load_person_first_footstep(os.path.join(raw_root, pid), gmm)
        if foot is None:
            for r in range(2):
                axes[r, j].set_title("{}: no detection".format(pid))
            continue
        coefs = cwt(foot)
        rgb = cwt_to_rgb_direct(coefs, (224, 224))

        ax_t = axes[0, j]
        ax_t.plot(t_ms, foot, lw=0.8, color="#1f4e9c")
        ax_t.set_title("{}  —  raw footstep".format(lab),
                       fontweight="bold")
        ax_t.set_xlabel("Time (ms)")
        if j == 0:
            ax_t.set_ylabel("Geo amplitude (a.u.)")
        ax_t.grid(alpha=0.25)
        ax_t.set_xlim(t_ms[0], t_ms[-1])

        ax_i = axes[1, j]
        ax_i.imshow(rgb, interpolation="nearest")
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        for s in ax_i.spines.values():
            s.set_edgecolor("#444")
        ax_i.set_title("{}  —  224×224 model input".format(lab),
                       fontweight="bold")

    fig.suptitle(
        "Cross-person comparison  ·  VIBeID P14 / P50 / P100, "
        "strongest footstep from session 1",
        fontsize=13.5, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, "fig7_cross_person.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig7]", out)


def main():
    fig5_cwt_3panel()
    fig6_gmm_detection()
    fig7_cross_person()
    print("\nAll CWT figures written to", OUT)


if __name__ == "__main__":
    main()
