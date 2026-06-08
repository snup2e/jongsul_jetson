"""Generate 4 signal transmission/conversion figures for interim presentation.

Inputs:
    data/stm32_30s_raw.npy  — 30s raw int24 capture (float64), 14998 SPS nominal

Outputs (PNG, 200 dpi) under 중간발표/figures/:
    fig1_frame_structure.png    — STM32 binary frame layout
    fig2_timeseries_30s.png     — 30 s calibrated voltage trace
    fig3_footstep_zoom_1s.png   — 1 s zoom on the strongest footstep
    fig4_polyphase_15k_to_8k.png — waveform + spectrum before/after resample
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "중간발표", "figures")
os.makedirs(OUT, exist_ok=True)

CALIBRATED_SCALE = 3.7253e-7   # V/LSB (CLAUDE.md S.5)
FS_IN = 15000                  # ADS1256 native (nominal)
FS_OUT = 8000                  # pipeline output

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_frame_structure():
    """Equal-width box diagram of [SYNC | SEQ | N | int24 BE × 64 | CRC8]."""
    fields = [
        ("SYNC",    "0xA5 0x5A",      "2 B",    "#4C78A8"),
        ("SEQ",     "uint16 LE",      "2 B",    "#72B7B2"),
        ("N",       "= 64",           "1 B",    "#F58518"),
        ("Payload", "int24 BE × 64",  "192 B",  "#54A24B"),
        ("CRC8",    "poly 0x07",      "1 B",    "#E45756"),
    ]
    offsets = [0, 2, 4, 5, 197, 198]  # cumulative byte offsets

    fig, ax = plt.subplots(figsize=(12, 3.6))
    H = 1.0
    W = 1.0
    gap = 0.04
    for i, (name, sub, sz, color) in enumerate(fields):
        x = i * (W + gap)
        rect = FancyBboxPatch(
            (x, 0), W, H,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            linewidth=1.4, edgecolor="black",
            facecolor=color, alpha=0.6,
        )
        ax.add_patch(rect)
        ax.text(x + W / 2, H * 0.66, name, ha="center", va="center",
                fontsize=14, fontweight="bold")
        ax.text(x + W / 2, H * 0.40, sub, ha="center", va="center",
                fontsize=10, color="#222")
        ax.text(x + W / 2, H * 0.16, sz, ha="center", va="center",
                fontsize=10, color="#444", style="italic")
        # byte-offset markers
        ax.text(x, -0.10, str(offsets[i]), ha="center", va="top",
                fontsize=9, color="#444")
        if i == len(fields) - 1:
            ax.text(x + W, -0.10, str(offsets[i + 1]),
                    ha="center", va="top", fontsize=9, color="#444")

    total_w = len(fields) * W + (len(fields) - 1) * gap
    ax.text(total_w / 2, -0.34, "byte offset", ha="center", va="top",
            fontsize=10, color="#444")

    ax.set_xlim(-0.15, total_w + 0.15)
    ax.set_ylim(-0.55, H + 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "STM32 → Jetson UART frame  (total 198 B, 921 600 baud, 234 frame/s @ 15 kSPS)",
        pad=14, fontweight="bold", fontsize=14)
    ax.text(total_w / 2, H + 0.30,
            "64 samples / frame   ·   ~4.27 ms wall-time   ·   189 kB/s sustained",
            ha="center", va="center", fontsize=11, color="#333")
    fig.tight_layout()
    out = os.path.join(OUT, "fig1_frame_structure.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig1]", out)


def _load_capture():
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy"))
    return raw.astype(np.float64)


def fig2_timeseries(raw):
    """30 s calibrated voltage trace."""
    volts = raw * CALIBRATED_SCALE
    t = np.arange(len(raw)) / FS_IN

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, volts * 1000, lw=0.5, color="#1f4e9c")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Geophone voltage (mV)")
    ax.set_title(
        "STM32 → Jetson 30 s capture  (15 kSPS, ADS1256 PGA=16, "
        "calibrated 3.73 × 10⁻⁷ V/LSB)",
        pad=10, fontweight="bold")
    ax.set_xlim(0, t[-1])
    ax.grid(alpha=0.25)

    # annotate stats
    pkpk = float(volts.max() - volts.min()) * 1000
    rms = float(np.sqrt(np.mean(volts ** 2))) * 1000
    ax.text(
        0.99, 0.96,
        "len = {} samples\npk-pk = {:.2f} mV\nRMS = {:.3f} mV".format(
            len(raw), pkpk, rms),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9),
    )
    fig.tight_layout()
    out = os.path.join(OUT, "fig2_timeseries_30s.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig2]", out)
    return volts, t


def fig3_zoom(volts, t):
    """1 s window around the strongest footstep peak."""
    # locate the strongest excursion
    idx = int(np.argmax(np.abs(volts - np.median(volts))))
    half = FS_IN // 2
    lo = max(0, idx - half)
    hi = min(len(volts), idx + half)
    seg = volts[lo:hi] * 1000   # mV
    tseg = t[lo:hi]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(tseg, seg, lw=0.8, color="#c1442c")
    ax.axvline(t[idx], color="#444", lw=0.7, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Geophone voltage (mV)")
    ax.set_title(
        "Single-footstep zoom  ({:.2f} s window around peak @ t = {:.3f} s)".format(
            (hi - lo) / FS_IN, t[idx]),
        pad=10, fontweight="bold")
    ax.grid(alpha=0.25)

    pk = float(np.max(np.abs(seg)))
    dur_ms = 1000.0 * (hi - lo) / FS_IN
    ax.text(
        0.99, 0.96,
        "peak |v| = {:.2f} mV\nwindow = {:.0f} ms\nfs = 15 kSPS native".format(
            pk, dur_ms),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9),
    )
    fig.tight_layout()
    out = os.path.join(OUT, "fig3_footstep_zoom_1s.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig3]", out, "(peak at idx={}, t={:.3f}s)".format(idx, t[idx]))
    return idx


def _fft_mag(x, fs):
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = 20 * np.log10(np.abs(X) + 1e-12)
    # normalize to peak = 0 dB
    mag -= mag.max()
    return f, mag


def fig4_polyphase(volts, t, peak_idx):
    """15 kSPS → 8 kSPS polyphase: waveform overlay + spectrum comparison."""
    # Run the actual production resampler
    sp = StreamingPolyphase(p=8, q=15)
    y8k = sp.feed(volts)
    t8k = np.arange(len(y8k)) / FS_OUT

    # Time-domain overlay: 0.4 s around the peak found in fig3
    half_s = 0.20
    in_lo = max(0, peak_idx - int(half_s * FS_IN))
    in_hi = min(len(volts), peak_idx + int(half_s * FS_IN))
    t_in_seg = t[in_lo:in_hi]
    v_in_seg = volts[in_lo:in_hi] * 1000

    # find corresponding 8k samples by time
    t_center = t[peak_idx]
    out_lo = max(0, int((t_center - half_s) * FS_OUT))
    out_hi = min(len(y8k), int((t_center + half_s) * FS_OUT))
    t_out_seg = t8k[out_lo:out_hi]
    v_out_seg = y8k[out_lo:out_hi] * 1000

    # FFT on full 30 s
    f_in, m_in = _fft_mag(volts, FS_IN)
    f_out, m_out = _fft_mag(y8k, FS_OUT)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(11, 7.2))

    ax_t.plot(t_in_seg, v_in_seg, lw=0.6, color="#4C78A8",
              label="raw 15 kSPS (input)")
    ax_t.plot(t_out_seg, v_out_seg, lw=1.1, color="#E45756",
              label="polyphase 8 kSPS (output)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Voltage (mV)")
    ax_t.set_title("Time-domain overlay  (400 ms around footstep)",
                   fontweight="bold", pad=8)
    ax_t.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax_t.grid(alpha=0.25)

    ax_f.plot(f_in, m_in, lw=0.7, color="#4C78A8",
              label="raw 15 kSPS spectrum")
    ax_f.plot(f_out, m_out, lw=0.9, color="#E45756",
              label="polyphase 8 kSPS spectrum")
    ax_f.axvline(FS_OUT / 2, color="#444", lw=0.8, ls="--",
                 label="output Nyquist = 4 kHz")
    ax_f.set_xlim(0, FS_IN / 2)
    ax_f.set_ylim(-90, 5)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude (dB, peak-normalised)")
    ax_f.set_title("Spectrum  (Kaiser-windowed polyphase FIR, anti-alias to 4 kHz)",
                   fontweight="bold", pad=8)
    ax_f.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax_f.grid(alpha=0.25)

    fig.suptitle(
        "Polyphase resample  15 kSPS → 8 kSPS  (p = 8, q = 15)",
        fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = os.path.join(OUT, "fig4_polyphase_15k_to_8k.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig4]", out,
          "(in={} samp, out={} samp, ratio={:.4f})".format(
              len(volts), len(y8k), len(y8k) / len(volts)))


def main():
    fig1_frame_structure()
    raw = _load_capture()
    volts, t = fig2_timeseries(raw)
    peak_idx = fig3_zoom(volts, t)
    fig4_polyphase(volts, t, peak_idx)
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()
