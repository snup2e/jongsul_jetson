"""Analyze fig6 GMM peaks: which are real footsteps vs noise, and find the
300 ms-period 4-peak pattern.

Reference (PGA=16, ADS1256, ±312.5 mV FS at ADC input):
    raw geophone (after CALIBRATED_SCALE = ADC LSB × VIBeID amp gain 10)
    plotted as mV in fig6.
    1% FS at ADC input == 31.25 mV in the plotted voltage axis.

Pre-S.5 PGA reference from CLAUDE.md (in % FS at ADC input):
    idle noise   : 0.78%  → 2.44 mV ADC pin → 24.4 mV in plot
    normal step  : 2~7%   → 6.25–21.9 mV   → 62.5–219 mV in plot
    hard stomp   : 5~11%  → 15.6–34.4 mV   → 156–344 mV in plot
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase
from extract import (
    matlab_smooth, slide_features, assign_clusters_frozen,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATED_SCALE = 3.7253e-7
FS_OUT = 8000


def detect_spans(signal, gmm, fs=FS_OUT, footfall_len=1500, peak_offset=400):
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


def main():
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
    volts = raw * CALIBRATED_SCALE
    sp = StreamingPolyphase(p=8, q=15)
    sig8k = sp.feed(volts)
    gmm = dict(np.load(os.path.join(os.path.dirname(__file__), "gmm_params.npz")))
    spans, g = detect_spans(sig8k, gmm)

    print("=" * 78)
    print("Frozen-GMM detected peaks ({} total) — sorted by time".format(len(spans)))
    print("=" * 78)
    print("{:>3} {:>8} {:>10} {:>10} {:>10}".format(
        "#", "t (s)", "peak (mV)", "|peak| mV", "% FS@PGA16"))
    print("-" * 78)
    pf_idx = []
    for k, (s, e, mn) in enumerate(spans):
        t_s = mn / FS_OUT
        pk_mv = sig8k[mn] * 1000.0  # plotted voltage = mV
        pk_fs_pct = abs(pk_mv) / 10.0 / 312.5 * 100.0  # divide by amp gain 10 → ADC pin volts → % of 312.5 mV FS
        print("{:>3d} {:>8.3f} {:>+10.2f} {:>10.2f} {:>9.3f}%".format(
            k + 1, t_s, pk_mv, abs(pk_mv), pk_fs_pct))
        pf_idx.append((t_s, pk_mv))

    # ---- Look for a 4-peak ~300 ms-cadence cluster ----
    print()
    print("=" * 78)
    print("Inter-peak gaps (Δt between consecutive detected peaks)")
    print("=" * 78)
    print("{:>3}->{:<3} {:>8} {:>8} {:>10}".format(
        "i", "i+1", "t_i", "Δt(ms)", "note"))
    print("-" * 78)
    for k in range(len(spans) - 1):
        dt = (spans[k + 1][2] - spans[k][2]) / FS_OUT
        note = "<-- ~300 ms" if 0.25 < dt < 0.35 else (
            "<-- ~150 ms" if 0.12 < dt < 0.18 else "")
        print("{:>3d}->{:<3d} {:>8.3f} {:>8.1f} ms  {}".format(
            k + 1, k + 2, spans[k][2] / FS_OUT, dt * 1000, note))

    # ---- Run autocorrelation on the dense-burst region (5..9 s) to find any
    # ---- periodicity around 300 ms ----
    print()
    print("=" * 78)
    print("Periodicity check: dense-burst window (signal envelope autocorr)")
    print("=" * 78)
    seg_t0 = 5.0
    seg_t1 = 9.0
    s0, s1 = int(seg_t0 * FS_OUT), int(seg_t1 * FS_OUT)
    env = np.abs(sig8k[s0:s1])
    env -= env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    ac /= ac[0]
    lags_ms = np.arange(len(ac)) / FS_OUT * 1000.0
    # peaks of autocorr in 80..600 ms range
    mask = (lags_ms > 80) & (lags_ms < 600)
    sub_ac = ac[mask]
    sub_lag = lags_ms[mask]
    # local maxima
    peaks = []
    for i in range(1, len(sub_ac) - 1):
        if sub_ac[i] > sub_ac[i - 1] and sub_ac[i] > sub_ac[i + 1]:
            peaks.append((sub_lag[i], sub_ac[i]))
    peaks.sort(key=lambda x: -x[1])
    print("Top 8 autocorrelation peaks in 80..600 ms (envelope of t={:.1f}..{:.1f} s):".format(seg_t0, seg_t1))
    for lag, val in peaks[:8]:
        print("  lag = {:6.1f} ms   ac = {:.3f}".format(lag, val))

    # ---- Also: check for any 300-ms quadruplet in the WHOLE 30 s
    print()
    print("=" * 78)
    print("Quadruplet search (4 consecutive peaks at ~300 ms each, tol ±60 ms)")
    print("=" * 78)
    times = [s[2] / FS_OUT for s in spans]
    for i in range(len(times) - 3):
        d1 = times[i + 1] - times[i]
        d2 = times[i + 2] - times[i + 1]
        d3 = times[i + 3] - times[i + 2]
        if all(0.24 < d < 0.36 for d in (d1, d2, d3)):
            print("  Found! peaks #{}..#{} at t = {:.2f}..{:.2f} s, gaps = {:.0f}/{:.0f}/{:.0f} ms".format(
                i + 1, i + 4, times[i], times[i + 3],
                d1 * 1000, d2 * 1000, d3 * 1000))


if __name__ == "__main__":
    main()
