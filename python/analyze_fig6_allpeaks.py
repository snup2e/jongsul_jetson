"""Find ALL visible peaks in the 30s capture (not just GMM hits), then search
for the user's ~300 ms × 4 cluster.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase
from scipy.signal import hilbert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATED_SCALE = 3.7253e-7
FS_OUT = 8000


def load_8k():
    raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
    volts = raw * CALIBRATED_SCALE
    sp = StreamingPolyphase(p=8, q=15)
    return sp.feed(volts)


def find_peaks_envelope(sig, fs=FS_OUT, min_sep_ms=80, thresh_mv=30):
    """Hilbert envelope peak finder — same idea as production envelope detector."""
    env = np.abs(hilbert(sig))
    # short smoothing (~30 ms)
    w = int(0.030 * fs)
    kern = np.ones(w) / w
    env_s = np.convolve(env, kern, mode="same")
    thr = thresh_mv * 1e-3  # mV → V
    min_sep = int(min_sep_ms * 1e-3 * fs)

    peaks = []
    i = 0
    while i < len(env_s):
        if env_s[i] > thr:
            # find local maximum
            j = i
            best = i
            while j < len(env_s) and env_s[j] > thr:
                if env_s[j] > env_s[best]:
                    best = j
                j += 1
            # refine to the actual signal peak (use |sig|, not envelope)
            lo = max(0, best - int(0.020 * fs))
            hi = min(len(sig), best + int(0.020 * fs))
            local = int(np.argmax(np.abs(sig[lo:hi])))
            peaks.append(lo + local)
            i = j + min_sep
        else:
            i += 1
    # dedupe by min_sep
    out = []
    for p in peaks:
        if out and p - out[-1] < min_sep:
            if abs(sig[p]) > abs(sig[out[-1]]):
                out[-1] = p
        else:
            out.append(p)
    return np.array(out, dtype=int)


def main():
    sig = load_8k()
    peaks = find_peaks_envelope(sig, thresh_mv=20)
    print("Found {} peaks above 20 mV envelope threshold".format(len(peaks)))
    print()
    print("{:>3} {:>8} {:>10} {:>10}".format("#", "t(s)", "peak(mV)", "%FS"))
    for k, p in enumerate(peaks):
        mv = sig[p] * 1000.0
        pct = abs(mv) / 10.0 / 312.5 * 100
        print("{:>3d} {:>8.3f} {:>+10.2f} {:>9.2f}%".format(
            k + 1, p / FS_OUT, mv, pct))

    # Look at the dense burst between 7.5..9 s for ~300 ms cadence
    print()
    print("=" * 70)
    print("Sub-window 7.5..9.0 s  (the second visible 'cluster of 4')")
    print("=" * 70)
    in_window = [(p, p / FS_OUT) for p in peaks if 7.5 < p / FS_OUT < 9.0]
    for k, (p, t) in enumerate(in_window):
        mv = sig[p] * 1000.0
        print("  peak {} @ t={:.3f}s  |v|={:.1f} mV".format(k + 1, t, abs(mv)))
    if len(in_window) >= 2:
        gaps = np.diff([t for _, t in in_window]) * 1000
        print("  gaps (ms):", " ".join("{:.1f}".format(g) for g in gaps))

    print()
    print("Sub-window 5.5..7.0 s  (the first visible 'cluster of 4')")
    print("=" * 70)
    in_window = [(p, p / FS_OUT) for p in peaks if 5.5 < p / FS_OUT < 7.0]
    for k, (p, t) in enumerate(in_window):
        mv = sig[p] * 1000.0
        print("  peak {} @ t={:.3f}s  |v|={:.1f} mV".format(k + 1, t, abs(mv)))
    if len(in_window) >= 2:
        gaps = np.diff([t for _, t in in_window]) * 1000
        print("  gaps (ms):", " ".join("{:.1f}".format(g) for g in gaps))

    # Also: any quadruplet of any peaks at ~300 ms?
    print()
    print("=" * 70)
    print("All-peak quadruplet scan (4 in a row, each gap 250..350 ms)")
    print("=" * 70)
    ts = peaks / FS_OUT
    found = 0
    for i in range(len(ts) - 3):
        d = np.diff(ts[i:i + 4]) * 1000
        if np.all((d > 250) & (d < 380)):
            print("  peaks #{}..#{}: t={:.2f}..{:.2f}, gaps={}".format(
                i + 1, i + 4, ts[i], ts[i + 3],
                " ".join("{:.0f}".format(x) for x in d)))
            found += 1
    if found == 0:
        print("  none in strict band; relaxing to 200..400 ms")
        for i in range(len(ts) - 3):
            d = np.diff(ts[i:i + 4]) * 1000
            if np.all((d > 200) & (d < 400)):
                print("  peaks #{}..#{}: t={:.2f}..{:.2f}, gaps={}".format(
                    i + 1, i + 4, ts[i], ts[i + 3],
                    " ".join("{:.0f}".format(x) for x in d)))


if __name__ == "__main__":
    main()
