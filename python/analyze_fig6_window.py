"""Check why frozen GMM missed peaks in the 5.8..8.9 s dense bursts.
Hypothesis: 0.35 s window often contains 2 peaks (~330 ms apart) so its
feature vector falls outside the trained 1-peak event distribution.
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
FS = 8000

raw = np.load(os.path.join(ROOT, "data", "stm32_30s_raw.npy")).astype(np.float64)
sp = StreamingPolyphase(p=8, q=15)
sig = sp.feed(raw * CALIBRATED_SCALE)
g = matlab_smooth(sig, span=5)
feats, idx = slide_features(g, fs=FS)
gmm = dict(np.load(os.path.join(os.path.dirname(__file__), "gmm_params.npz")))
labels = assign_clusters_frozen(
    feats, gmm["mu"], gmm["cov"], gmm["phi"],
    gmm["event_cluster"], gmm["threshold"],
)

W = int(0.35 * FS)
step = idx[1, 0] - idx[0, 0]
print("window = {} samples ({:.0f} ms),  step = {} samples ({:.0f} ms)".format(
    W, W / FS * 1000, step, step / FS * 1000))

# Real peak locations (from previous analysis)
peaks_t = [5.800, 6.131, 6.502, 6.795,
           7.834, 8.224, 8.559, 8.909]

print("\nWindow-by-window scan over t = 5.5 .. 9.2 s")
print("=" * 86)
print("{:>4} {:>9} {:>9}  {:>2}  {:>20}  {:>9}".format(
    "win#", "t_start", "t_end", "lbl", "peaks inside (t,mV)", "peak_max"))
print("-" * 86)
for i, (s, e) in enumerate(idx):
    t_s, t_e = s / FS, e / FS
    if t_e < 5.5 or t_s > 9.2:
        continue
    inside = [(pt, sig[int(pt * FS)] * 1000)
              for pt in peaks_t if t_s <= pt < t_e]
    if not inside:
        continue
    # Also compute window |max|
    seg = g[s:e]
    wmax = float(np.max(np.abs(seg))) * 1000
    pk_str = ", ".join("({:.2f},{:+.0f})".format(p, v) for p, v in inside)
    print("{:>4d} {:>9.3f} {:>9.3f}  {:>2d}  {:>20}  {:>+9.1f}".format(
        i, t_s, t_e, int(labels[i]), pk_str, wmax))

# Summarize: of windows containing peaks, how many got label 1?
print("\nSummary in 5.5..9.2 s:")
mask = (idx[:, 0] >= 5.5 * FS) & (idx[:, 1] <= 9.2 * FS)
sub_lab = labels[mask]
sub_idx = idx[mask]
total = 0
hit = 0
for i, (s, e) in enumerate(sub_idx):
    t_s, t_e = s / FS, e / FS
    inside = [pt for pt in peaks_t if t_s <= pt < t_e]
    if inside:
        total += 1
        if sub_lab[i] == 1:
            hit += 1
print("  windows containing >=1 real peak : {}".format(total))
print("  of those, labeled event          : {}  ({:.0f}%)".format(
    hit, 100 * hit / max(total, 1)))

# Now: how many peaks-per-window?
print("\nPeaks-per-window histogram in 5.5..9.2 s (windows with peaks):")
from collections import Counter
counts = []
for s, e in sub_idx:
    t_s, t_e = s / FS, e / FS
    n = sum(1 for pt in peaks_t if t_s <= pt < t_e)
    if n > 0:
        counts.append(n)
c = Counter(counts)
for k in sorted(c):
    print("  {} peak(s) in window : {} window(s)".format(k, c[k]))

# For each peak, how many DIFFERENT windows contain it?
print("\nFor each real peak, count of windows containing it & their labels:")
for pt in peaks_t:
    s_i = int(pt * FS)
    cover = []
    for i, (s, e) in enumerate(idx):
        if s <= s_i < e:
            cover.append((i, int(labels[i])))
    lbl_str = " ".join("[{}:{}]".format(i, l) for i, l in cover)
    n_event = sum(1 for _, l in cover if l == 1)
    print("  peak @ {:.3f}s : {} windows, {} event   {}".format(
        pt, len(cover), n_event, lbl_str))
