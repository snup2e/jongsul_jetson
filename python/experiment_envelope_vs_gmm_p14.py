"""Compare frozen GMM vs Hilbert-envelope detection on P14 raw .mat files.

Goal: verify that the envelope detector recovers a similar set of footsteps as
the frozen GMM on P14 (the data the classifier was trained on). If counts and
peak positions roughly align, the envelope detector can replace GMM without
retraining the classifier (crops will fall on the same peaks ± a few samples).
"""
import os
import sys

import numpy as np
from scipy.io import loadmat
import scipy.signal as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import (
    matlab_smooth, slide_features, assign_clusters_frozen,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 8000
FOOTFALL_LEN = 1500
PEAK_OFFSET = 400


def load_gmm():
    return dict(np.load(os.path.join(os.path.dirname(__file__), "gmm_params.npz")))


def load_p14_mat(idx):
    """Load P14_{idx}.mat, return 1-D geo signal at 8 kHz."""
    path = os.path.join(ROOT, "data", "raw", "P14", "P14_{}.mat".format(idx))
    m = loadmat(path)
    # VIBeID: 'data' or 'geo' or 'sig' — try common keys
    for k in ("data", "geo", "sig", "signal"):
        if k in m:
            return np.asarray(m[k]).ravel().astype(np.float64)
    # fallback: first non-meta key
    for k, v in m.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return v.ravel().astype(np.float64)
    raise KeyError("no signal key in " + path)


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
    return peaks


def envelope_peaks(signal, env_smooth_ms=30.0, k_mad=8.0, min_sep_ms=120.0):
    g = matlab_smooth(signal, span=5)
    env = np.abs(ss.hilbert(g))
    n_smooth = max(1, int(env_smooth_ms * 1e-3 * FS))
    if n_smooth > 1:
        kern = np.ones(n_smooth) / n_smooth
        env = np.convolve(env, kern, mode="same")
    med = np.median(env)
    mad = np.median(np.abs(env - med))
    thr = med + k_mad * mad * 1.4826
    min_dist = max(1, int(min_sep_ms * 1e-3 * FS))
    peaks, _ = ss.find_peaks(env, height=thr, distance=min_dist)
    return peaks.tolist()


def match_peaks(gmm_p, env_p, tol_samples=400):
    """For each GMM peak, find nearest envelope peak within tol. Return match stats."""
    matches = 0
    deltas = []
    env_arr = np.asarray(env_p)
    used = set()
    for gp in gmm_p:
        if len(env_arr) == 0:
            continue
        d = np.abs(env_arr - gp)
        k = int(np.argmin(d))
        if d[k] <= tol_samples and k not in used:
            matches += 1
            deltas.append(int(env_arr[k] - gp))
            used.add(k)
    return matches, deltas


def main():
    gmm = load_gmm()
    print("{:<6} {:>10} {:>10} {:>10} {:>20}".format(
        "P14_x", "len_s", "GMM peaks", "env peaks", "match (tol=50ms)"))
    print("-" * 70)
    all_gmm = 0
    all_env = 0
    all_match = 0
    for idx in (1, 2, 3, 4, 5, 6):
        try:
            sig = load_p14_mat(idx)
        except (FileNotFoundError, KeyError) as e:
            print("{:<6} {}".format(idx, e))
            continue
        gp = gmm_peaks(sig, gmm)
        ep = envelope_peaks(sig)
        matches, deltas = match_peaks(gp, ep, tol_samples=int(0.05 * FS))
        print("{:<6} {:>10.1f} {:>10} {:>10} {:>15} ({}/{})".format(
            idx, len(sig) / FS, len(gp), len(ep),
            "{}".format(matches), matches, len(gp)))
        if deltas:
            print("       median Δpeak: {:+d} samples ({:.1f} ms), max |Δ|: {} ({:.1f} ms)".format(
                int(np.median(deltas)), np.median(deltas) / FS * 1000,
                int(max(abs(d) for d in deltas)),
                max(abs(d) for d in deltas) / FS * 1000))
        all_gmm += len(gp)
        all_env += len(ep)
        all_match += matches

    print("-" * 70)
    print("TOTAL  {:>21} {:>10} {:>15}".format(all_gmm, all_env, all_match))
    if all_gmm:
        print("Envelope recall vs GMM: {:.1f}% ({}/{})".format(
            100.0 * all_match / all_gmm, all_match, all_gmm))
    if all_env:
        print("Envelope extras (potential FP or missed by GMM): {} ({:.1f}%)".format(
            all_env - all_match, 100.0 * (all_env - all_match) / all_env))


if __name__ == "__main__":
    main()
