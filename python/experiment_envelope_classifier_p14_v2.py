"""V2: envelope detector with peak alignment correction.

After envelope identifies a candidate region, find argmax(|signal|) within a
small window around the envelope peak (matching extract_footsteps logic).
Also split accuracy into: matched-with-GMM / envelope-only / GMM-only.
"""
import os
import sys

import numpy as np
import scipy.signal as ss
import torch
from torch import nn
import torchvision.models as tvm
import torchvision.transforms as T
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import matlab_smooth, slide_features, assign_clusters_frozen
from cwt_fast import cwt as fast_cwt
from render_lut import cwt_to_rgb_direct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 8000
FOOTFALL_LEN = 1500
PEAK_OFFSET = 400
WEIGHTS = os.path.join(ROOT, "weights", "mobilenet_v3_large_v2_best.pth")
TUKEY = ss.windows.tukey(FOOTFALL_LEN, 0.5)
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DEVICE = "cpu"


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
    """Find envelope peaks, then within ±align_radius_ms snap to argmax(|signal|).

    This matches what extract_footsteps does inside an event window.
    """
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
    return aligned, g


def crops_from_peaks(g, peaks):
    L = len(g)
    out = []
    for p in peaks:
        strt = max(0, p - PEAK_OFFSET)
        stop = min(L, strt + FOOTFALL_LEN)
        seg = g[strt:stop]
        win = TUKEY[:len(seg)] if len(seg) < FOOTFALL_LEN else TUKEY
        windowed = seg * win
        if len(windowed) < FOOTFALL_LEN:
            padded = np.zeros(FOOTFALL_LEN)
            padded[:len(windowed)] = windowed
            out.append(padded)
        else:
            out.append(windowed)
    return np.asarray(out) if out else np.zeros((0, FOOTFALL_LEN))


def crops_to_rgb(crops):
    M = len(crops)
    out = np.empty((M, 224, 224, 3), dtype=np.uint8)
    for i in range(M):
        coeffs = fast_cwt(crops[i])
        out[i] = cwt_to_rgb_direct(coeffs, (224, 224))
    return out


def load_model():
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    nc = int(ckpt["num_classes"])
    m = tvm.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval().to(DEVICE)
    return m, nc


@torch.no_grad()
def predict(model, imgs_uint8, batch=32):
    if len(imgs_uint8) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    preds = np.empty(len(imgs_uint8), dtype=np.int64)
    confs = np.empty(len(imgs_uint8), dtype=np.float32)
    for i in range(0, len(imgs_uint8), batch):
        chunk = imgs_uint8[i:i+batch]
        t = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        t = NORMALIZE(t)
        logits = model(t)
        prob = torch.softmax(logits, dim=1)
        c, p = prob.max(dim=1)
        preds[i:i+batch] = p.cpu().numpy()
        confs[i:i+batch] = c.cpu().numpy()
    return preds, confs


def split_by_match(gmm_pks, env_pks, tol_samples=400):
    """Split envelope peaks into (matched_env, env_only). Also (matched_gmm, gmm_only)."""
    env_arr = np.asarray(env_pks)
    gmm_arr = np.asarray(gmm_pks)
    matched_env = []
    env_only = []
    used_gmm = set()
    for ep in env_pks:
        if len(gmm_arr) == 0:
            env_only.append(ep)
            continue
        d = np.abs(gmm_arr - ep)
        k = int(np.argmin(d))
        if d[k] <= tol_samples and k not in used_gmm:
            matched_env.append(ep)
            used_gmm.add(k)
        else:
            env_only.append(ep)
    matched_gmm = [gmm_pks[k] for k in sorted(used_gmm)]
    gmm_only = [gmm_pks[k] for k in range(len(gmm_pks)) if k not in used_gmm]
    return matched_env, env_only, matched_gmm, gmm_only


def main():
    gmm = load_gmm()
    print("loading model ...")
    model, nc = load_model()
    print("  num_classes={}".format(nc))
    target_idx = 13

    print("\n{:<6}  {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "P14_x", "GMM acc", "env acc", "matched", "env-only", "gmm-only"))
    print("-" * 70)

    sums = dict(g_n=0, g_h=0, e_n=0, e_h=0,
                m_n=0, m_h=0, eo_n=0, eo_h=0, go_n=0, go_h=0)

    for idx in (1, 2, 3, 4, 5, 6):
        path = os.path.join(ROOT, "data", "raw", "P14", "P14_{}.mat".format(idx))
        sig = loadmat(path)["geo_data"].ravel().astype(np.float64)
        gp, g_sm = gmm_peaks(sig, gmm)
        ep, _ = envelope_peaks_aligned(sig)

        # split
        m_env, eo, m_gmm, go = split_by_match(gp, ep, tol_samples=int(0.05 * FS))

        def acc(peaks):
            if not peaks:
                return None, 0, 0
            imgs = crops_to_rgb(crops_from_peaks(g_sm, peaks))
            preds, _ = predict(model, imgs)
            hits = int((preds == target_idx).sum())
            return hits / len(preds), hits, len(preds)

        g_acc, g_h, g_n = acc(gp)
        e_acc, e_h, e_n = acc(ep)
        m_acc, m_h, m_n = acc(m_env)
        eo_acc, eo_h, eo_n = acc(eo)
        go_acc, go_h, go_n = acc(go)

        sums["g_n"] += g_n; sums["g_h"] += g_h
        sums["e_n"] += e_n; sums["e_h"] += e_h
        sums["m_n"] += m_n; sums["m_h"] += m_h
        sums["eo_n"] += eo_n; sums["eo_h"] += eo_h
        sums["go_n"] += go_n; sums["go_h"] += go_h

        fmt = lambda a, h, n: ("{:>5.1%} ({:>3}/{:>3})".format(a, h, n) if a is not None else "—")
        print("{:<6}  {:>16} {:>16} {:>16} {:>16} {:>16}".format(
            idx, fmt(g_acc, g_h, g_n), fmt(e_acc, e_h, e_n),
            fmt(m_acc, m_h, m_n) if m_n else "—",
            fmt(eo_acc, eo_h, eo_n) if eo_n else "—",
            fmt(go_acc, go_h, go_n) if go_n else "—"))

    print("-" * 70)
    print("TOTAL")
    for label, key in [("GMM", "g"), ("Envelope", "e"), ("matched", "m"),
                       ("env-only", "eo"), ("GMM-only", "go")]:
        n = sums[key + "_n"]; h = sums[key + "_h"]
        if n:
            print("  {:<10} {:>5} crops  {:>5} correct  {:>6.2%}".format(label, n, h, h / n))


if __name__ == "__main__":
    main()
