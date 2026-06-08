"""Validate Hilbert-envelope detector on P14 by measuring classifier accuracy.

For P14_1..P14_6, extract footstep crops two ways:
  (a) Frozen GMM (current pipeline)
  (b) Hilbert envelope + peak-find
Then CWT → LUT → MobileNetV3-Large v2 → top-1 → compare accuracy on label=13.

If envelope accuracy ≈ GMM accuracy AND envelope finds more crops, it's a clean
replacement that doesn't need classifier retraining.
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
from extract import (
    matlab_smooth, slide_features, assign_clusters_frozen,
)
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


# ---------------------------------------------------------------- detectors

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


def envelope_peaks(signal, k_mad=8.0, env_smooth_ms=30.0, min_sep_ms=120.0):
    g = matlab_smooth(signal, span=5)
    env = np.abs(ss.hilbert(g))
    n_smooth = max(1, int(env_smooth_ms * 1e-3 * FS))
    kern = np.ones(n_smooth) / n_smooth
    env = np.convolve(env, kern, mode="same")
    med = np.median(env)
    mad = np.median(np.abs(env - med))
    thr = med + k_mad * mad * 1.4826
    min_dist = max(1, int(min_sep_ms * 1e-3 * FS))
    peaks, _ = ss.find_peaks(env, height=thr, distance=min_dist)
    return peaks.tolist(), g


def crops_from_peaks(g, peaks):
    """Replicate extract_footsteps cropping: peak-400 start, 1500 samples, Tukey."""
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


# ---------------------------------------------------------------- classifier

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


# ---------------------------------------------------------------- main

def main():
    gmm = load_gmm()
    print("loading model ...")
    model, nc = load_model()
    print("  num_classes={}".format(nc))

    target_idx = 13  # P14 (0-indexed)

    print("\n{:<6}  {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}".format(
        "P14_x",
        "GMM #", "GMM acc", "GMM cf",
        "env #", "env acc", "env cf"))
    print("-" * 78)

    tot_g_n = tot_g_hit = 0
    tot_e_n = tot_e_hit = 0
    for idx in (1, 2, 3, 4, 5, 6):
        path = os.path.join(ROOT, "data", "raw", "P14", "P14_{}.mat".format(idx))
        sig = loadmat(path)["geo_data"].ravel().astype(np.float64)
        gp, g_sm = gmm_peaks(sig, gmm)
        ep, _ = envelope_peaks(sig)

        g_crops = crops_from_peaks(g_sm, gp)
        e_crops = crops_from_peaks(g_sm, ep)
        g_imgs = crops_to_rgb(g_crops)
        e_imgs = crops_to_rgb(e_crops)
        g_pred, g_conf = predict(model, g_imgs)
        e_pred, e_conf = predict(model, e_imgs)

        g_acc = (g_pred == target_idx).mean() if len(g_pred) else 0
        e_acc = (e_pred == target_idx).mean() if len(e_pred) else 0
        g_cf = g_conf[g_pred == target_idx].mean() if (g_pred == target_idx).any() else 0
        e_cf = e_conf[e_pred == target_idx].mean() if (e_pred == target_idx).any() else 0

        tot_g_n += len(g_pred)
        tot_g_hit += int((g_pred == target_idx).sum())
        tot_e_n += len(e_pred)
        tot_e_hit += int((e_pred == target_idx).sum())

        print("{:<6}  {:>8} {:>7.1%} {:>8.3f}   {:>8} {:>7.1%} {:>8.3f}".format(
            idx, len(gp), g_acc, g_cf, len(ep), e_acc, e_cf))

    print("-" * 78)
    g_total_acc = tot_g_hit / tot_g_n if tot_g_n else 0
    e_total_acc = tot_e_hit / tot_e_n if tot_e_n else 0
    print("TOTAL    {:>8} {:>7.1%}            {:>8} {:>7.1%}".format(
        tot_g_n, g_total_acc, tot_e_n, e_total_acc))
    print("\nGMM:      {} crops, {} correct → {:.2%}".format(tot_g_n, tot_g_hit, g_total_acc))
    print("Envelope: {} crops, {} correct → {:.2%}".format(tot_e_n, tot_e_hit, e_total_acc))
    print("\nEnvelope extras: +{} crops vs GMM ({:+.1%})".format(
        tot_e_n - tot_g_n, (tot_e_n - tot_g_n) / max(tot_g_n, 1)))


if __name__ == "__main__":
    main()
