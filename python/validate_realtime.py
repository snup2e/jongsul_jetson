"""Validate jetson_realtime.py components against offline apply_frozen_gmm.

Runs on Windows (no TRT, no CUDA needed) to catch logic errors before
deploying to Jetson.

Targets:
  1. StreamingSmoother == matlab_smooth except first (span-1) samples
  2. SlidingDetector fed with pre-smoothed signal == offline extract_footsteps,
     both for single-chunk and multi-chunk feed.
"""
import numpy as np
import scipy.io

from extract import (
    matlab_smooth,
    slide_features,
    assign_clusters_frozen,
    extract_footsteps,
    load_gmm_npz,
)
from jetson_realtime import SlidingDetector, StreamingSmoother, MatFileSource


def test_smoother():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000)
    y_off = matlab_smooth(x, span=5)

    for cs in [137, 1024, 1, 9999]:
        sm = StreamingSmoother(span=5)
        parts = []
        for i in range(0, len(x), cs):
            out = sm(x[i : i + cs])
            if len(out):
                parts.append(out)
        tail = sm.flush()
        if len(tail):
            parts.append(tail)
        y_stream = np.concatenate(parts) if parts else np.array([])
        ok = (len(y_stream) == len(y_off))
        diff = np.abs(y_off - y_stream).max() if ok else float("nan")
        print("[smoother] cs={:>4d}  len match={}  max|diff|={:.3e}".format(
            cs, ok, diff,
        ))


def test_detector(mat_path, gmm_path):
    print("[detector] loading {}".format(mat_path))
    geo = scipy.io.loadmat(mat_path)["geo_data"].ravel().astype(np.float64)
    gmm = load_gmm_npz(gmm_path)
    g_off = matlab_smooth(geo, span=5)

    # Offline reference
    feats, indices = slide_features(g_off)
    labels = assign_clusters_frozen(
        feats, gmm["mu"], gmm["cov"], gmm["phi"],
        gmm["event_cluster"], gmm["threshold"],
    )
    fs_off = extract_footsteps(g_off, indices, labels)
    print("[detector] offline:        {} footsteps".format(len(fs_off)))

    # Streaming, single chunk
    det = SlidingDetector(gmm)
    fs_one = list(det.feed(g_off)) + list(det.flush())
    fs_one = np.asarray(fs_one) if fs_one else np.zeros((0, 1500))
    print("[detector] streaming-1ch:  {} footsteps".format(len(fs_one)))
    if len(fs_one) == len(fs_off):
        print("[detector]   max |diff|: {:.3e}".format(np.abs(fs_off - fs_one).max()))
    else:
        print("[detector]   COUNT MISMATCH")

    # Streaming, many chunks (1024 = ADS1256-like batch)
    det2 = SlidingDetector(gmm)
    fs_many = []
    cs = 1024
    for i in range(0, len(g_off), cs):
        fs_many.extend(det2.feed(g_off[i : i + cs]))
    fs_many.extend(det2.flush())
    fs_many = np.asarray(fs_many) if fs_many else np.zeros((0, 1500))
    print("[detector] streaming-Nch:  {} footsteps".format(len(fs_many)))
    if len(fs_many) == len(fs_off):
        print("[detector]   max |diff|: {:.3e}".format(np.abs(fs_off - fs_many).max()))
    else:
        print("[detector]   COUNT MISMATCH")

    # Streaming, weird small chunk (stress test)
    det3 = SlidingDetector(gmm)
    fs_small = []
    cs = 137
    for i in range(0, len(g_off), cs):
        fs_small.extend(det3.feed(g_off[i : i + cs]))
    fs_small.extend(det3.flush())
    fs_small = np.asarray(fs_small) if fs_small else np.zeros((0, 1500))
    print("[detector] streaming-137:  {} footsteps".format(len(fs_small)))
    if len(fs_small) == len(fs_off):
        print("[detector]   max |diff|: {:.3e}".format(np.abs(fs_off - fs_small).max()))
    else:
        print("[detector]   COUNT MISMATCH")


def test_full_pipeline(mat_path, gmm_path):
    """raw .mat → MatFileSource → StreamingSmoother → SlidingDetector
       must match raw .mat → matlab_smooth → offline detector.
    """
    print("[full] loading {}".format(mat_path))
    geo = scipy.io.loadmat(mat_path)["geo_data"].ravel().astype(np.float64)
    gmm = load_gmm_npz(gmm_path)
    g_off = matlab_smooth(geo, span=5)
    feats, indices = slide_features(g_off)
    labels = assign_clusters_frozen(
        feats, gmm["mu"], gmm["cov"], gmm["phi"],
        gmm["event_cluster"], gmm["threshold"],
    )
    fs_off = extract_footsteps(g_off, indices, labels)
    print("[full] offline:    {} footsteps".format(len(fs_off)))

    # End-to-end streaming via MatFileSource
    src = MatFileSource(mat_path, chunk_size=1024)
    sm = StreamingSmoother(span=5)
    det = SlidingDetector(gmm)
    fs_stream = []
    for chunk in src.chunks():
        smoothed = sm(chunk)
        if len(smoothed):
            fs_stream.extend(det.feed(smoothed))
    tail = sm.flush()
    if len(tail):
        fs_stream.extend(det.feed(tail))
    fs_stream.extend(det.flush())
    fs_stream = np.asarray(fs_stream) if fs_stream else np.zeros((0, 1500))
    print("[full] streaming:  {} footsteps".format(len(fs_stream)))
    if len(fs_stream) == len(fs_off):
        print("[full]   max |diff|: {:.3e}".format(np.abs(fs_off - fs_stream).max()))
    else:
        print("[full]   COUNT MISMATCH — pipeline integration broken")


if __name__ == "__main__":
    print("=== test_smoother ===")
    test_smoother()
    print()
    print("=== test_detector (pre-smoothed input) ===")
    test_detector("E:/Terra/data/raw/P14/P14_1.mat", "E:/Terra/python/gmm_params.npz")
    print()
    print("=== test_full_pipeline (raw input via MatFileSource) ===")
    test_full_pipeline("E:/Terra/data/raw/P14/P14_1.mat", "E:/Terra/python/gmm_params.npz")
