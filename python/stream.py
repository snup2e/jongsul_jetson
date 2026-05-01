"""Jetson-equivalent inference frontend (no model, no I/O concerns).

Glues the pre-validated stages together so a raw 8 kHz signal becomes a stack
of model-ready 224x224 RGB tensors:

    geo_data (float64, 8 kHz)
        --[matlab_smooth + slide_features + frozen GMM + Event_Extract]-->
    footsteps (M, 1500) float64
        --[pywt.cwt(scales=1..256, morl) + min-max + jet LUT + cv2.resize]-->
    inputs (M, 224, 224, 3) uint8 RGB

The actual ADS1256 SPI streaming on Jetson plugs into raw_to_inputs() by
chunking incoming samples and re-running it; the pipeline itself is stateless
beyond the per-call signal buffer.
"""
from pathlib import Path
from typing import Tuple

import numpy as np

from cwt_fast import cwt as fast_cwt
from extract import apply_frozen_gmm, load_gmm_npz
from render_lut import cwt_to_rgb_direct


def footsteps_to_inputs(footsteps: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """(M, 1500) float footsteps -> (M, H, W, 3) uint8 RGB.

    CWT + LUT per footstep. cwt_fast caches FFT'd wavelets across calls so
    the per-call cost is dominated by one FFT(signal) + 256-batch IFFT.
    """
    M = len(footsteps)
    H, W = size
    out = np.empty((M, H, W, 3), dtype=np.uint8)
    for i in range(M):
        coeffs = fast_cwt(footsteps[i])
        out[i] = cwt_to_rgb_direct(coeffs, size)
    return out


def raw_to_inputs(geo_data: np.ndarray, gmm_params: dict,
                  size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Full pipeline for a chunk of raw signal.

    Returns (footsteps (M, 1500), inputs (M, H, W, 3) uint8).
    """
    footsteps = apply_frozen_gmm(geo_data, gmm_params)
    inputs = footsteps_to_inputs(footsteps, size=size)
    return footsteps, inputs


def load_default_gmm() -> dict:
    return load_gmm_npz("E:/Terra/python/gmm_params.npz")


if __name__ == "__main__":
    # Smoke test: run on P14_1.mat alone (5 minutes of signal).
    import scipy.io
    import time

    p14 = scipy.io.loadmat("E:/Terra/data/raw/P14/P14_1.mat")["geo_data"].ravel()
    print(f"loaded P14_1.mat: {p14.shape}, {len(p14)/8000:.1f}s")

    gmm = load_default_gmm()
    t0 = time.time()
    footsteps, inputs = raw_to_inputs(p14, gmm)
    dt = time.time() - t0
    print(f"raw -> {len(footsteps)} footsteps -> inputs {inputs.shape} ({dt:.1f}s)")
    print(f"  per-footstep: {1000 * dt / max(len(footsteps), 1):.1f} ms (CWT+LUT+resize)")
