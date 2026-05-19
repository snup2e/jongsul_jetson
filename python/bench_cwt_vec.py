"""Verify vectorized post-loop matches the legacy per-scale Python loop.

Runs cwt_fast.cwt (vectorized) and a reference reproduction of the previous
per-scale loop using the same cached FFT product. They must match bit-for-bit.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cwt_fast
from extract import apply_envelope_detector
from render_lut import JET_LUT_RGB_U8, coeffs_to_indices


def cwt_legacy(sig):
    """Reference: pre-vectorization per-scale Python loop."""
    n = sig.size
    if cwt_fast._fft_psi is None or n != cwt_fast._n_signal:
        cwt_fast._build_caches(n)

    sig_padded = np.zeros(cwt_fast._fft_len, dtype=np.float64)
    sig_padded[:n] = sig
    fft_sig = cwt_fast._rfft(sig_padded)

    product = cwt_fast._fft_psi * fft_sig[None, :]
    conv_full = cwt_fast._irfft(product, n=cwt_fast._fft_len, axis=1)

    out = np.empty((len(cwt_fast.SCALES), n), dtype=np.float64)
    for i in range(len(cwt_fast.SCALES)):
        L = cwt_fast._psi_lens[i]
        conv_i = conv_full[i, :n + L - 1]
        coef = -cwt_fast._sqrt_scales[i] * np.diff(conv_i)
        d = (coef.size - n) / 2.0
        if d > 0:
            coef = coef[int(np.floor(d)):-int(np.ceil(d))]
        out[i] = coef[:n]
    return out


def render(coeffs):
    idx = coeffs_to_indices(coeffs)
    rgb = JET_LUT_RGB_U8[idx]
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)


def main(mat_path, n_measure=50):
    geo = scipy.io.loadmat(mat_path)["geo_data"].ravel().astype(np.float64)
    footsteps = apply_envelope_detector(geo)
    M = min(len(footsteps), n_measure)
    print(f"{mat_path}: {len(footsteps)} footsteps, profiling first {M}\n")

    # warmup
    for i in range(3):
        cwt_fast.cwt(footsteps[i])
        cwt_legacy(footsteps[i])

    max_abs = 0.0
    img_mismatch = 0
    pixel_max = 0
    t_vec, t_leg = [], []
    for i in range(M):
        sig = footsteps[i % len(footsteps)]

        t = time.perf_counter()
        c_vec = cwt_fast.cwt(sig).copy()
        t_vec.append(time.perf_counter() - t)

        t = time.perf_counter()
        c_leg = cwt_legacy(sig)
        t_leg.append(time.perf_counter() - t)

        max_abs = max(max_abs, float(np.abs(c_vec - c_leg).max()))
        img1, img2 = render(c_vec), render(c_leg)
        if not np.array_equal(img1, img2):
            img_mismatch += 1
            pixel_max = max(pixel_max, int(np.abs(img1.astype(int) - img2.astype(int)).max()))

    print(f"per-footstep median time:")
    print(f"  legacy (python loop) : {np.median(t_leg) * 1000:6.1f} ms")
    print(f"  vectorized           : {np.median(t_vec) * 1000:6.1f} ms")
    print(f"  speedup              : {np.median(t_leg) / np.median(t_vec):.2f}x\n")
    print(f"equivalence:")
    print(f"  max |abs diff| (float64): {max_abs:.3e}")
    print(f"  uint8 LUT image mismatches : {img_mismatch} / {M}")
    print(f"  uint8 LUT max pixel diff   : {pixel_max}")


if __name__ == "__main__":
    mat = sys.argv[1] if len(sys.argv) > 1 else "E:/Terra/data/raw/P14/P14_1.mat"
    main(mat)
