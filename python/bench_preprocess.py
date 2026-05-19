"""Per-stage profiling for the preprocessing pipeline.

Runs raw -> envelope -> CWT -> LUT -> resize on real data and reports
per-footstep median time for each stage. Used to find the bottleneck
inside the ~336 ms render budget claimed in CLAUDE.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import apply_envelope_detector
from cwt_fast import cwt as fast_cwt
from render_lut import JET_LUT_RGB_U8, coeffs_to_indices


def median_ms(times):
    return float(np.median(times) * 1000.0)


def run(mat_path: str, n_warmup: int = 3, max_footsteps: int = 50):
    geo = scipy.io.loadmat(mat_path)["geo_data"].ravel().astype(np.float64)
    print(f"loaded {mat_path}: {geo.shape}, {geo.size / 8000:.1f}s")

    t0 = time.perf_counter()
    footsteps = apply_envelope_detector(geo)
    t_extract = time.perf_counter() - t0
    M = min(len(footsteps), max_footsteps)
    print(f"envelope -> {len(footsteps)} footsteps in {t_extract * 1000:.0f} ms "
          f"(per-footstep extract: {1000 * t_extract / max(len(footsteps), 1):.1f} ms)")
    print(f"profiling first {M} footsteps...\n")

    cwt_times = []
    norm_times = []
    lut_times = []
    resize_times = []

    for i in range(n_warmup + M):
        sig = footsteps[i % len(footsteps)]

        t = time.perf_counter()
        coeffs = fast_cwt(sig)
        cwt_t = time.perf_counter() - t

        t = time.perf_counter()
        idx = coeffs_to_indices(coeffs)
        norm_t = time.perf_counter() - t

        t = time.perf_counter()
        rgb = JET_LUT_RGB_U8[idx]
        lut_t = time.perf_counter() - t

        t = time.perf_counter()
        out = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        resize_t = time.perf_counter() - t

        if i >= n_warmup:
            cwt_times.append(cwt_t)
            norm_times.append(norm_t)
            lut_times.append(lut_t)
            resize_times.append(resize_t)

    total_ms = median_ms(cwt_times) + median_ms(norm_times) + median_ms(lut_times) + median_ms(resize_times)
    print(f"  cwt_fast.cwt            : {median_ms(cwt_times):6.1f} ms")
    print(f"  coeffs_to_indices       : {median_ms(norm_times):6.1f} ms")
    print(f"  LUT lookup              : {median_ms(lut_times):6.1f} ms")
    print(f"  cv2.resize INTER_AREA   : {median_ms(resize_times):6.1f} ms")
    print(f"  --------------------------------")
    print(f"  total (render only)     : {total_ms:6.1f} ms / footstep")
    print(f"  envelope amortized      : {1000 * t_extract / max(len(footsteps), 1):6.1f} ms / footstep")


if __name__ == "__main__":
    mat = sys.argv[1] if len(sys.argv) > 1 else "E:/Terra/data/raw/P14/P14_1.mat"
    run(mat)
