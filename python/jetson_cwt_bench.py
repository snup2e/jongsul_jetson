"""CWT optimization benchmark for Jetson Nano.

Measures variants of CWT(scales=1..256, 'morl') on length-1500 signals:

    baseline      pywt.cwt(..., method='conv')   [current Stage 5]
    pywt_fft      pywt.cwt(..., method='fft')    [if pywt 1.1.1 supports]
    cached_psi    manual time-domain conv with precomputed int_psi_scale
    fft_cached    manual FFT-based conv with cached freq-domain wavelets

Each variant is timed (median over N runs after warm-up) and verified against
the baseline:
    - max|Δ| / mean|Δ| in coefficient space
    - max pixel diff after 224x224 RGB rendering (render_lut.cwt_to_rgb_twostep)

Pixel diff = 0 means we can drop the variant into the inference pipeline
without retraining. >0 but small means we're within v2 augmentation margin.

Usage on Jetson:
    cd ~/terra
    python3 jetson_cwt_bench.py                       # synthetic chirp
    python3 jetson_cwt_bench.py --signal P14_1.mat    # real footstep window
"""
import argparse
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pywt


SCALES = np.arange(1, 257)
WAVELET = 'morl'
SIG_LEN = 1500
# pywt 1.1.1 (Jetson) hard-codes precision=10. pywt >=1.4 defaults to 12 but
# accepts a kwarg.  Force 10 everywhere so this benchmark matches the actual
# Jetson Stage 5 output (and so cache-vs-baseline diffs reflect implementation,
# not an apples-to-oranges precision change).
PRECISION = 10


def load_signal(path):
    # type: (Optional[str]) -> np.ndarray
    if path:
        import scipy.io
        geo = scipy.io.loadmat(path)["geo_data"].ravel().astype(np.float64)
        # take a chunk past the silent intro
        start = len(geo) // 4
        return geo[start:start + SIG_LEN].copy()
    # synthetic broadband chirp 1 -> 200 Hz at Fs=8000
    t = np.arange(SIG_LEN) / 8000.0
    chirp = np.sin(2 * np.pi * (1 + 199 * t / t[-1]) * t)
    noise = 0.3 * np.random.RandomState(0).standard_normal(SIG_LEN)
    return chirp + noise


# ------------------------------------------------------------------ variants

def _pywt_cwt(sig, method):
    # type: (np.ndarray, str) -> np.ndarray
    try:
        coef, _ = pywt.cwt(sig, SCALES, WAVELET, method=method, precision=PRECISION)
    except TypeError:
        # pywt 1.1.x has no `precision` kwarg; default is 10 which is what we want
        coef, _ = pywt.cwt(sig, SCALES, WAVELET, method=method)
    return coef


def cwt_baseline(sig):
    # type: (np.ndarray) -> np.ndarray
    return _pywt_cwt(sig, 'conv')


def cwt_pywt_fft(sig):
    # type: (np.ndarray) -> np.ndarray
    return _pywt_cwt(sig, 'fft')


# Cached integrated-wavelet tables (one entry per scale).  These do not
# depend on the input signal, so we build them once and reuse.
_PSI_CACHE = None    # type: Optional[Tuple[List[np.ndarray], np.ndarray]]
_FFT_CACHE = None    # type: Optional[Tuple[np.ndarray, int, np.ndarray, List[int]]]


def _build_psi_cache():
    # type: () -> Tuple[List[np.ndarray], np.ndarray]
    """Replicate pywt.cwt's per-scale int_psi slicing.

    Returns (psi_scaled list of (L_s,) arrays, sqrt(scales) array).
    Matches pywt 1.1.x source: arange(scale*support+1)/(scale*step) -> int -> reverse.
    """
    wavelet = pywt.ContinuousWavelet(WAVELET)
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=PRECISION)
    int_psi = np.asarray(int_psi)
    if getattr(wavelet, 'complex_cwt', False):
        int_psi = np.conj(int_psi)
    step = x[1] - x[0]
    psi_list = []
    for scale in SCALES:
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)
        j = j[j < int_psi.size]
        psi_list.append(int_psi[j][::-1].copy())
    return psi_list, np.sqrt(SCALES.astype(np.float64))


def _trim(coef, n_data):
    # type: (np.ndarray, int) -> np.ndarray
    """Match pywt's post-diff trim to data size."""
    d = (coef.size - n_data) / 2.0
    if d > 0:
        coef = coef[int(np.floor(d)):-int(np.ceil(d))]
    return coef[:n_data]


def cwt_cached_psi(sig):
    # type: (np.ndarray) -> np.ndarray
    """Time-domain conv but with cached int_psi_scale tables."""
    global _PSI_CACHE
    if _PSI_CACHE is None:
        _PSI_CACHE = _build_psi_cache()
    psi_list, sqrt_scales = _PSI_CACHE
    n = len(sig)
    out = np.empty((len(SCALES), n), dtype=np.float64)
    for i in range(len(SCALES)):
        conv = np.convolve(sig, psi_list[i])
        coef = -sqrt_scales[i] * np.diff(conv)
        out[i] = _trim(coef, n)
    return out


def cwt_fft_cached(sig):
    # type: (np.ndarray) -> np.ndarray
    """FFT-based conv with cached freq-domain wavelets.

    Per call: 1 FFT of the signal + (256 broadcast multiplies) + 1 batched IFFT.
    Per scale: trim/diff in time domain.
    """
    global _PSI_CACHE, _FFT_CACHE
    if _PSI_CACHE is None:
        _PSI_CACHE = _build_psi_cache()
    psi_list, sqrt_scales = _PSI_CACHE
    n = len(sig)

    if _FFT_CACHE is None:
        psi_lens = [p.size for p in psi_list]
        max_conv = n + max(psi_lens) - 1
        fft_len = 1
        while fft_len < max_conv:
            fft_len *= 2
        fft_psi = np.zeros((len(SCALES), fft_len), dtype=np.complex128)
        for i, p in enumerate(psi_list):
            fft_psi[i, :p.size] = p
        fft_psi = np.fft.fft(fft_psi, axis=1)
        _FFT_CACHE = (fft_psi, fft_len, sqrt_scales, psi_lens)

    fft_psi, fft_len, sqrt_scales, psi_lens = _FFT_CACHE

    sig_padded = np.zeros(fft_len, dtype=np.float64)
    sig_padded[:n] = sig
    fft_sig = np.fft.fft(sig_padded)

    conv_full = np.fft.ifft(fft_psi * fft_sig[None, :], axis=1).real

    out = np.empty((len(SCALES), n), dtype=np.float64)
    for i in range(len(SCALES)):
        conv_i = conv_full[i, :n + psi_lens[i] - 1]
        coef = -sqrt_scales[i] * np.diff(conv_i)
        out[i] = _trim(coef, n)
    return out


# ------------------------------------------------------------------ harness

def time_call(fn, sig, n_warm=3, n_run=20):
    # type: (Callable, np.ndarray, int, int) -> float
    for _ in range(n_warm):
        fn(sig)
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        fn(sig)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def diff_stats(a, b):
    # type: (np.ndarray, np.ndarray) -> Dict[str, float]
    d = np.abs(a - b)
    scale = float(np.abs(a).max()) + 1e-12
    return {'max': float(d.max()), 'mean': float(d.mean()), 'rel': float(d.max() / scale)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--signal", default=None,
                    help=".mat file with geo_data (uses one window). default=synthetic")
    ap.add_argument("--n_warm", type=int, default=3)
    ap.add_argument("--n_run", type=int, default=20)
    args = ap.parse_args()

    print("pywt  : %s" % pywt.__version__)
    print("numpy : %s" % np.__version__)

    sig = load_signal(args.signal)
    src = args.signal if args.signal else "synthetic chirp"
    print("signal: %s, shape %s, range [%.3f, %.3f]" %
          (src, sig.shape, sig.min(), sig.max()))
    print("scales: 1..%d, wavelet '%s'" % (SCALES[-1], WAVELET))
    print("timing: %d warm-up + %d runs (median reported)\n" % (args.n_warm, args.n_run))

    coef_ref = cwt_baseline(sig)
    print("baseline output: shape %s, dtype %s\n" % (coef_ref.shape, coef_ref.dtype))

    variants = [
        ('baseline (pywt method=conv)', cwt_baseline),
    ]
    try:
        cwt_pywt_fft(sig)
        variants.append(('pywt method=fft', cwt_pywt_fft))
        print("[ok] pywt.cwt(method='fft') is available")
    except (TypeError, ValueError) as e:
        print("[skip] pywt.cwt(method='fft'): %s" % e)
    variants.append(('cached_psi (manual conv)', cwt_cached_psi))
    variants.append(('fft_cached (manual FFT)',  cwt_fft_cached))
    print()

    print("%-32s %10s   %10s  %10s  %10s" %
          ("variant", "ms/call", "max|Δ|", "mean|Δ|", "rel|Δ|"))
    print("-" * 80)
    results = []
    for name, fn in variants:
        ms = time_call(fn, sig, args.n_warm, args.n_run)
        coef = fn(sig)
        if coef.shape != coef_ref.shape:
            print("%-32s %9.1f   shape mismatch %s vs %s" %
                  (name, ms, coef.shape, coef_ref.shape))
            continue
        d = diff_stats(coef_ref, coef)
        print("%-32s %9.1f   %10.2e  %10.2e  %10.2e" %
              (name, ms, d['max'], d['mean'], d['rel']))
        results.append((name, ms, coef))

    # Pixel-level check after LUT (loose: confirms inference pipeline unaffected)
    print("\n--- pixel diff after render_lut.cwt_to_rgb_twostep (224x224 RGB uint8) ---")
    try:
        from render_lut import cwt_to_rgb_twostep
        rgb_ref = cwt_to_rgb_twostep(coef_ref)
        for name, _ms, coef in results[1:]:
            rgb = cwt_to_rgb_twostep(coef)
            d = np.abs(rgb.astype(np.int32) - rgb_ref.astype(np.int32))
            print("%-32s max|Δpx|=%3d  mean|Δpx|=%.4f" %
                  (name, int(d.max()), float(d.mean())))
    except Exception as e:
        print("[skip] LUT pixel check: %s" % e)

    # Speedup summary
    print("\n--- speedup vs baseline ---")
    base_ms = results[0][1]
    for name, ms, _ in results:
        print("%-32s %6.2fx  (%.1f ms)" % (name, base_ms / ms, ms))


if __name__ == "__main__":
    main()
