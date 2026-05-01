"""Fast CWT for the Terra inference pipeline.

Replaces pywt.cwt(scales=1..256, 'morl', method='conv') with a manual
FFT-based convolution that caches the integrated wavelet's frequency-domain
representation across calls.

Pixel-equivalent to the original pipeline (max|Δpx|=0 vs pywt method='conv',
verified by jetson_cwt_bench.py on Jetson Nano with pywt 1.1.1, numpy 1.19.5).
~7.6× faster on Jetson (1486 ms -> 195 ms per footstep on P14 data).

Constraints:
    - Fixed scales = 1..256 and wavelet = 'morl' (matches v2 training).
    - Fixed precision=10 (matches pywt 1.1.1 default; pywt >=1.4 uses 12).
    - Caches are sized to the first signal length seen.  Mixing lengths
      triggers a full cache rebuild, which is fine for 1500-sample windows.
"""
from typing import List, Optional

import numpy as np
import pywt


SCALES = np.arange(1, 257)
WAVELET = 'morl'
PRECISION = 10


_psi_list = None      # type: Optional[List[np.ndarray]]
_psi_lens = None      # type: Optional[List[int]]
_sqrt_scales = None   # type: Optional[np.ndarray]
_fft_psi = None       # type: Optional[np.ndarray]
_fft_len = 0          # type: int
_n_signal = 0         # type: int


def _build_caches(n_signal):
    # type: (int) -> None
    global _psi_list, _psi_lens, _sqrt_scales, _fft_psi, _fft_len, _n_signal

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

    psi_lens = [p.size for p in psi_list]
    max_conv = n_signal + max(psi_lens) - 1
    fft_len = 1
    while fft_len < max_conv:
        fft_len *= 2

    fft_psi = np.zeros((len(SCALES), fft_len), dtype=np.complex128)
    for i, p in enumerate(psi_list):
        fft_psi[i, :p.size] = p
    fft_psi = np.fft.fft(fft_psi, axis=1)

    _psi_list = psi_list
    _psi_lens = psi_lens
    _sqrt_scales = np.sqrt(SCALES.astype(np.float64))
    _fft_psi = fft_psi
    _fft_len = fft_len
    _n_signal = n_signal


def cwt(sig):
    # type: (np.ndarray) -> np.ndarray
    """CWT(sig, scales=1..256, 'morl') with cached FFT wavelets.

    Pixel-equivalent to pywt.cwt(sig, np.arange(1, 257), 'morl',
    method='conv', precision=10). Returns float64 (256, len(sig)).
    """
    n = sig.size
    if _fft_psi is None or n != _n_signal:
        _build_caches(n)

    sig_padded = np.zeros(_fft_len, dtype=np.float64)
    sig_padded[:n] = sig
    fft_sig = np.fft.fft(sig_padded)

    conv_full = np.fft.ifft(_fft_psi * fft_sig[None, :], axis=1).real

    out = np.empty((len(SCALES), n), dtype=np.float64)
    for i in range(len(SCALES)):
        L = _psi_lens[i]
        conv_i = conv_full[i, :n + L - 1]
        coef = -_sqrt_scales[i] * np.diff(conv_i)
        d = (coef.size - n) / 2.0
        if d > 0:
            coef = coef[int(np.floor(d)):-int(np.ceil(d))]
        out[i] = coef[:n]
    return out
