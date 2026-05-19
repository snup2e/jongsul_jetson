"""Fast CWT for the Terra inference pipeline.

Replaces pywt.cwt(scales=1..256, 'morl', method='conv') with an FFT-based
convolution that caches the integrated wavelet's frequency-domain
representation across calls.

Uses scipy.fft.rfft / irfft with workers=-1 (multi-threaded pocketfft) since
both signal and morl wavelet are real-valued. scipy.fft falls back to
np.fft.rfft on environments without scipy.fft. Verified pixel-equivalent
(uint8 LUT image, max |Δpx|=0) on P14/P50/P100.

Constraints:
    - Fixed scales = 1..256 and wavelet = 'morl' (matches v2 training).
    - Fixed precision=10 (matches pywt 1.1.1 default; pywt >=1.4 uses 12).
    - Caches are sized to the first signal length seen. Mixing lengths
      triggers a full cache rebuild, which is fine for 1500-sample windows.
"""
from typing import List, Optional

import numpy as np
import pywt

try:
    import scipy.fft as _sfft
    _HAVE_SCIPY_FFT = True
except ImportError:
    _HAVE_SCIPY_FFT = False


def _rfft(x, n=None, axis=-1):
    if _HAVE_SCIPY_FFT:
        return _sfft.rfft(x, n=n, axis=axis, workers=-1)
    return np.fft.rfft(x, n=n, axis=axis)


def _irfft(x, n=None, axis=-1):
    if _HAVE_SCIPY_FFT:
        return _sfft.irfft(x, n=n, axis=axis, workers=-1)
    return np.fft.irfft(x, n=n, axis=axis)


SCALES = np.arange(1, 257)
WAVELET = 'morl'
PRECISION = 10


_psi_lens = None         # type: Optional[List[int]]
_sqrt_scales = None      # type: Optional[np.ndarray]
_fft_psi = None          # type: Optional[np.ndarray]
_product_buf = None      # type: Optional[np.ndarray]
_neg_sqrt_col = None     # type: Optional[np.ndarray]
_row_idx = None          # type: Optional[np.ndarray]
_col_idx_A = None        # type: Optional[np.ndarray]
_col_idx_B = None        # type: Optional[np.ndarray]
_fft_len = 0             # type: int
_n_signal = 0            # type: int


def _build_caches(n_signal):
    # type: (int) -> None
    global _psi_lens, _sqrt_scales, _fft_psi, _product_buf
    global _neg_sqrt_col, _row_idx, _col_idx_A, _col_idx_B
    global _fft_len, _n_signal

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

    psi_padded = np.zeros((len(SCALES), fft_len), dtype=np.float64)
    for i, p in enumerate(psi_list):
        psi_padded[i, :p.size] = p
    fft_psi = _rfft(psi_padded, axis=1)

    sqrt_scales = np.sqrt(SCALES.astype(np.float64))

    # Vectorize post-loop: each scale's trimmed output is
    #   out[i, k] = -sqrt[i] * (conv_full[i, s_i + k + 1] - conv_full[i, s_i + k])
    # where s_i = (psi_lens[i] - 2) // 2 (always > 0 for morl scales 1..256).
    # Pre-build (256, n) index arrays so per-call work is two fancy-index gathers
    # + a subtract-multiply, no python loop over scales.
    starts = np.array([(L - 2) // 2 for L in psi_lens], dtype=np.intp)
    arange_n = np.arange(n_signal, dtype=np.intp)
    col_idx_A = starts[:, None] + arange_n[None, :]
    col_idx_B = col_idx_A + 1
    row_idx = np.arange(len(SCALES), dtype=np.intp)[:, None]

    _psi_lens = psi_lens
    _sqrt_scales = sqrt_scales
    _fft_psi = fft_psi
    _product_buf = np.empty_like(fft_psi)
    _neg_sqrt_col = -sqrt_scales[:, None]
    _row_idx = row_idx
    _col_idx_A = col_idx_A
    _col_idx_B = col_idx_B
    _fft_len = fft_len
    _n_signal = n_signal


def cwt(sig):
    # type: (np.ndarray) -> np.ndarray
    """CWT(sig, scales=1..256, 'morl') with cached rfft wavelets.

    Pixel-equivalent (after uint8 LUT quantize) to the full-complex variant
    and to pywt.cwt(sig, np.arange(1, 257), 'morl', method='conv',
    precision=10). Returns float64 (256, len(sig)).
    """
    n = sig.size
    if _fft_psi is None or n != _n_signal:
        _build_caches(n)

    sig_padded = np.zeros(_fft_len, dtype=np.float64)
    sig_padded[:n] = sig
    fft_sig = _rfft(sig_padded)

    np.multiply(_fft_psi, fft_sig[None, :], out=_product_buf)
    conv_full = _irfft(_product_buf, n=_fft_len, axis=1)

    # Vectorized post-loop using pre-built indices (see _build_caches).
    A = conv_full[_row_idx, _col_idx_A]
    B = conv_full[_row_idx, _col_idx_B]
    np.subtract(B, A, out=A)
    np.multiply(A, _neg_sqrt_col, out=A)
    return A
