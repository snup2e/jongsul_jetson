"""matplotlib-free CWT → RGB rendering for Jetson inference.

Replaces specmeaker.py's matplotlib pipeline with a pure numpy/OpenCV LUT.
Training PNGs were rendered as 84x496 RGBA via matplotlib jet, then loaded as
RGB and resized to 224x224 via PIL BILINEAR.

This module skips disk I/O and matplotlib entirely:
    coefficients (256, 1500)  -- pywt.cwt output
        |  per-image min-max normalize -> uint8 indices
        v
    indices (256, 1500) uint8
        |  LUT[indices]  (matplotlib's 'jet' colormap, 256 entries, RGB uint8)
        v
    rgb (256, 1500, 3) uint8
        |  cv2.resize -> 224x224
        v
    final (224, 224, 3) uint8

Two resize variants are exposed so we can pick whichever matches training PNGs
better in validate_stage4.py:

    direct:    256x1500 --INTER_AREA--> 224x224
    two_step:  256x1500 --INTER_AREA--> 84x496 --INTER_LINEAR--> 224x224
               (mimics matplotlib's 84x496 PNG + PIL BILINEAR)
"""
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


# Load the jet LUT from a frozen .npy that was exported from the training
# environment (Colab matplotlib 3.7+). matplotlib LUTs drift across versions
# (3.7 deprecated cm.get_cmap; 3.10 returns slightly different bytes), so we
# refuse to silently use a local matplotlib LUT for inference — that's how the
# v1 brittleness happened.
#
# Override path with $TERRA_JET_LUT. Otherwise check standard candidate paths
# (co-located on Jetson, sibling weights/ folder on Windows).
def _find_lut() -> Path:
    env = os.environ.get("TERRA_JET_LUT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "jet_lut_v2.npy",                  # Jetson: ~/terra/jet_lut_v2.npy
        here.parent / "weights" / "jet_lut_v2.npy",  # Windows: E:/Terra/weights/...
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default (triggers warning below)


LUT_PATH = _find_lut()


def _load_lut() -> np.ndarray:
    if LUT_PATH.exists():
        lut = np.load(str(LUT_PATH))
        if lut.shape != (256, 3) or lut.dtype != np.uint8:
            raise ValueError(f"bad LUT at {LUT_PATH}: shape={lut.shape}, dtype={lut.dtype}")
        return lut
    # Fallback: local matplotlib. Loud warning — this WILL drift from training.
    warnings.warn(
        f"\n{LUT_PATH} not found, falling back to local matplotlib jet.\n"
        f"This may not match the training LUT (Colab matplotlib 3.7+).\n"
        f"Export from Colab via notebook/colab_v2_export_lut.ipynb and place at the path.",
        stacklevel=2,
    )
    import matplotlib.cm as _cm
    return (_cm.get_cmap("jet")(np.arange(256) / 255.0)[:, :3] * 255.0
            ).round().astype(np.uint8)


JET_LUT_RGB_U8: np.ndarray = _load_lut()


def coeffs_to_indices(coefficients: np.ndarray) -> np.ndarray:
    """Per-image min-max normalize to [0, 255] uint8 indices.

    Matches matplotlib imshow's default Normalize: vmin/vmax = data min/max,
    linear scale, then quantize to LUT index.
    """
    cmin = float(coefficients.min())
    cmax = float(coefficients.max())
    if cmax - cmin < 1e-12:
        return np.zeros(coefficients.shape, dtype=np.uint8)
    scaled = (coefficients - cmin) / (cmax - cmin) * 255.0
    return np.clip(scaled.round(), 0, 255).astype(np.uint8)


def cwt_to_rgb_direct(coefficients: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """coefficients (256, 1500) -> (H, W, 3) uint8 RGB, single resize.

    size is (H, W).
    """
    idx = coeffs_to_indices(coefficients)
    rgb = JET_LUT_RGB_U8[idx]  # (256, 1500, 3)
    H, W = size
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


def cwt_to_rgb_twostep(coefficients: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Mimic training pipeline: render at 84x496, then resize to 224x224.

    Uses INTER_AREA for the first downsample (256x1500 -> 84x496) and
    INTER_LINEAR for the upsample/downsample mix (84x496 -> 224x224, since
    height grows and width shrinks). PIL Image.resize default is BILINEAR.
    """
    idx = coeffs_to_indices(coefficients)
    rgb_full = JET_LUT_RGB_U8[idx]  # (256, 1500, 3)
    rgb_84 = cv2.resize(rgb_full, (496, 84), interpolation=cv2.INTER_AREA)
    H, W = size
    return cv2.resize(rgb_84, (W, H), interpolation=cv2.INTER_LINEAR)


def render_from_signal(sig_1500: np.ndarray, size: Tuple[int, int] = (224, 224),
                       method: str = "twostep") -> np.ndarray:
    """End-to-end: 1500-sample footstep -> (H, W, 3) uint8 RGB."""
    from cwt_fast import cwt
    coefficients = cwt(sig_1500)
    if method == "direct":
        return cwt_to_rgb_direct(coefficients, size)
    if method == "twostep":
        return cwt_to_rgb_twostep(coefficients, size)
    raise ValueError(f"unknown method: {method}")
