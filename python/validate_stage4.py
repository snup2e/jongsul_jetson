"""Stage 4 validation: matplotlib-free LUT pipeline vs training PNG path.

Reference path (what the model was trained on):
    processed/Pn/cwt_image_*.png (84x496 RGBA on disk)
        -> PIL Image.open(path).convert('RGB')
        -> resize((224, 224))   # PIL default BILINEAR
        -> (224, 224, 3) uint8

Test path (Jetson inference):
    a1.mat row -> CWT (256x1500 float)
        -> render_lut.cwt_to_rgb_{direct,twostep}((224, 224))
        -> (224, 224, 3) uint8

Acceptance: pixel RMSE on 224x224 RGB << 5/255 (~2%) means the LUT path
produces effectively the same model input as the training path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import pywt
from PIL import Image

from render_lut import cwt_to_rgb_direct, cwt_to_rgb_twostep


DATA = Path("E:/Terra/data")
A1 = scipy.io.loadmat(str(DATA / "interim/a1.mat"))["footstep_feat"]
LABELS = A1[:, -1].astype(int)


def reference_224(png_path: Path) -> np.ndarray:
    """Training-equivalent loader: PNG -> RGB -> 224x224 BILINEAR."""
    img = Image.open(png_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    return np.asarray(img)  # (224, 224, 3) uint8


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(
        max_abs=int(d.max()),
        mean_abs=float(d.mean()),
        rmse=float(np.sqrt((d.astype(np.float64) ** 2).mean())),
    )


def main():
    print("=" * 78)
    print("STAGE 4 - LUT pipeline (no matplotlib) vs training PNG path @ 224x224")
    print("=" * 78)

    cases = [
        ("P1", 1, 0),
        ("P1", 1, 100),
        ("P1", 1, 1000),
        ("P14", 14, 0),
        ("P14", 14, 500),
        ("P50", 50, 0),
        ("P50", 50, 200),
        ("P100", 100, 0),
        ("P100", 100, 500),
    ]

    print(f"\n{'person':<7} {'idx':>5}  "
          f"{'direct max':>10} {'direct RMSE':>11}  "
          f"{'twostep max':>11} {'twostep RMSE':>12}")
    print("-" * 78)

    rmse_direct, rmse_two = [], []
    scales = np.arange(1, 257)

    for person, pid, idx in cases:
        rows = A1[LABELS == pid]
        if idx >= len(rows):
            continue
        sig = rows[idx, :-1]

        ref_png = DATA / f"processed/{person}/cwt_image_{pid - 1}_{idx}.png"
        if not ref_png.exists():
            print(f"{person:<7} {idx:>5}  ref missing: {ref_png}")
            continue

        ref = reference_224(ref_png)
        coeffs, _ = pywt.cwt(sig, scales, "morl")

        out_d = cwt_to_rgb_direct(coeffs, (224, 224))
        out_t = cwt_to_rgb_twostep(coeffs, (224, 224))
        sd = diff_stats(out_d, ref)
        st = diff_stats(out_t, ref)

        rmse_direct.append(sd["rmse"])
        rmse_two.append(st["rmse"])

        print(f"{person:<7} {idx:>5}  "
              f"{sd['max_abs']:>10} {sd['rmse']:>11.3f}  "
              f"{st['max_abs']:>11} {st['rmse']:>12.3f}")

    if rmse_direct:
        print()
        print(f"  direct  : mean RMSE = {np.mean(rmse_direct):6.3f}  "
              f"({100*np.mean(rmse_direct)/255:.2f}% of 0-255 range)")
        print(f"  twostep : mean RMSE = {np.mean(rmse_two):6.3f}  "
              f"({100*np.mean(rmse_two)/255:.2f}% of 0-255 range)")
        print(f"\n  acceptance: RMSE < 5 (2%)  -> training-equivalent input")
        print(f"              RMSE > 15 (6%)  -> investigate LUT or interp choice")


if __name__ == "__main__":
    main()
