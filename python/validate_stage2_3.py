"""Stage 2 & 3 validation: replicate specmeaker.py output and compare with processed/Pn PNGs.

Per CLAUDE.md & specmeaker.py:
  - input:  a1.mat row (1500-sample footstep)
  - process: pywt.cwt(scales=1..256, morl) → matplotlib imshow(jet, aspect='auto')
            → savefig(transparent=True, bbox_inches='tight', pad_inches=0)
  - target: processed/Pn/cwt_image_{n-1}_{idx}.png  (496×84 RGBA)

Since this re-runs the EXACT original Python code, we expect (near-)byte-equal output.
If pixel RMSE is small, Stage 2+3 are validated and we focus optimization on the inference path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import pywt
import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
from PIL import Image


DATA = Path("E:/Terra/data")
A1 = scipy.io.loadmat(str(DATA / "interim/a1.mat"))["footstep_feat"]
LABELS = A1[:, -1].astype(int)


def render_footstep(sig_1500: np.ndarray, out_path: str,
                    figsize=(4.96, 0.84), dpi: int = 100):
    """specmeaker.py recipe + explicit figsize so output is 496×84.

    Original specmeaker.py relied on env rcParams to land at this size; we
    fix it explicitly to make the comparison reproducible.
    """
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig_1500, scales, "morl")
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed
    ax.imshow(coefficients, cmap="jet", aspect="auto")
    ax.axis("off")
    fig.savefig(out_path, transparent=True, pad_inches=0)
    plt.close(fig)


def render_footstep_naive(sig_1500: np.ndarray, out_path: str):
    """Original specmeaker.py recipe verbatim (for comparison)."""
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig_1500, scales, "morl")
    plt.imshow(coefficients, cmap="jet", aspect="auto")
    plt.axis("off")
    plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()


def compare_pngs(p1: str, p2: str) -> dict:
    a = np.asarray(Image.open(p1)).astype(np.int16)
    b = np.asarray(Image.open(p2)).astype(np.int16)
    if a.shape != b.shape:
        return dict(shape_a=a.shape, shape_b=b.shape, mismatch=True)
    diff = np.abs(a - b)
    return dict(
        shape=a.shape,
        max_abs=int(diff.max()),
        mean_abs=float(diff.mean()),
        rmse=float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
        n_pixels_diff=int((diff.sum(axis=-1) > 0).sum()),
        total_pixels=a.shape[0] * a.shape[1],
    )


def main():
    out_dir = Path("E:/Terra/python/_out_stage23")
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("STAGE 2/3 — replicate specmeaker.py on a1.mat → compare with processed/")
    print("=" * 70)

    # Test on a few footsteps from each available person
    test_cases = [
        ("P1", 1, 0),         # P1 first
        ("P1", 1, 100),       # P1 mid
        ("P14", 14, 0),
        ("P50", 50, 0),
        ("P50", 50, 200),
        ("P100", 100, 500),
    ]

    print(f"\n{'Person':<7} {'idx':>5} {'shape':>14} "
          f"{'max|Δ|':>7} {'mean|Δ|':>9} {'RMSE':>7} {'%diff':>7}")
    print("-" * 70)

    rmses = []
    for person, pid, idx in test_cases:
        # Get the matching row in a1.mat
        rows = A1[LABELS == pid]
        if idx >= len(rows):
            print(f"{person:<7} {idx:>5d}  skip (only {len(rows)} rows)")
            continue
        sig = rows[idx, :-1]

        # Render via specmeaker recipe
        ours = out_dir / f"{person}_{idx}.png"
        render_footstep(sig, str(ours))

        # Reference PNG path
        ref = DATA / f"processed/{person}/cwt_image_{pid-1}_{idx}.png"
        if not ref.exists():
            print(f"{person:<7} {idx:>5d}  ref missing: {ref}")
            continue

        d = compare_pngs(str(ours), str(ref))
        if d.get("mismatch"):
            print(f"{person:<7} {idx:>5d}  shape mismatch: {d['shape_a']} vs {d['shape_b']}")
            continue
        pct_diff = 100.0 * d["n_pixels_diff"] / d["total_pixels"]
        rmses.append(d["rmse"])
        print(f"{person:<7} {idx:>5d}  {str(d['shape']):>14} "
              f"{d['max_abs']:>7} {d['mean_abs']:>9.3f} "
              f"{d['rmse']:>7.3f} {pct_diff:>6.2f}%")

    if rmses:
        print(f"\n  mean RMSE across {len(rmses)} samples: {np.mean(rmses):.3f}")
        print(f"  acceptance: RMSE < ~5 (out of 0-255) ⇒ effectively identical for training")
        print(f"             RMSE > 20 ⇒ matplotlib version / dpi / bbox computation differs")


if __name__ == "__main__":
    main()
