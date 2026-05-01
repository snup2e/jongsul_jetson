"""Isolate where Path C diverges from Path A.

Path A:  processed PNG (matplotlib) -> PIL resize -> model            [reference]
Path B:  a1.mat row  -> LUT(direct) -> 224x224 -> model                [LUT only]
Path B': a1.mat row  -> matplotlib in-memory -> 224x224 -> model       [matplotlib in-memory only]
Path C:  raw .mat   -> apply_frozen_gmm -> CWT -> LUT -> 224 -> model  [full]

If B and B' both ~ A: rendering is fine, problem is in raw->footstep extraction.
If B << A:           LUT itself misclassifies.
If B' << A:          PNG-on-disk vs in-memory render differ (unlikely).
"""
from __future__ import annotations

import io
import time

import numpy as np
import scipy.io
import torch
import pywt
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer import build_model, predict_batch, NORMALIZE, WEIGHTS, DEVICE
from render_lut import cwt_to_rgb_direct


A1 = scipy.io.loadmat("E:/Terra/data/interim/a1.mat")["footstep_feat"]
LABELS = A1[:, -1].astype(int)
PERSONS = [("P1", 1), ("P14", 14), ("P50", 50), ("P100", 100)]
N = 100  # samples per person (matplotlib path is slow)


def render_matplotlib_inmem(sig_1500: np.ndarray) -> np.ndarray:
    """Replicate specmeaker.py exactly, in memory, return (224, 224, 3) uint8."""
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig_1500, scales, "morl")
    fig = plt.figure(figsize=(4.96, 0.84), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(coefficients, cmap="jet", aspect="auto")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, pad_inches=0, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(img)


def render_lut_inmem(sig_1500: np.ndarray) -> np.ndarray:
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig_1500, scales, "morl")
    return cwt_to_rgb_direct(coefficients, (224, 224))


def main():
    print("loading model...")
    import torch as _t
    ckpt = _t.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_large")
    model = build_model(arch, int(ckpt["num_classes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)

    print(f"\n{'person':<7} {'n':>4}   "
          f"{'B (LUT)':>8} {'B prime (mpl)':>14}")
    print("-" * 50)

    for person, pid in PERSONS:
        rows = A1[LABELS == pid][:, :-1][:N]
        target = pid - 1

        t0 = time.time()
        b_inputs = np.empty((len(rows), 224, 224, 3), dtype=np.uint8)
        for i, sig in enumerate(rows):
            b_inputs[i] = render_lut_inmem(sig)
        b_preds = predict_batch(model, b_inputs)
        b_acc = (b_preds == target).mean() * 100
        tb = time.time() - t0

        t0 = time.time()
        bp_inputs = np.empty((len(rows), 224, 224, 3), dtype=np.uint8)
        for i, sig in enumerate(rows):
            bp_inputs[i] = render_matplotlib_inmem(sig)
        bp_preds = predict_batch(model, bp_inputs)
        bp_acc = (bp_preds == target).mean() * 100
        tbp = time.time() - t0

        print(f"{person:<7} {len(rows):>4}   "
              f"{b_acc:>7.2f}% {bp_acc:>13.2f}%   "
              f"(B {tb:.1f}s, B' {tbp:.1f}s)")

        # show top miscls for B
        if b_acc < 90:
            wrong = b_preds[b_preds != target]
            if len(wrong):
                vals, cnt = np.unique(wrong, return_counts=True)
                topk = np.argsort(-cnt)[:3]
                print(f"        B miscls: " +
                      ", ".join(f"P{vals[i] + 1}({cnt[i]})" for i in topk))


if __name__ == "__main__":
    main()
