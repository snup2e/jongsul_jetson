"""Why does disk PNG (Path A) classify P50 correctly but in-memory matplotlib of
the SAME a1.mat row classify it as P10?

For a single footstep idx, render four ways and compare predictions:
    A:  PIL.open(processed/.../cwt_image_*.png).convert('RGB').resize(224)
    B:  a1.mat row -> LUT direct -> 224
    B': a1.mat row -> matplotlib in-mem PNG -> PIL.convert('RGB').resize(224)
    B'': processed PNG bytes -> in-mem (same as A, sanity)

Compute pairwise pixel RMSE at 224x224 and print top-1 prediction for each.
"""
from __future__ import annotations

import io

import numpy as np
import scipy.io
import torch
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from infer import build_model, NORMALIZE, WEIGHTS, DEVICE
from render_lut import cwt_to_rgb_direct


A1 = scipy.io.loadmat("E:/Terra/data/interim/a1.mat")["footstep_feat"]
LABELS = A1[:, -1].astype(int)


def render_mpl_pil(sig):
    scales = np.arange(1, 257)
    c, _ = pywt.cwt(sig, scales, "morl")
    fig = plt.figure(figsize=(4.96, 0.84), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(c, cmap="jet", aspect="auto")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, pad_inches=0, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((224, 224), Image.BILINEAR), buf.getvalue()


def render_lut(sig):
    scales = np.arange(1, 257)
    c, _ = pywt.cwt(sig, scales, "morl")
    return cwt_to_rgb_direct(c, (224, 224))


def to_tensor(arr):
    t = torch.from_numpy(arr).to(DEVICE)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return NORMALIZE(t.permute(0, 3, 1, 2).float() / 255.0)


@torch.no_grad()
def topk(model, arr, k=3):
    logits = model(to_tensor(arr))[0]
    p = torch.softmax(logits, dim=0)
    top = torch.topk(p, k)
    return [(int(top.indices[i]), float(top.values[i])) for i in range(k)]


def main():
    print("loading model...")
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_large")
    model = build_model(arch, int(ckpt["num_classes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)

    for person, pid in [("P1", 1), ("P14", 14), ("P50", 50), ("P100", 100)]:
        idx = 0
        target = pid - 1
        rows = A1[LABELS == pid]
        sig = rows[idx, :-1]

        # Path A: disk PNG
        disk_png = f"E:/Terra/data/processed/{person}/cwt_image_{pid - 1}_{idx}.png"
        a_img = Image.open(disk_png).convert("RGB").resize((224, 224), Image.BILINEAR)
        a_arr = np.asarray(a_img)

        # Path B: LUT
        b_arr = render_lut(sig)

        # Path B': matplotlib in-memory
        bp_img, bp_bytes = render_mpl_pil(sig)
        bp_arr = np.asarray(bp_img)

        # Pairwise RMSE
        def rmse(x, y):
            return float(np.sqrt(((x.astype(np.float64) - y) ** 2).mean()))

        print(f"\n=== {person} idx={idx}, target_idx={target} (P{pid}) ===")
        print(f"  RMSE A vs B  (disk vs LUT):           {rmse(a_arr, b_arr):.2f}")
        print(f"  RMSE A vs B' (disk vs in-mem mpl):    {rmse(a_arr, bp_arr):.2f}")
        print(f"  RMSE B vs B' (LUT vs in-mem mpl):     {rmse(b_arr, bp_arr):.2f}")

        # Predictions
        ka = topk(model, a_arr)
        kb = topk(model, b_arr)
        kbp = topk(model, bp_arr)

        def fmt(k):
            return ", ".join(f"P{c + 1}={p * 100:.1f}%" for c, p in k)

        print(f"  A  top3: {fmt(ka)}")
        print(f"  B  top3: {fmt(kb)}")
        print(f"  B' top3: {fmt(kbp)}")

        # Compare PNG sizes / disk vs in-mem
        with open(disk_png, "rb") as f:
            disk_bytes = f.read()
        a_raw = np.asarray(Image.open(disk_png))
        bp_raw = np.asarray(Image.open(io.BytesIO(bp_bytes)))
        print(f"  disk PNG: {a_raw.shape} {a_raw.dtype}, {len(disk_bytes)} bytes")
        print(f"  in-mem PNG: {bp_raw.shape} {bp_raw.dtype}, {len(bp_bytes)} bytes")
        if a_raw.shape == bp_raw.shape:
            d = np.abs(a_raw.astype(np.int16) - bp_raw.astype(np.int16))
            print(f"  raw 84x496 RGBA RMSE: {np.sqrt((d.astype(np.float64)**2).mean()):.2f}, max: {d.max()}")


if __name__ == "__main__":
    main()
