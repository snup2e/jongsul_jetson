"""Try specmeaker.py verbatim (no figsize, no add_axes) and compare.

If even verbatim local rendering disagrees with disk PNG, the gap is purely
matplotlib/pywt version drift between this machine and the Colab session
that produced processed/. In that case the answer is: retrain the backbone
on a deterministic pipeline (LUT), not chase matplotlib parity.
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


print("matplotlib:", matplotlib.__version__)
print("pywt     :", pywt.__version__)
print("PIL      :", Image.__version__)


A1 = scipy.io.loadmat("E:/Terra/data/interim/a1.mat")["footstep_feat"]
LABELS = A1[:, -1].astype(int)


def render_specmeaker_verbatim(sig):
    """Exactly specmeaker.py — no figsize, no add_axes."""
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig, scales, "morl")
    fig = plt.figure()
    plt.imshow(coefficients, cmap="jet", aspect="auto")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, transparent=True, bbox_inches="tight", pad_inches=0, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


@torch.no_grad()
def predict_top3(model, arr_224):
    t = torch.from_numpy(np.ascontiguousarray(arr_224)).to(DEVICE).unsqueeze(0)
    t = NORMALIZE(t.permute(0, 3, 1, 2).float() / 255.0)
    p = torch.softmax(model(t)[0], dim=0)
    top = torch.topk(p, 3)
    return [(int(top.indices[i]), float(top.values[i])) for i in range(3)]


def main():
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_large")
    model = build_model(arch, int(ckpt["num_classes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)

    for person, pid in [("P50", 50), ("P100", 100)]:
        sig = A1[LABELS == pid][0, :-1]

        verb_img = render_specmeaker_verbatim(sig)
        print(f"\n=== {person} idx=0 ===")
        print(f"  verbatim render size: {verb_img.size}, mode: {verb_img.mode}")

        # match to disk PNG ratio
        verb_arr = np.asarray(verb_img.convert("RGB").resize((224, 224), Image.BILINEAR))

        disk_png = f"E:/Terra/data/processed/{person}/cwt_image_{pid - 1}_0.png"
        disk_img = Image.open(disk_png)
        print(f"  disk PNG size: {disk_img.size}, mode: {disk_img.mode}")
        disk_arr = np.asarray(disk_img.convert("RGB").resize((224, 224), Image.BILINEAR))

        d = np.abs(disk_arr.astype(np.int16) - verb_arr.astype(np.int16))
        print(f"  224x224 RGB RMSE = {np.sqrt((d.astype(np.float64)**2).mean()):.2f}, max = {d.max()}")

        ka = predict_top3(model, disk_arr)
        kv = predict_top3(model, verb_arr)
        fmt = lambda k: ", ".join(f"P{c + 1}={p * 100:.1f}%" for c, p in k)
        print(f"  disk     top3: {fmt(ka)}")
        print(f"  verbatim top3: {fmt(kv)}")


if __name__ == "__main__":
    main()
