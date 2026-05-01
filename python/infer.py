"""End-to-end inference test: training-equivalent vs Jetson-equivalent paths.

Set TERRA_WEIGHTS env to override the default v2 .pth.
Set TERRA_JET_LUT env to override the LUT npy path (see render_lut.py).

Path A (reference — what the model was trained on):
    processed/Pn/cwt_image_*.png
        -> PIL.convert('RGB').resize(224, BILINEAR)
        -> ImageNet normalize -> MobileNetV3-Large -> top-1

Path C (full Jetson pipeline):
    raw/Pn/Pn_*.mat
        -> extract.apply_frozen_gmm(gmm_params.npz) -> 1500-sample footsteps
        -> render_lut.cwt_to_rgb_direct(224x224)
        -> ImageNet normalize -> MobileNetV3-Large -> top-1

If Path A acc ~= Path C acc per person, the LUT + frozen-GMM pipeline produces
inputs the model treats the same as its training distribution.

Class index convention (from notebook): folders are named "0".."99" and sorted
by int(name), so class_idx == person_id - 1. Verified: cwt_image_{pid-1}_*.png.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from extract import load_person_raw
from stream import load_default_gmm, raw_to_inputs


WEIGHTS_DEFAULT = "E:/Terra/weights/mobilenet_v3_large_v2_best.pth"  # v2
WEIGHTS = os.environ.get("TERRA_WEIGHTS", WEIGHTS_DEFAULT)
DATA = Path("E:/Terra/data")
PERSONS = [("P1", 1), ("P14", 14), ("P50", 50), ("P100", 100)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"unknown arch: {arch}")


def load_model() -> tuple[nn.Module, int]:
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_large")  # v1 ckpt has no 'arch'
    nc = int(ckpt["num_classes"])
    m = build_model(arch, nc)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval().to(DEVICE)
    return m, nc


@torch.no_grad()
def predict_batch(model: nn.Module, imgs_uint8: np.ndarray, batch: int = 64) -> np.ndarray:
    """imgs_uint8: (N, 224, 224, 3) uint8 RGB -> (N,) int64 top-1 class idx."""
    N = len(imgs_uint8)
    preds = np.empty(N, dtype=np.int64)
    for i in range(0, N, batch):
        chunk = imgs_uint8[i : i + batch]
        # NHWC uint8 -> NCHW float [0,1] -> normalize
        t = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        t = NORMALIZE(t)
        logits = model(t)
        preds[i : i + batch] = logits.argmax(dim=1).cpu().numpy()
    return preds


def path_a_inputs(person: str, pid: int, max_n: int) -> np.ndarray:
    """Load processed PNGs the way training did. Returns (N, 224, 224, 3) uint8."""
    folder = DATA / f"processed/{person}"
    files = sorted(folder.glob("cwt_image_*.png"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    files = files[:max_n]
    out = np.empty((len(files), 224, 224, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)
        out[i] = np.asarray(img)
    return out


def path_c_inputs(person: str, max_n: int, gmm) -> tuple[np.ndarray, int]:
    """Full Jetson pipeline from raw .mat. Returns (inputs, total_extracted)."""
    geo = load_person_raw(person, n_files=4)
    _, inputs = raw_to_inputs(geo, gmm)
    total = len(inputs)
    return inputs[:max_n], total


def accuracy(preds: np.ndarray, target_class: int) -> float:
    if len(preds) == 0:
        return float("nan")
    return float((preds == target_class).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=200,
                    help="footsteps per person per path (default 200)")
    ap.add_argument("--skip-c", action="store_true",
                    help="skip Path C (slow CWT extraction) for quick A-only check")
    args = ap.parse_args()

    print("=" * 78)
    print("E2E inference: training path (A) vs Jetson path (C)")
    print(f"  device={DEVICE}, max_n={args.max_n}")
    print("=" * 78)

    print(f"\nloading model from {WEIGHTS} ...")
    model, nc = load_model()
    print(f"  num_classes={nc}")

    gmm = None if args.skip_c else load_default_gmm()

    print(f"\n{'person':<7} {'pid':>4} {'tgt_idx':>8}   "
          f"{'A: n':>5} {'A: top1':>8}   "
          f"{'C: n':>5} {'C/total':>10} {'C: top1':>8}")
    print("-" * 78)

    summary = []
    for person, pid in PERSONS:
        target_idx = pid - 1  # folders 0..99 sorted by int

        # Path A
        a_inputs = path_a_inputs(person, pid, args.max_n)
        t0 = time.time()
        a_preds = predict_batch(model, a_inputs)
        ta = time.time() - t0
        a_acc = accuracy(a_preds, target_idx)

        # Path C
        if args.skip_c:
            c_n = c_total = 0
            c_acc = float("nan")
            tc = 0.0
        else:
            t0 = time.time()
            c_inputs, c_total = path_c_inputs(person, args.max_n, gmm)
            c_preds = predict_batch(model, c_inputs)
            tc = time.time() - t0
            c_n = len(c_inputs)
            c_acc = accuracy(c_preds, target_idx)

        print(f"{person:<7} {pid:>4} {target_idx:>8}   "
              f"{len(a_preds):>5} {a_acc * 100:>7.2f}%   "
              f"{c_n:>5} {c_total:>10} {c_acc * 100:>7.2f}%   "
              f"(A {ta:.1f}s, C {tc:.1f}s)")

        # confusion: top-3 wrong predictions
        if a_acc < 1.0:
            wrong = a_preds[a_preds != target_idx]
            if len(wrong):
                vals, counts = np.unique(wrong, return_counts=True)
                topk = np.argsort(-counts)[:3]
                miscl = ", ".join(f"P{vals[i] + 1}({counts[i]})" for i in topk)
                print(f"        Path A misclassified as: {miscl}")
        if not args.skip_c and c_acc < 1.0:
            wrong = c_preds[c_preds != target_idx]
            if len(wrong):
                vals, counts = np.unique(wrong, return_counts=True)
                topk = np.argsort(-counts)[:3]
                miscl = ", ".join(f"P{vals[i] + 1}({counts[i]})" for i in topk)
                print(f"        Path C misclassified as: {miscl}")

        summary.append((person, a_acc, c_acc))

    print()
    a_mean = np.nanmean([s[1] for s in summary])
    c_mean = np.nanmean([s[2] for s in summary])
    print(f"  mean Path A acc = {a_mean * 100:.2f}%")
    if not args.skip_c:
        print(f"  mean Path C acc = {c_mean * 100:.2f}%")
        print(f"  delta (A - C)   = {(a_mean - c_mean) * 100:+.2f} pp")
        print()
        print("  acceptance: |delta| < 5pp -> Jetson pipeline equivalent to training")


if __name__ == "__main__":
    main()
