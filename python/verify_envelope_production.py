"""Verify production apply_envelope_detector matches the experiment script
(same crops -> same classifier accuracy as experiment_envelope_classifier_p14_v2.py).
"""
import os
import sys

import numpy as np
import torch
from torch import nn
import torchvision.models as tvm
import torchvision.transforms as T
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import apply_envelope_detector, apply_frozen_gmm, load_gmm_npz
from stream import footsteps_to_inputs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights", "mobilenet_v3_large_v2_best.pth")
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_model():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    nc = int(ckpt["num_classes"])
    m = tvm.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, nc


@torch.no_grad()
def predict(model, imgs):
    if len(imgs) == 0:
        return np.empty(0, dtype=np.int64)
    out = np.empty(len(imgs), dtype=np.int64)
    for i in range(0, len(imgs), 32):
        c = imgs[i:i+32]
        t = torch.from_numpy(c).permute(0, 3, 1, 2).float() / 255.0
        t = NORMALIZE(t)
        out[i:i+32] = model(t).argmax(dim=1).numpy()
    return out


def main():
    gmm = load_gmm_npz(os.path.join(os.path.dirname(__file__), "gmm_params.npz"))
    print("loading model...")
    model, nc = load_model()
    target = 13

    print("\n{:<6} {:>10} {:>10} {:>10} {:>10}".format(
        "P14_x", "GMM #", "GMM acc", "env #", "env acc"))
    print("-" * 60)

    tot_g_n = tot_g_h = tot_e_n = tot_e_h = 0
    for idx in (1, 2, 3, 4, 5, 6):
        sig = loadmat(os.path.join(ROOT, "data", "raw", "P14",
                                   "P14_{}.mat".format(idx)))["geo_data"].ravel().astype(np.float64)
        # production paths
        fs_gmm = apply_frozen_gmm(sig, gmm)
        fs_env = apply_envelope_detector(sig)
        imgs_g = footsteps_to_inputs(fs_gmm) if len(fs_gmm) else np.empty((0, 224, 224, 3), np.uint8)
        imgs_e = footsteps_to_inputs(fs_env) if len(fs_env) else np.empty((0, 224, 224, 3), np.uint8)
        pred_g = predict(model, imgs_g)
        pred_e = predict(model, imgs_e)
        g_n, g_h = len(pred_g), int((pred_g == target).sum())
        e_n, e_h = len(pred_e), int((pred_e == target).sum())
        tot_g_n += g_n; tot_g_h += g_h
        tot_e_n += e_n; tot_e_h += e_h
        print("{:<6} {:>10} {:>9.1%} {:>10} {:>9.1%}".format(
            idx, g_n, g_h / max(g_n, 1), e_n, e_h / max(e_n, 1)))
    print("-" * 60)
    print("TOTAL   {:>10} {:>9.2%} {:>10} {:>9.2%}".format(
        tot_g_n, tot_g_h / max(tot_g_n, 1), tot_e_n, tot_e_h / max(tot_e_n, 1)))
    print("\nExpect (from previous experiment): GMM=83.38%, env=84.89%")


if __name__ == "__main__":
    main()
