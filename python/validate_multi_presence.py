"""validate_multi_presence.py — multi-person synthesis validation (Stage 5e B)

Mixes P14 + P50 raw signals at various amplitude / time-offset combinations
and verifies the multi-presence Voter (Stage 5e) confirms BOTH individuals
when they would walk simultaneously.

Pipeline (offline batch — voter logic identical to streaming jetson_realtime):
    raw A + B (mixed) → apply_frozen_gmm → footsteps
        → CWT (cwt_fast) → LUT → MobileNetV3-Large → top1 + conf
        → multi-presence Voter

Acceptance per case:
    - REQUIRED pids must all be in voter.confirmed
    - No pids OTHER than required ∪ allowed → false-positive

Usage (from python/ directory):
    python validate_multi_presence.py                  # 90s default
    python validate_multi_presence.py --full           # full 5min
    python validate_multi_presence.py --vote-k 8       # parameter sweep
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torchvision import models, transforms

from cwt_fast import cwt as fast_cwt
from extract import apply_frozen_gmm, load_gmm_npz
from jetson_realtime import Voter
from render_lut import cwt_to_rgb_direct


DATA_ROOT = Path("E:/Terra/data/raw")
WEIGHTS = os.environ.get("TERRA_WEIGHTS", "E:/Terra/weights/mobilenet_v3_large_v2_best.pth")
GMM_NPZ = "E:/Terra/python/gmm_params.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

P14_TARGET = 13   # 0-indexed (a1.mat label==14 → folder cwt_image_13_*)
P50_TARGET = 49
P100_TARGET = 99


def load_raw(person: str, fmax: int = 4) -> np.ndarray:
    parts = []
    for i in range(1, fmax + 1):
        p = DATA_ROOT / person / f"{person}_{i}.mat"
        if not p.exists():
            break
        parts.append(scipy.io.loadmat(str(p))["geo_data"].ravel().astype(np.float64))
    return np.concatenate(parts)


def build_model() -> Tuple[nn.Module, int]:
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    nc = int(ckpt["num_classes"])
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval().to(DEVICE), nc


@torch.no_grad()
def predict_footsteps(model: nn.Module, footsteps: np.ndarray, batch: int = 64
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """(M, 1500) → (M,) top1, (M,) conf via CWT+LUT+MobileNetV3."""
    M = len(footsteps)
    if M == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    imgs = np.empty((M, 224, 224, 3), dtype=np.uint8)
    for i in range(M):
        imgs[i] = cwt_to_rgb_direct(fast_cwt(footsteps[i]), (224, 224))
    top1 = np.empty(M, dtype=np.int64)
    conf = np.empty(M, dtype=np.float32)
    for i in range(0, M, batch):
        chunk = imgs[i : i + batch]
        t = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        t = NORMALIZE(t)
        probs = torch.softmax(model(t), dim=1)
        c, p = probs.max(dim=1)
        top1[i : i + batch] = p.cpu().numpy()
        conf[i : i + batch] = c.cpu().numpy()
    return top1, conf


def synthesize(sig_a: np.ndarray, sig_b: np.ndarray, alpha: float, beta: float,
               offset_samples: int = 0) -> np.ndarray:
    """Sample-level superposition (alpha*A + beta*B). Models true temporal
    overlap. NOTE: classifier was trained on single-person CWTs only — summed
    inputs are OOD and produce arbitrary class outputs. Useful only to
    demonstrate the limitation."""
    n = min(len(sig_a), len(sig_b))
    a = sig_a[:n] * alpha
    b = sig_b[:n] * beta
    if offset_samples > 0:
        b = np.concatenate([np.zeros(offset_samples), b[:n - offset_samples]])
    elif offset_samples < 0:
        b = np.concatenate([b[-offset_samples:], np.zeros(-offset_samples)])
    return a + b


def temporal_interleave(sig_a: np.ndarray, sig_b: np.ndarray, block_seconds: float,
                        fs: int = 8000) -> np.ndarray:
    """Splice in alternating time blocks: A[0:T] B[T:2T] A[2T:3T] B[3T:4T] ...
    Models two people walking in turns (each footstep belongs to a single
    person — what most home multi-person scenarios actually look like).
    """
    block = int(block_seconds * fs)
    n = min(len(sig_a), len(sig_b))
    out = np.zeros(n, dtype=np.float64)
    pos = 0
    flip = False
    while pos < n:
        end = min(pos + block, n)
        out[pos:end] = (sig_b if flip else sig_a)[pos:end]
        pos = end
        flip = not flip
    return out


def run_voter(top1: np.ndarray, conf: np.ndarray, K: int, M: int, conf_min: float = 0.5):
    voter = Voter(K=K, M=M, conf_min=conf_min)
    timeline = []
    for i, (t, c) in enumerate(zip(top1.tolist(), conf.tolist())):
        new = voter.update(int(t), float(c))
        for pid in new:
            timeline.append((i + 1, pid))   # 1-indexed footstep
    confirmed = {pid for _, pid in timeline}
    return timeline, confirmed


def report_case(name: str, sig: np.ndarray, model, gmm,
                required: set, allowed: set, K: int, M: int) -> bool:
    footsteps = apply_frozen_gmm(sig, gmm)
    if len(footsteps) == 0:
        print("  {:<28} NO FOOTSTEPS DETECTED (GMM threshold)".format(name))
        return False
    top1, conf = predict_footsteps(model, footsteps)
    timeline, confirmed = run_voter(top1, conf, K=K, M=M)

    vals, counts = np.unique(top1, return_counts=True)
    order = np.argsort(-counts)[:3]
    dist = " ".join("P{}={}".format(int(vals[i]) + 1, int(counts[i])) for i in order)

    miss = required - confirmed
    extras = confirmed - required - allowed
    ok = (not miss) and (not extras)

    p14 = "✓" if P14_TARGET in confirmed else "✗"
    p50 = "✓" if P50_TARGET in confirmed else "✗"
    extra_str = ",".join("P{}".format(p + 1) for p in sorted(extras)) if extras else "—"

    timing = ", ".join("#{}→P{}".format(i, p + 1) for i, p in timeline[:4])
    if len(timeline) > 4:
        timing += " …"

    flag = "PASS" if ok else "FAIL"
    print("  {:<28} fs={:>3} | top3 {:<24} | P14:{}  P50:{}  extras:{:<6} | {}".format(
        name, len(footsteps), dist, p14, p50, extra_str, flag,
    ))
    if timing:
        print("    confirm timing: {}".format(timing))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=8000 * 90,
                    help="trim to first N samples (default 90s)")
    ap.add_argument("--full", action="store_true", help="use full P*_1.mat (5 min)")
    ap.add_argument("--vote-k", type=int, default=10)
    ap.add_argument("--vote-m", type=int, default=3)
    ap.add_argument("--vote-conf", type=float, default=0.50)
    args = ap.parse_args()

    print("loading model + GMM ...")
    model, nc = build_model()
    gmm = load_gmm_npz(GMM_NPZ)

    print("loading raw P14, P50, P100 ...")
    p14 = load_raw("P14")
    p50 = load_raw("P50")
    p100 = load_raw("P100")
    if not args.full:
        p14 = p14[: args.max_samples]
        p50 = p50[: args.max_samples]
        p100 = p100[: args.max_samples]
    print("  P14: {:.1f}s  P50: {:.1f}s  P100: {:.1f}s".format(
        len(p14)/8000, len(p50)/8000, len(p100)/8000))
    print("  device={}, num_classes={}, K={} M={} conf_min={:.2f}".format(
        DEVICE, nc, args.vote_k, args.vote_m, args.vote_conf))

    realistic_cases = [
        # name, signal, required, allowed
        ("solo P14 control",            p14,                                {P14_TARGET},              set()),
        ("solo P50 control",            p50,                                {P50_TARGET},              set()),
        ("solo P100 control",           p100,                               {P100_TARGET},             set()),
        # P14 + P50 (P50 sparse — only longer blocks accumulate ≥M hits)
        ("interleave 45s  P14+P50",     temporal_interleave(p14, p50, 45),  {P14_TARGET, P50_TARGET},  set()),
        # P14 + P100 (both dense, smaller blocks should still confirm)
        ("interleave 30s  P14+P100",    temporal_interleave(p14, p100, 30), {P14_TARGET, P100_TARGET}, set()),
        ("interleave 15s  P14+P100",    temporal_interleave(p14, p100, 15), {P14_TARGET, P100_TARGET}, set()),
    ]
    superposition_cases = [
        # OOD inputs — classifier never saw summed CWTs in training. Reported
        # for transparency, not asserted to PASS. Single-channel hard limit.
        ("(OOD) sum 1:1 offset 0",      synthesize(p14, p50, 1.0, 1.0, 0),    set(), set()),
        ("(OOD) sum 1:0.5 P50 quieter", synthesize(p14, p50, 1.0, 0.5, 0),    set(), set()),
    ]

    print("\n" + "=" * 110)
    print("REALISTIC multi-person scenarios  (K={}, M={}, conf_min={:.2f})".format(
        args.vote_k, args.vote_m, args.vote_conf))
    print("=" * 110)
    n_pass = 0
    for name, sig, req, allow in realistic_cases:
        if report_case(name, sig, model, gmm, req, allow,
                       K=args.vote_k, M=args.vote_m):
            n_pass += 1

    print("\n" + "=" * 110)
    print("OOD signal superposition (fundamental single-channel limitation, INFO ONLY)")
    print("=" * 110)
    for name, sig, req, allow in superposition_cases:
        # Don't count toward pass total — informational
        report_case(name, sig, model, gmm, req, allow,
                    K=args.vote_k, M=args.vote_m)

    print("\n" + "=" * 110)
    print("{}/{} REALISTIC cases PASS".format(n_pass, len(realistic_cases)))
    print("=" * 110)


if __name__ == "__main__":
    main()
