"""Diagnose Stage 1b under-extraction + P1 corr drop.

A) Try GMM trained on P1 (matching main.m hardcoding) vs P14 (paper claim).
B) Try multiple GMM seeds.
C) Inspect the actual waveform difference for one P1 mismatch.
"""
from __future__ import annotations

import time
import numpy as np
import scipy.io
import extract as ex


a1 = scipy.io.loadmat("E:/Terra/data/interim/a1.mat")["footstep_feat"]
a1_labels = a1[:, -1].astype(int)


def evaluate(gmm, persons=("P1", "P14", "P50", "P100"), threshold=0.90):
    rows = []
    for p in persons:
        pid = int(p[1:])
        geo = ex.load_person_raw(p, "E:/Terra/data/raw", n_files=4)
        ours = ex.apply_to_person(geo, gmm, fs=8000, threshold=threshold)
        gt = a1[a1_labels == pid][:, :-1]
        if len(ours) == 0 or len(gt) == 0:
            rows.append((p, len(gt), len(ours), 0.0, 0.0))
            continue
        # best-match corr
        smaller, bigger = (ours, gt) if len(ours) <= len(gt) else (gt, ours)
        Az = (smaller - smaller.mean(1, keepdims=True)) / (smaller.std(1, keepdims=True) + 1e-12)
        Bz = (bigger - bigger.mean(1, keepdims=True)) / (bigger.std(1, keepdims=True) + 1e-12)
        C = (Az @ Bz.T) / Az.shape[1]
        corrs = C.max(axis=1)
        rows.append((p, len(gt), len(ours), float(np.median(corrs)),
                     float(np.percentile(corrs, 5))))
    return rows


def print_table(label, rows):
    print(f"\n  {label}")
    print(f"  {'Pn':<6} {'gt':>6} {'ours':>6} {'Δ%':>7} {'med corr':>10} {'p5 corr':>10}")
    for p, n_gt, n_ours, med, p5 in rows:
        d = 100.0 * (n_ours - n_gt) / max(n_gt, 1)
        print(f"  {p:<6} {n_gt:>6d} {n_ours:>6d} {d:>+6.1f}%  {med:>10.4f} {p5:>10.4f}")


print("=" * 70)
print("Diagnostic A: train on P1 (matching main.m) vs P14 (matching paper)")
print("=" * 70)

for train_p in ["P1", "P14"]:
    print(f"\n  Training GMM on {train_p} first 100s, seed=0 ...")
    t0 = time.time()
    geo_t = ex.load_person_raw(train_p, "E:/Terra/data/raw", n_files=4)
    gmm, _ = ex.train_gmm_on_first_seconds(geo_t, n_seconds=100, fs=8000, seed=0)
    dets = [float(np.linalg.det(gmm.cov[k])) for k in range(2)]
    print(f"  EM iters: {gmm.iters}, π={gmm.phi}, |Σ| ratio={dets[0]/dets[1]:.2e}")
    print(f"  GMM time: {time.time()-t0:.1f}s")
    rows = evaluate(gmm)
    print_table(f"trained on {train_p}", rows)


print("\n" + "=" * 70)
print("Diagnostic B: GMM seed sensitivity (train on P14)")
print("=" * 70)
geo_t = ex.load_person_raw("P14", "E:/Terra/data/raw", n_files=4)
for seed in [1, 2, 3, 7, 42]:
    print(f"\n  seed={seed}")
    gmm, _ = ex.train_gmm_on_first_seconds(geo_t, n_seconds=100, fs=8000, seed=seed)
    dets = [float(np.linalg.det(gmm.cov[k])) for k in range(2)]
    print(f"    iters={gmm.iters}, π={gmm.phi}, |Σ|₀/|Σ|₁={dets[0]/dets[1]:.2e}")
    rows = evaluate(gmm)
    print_table(f"P14 seed={seed}", rows)


print("\n" + "=" * 70)
print("Diagnostic C: relax posterior threshold")
print("=" * 70)
geo_t = ex.load_person_raw("P14", "E:/Terra/data/raw", n_files=4)
gmm, _ = ex.train_gmm_on_first_seconds(geo_t, n_seconds=100, fs=8000, seed=0)
for thr in [0.50, 0.70, 0.85, 0.90, 0.95]:
    rows = evaluate(gmm, threshold=thr)
    print_table(f"threshold={thr}", rows)
