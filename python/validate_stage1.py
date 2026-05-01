"""Stage 1 validation: Python port (extract.py) vs original a1.mat.

Stage 1a — a1.mat ground-truth sanity check
Stage 1b — full pipeline: GMM train on P14 first 100s → apply to {P1, P14, P50, P100}
           → compare extracted footsteps with a1.mat rows.

Acceptance (per CLAUDE.md):
  - count match within ±10%
  - matched waveform corr > 0.99 for the bulk
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.io

import extract as ex


DATA_ROOT = Path("E:/Terra/data")
PROCESSED_ROOT = DATA_ROOT / "processed"
INTERIM = DATA_ROOT / "interim/a1.mat"
RAW_ROOT = DATA_ROOT / "raw"

PERSONS = ["P1", "P14", "P50", "P100"]      # subset on disk
GMM_TRAIN_PERSON = "P14"                     # paper's choice
GMM_SEED = 0


def stage_1a():
    print("=" * 70)
    print("STAGE 1a — a1.mat ground-truth sanity check")
    print("=" * 70)
    d = scipy.io.loadmat(str(INTERIM))
    ff = d["footstep_feat"]
    print(f"  footstep_feat: shape={ff.shape}, dtype={ff.dtype}")
    labels = ff[:, -1].astype(int)
    print(f"  unique labels: {labels.min()}..{labels.max()}  "
          f"(n={len(np.unique(labels))})")

    print("\n  per-person counts vs processed PNG counts:")
    for p in PERSONS:
        pid = int(p[1:])
        a1_count = int((labels == pid).sum())
        png_count = len(list((PROCESSED_ROOT / p).glob("*.png")))
        ok = "OK" if a1_count == png_count else "MISMATCH"
        print(f"    {p}: a1.mat={a1_count}, PNG={png_count}  [{ok}]")

    # First footstep of P1
    p1_rows = ff[labels == 1]
    sig0 = p1_rows[0, :-1]
    print(f"\n  P1 first footstep stats: "
          f"min={sig0.min():.4f}, max={sig0.max():.4f}, "
          f"std={sig0.std():.4f}, last20 zero? {np.all(sig0[-20:] == 0)}")
    print()


def stage_1b():
    print("=" * 70)
    print(f"STAGE 1b — port pipeline, GMM trained on {GMM_TRAIN_PERSON} first 100s")
    print("=" * 70)

    # --- Train GMM ---
    t0 = time.time()
    print(f"\n  loading raw {GMM_TRAIN_PERSON}...")
    geo_train = ex.load_person_raw(GMM_TRAIN_PERSON, str(RAW_ROOT), n_files=4)
    print(f"    geo_data length: {len(geo_train)} samples "
          f"({len(geo_train)/8000:.1f} sec)")

    print(f"  training GMM on first 100s of {GMM_TRAIN_PERSON} (seed={GMM_SEED})...")
    gmm, train_feats = ex.train_gmm_on_first_seconds(
        geo_train, n_seconds=100, fs=8000, seed=GMM_SEED
    )
    dets = [float(np.linalg.det(gmm.cov[k])) for k in range(2)]
    event_cluster = int(np.argmax(dets))
    print(f"    EM iters: {gmm.iters}, converged: {gmm.converged}")
    print(f"    π = {gmm.phi}")
    print(f"    |Σ₀|={dets[0]:.3e}, |Σ₁|={dets[1]:.3e}  "
          f"→ event cluster = {event_cluster}")
    print(f"    train feature stats: shape={train_feats.shape}, "
          f"NaN={np.isnan(train_feats).any()}")
    print(f"  GMM train time: {time.time()-t0:.1f}s")

    # --- Load a1.mat once ---
    print("\n  loading a1.mat...")
    a1 = scipy.io.loadmat(str(INTERIM))["footstep_feat"]
    a1_labels = a1[:, -1].astype(int)

    # --- Apply per person ---
    print(f"\n{'Person':<8} {'a1 #':>8} {'ours #':>8} {'Δ%':>7} "
          f"{'med corr':>10} {'p95 corr':>10} {'p<0.99':>8}")
    print("-" * 70)
    results = {}
    for p in PERSONS:
        t1 = time.time()
        pid = int(p[1:])
        if p == GMM_TRAIN_PERSON:
            geo = geo_train
        else:
            geo = ex.load_person_raw(p, str(RAW_ROOT), n_files=4)

        ours = ex.apply_to_person(geo, gmm, fs=8000, threshold=0.90)
        gt = a1[a1_labels == pid][:, :-1]   # (n_gt, 1500)

        n_ours = len(ours)
        n_gt = len(gt)
        delta_pct = 100.0 * (n_ours - n_gt) / n_gt if n_gt > 0 else 0.0

        # Match best-correlated row in gt for each ours (or vice versa)
        # Use the smaller side as anchor to keep cost bounded.
        corrs = best_match_correlations(ours, gt)
        med_c = np.median(corrs) if len(corrs) else float("nan")
        p95_c = np.percentile(corrs, 5) if len(corrs) else float("nan")  # 5th percentile
        n_below = int((corrs < 0.99).sum())

        results[p] = dict(n_ours=n_ours, n_gt=n_gt, corrs=corrs)
        print(f"{p:<8} {n_gt:>8d} {n_ours:>8d} {delta_pct:>+6.1f}% "
              f"{med_c:>10.4f} {p95_c:>10.4f} {n_below:>8d}  "
              f"({time.time()-t1:.1f}s)")

    # Save raw correlations for later inspection
    np.savez_compressed(
        "E:/Terra/python/stage1b_results.npz",
        **{f"{p}_corrs": results[p]["corrs"] for p in PERSONS},
        **{f"{p}_n_ours": np.array(results[p]["n_ours"]) for p in PERSONS},
        **{f"{p}_n_gt": np.array(results[p]["n_gt"]) for p in PERSONS},
    )
    print("\n  saved correlations → python/stage1b_results.npz")

    # Acceptance summary
    print("\n  ACCEPTANCE:")
    for p in PERSONS:
        r = results[p]
        delta_ok = abs(r["n_ours"] - r["n_gt"]) <= 0.10 * r["n_gt"]
        corrs = r["corrs"]
        corr_ok = (np.median(corrs) > 0.99) if len(corrs) else False
        flag = "PASS" if (delta_ok and corr_ok) else "REVIEW"
        print(f"    {p}: count_ok={delta_ok}, median_corr_ok={corr_ok}  → {flag}")


def best_match_correlations(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row in A, find the best Pearson corr against any row in B.

    Returns array of length min(|A|, |B|) using the smaller side as anchor.
    Truncates rows to common length and zero-pads if needed.
    """
    if len(A) == 0 or len(B) == 0:
        return np.array([])
    # Anchor on the smaller set
    if len(A) > len(B):
        A, B = B, A
    Az = (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-12)
    Bz = (B - B.mean(axis=1, keepdims=True)) / (B.std(axis=1, keepdims=True) + 1e-12)
    # cross-correlation matrix (|A| × |B|), normalized
    C = (Az @ Bz.T) / Az.shape[1]
    # best B for each A
    return C.max(axis=1)


if __name__ == "__main__":
    stage_1a()
    stage_1b()
