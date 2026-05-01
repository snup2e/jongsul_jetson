"""Verify gmm_params.npz reproduces the freshly-trained pipeline byte-exactly.

Path A: train_gmm_on_first_seconds(P14, seed=0) -> apply_to_person(P14)
Path B: load_gmm_npz() -> apply_frozen_gmm(P14)

If A and B produce different footsteps, our .npz is missing some state.
Expected: identical shape, max|diff| == 0.
"""
from __future__ import annotations

import numpy as np

from extract import (
    apply_frozen_gmm,
    apply_to_person,
    load_gmm_npz,
    load_person_raw,
    train_gmm_on_first_seconds,
)


def main():
    print("=" * 72)
    print("Verify gmm_params.npz <-> fresh training")
    print("=" * 72)

    # Path A: fresh training (matches Stage 1b setup)
    print("\nPath A: fresh GMM(P14, seed=0) -> apply on P14")
    geo_p14 = load_person_raw("P14", n_files=4)
    gmm, _ = train_gmm_on_first_seconds(geo_p14, n_seconds=100, seed=0)
    a = apply_to_person(geo_p14, gmm)
    print(f"  footsteps shape: {a.shape}")

    # Path B: load .npz
    print("\nPath B: load gmm_params.npz -> apply on P14")
    params = load_gmm_npz("E:/Terra/python/gmm_params.npz")
    b = apply_frozen_gmm(geo_p14, params)
    print(f"  footsteps shape: {b.shape}")

    # Compare
    print("\nComparison:")
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: {a.shape} vs {b.shape}")
        return
    diff = np.abs(a - b)
    print(f"  max|diff| = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}")
    if diff.max() == 0:
        print(f"\n  PASS: byte-equal. .npz captures all state.")
    elif diff.max() < 1e-10:
        print(f"\n  PASS: numerically identical (float roundoff).")
    else:
        print(f"\n  FAIL: investigate which params drift.")


if __name__ == "__main__":
    main()
