"""Test: is raw/P1/ actually P1 data, or mislabeled (e.g. P2)?

For each of N random extracted footsteps from raw/P1/, find the best-matching
row across the ENTIRE a1.mat (not just label==1). Tally the labels of best
matches. If most match label==2, the raw P1 folder is mislabeled as P2.

Same test on P14 as control — should match label==14.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

import extract as ex


DATA = Path("E:/Terra/data")


def best_match_labels(extracted: np.ndarray, a1: np.ndarray, a1_labels: np.ndarray,
                      n_samples: int = 100, seed: int = 0) -> np.ndarray:
    """For n_samples random rows of `extracted`, return the label of best
    matching row across all of a1."""
    rng = np.random.default_rng(seed)
    if len(extracted) > n_samples:
        idx = rng.choice(len(extracted), n_samples, replace=False)
        E = extracted[idx]
    else:
        E = extracted
    Ez = (E - E.mean(axis=1, keepdims=True)) / (E.std(axis=1, keepdims=True) + 1e-12)
    A = a1
    Az = (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-12)
    # corr matrix (n_samples × N) — could be heavy. n_samples=100, N=144371,
    # 1500 cols → 100 × 144K matmul = 21G ops. ~20s on CPU.
    C = (Ez @ Az.T) / Ez.shape[1]
    best_a1 = C.argmax(axis=1)
    return a1_labels[best_a1], C.max(axis=1)


def main():
    print("loading a1.mat...")
    a1_full = scipy.io.loadmat(str(DATA / "interim/a1.mat"))["footstep_feat"]
    a1_signals = a1_full[:, :-1]
    a1_labels = a1_full[:, -1].astype(int)
    print(f"  a1: {a1_full.shape}, labels {a1_labels.min()}..{a1_labels.max()}")

    # Use the frozen GMM (P14 trained, gmm_params.npz)
    gmm_params = ex.load_gmm_npz("E:/Terra/python/gmm_params.npz")

    for person, expected_pid in [("P1", 1), ("P14", 14), ("P50", 50), ("P100", 100)]:
        geo = ex.load_person_raw(person, n_files=4)
        footsteps = ex.apply_frozen_gmm(geo, gmm_params)
        print(f"\n{person}: extracted {len(footsteps)} footsteps from raw")

        labels_matched, corrs = best_match_labels(footsteps, a1_signals, a1_labels,
                                                   n_samples=100, seed=0)
        unique, counts = np.unique(labels_matched, return_counts=True)
        order = np.argsort(-counts)
        print(f"  best-match label distribution (top 5):")
        for j in order[:5]:
            mark = "  <-- expected" if unique[j] == expected_pid else ""
            print(f"    label P{unique[j]:<3d}: {counts[j]:>3d} / 100  "
                  f"(median corr = {np.median(corrs[labels_matched == unique[j]]):.3f}){mark}")
        n_correct = int((labels_matched == expected_pid).sum())
        verdict = "OK" if n_correct >= 60 else "MISLABEL SUSPECTED"
        print(f"  -> matched expected P{expected_pid}: {n_correct}/100  [{verdict}]")


if __name__ == "__main__":
    main()
