"""Train the footstep-detection GMM once and freeze parameters to .npz.

Per main.m + Stage 1b validation: GMM is trained on the first 100 seconds of
P14 (geo_data smoothed with span=5) using seed=0. This produces a 2-component
2-D... wait, 7-D GMM. The cluster with larger |Sigma| is the "event" cluster.

The exported .npz holds everything inference needs:
    mu                (2, 7)  float64
    cov               (2, 7, 7) float64
    phi               (2,)    float64
    event_cluster     int64   index into mu/cov/phi
    threshold         float64 (default 0.90)
    iters, converged  bookkeeping

Inference loads this once and skips the EM step entirely.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from extract import (
    load_person_raw,
    train_gmm_on_first_seconds,
)


OUT = Path("E:/Terra/python/gmm_params.npz")
TRAIN_PERSON = "P14"   # Per paper / CLAUDE.md
SEED = 0
N_SECONDS = 100
N_FILES = 4            # NewDatasetCreation2.m default; same as Stage 1b
THRESHOLD = 0.90       # main.m posterior threshold


def main():
    print("=" * 72)
    print(f"GMM export: train on {TRAIN_PERSON}, first {N_SECONDS}s, seed={SEED}")
    print("=" * 72)

    print(f"\nloading raw {TRAIN_PERSON} ({N_FILES} files)...")
    geo = load_person_raw(TRAIN_PERSON, n_files=N_FILES)
    print(f"  geo_data shape={geo.shape}, {len(geo) / 8000:.1f}s")

    print(f"\ntraining GMM on first {N_SECONDS}s...")
    gmm, feats = train_gmm_on_first_seconds(geo, n_seconds=N_SECONDS, seed=SEED)
    print(f"  EM: iters={gmm.iters}, converged={gmm.converged}")
    print(f"  features shape={feats.shape}")

    dets = np.array([np.linalg.det(gmm.cov[k]) for k in range(2)])
    event_cluster = int(np.argmax(dets))
    print(f"  |cov[0]|={dets[0]:.3e}, |cov[1]|={dets[1]:.3e}")
    print(f"  event_cluster = {event_cluster} (larger |Sigma|)")
    print(f"  phi = {gmm.phi}")
    print(f"  mu[event] = {gmm.mu[event_cluster]}")
    print(f"  mu[noise] = {gmm.mu[1 - event_cluster]}")

    np.savez(
        OUT,
        mu=gmm.mu,
        cov=gmm.cov,
        phi=gmm.phi,
        event_cluster=np.int64(event_cluster),
        threshold=np.float64(THRESHOLD),
        iters=np.int64(gmm.iters),
        converged=np.bool_(gmm.converged),
        train_person=np.array(TRAIN_PERSON),
        train_seconds=np.int64(N_SECONDS),
        train_seed=np.int64(SEED),
    )
    print(f"\nsaved to {OUT}")
    print(f"  size = {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
