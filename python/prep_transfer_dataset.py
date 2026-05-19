"""D.3: prepare transfer-learning dataset for family deployment.

Inputs:
  - data/recordings/<pid>/*.npz from record_session.py
        raw_int32 (15 kHz), scale_v_per_lsb, ... (see record_session.py header)
  - data/raw/P<N>/P<N>_*.mat   VIBeID raw geo_data (8 kHz, float64) for unknown class

Pipeline per source (matches Jetson inference exactly — CLAUDE.md 추론 파이프라인):
  raw_int32 * scale → V (15 kHz)
    → StreamingPolyphase 8/15 → 8 kHz (recordings only; VIBeID already 8 kHz)
    → apply_envelope_detector (Hilbert + MAD threshold + peak align)
    → footsteps (M, 1500) float64
    → cwt_fast.cwt + render_lut.cwt_to_rgb_direct → (M, 224, 224, 3) uint8

Default labels (5-class):
  0..N-1 = family pids (--pids me mom dad sis → 0=me, 1=mom, 2=dad, 3=sis)
  N      = unknown (VIBeID + noise recordings merged)

With --separate-noise: 6-class, last label = noise (still separate from unknown).

Output: data/transfer/family_v1.npz
  X_train (T, 224, 224, 3) uint8
  y_train (T,) int32
  X_val   (V, 224, 224, 3) uint8
  y_val   (V,) int32
  pid_map (dict-as-object_array)  e.g. {0:"me", 1:"mom", ..., 4:"unknown"}
  meta    (dict)  counts per class, source summary, seed, args

Usage:
  python python/prep_transfer_dataset.py --pids me mom dad sis
  python python/prep_transfer_dataset.py --pids me mom dad sis --separate-noise
  python python/prep_transfer_dataset.py --pids me mom dad sis --unknown-source P14 P50
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase  # noqa: E402
from cwt_fast import cwt as fast_cwt  # noqa: E402
from extract import apply_envelope_detector  # noqa: E402
from render_lut import cwt_to_rgb_direct  # noqa: E402


def footsteps_to_inputs(footsteps):
    """(M, 1500) float -> (M, 224, 224, 3) uint8 via CWT + LUT."""
    M = len(footsteps)
    out = np.empty((M, 224, 224, 3), dtype=np.uint8)
    for i in range(M):
        out[i] = cwt_to_rgb_direct(fast_cwt(footsteps[i]), (224, 224))
    return out


def load_recording_npz(path):
    """Load a record_session.py .npz, return 8 kHz float64 signal in volts."""
    d = np.load(path)
    raw = d["raw_int32"].astype(np.float64)
    scale = float(d["scale_v_per_lsb"])
    fs_in = int(d["fs_in"])
    sig_v_15k = raw * scale
    if fs_in == 8000:
        return sig_v_15k
    rs = StreamingPolyphase(p=8, q=fs_in // 1000)  # supports 15→8 only currently
    if fs_in != 15000:
        raise ValueError("only 15 kHz recordings supported (got {})".format(fs_in))
    rs = StreamingPolyphase(p=8, q=15)
    sig_8k = rs.feed(sig_v_15k)
    return sig_8k


def collect_pid_footsteps(pid, recordings_dir, label):
    """Find all <pid>_s*.npz under recordings_dir/<pid>, run envelope, return
    (footsteps (M, 1500), labels (M,))."""
    pdir = os.path.join(recordings_dir, pid)
    if not os.path.isdir(pdir):
        print("  [skip] no directory {} — no sessions recorded yet".format(pdir))
        return np.empty((0, 1500), dtype=np.float64), np.empty(0, dtype=np.int32)
    files = sorted(glob.glob(os.path.join(pdir, "{}_s*.npz".format(pid))))
    if not files:
        print("  [skip] no sessions in {}".format(pdir))
        return np.empty((0, 1500), dtype=np.float64), np.empty(0, dtype=np.int32)

    per_session = []
    for f in files:
        try:
            sig = load_recording_npz(f)
        except Exception as e:
            print("  [warn] {} failed to load: {}".format(os.path.basename(f), e))
            continue
        fps = apply_envelope_detector(sig)
        per_session.append(fps)
        print("  {}: {:.0f}s -> {} footsteps".format(
            os.path.basename(f), len(sig) / 8000, len(fps)))

    if not per_session:
        return np.empty((0, 1500), dtype=np.float64), np.empty(0, dtype=np.int32)

    all_fps = np.concatenate(per_session, axis=0)
    labels = np.full(len(all_fps), label, dtype=np.int32)
    return all_fps, labels


def collect_vibeid_footsteps(raw_dir, pid_folder, label):
    """Load all data/raw/<pid_folder>/<pid_folder>_*.mat, apply envelope."""
    pdir = os.path.join(raw_dir, pid_folder)
    files = sorted(glob.glob(os.path.join(pdir, "{}_*.mat".format(pid_folder))))
    if not files:
        print("  [skip] no .mat in {}".format(pdir))
        return np.empty((0, 1500), dtype=np.float64), np.empty(0, dtype=np.int32)

    per_file = []
    for f in files:
        d = scipy.io.loadmat(f)
        if "geo_data" not in d:
            print("  [warn] {} missing geo_data".format(os.path.basename(f)))
            continue
        geo = d["geo_data"].ravel().astype(np.float64)
        fps = apply_envelope_detector(geo)
        per_file.append(fps)
        print("  {}: {:.0f}s -> {} footsteps".format(
            os.path.basename(f), len(geo) / 8000, len(fps)))

    if not per_file:
        return np.empty((0, 1500), dtype=np.float64), np.empty(0, dtype=np.int32)

    all_fps = np.concatenate(per_file, axis=0)
    labels = np.full(len(all_fps), label, dtype=np.int32)
    return all_fps, labels


def stratified_split(X, y, train_frac, seed):
    """Per-class shuffle + first `train_frac` train, rest val. Returns
    (X_train, y_train, X_val, y_val)."""
    rng = np.random.default_rng(seed)
    Xt, yt, Xv, yv = [], [], [], []
    for cls in sorted(np.unique(y)):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        Xt.append(X[idx[:n_train]])
        yt.append(y[idx[:n_train]])
        Xv.append(X[idx[n_train:]])
        yv.append(y[idx[n_train:]])
    return (
        np.concatenate(Xt) if Xt else np.empty((0, 224, 224, 3), np.uint8),
        np.concatenate(yt) if yt else np.empty(0, np.int32),
        np.concatenate(Xv) if Xv else np.empty((0, 224, 224, 3), np.uint8),
        np.concatenate(yv) if yv else np.empty(0, np.int32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="+", required=True,
                    help="family person IDs (in label order, 0..N-1)")
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--raw-vibeid", default="data/raw")
    ap.add_argument("--unknown-source", nargs="*", default=["P14"],
                    help="VIBeID pid folders to use as unknown class (default: P14)")
    ap.add_argument("--noise-pid", default="noise",
                    help="recording pid used for non-footstep impulses")
    ap.add_argument("--separate-noise", action="store_true",
                    help="give noise its own label (else merged into unknown)")
    ap.add_argument("--out", default="data/transfer/family_v1.npz")
    ap.add_argument("--split", type=float, default=0.8,
                    help="train fraction (per-class stratified)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if any(p == args.noise_pid for p in args.pids):
        raise ValueError("--pids must not contain --noise-pid ({})".format(args.noise_pid))

    # Build label map
    n_family = len(args.pids)
    unknown_label = n_family
    noise_label = n_family + 1 if args.separate_noise else unknown_label
    pid_map = {i: pid for i, pid in enumerate(args.pids)}
    pid_map[unknown_label] = "unknown"
    if args.separate_noise:
        pid_map[noise_label] = "noise"

    print("[plan] labels:")
    for k in sorted(pid_map):
        print("  {} -> {}".format(k, pid_map[k]))
    print()

    # ----- Family recordings -----
    fp_pool = []
    y_pool = []
    counts = {}
    for label, pid in enumerate(args.pids):
        print("[family] pid={} label={}".format(pid, label))
        fps, ys = collect_pid_footsteps(pid, args.recordings, label)
        fp_pool.append(fps)
        y_pool.append(ys)
        counts[pid] = len(fps)

    # ----- Noise recordings -----
    print("[noise] pid={} label={} ({}separate)".format(
        args.noise_pid, noise_label, "" if args.separate_noise else "merged-to-unknown; "))
    fps_noise, ys_noise = collect_pid_footsteps(args.noise_pid, args.recordings, noise_label)
    fp_pool.append(fps_noise)
    y_pool.append(ys_noise)
    counts["noise"] = len(fps_noise)

    # ----- VIBeID unknown -----
    fps_unk = []
    ys_unk = []
    for vbpid in args.unknown_source:
        print("[unknown] VIBeID {} label={}".format(vbpid, unknown_label))
        fps, ys = collect_vibeid_footsteps(args.raw_vibeid, vbpid, unknown_label)
        fps_unk.append(fps)
        ys_unk.append(ys)
    if fps_unk:
        fp_pool.append(np.concatenate(fps_unk))
        y_pool.append(np.concatenate(ys_unk))
        counts["vibeid_unknown"] = sum(len(f) for f in fps_unk)

    footsteps = np.concatenate(fp_pool, axis=0) if fp_pool else np.empty((0, 1500))
    labels = np.concatenate(y_pool) if y_pool else np.empty(0, np.int32)
    if len(footsteps) == 0:
        print("\n[abort] no footsteps collected. Check --recordings and --raw-vibeid paths.")
        sys.exit(1)

    print("\n[render] CWT + LUT for {} footsteps...".format(len(footsteps)))
    t0 = time.time()
    X = footsteps_to_inputs(footsteps)
    dt = time.time() - t0
    print("  rendered {} images in {:.1f}s ({:.1f} ms/each)".format(
        len(X), dt, dt * 1000 / max(len(X), 1)))

    print("\n[split] stratified {:.0%} train / {:.0%} val (seed={})".format(
        args.split, 1 - args.split, args.seed))
    X_train, y_train, X_val, y_val = stratified_split(X, labels, args.split, args.seed)

    print("\n[summary] per-class counts:")
    print("  {:<22s} {:>8s} {:>8s} {:>8s}".format("label", "train", "val", "total"))
    for k in sorted(pid_map):
        name = pid_map[k]
        tr = int((y_train == k).sum())
        va = int((y_val == k).sum())
        print("  {:<22s} {:>8d} {:>8d} {:>8d}".format(
            "{} ({})".format(k, name), tr, va, tr + va))
    print("  {:<22s} {:>8d} {:>8d} {:>8d}".format("TOTAL", len(y_train), len(y_val), len(labels)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pid_map_array = np.array(list(pid_map.items()), dtype=object)
    np.savez_compressed(
        args.out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        pid_map=pid_map_array,
        meta=np.array({
            "created_ts": time.time(),
            "n_family": n_family,
            "unknown_label": unknown_label,
            "noise_label": noise_label if args.separate_noise else None,
            "counts": counts,
            "seed": args.seed,
            "split": args.split,
            "args": vars(args),
        }, dtype=object),
    )
    sz_mb = os.path.getsize(args.out) / 1e6
    print("\nsaved -> {}  ({:.1f} MB)".format(args.out, sz_mb))


if __name__ == "__main__":
    main()
