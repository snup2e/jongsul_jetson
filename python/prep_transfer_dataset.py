"""D.3: prepare transfer-learning dataset for family deployment.

Inputs:
  - data/recordings/<pid>/*.npz from record_session.py
        raw_int32 (15 kHz), scale_v_per_lsb, ... (see record_session.py header)
  - data/raw/P<N>/P<N>_*.mat   VIBeID raw geo_data (8 kHz, float64) for unknown class

Pipeline per source (matches Jetson inference exactly — CLAUDE.md 추론 파이프라인):
  raw_int32 * scale → V (15 kHz)
    → StreamingPolyphase 8/15 → 8 kHz (recordings only; VIBeID already 8 kHz)
    → apply_envelope_detector (Hilbert + MAD threshold + peak align)
    → footsteps (M, 1500) float64  [temporally ordered within each recording]
    → cwt_fast.cwt + render_lut.cwt_to_rgb_direct → (M, 224, 224, 3) uint8

Default labels (5-class):
  0..N-1 = family pids (--pids me mom dad sis → 0=me, 1=mom, 2=dad, 3=sis)
  N      = unknown (VIBeID + noise recordings merged)

With --separate-noise: 6-class, last label = noise (still separate from unknown).

Split (train / val / test) — TEMPORAL, PER RECORDING SESSION (no leakage):
  각 녹음 세션(파일)의 발걸음을 시간순으로 train_frac / val_frac / 나머지(test) 로 분할.
  세션=페이스(평상시/느린/빠른)라 세 페이스가 train·val·test 에 모두 포함(같은 분포).
  블록 경계에 `--gap` 발걸음을 버려 인접-발걸음 leakage 까지 제거.
  → 무작위 per-footstep split(구버전)과 달리, test 는 학습에 안 쓴 시간 구간 = 정직한 일반화 추정.
  test 는 학습/모델선택에 절대 사용 금지 (val 만 모델·에폭 선택용).

Output: data/transfer/family_v1.npz
  X_train/y_train, X_val/y_val, X_test/y_test  (uint8 (n,224,224,3) / int32 (n,))
  pid_map (dict-as-object_array)  e.g. {0:"me", 1:"mom", ..., 4:"unknown"}
  meta    (dict)  counts per class, split fracs, gap, args

Usage:
  python python/prep_transfer_dataset.py --pids P1 P2 P3 P4
  python python/prep_transfer_dataset.py --pids me mom dad sis --train-frac 0.7 --val-frac 0.15 --gap 3
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


def collect_pid_sessions(pid, recordings_dir):
    """Find all <pid>_s*.npz under recordings_dir/<pid>, run envelope per session.
    Returns a LIST of (Mi, 1500) footstep arrays — one per recording session,
    footsteps temporally ordered within each. (Session identity preserved so the
    split can be grouped/temporal instead of random per-footstep.)"""
    pdir = os.path.join(recordings_dir, pid)
    if not os.path.isdir(pdir):
        print("  [skip] no directory {} — no sessions recorded yet".format(pdir))
        return []
    files = sorted(glob.glob(os.path.join(pdir, "{}_s*.npz".format(pid))))
    files = [f for f in files if not f.endswith(".bak")]
    if not files:
        print("  [skip] no sessions in {}".format(pdir))
        return []
    per_session = []
    for f in files:
        try:
            sig = load_recording_npz(f)
        except Exception as e:
            print("  [warn] {} failed to load: {}".format(os.path.basename(f), e))
            continue
        fps = apply_envelope_detector(sig)
        per_session.append(np.asarray(fps))
        print("  {}: {:.0f}s -> {} footsteps".format(
            os.path.basename(f), len(sig) / 8000, len(fps)))
    return per_session


def collect_vibeid_sessions(raw_dir, pid_folder):
    """Load all data/raw/<pid_folder>/<pid_folder>_*.mat, apply envelope per file.
    Returns a LIST of (Mi, 1500) footstep arrays — one per .mat file (treated as a
    'session' for the temporal split)."""
    pdir = os.path.join(raw_dir, pid_folder)
    files = sorted(glob.glob(os.path.join(pdir, "{}_*.mat".format(pid_folder))))
    if not files:
        print("  [skip] no .mat in {}".format(pdir))
        return []
    per_file = []
    for f in files:
        d = scipy.io.loadmat(f)
        if "geo_data" not in d:
            print("  [warn] {} missing geo_data".format(os.path.basename(f)))
            continue
        geo = d["geo_data"].ravel().astype(np.float64)
        fps = apply_envelope_detector(geo)
        per_file.append(np.asarray(fps))
        print("  {}: {:.0f}s -> {} footsteps".format(
            os.path.basename(f), len(geo) / 8000, len(fps)))
    return per_file


def temporal_split_sessions(sessions, train_frac, val_frac, gap):
    """sessions: list of (Mi,1500) temporally-ordered footstep arrays (one per recording).
    Split EACH session chronologically: first `train_frac` -> train, next `val_frac` -> val,
    remainder -> test. `gap` footsteps are dropped before val and before test (at the block
    boundaries) so no train footstep is temporally adjacent to a val/test footstep.
    Returns (train, val, test) each (n, 1500)."""
    tr, va, te = [], [], []
    for fps in sessions:
        n = len(fps)
        if n == 0:
            continue
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        n_tr = max(0, min(n_tr, n))
        n_va = max(0, min(n_va, n - n_tr))
        g = gap if n > 5 * max(gap, 1) else 0   # skip gap on tiny sessions
        tr.append(fps[: max(0, n_tr - g)])
        va.append(fps[n_tr: max(n_tr, n_tr + n_va - g)])
        te.append(fps[n_tr + n_va:])

    def cat(lst):
        lst = [a for a in lst if len(a)]
        return np.concatenate(lst) if lst else np.empty((0, 1500))
    return cat(tr), cat(va), cat(te)


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
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="per-session chronological train fraction")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="per-session chronological val fraction (test = 1 - train - val)")
    ap.add_argument("--gap", type=int, default=3,
                    help="footsteps dropped at each block boundary (anti-leakage)")
    ap.add_argument("--seed", type=int, default=0, help="(kept for meta; split is deterministic)")
    args = ap.parse_args()

    if any(p == args.noise_pid for p in args.pids):
        raise ValueError("--pids must not contain --noise-pid ({})".format(args.noise_pid))
    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac <= 0:
        raise ValueError("train-frac + val-frac must be < 1 (got test_frac={:.2f})".format(test_frac))

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
    print("[plan] temporal per-session split: train {:.0%} / val {:.0%} / test {:.0%}  (gap={})".format(
        args.train_frac, args.val_frac, test_frac, args.gap))
    print()

    # (label, [per-session footstep arrays]) for every class source
    class_sessions = []
    counts = {}

    for label, pid in enumerate(args.pids):
        print("[family] pid={} label={}".format(pid, label))
        sess = collect_pid_sessions(pid, args.recordings)
        class_sessions.append((label, sess))
        counts[pid] = int(sum(len(s) for s in sess))

    print("[noise] pid={} label={} ({}separate)".format(
        args.noise_pid, noise_label, "" if args.separate_noise else "merged-to-unknown; "))
    sess_noise = collect_pid_sessions(args.noise_pid, args.recordings)
    class_sessions.append((noise_label, sess_noise))
    counts["noise"] = int(sum(len(s) for s in sess_noise))

    unk_sessions = []
    for vbpid in args.unknown_source:
        print("[unknown] VIBeID {} label={}".format(vbpid, unknown_label))
        unk_sessions += collect_vibeid_sessions(args.raw_vibeid, vbpid)
    if unk_sessions:
        class_sessions.append((unknown_label, unk_sessions))
        counts["vibeid_unknown"] = int(sum(len(s) for s in unk_sessions))

    # ----- temporal split per class, then pool -----
    f_tr, y_tr, f_va, y_va, f_te, y_te = [], [], [], [], [], []
    for label, sess in class_sessions:
        tr, va, te = temporal_split_sessions(sess, args.train_frac, args.val_frac, args.gap)
        for fps, fl, yl in ((tr, f_tr, y_tr), (va, f_va, y_va), (te, f_te, y_te)):
            if len(fps):
                fl.append(fps)
                yl.append(np.full(len(fps), label, dtype=np.int32))

    def catf(lst):
        return np.concatenate(lst) if lst else np.empty((0, 1500))

    def caty(lst):
        return np.concatenate(lst) if lst else np.empty(0, dtype=np.int32)

    foot_tr, labels_tr = catf(f_tr), caty(y_tr)
    foot_va, labels_va = catf(f_va), caty(y_va)
    foot_te, labels_te = catf(f_te), caty(y_te)
    n_total = len(foot_tr) + len(foot_va) + len(foot_te)
    if n_total == 0:
        print("\n[abort] no footsteps collected. Check --recordings and --raw-vibeid paths.")
        sys.exit(1)

    print("\n[render] CWT + LUT for {} footsteps (train {} / val {} / test {})...".format(
        n_total, len(foot_tr), len(foot_va), len(foot_te)))
    t0 = time.time()
    X_train = footsteps_to_inputs(foot_tr)
    X_val = footsteps_to_inputs(foot_va)
    X_test = footsteps_to_inputs(foot_te)
    y_train, y_val, y_test = labels_tr, labels_va, labels_te
    dt = time.time() - t0
    print("  rendered {} images in {:.1f}s ({:.1f} ms/each)".format(
        n_total, dt, dt * 1000 / max(n_total, 1)))

    print("\n[summary] per-class counts (train / val / test):")
    print("  {:<22s} {:>7s} {:>7s} {:>7s} {:>7s}".format("label", "train", "val", "test", "total"))
    for k in sorted(pid_map):
        tr = int((y_train == k).sum()); va = int((y_val == k).sum()); te = int((y_test == k).sum())
        print("  {:<22s} {:>7d} {:>7d} {:>7d} {:>7d}".format(
            "{} ({})".format(k, pid_map[k]), tr, va, te, tr + va + te))
    print("  {:<22s} {:>7d} {:>7d} {:>7d} {:>7d}".format(
        "TOTAL", len(y_train), len(y_val), len(y_test), n_total))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pid_map_array = np.array(list(pid_map.items()), dtype=object)
    np.savez_compressed(
        args.out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        pid_map=pid_map_array,
        meta=np.array({
            "created_ts": time.time(),
            "n_family": n_family,
            "unknown_label": unknown_label,
            "noise_label": noise_label if args.separate_noise else None,
            "counts": counts,
            "split_type": "temporal_per_session",
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": test_frac,
            "gap": args.gap,
            "seed": args.seed,
            "args": vars(args),
        }, dtype=object),
    )
    sz_mb = os.path.getsize(args.out) / 1e6
    print("\nsaved -> {}  ({:.1f} MB)".format(args.out, sz_mb))
    print("⚠ test 세트는 학습/모델선택에 사용 금지 — 최종 보고용으로만 1회 평가.")


if __name__ == "__main__":
    main()
