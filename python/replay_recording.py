"""replay_recording.py — offline replay of collected recordings through the
full inference pipeline (envelope/GMM detector -> CWT -> LUT -> ONNX -> Voter).

Reproduces the live demo offline, deterministically, on the family recordings.
Lets us answer "does it work?" without the STM32/Jetson in the loop, and
compare the training-time detector (envelope) vs the live-demo detector (GMM).

True label mapping (from data/transfer/family_v1.npz pid_map):
    0=김건형  1=김범수  2=성재용  3=박지훈  4=unknown

Pipeline parity:
  * envelope  -> matches the TRAINING crops (prep_transfer_dataset.py)
  * gmm       -> matches the LIVE DEMO (web_server.py: smoother span=5 + SlidingDetector)
  * Voter K=5/M=3/conf=0.50 -> matches the demo (web_server argparse defaults)
  * ONNX FP32 (onnxruntime) ~ TRT FP16 (Jetson); tiny numeric diff, irrelevant to top-1.

Usage:
  python replay_recording.py                       # all 4 family, s1, 15s @ 30s, both detectors
  python replay_recording.py --session 2 --dur 20 --offset 60
  python replay_recording.py --detector gmm        # only the live-demo path
  python replay_recording.py --pids 김범수 성재용
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stm32_source import CALIBRATED_SCALE  # noqa: E402
from resample_for_ads1256 import StreamingPolyphase  # noqa: E402
from jetson_realtime import StreamingSmoother, SlidingDetector, Voter  # noqa: E402
from extract import apply_envelope_detector, load_gmm_npz  # noqa: E402
from cwt_fast import cwt as fast_cwt  # noqa: E402
from render_lut import cwt_to_rgb_direct  # noqa: E402

LABELS = {0: "김건형", 1: "김범수", 2: "성재용", 3: "박지훈", 4: "unknown"}
NAME2LABEL = {v: k for k, v in LABELS.items()}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def to_input(img_hwc):
    x = img_hwc.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def load_slice(npz_path, offset_s, dur_s, fs_in=15000):
    d = np.load(npz_path, allow_pickle=True)
    raw = d["raw_int32"].astype(np.float64)
    i0 = int(offset_s * fs_in)
    i1 = int((offset_s + dur_s) * fs_in)
    seg = raw[i0:i1]
    return seg * CALIBRATED_SCALE  # -> volts


def detect_footsteps(sig_v, detector, gmm):
    """sig_v: raw geophone in volts @ 15 kHz -> footsteps (M, 1500) @ 8 kHz."""
    rs = StreamingPolyphase(p=8, q=15)
    sig_8k = rs.feed(sig_v)
    if detector == "envelope":
        return apply_envelope_detector(sig_8k)
    # gmm path = exact live web_server.py producer stages
    sm = StreamingSmoother(span=5)
    smoothed = np.concatenate([sm(sig_8k), sm.flush()])
    det = SlidingDetector(gmm, fs=8000)
    return np.array(det.feed(smoothed) + det.flush())


def run_person(sess, true_label, detector, gmm, session_obj, vote_k, vote_m, vote_conf):
    footsteps = detect_footsteps(sess, detector, gmm)
    M = len(footsteps)
    if M == 0:
        return {"n": 0, "preds": [], "confs": [], "confirms": [], "true": true_label}

    inp_name = session_obj.get_inputs()[0].name
    voter = Voter(K=vote_k, M=vote_m, conf_min=vote_conf)
    preds, confs, confirms = [], [], []
    for i, fp in enumerate(footsteps):
        coeffs = fast_cwt(fp)
        img = cwt_to_rgb_direct(coeffs, (224, 224))
        x = to_input(img)
        logits = session_obj.run(None, {inp_name: x})[0][0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        top1, conf = int(p.argmax()), float(p.max())
        preds.append(top1)
        confs.append(conf)
        for cpid in voter.update(top1, conf):
            confirms.append((i, cpid, conf))
    return {"n": M, "preds": preds, "confs": confs, "confirms": confirms, "true": true_label}


def pick_file(pids_dir, pid, session):
    cands = sorted(glob.glob(os.path.join(pids_dir, pid, "{}_s{}_*.npz".format(pid, session))))
    cands = [c for c in cands if not c.endswith(".bak")]
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--onnx", default="mobilenet_v3_family_v1.onnx")
    ap.add_argument("--pids", nargs="+", default=["김건형", "김범수", "성재용", "박지훈"])
    ap.add_argument("--session", default="1")
    ap.add_argument("--offset", type=float, default=30.0, help="start time in recording (s)")
    ap.add_argument("--dur", type=float, default=15.0, help="window length (s)")
    ap.add_argument("--detector", choices=["envelope", "gmm", "both"], default="both")
    ap.add_argument("--gmm", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmm_params.npz"))
    ap.add_argument("--vote-k", type=int, default=5)
    ap.add_argument("--vote-m", type=int, default=3)
    ap.add_argument("--vote-conf", type=float, default=0.50)
    args = ap.parse_args()

    session_obj = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    gmm = load_gmm_npz(args.gmm)
    detectors = ["envelope", "gmm"] if args.detector == "both" else [args.detector]

    print("=" * 72)
    print("REPLAY  onnx={}  window=[{:.0f}s, +{:.0f}s]  voter K={}/M={}/conf={:.2f}".format(
        os.path.basename(args.onnx), args.offset, args.dur,
        args.vote_k, args.vote_m, args.vote_conf))
    print("=" * 72)

    summary = {det: {"correct_steps": 0, "total_steps": 0, "confirmed_right": 0, "n": 0}
               for det in detectors}

    for pid in args.pids:
        true_label = NAME2LABEL.get(pid, 4)
        fpath = pick_file(args.recordings, pid, args.session)
        if not fpath:
            print("\n[{}] no s{} recording found — skip".format(pid, args.session))
            continue
        sess = load_slice(fpath, args.offset, args.dur)
        print("\n### {} (true label {})  file={}".format(pid, true_label, os.path.basename(fpath)))

        for det in detectors:
            r = run_person(sess, true_label, det, gmm, session_obj,
                           args.vote_k, args.vote_m, args.vote_conf)
            if r["n"] == 0:
                print("  [{:8s}] 0 footsteps detected in window".format(det))
                continue
            preds = np.array(r["preds"])
            confs = np.array(r["confs"])
            correct = int((preds == true_label).sum())
            maj = int(np.bincount(preds, minlength=5).argmax())
            # voter outcome: was the TRUE person confirmed, and when?
            true_confirm = next((c for c in r["confirms"] if c[1] == true_label), None)
            wrong_confirms = [c for c in r["confirms"] if c[1] != true_label]
            conf_str = "step {} (conf {:.0%})".format(true_confirm[0], true_confirm[2]) \
                if true_confirm else "NEVER"
            print("  [{:8s}] {:2d} steps | per-step acc {:2d}/{:2d} ({:3.0f}%) | "
                  "majority={}({}) | mean conf {:.0%}".format(
                      det, r["n"], correct, r["n"], 100 * correct / r["n"],
                      maj, LABELS[maj], confs.mean()))
            print("             voter -> {} confirmed: {}{}".format(
                LABELS[true_label], conf_str,
                "  [!] also wrong-confirmed: " +
                ", ".join("{}@{}".format(LABELS[c[1]], c[0]) for c in wrong_confirms)
                if wrong_confirms else ""))

            s = summary[det]
            s["correct_steps"] += correct
            s["total_steps"] += r["n"]
            s["confirmed_right"] += 1 if (true_confirm and not wrong_confirms) else 0
            s["n"] += 1

    print("\n" + "=" * 72)
    print("SUMMARY")
    for det in detectors:
        s = summary[det]
        if s["total_steps"] == 0:
            continue
        print("  {:8s}: per-step {}/{} ({:.0f}%) | clean-confirm {}/{} people".format(
            det, s["correct_steps"], s["total_steps"],
            100 * s["correct_steps"] / s["total_steps"],
            s["confirmed_right"], s["n"]))
    print("=" * 72)


if __name__ == "__main__":
    main()
