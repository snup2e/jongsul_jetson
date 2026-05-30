"""record_session.py — D.2: capture STM32 stream to .npz with live stats + footstep sanity.

Output: <outdir>/<pid>/<pid>_s<session>_<YYYYMMDD_HHMM>.npz
  raw_int32        : (N,) int32, raw ADC counts at fs_in (15 kHz native)
  fs_in            : int
  scale_v_per_lsb  : float (CALIBRATED_SCALE at recording time, informational)
  pid              : str
  session          : int
  duration_s       : float (actual elapsed)
  ts_start         : float (time.time() at first frame)
  frames_ok        : int
  frames_bad       : int (CRC + N errors)
  seq_drops        : int
  n_footsteps      : int (post-hoc detector; -1 if skipped)

Usage:
  python3 record_session.py --pid me --session 1 --duration 300
  python3 record_session.py --pid me --session 2 --duration 60 --port /dev/ttyACM0

Live: every 1 s prints elapsed/remaining + RMS_mV + peak_mV + headroom% + frames_ok + drops.
On Ctrl-C: stops cleanly, saves what was captured (marked as STOPPED EARLY).
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stm32_source import STM32Source, CALIBRATED_SCALE  # noqa: E402
from ads1256_source import StreamingPolyphase  # noqa: E402

ADC_FS_LSB = 8388608  # 2^23, ADS1256 24-bit signed full scale


def _fmt_eta(s):
    m, sec = divmod(int(max(0.0, s)), 60)
    return "{:d}:{:02d}".format(m, sec)


def _detect_footsteps(raw_int32, fs_in, gmm_path):
    """Resample to 8 kHz, smooth (span=5), run SlidingDetector, return count."""
    # Lazy imports — these pull torch-free deps but take ~1 s to import.
    from extract import load_gmm_npz
    from jetson_realtime import StreamingSmoother, SlidingDetector

    sig_v = raw_int32.astype(np.float64) * CALIBRATED_SCALE
    rs = StreamingPolyphase(p=8, q=15)
    sig_8k = rs.feed(sig_v)
    if len(sig_8k) < 8000:
        return -1

    gmm = load_gmm_npz(gmm_path)
    sm = StreamingSmoother(span=5)
    sm_out = sm(sig_8k)
    sm_tail = sm.flush()
    smoothed = np.concatenate([sm_out, sm_tail]) if len(sm_tail) else sm_out

    det = SlidingDetector(gmm, fs=8000)
    fps = det.feed(smoothed) + det.flush()
    return len(fps)


def record(pid, session, duration, port, baud, fs_in, outdir, gmm_path, skip_detect):
    src = STM32Source(
        port=port, baud=baud, fs_in=fs_in, fs_out=8000,
        chunk_size=64, scale=1.0, timeout=1.0,
    )
    src.open()
    print("[connect]", src.info())
    print("[record] pid={} session={} target={:.0f}s outdir={}".format(
        pid, session, duration, outdir))

    buffers = []
    t0 = time.time()
    deadline = t0 + duration
    next_print_t = t0 + 1.0
    sum_sq = 0.0
    n_acc = 0
    peak_acc = 0
    stopped_early = False

    try:
        for samples in src.chunks_raw():
            buffers.append(samples)
            sum_sq += float(np.dot(samples, samples))
            n_acc += len(samples)
            local_peak = int(np.max(np.abs(samples)))
            if local_peak > peak_acc:
                peak_acc = local_peak

            now = time.time()
            if now >= next_print_t:
                rms_counts = (sum_sq / n_acc) ** 0.5 if n_acc else 0.0
                rms_mv = rms_counts * CALIBRATED_SCALE * 1000.0
                peak_mv = peak_acc * CALIBRATED_SCALE * 1000.0
                headroom_pct = (1.0 - peak_acc / ADC_FS_LSB) * 100.0
                elapsed = now - t0
                remaining = deadline - now
                st = src.stats()
                print("[t={:6.1f}s eta={}] RMS={:6.2f}mV peak={:7.2f}mV ({:5.1f}% head) "
                      "ok={} drops={} rst={}".format(
                          elapsed, _fmt_eta(remaining), rms_mv, peak_mv, headroom_pct,
                          st["frames_ok"], st["seq_drops"], st["resets"]))
                sum_sq = 0.0
                n_acc = 0
                peak_acc = 0
                next_print_t += 1.0

            if now >= deadline:
                src.stop()
    except KeyboardInterrupt:
        print("\n[interrupt] stopping early at {:.1f}s".format(time.time() - t0))
        stopped_early = True
        src.stop()
    finally:
        src.close()

    elapsed = time.time() - t0
    if buffers:
        # samples come as float64 (scale=1.0 keeps integer values exact).
        raw = np.concatenate(buffers).astype(np.int32)
    else:
        raw = np.empty(0, dtype=np.int32)
    s = src.stats()

    print()
    print("[result] elapsed {:.1f}s, samples {} ({:.1f} sps actual)".format(
        elapsed, len(raw), len(raw) / max(elapsed, 1e-9)))
    print("  frames_ok={} bad_crc={} bad_n={} seq_drops={} resets={} bytes_dropped={}".format(
        s["frames_ok"], s["frames_bad_crc"], s["frames_bad_n"],
        s["seq_drops"], s["resets"], s["bytes_dropped"]))

    n_fp = -1
    if not skip_detect and len(raw) >= fs_in:
        try:
            n_fp = _detect_footsteps(raw, fs_in, gmm_path)
            print("  footsteps detected (post-hoc): {}".format(n_fp))
        except Exception as e:
            print("  footstep detection skipped: {}".format(e))

    os.makedirs(os.path.join(outdir, pid), exist_ok=True)
    fname = "{}_s{}_{}.npz".format(pid, session, datetime.now().strftime("%Y%m%d_%H%M"))
    fpath = os.path.join(outdir, pid, fname)
    np.savez(
        fpath,
        raw_int32=raw,
        fs_in=fs_in,
        scale_v_per_lsb=CALIBRATED_SCALE,
        pid=pid,
        session=session,
        duration_s=elapsed,
        ts_start=t0,
        frames_ok=s["frames_ok"],
        frames_bad=s["frames_bad_crc"] + s["frames_bad_n"],
        seq_drops=s["seq_drops"],
        resets=s["resets"],
        n_footsteps=n_fp,
    )
    sz_mb = os.path.getsize(fpath) / 1e6
    print("  saved -> {}  ({:.1f} MB)".format(fpath, sz_mb))

    # PASS = 데이터 무결성 직접 검증 (sample count 기준).
    # SEQ drops + resets counter 는 Linux cdc_acm + ST-Link VCP 조합에서
    # false-positive 가 빈번. 매우 짧은 STM32 brownout (1 frame ≈ 4 ms 손실)
    # 도 reset 으로 카운트되지만 sample count 가 맞으면 발걸음 분류 영향 0.
    # 진짜 fail 은 sample count 가 어긋날 때 (sps_ratio 밖) 또는 reset 다발
    # (≥ 1/min) — 후자는 USB power 불안정 신호.
    bad_total = s["frames_bad_crc"] + s["frames_bad_n"]
    err_ratio = bad_total / max(s["frames_ok"], 1)
    actual_sps = len(raw) / max(elapsed, 1e-9)
    sps_ratio = actual_sps / fs_in
    reset_rate_per_min = s["resets"] / max(elapsed / 60.0, 1e-9)
    pass_ = (
        not stopped_early
        and s["frames_ok"] > 100
        and err_ratio < 0.001
        and 0.995 <= sps_ratio <= 1.005    # ±0.5% sample-count 무결
        and reset_rate_per_min <= 1.0       # 분당 1번 이하 reset (=짧은 brownout 허용)
    )
    print()
    if stopped_early:
        print("[STOPPED EARLY]")
    else:
        print("[PASS]" if pass_ else "[FAIL]")
    return 0 if pass_ else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True, help="person ID (e.g., me, mom, dad)")
    ap.add_argument("--session", type=int, required=True, help="session number (1, 2, 3)")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds (default 300 = 5min)")
    ap.add_argument("--port", default=None, help="STM32 VCP path (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fs-in", type=int, default=15000)
    ap.add_argument("--outdir", default="data/recordings")
    ap.add_argument("--gmm", default=os.environ.get("TERRA_GMM", "gmm_params.npz"))
    ap.add_argument("--skip-detect", action="store_true",
                    help="skip post-hoc footstep detection (faster save)")
    args = ap.parse_args()
    sys.exit(record(
        args.pid, args.session, args.duration, args.port, args.baud,
        args.fs_in, args.outdir, args.gmm, args.skip_detect,
    ))


if __name__ == "__main__":
    main()
