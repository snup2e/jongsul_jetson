"""live_monitor.py — Real-time geophone monitor (standalone tool).

본 프로젝트의 record_session.py / web_server.py 와 독립. 같은 STM32 포트를
점유하므로 동시에 실행하지는 못 함. 용도:

  1) 브라우저에서 raw 파형 실시간 보기 (마지막 5초, mV)
  2) 동시에 .npz 로 저장 (record_session.py 스키마 호환)
  3) 끝나면 --analyze 로 사후 진단, 또는 Claude 에게 .npz 던지기

Usage:
  # 모니터 + 저장 (Ctrl-C 까지)
  python3 live_monitor.py
  # pid 라벨 + 30초 자동 종료
  python3 live_monitor.py --pid presanity --duration 30
  # 저장 끄고 모니터만
  python3 live_monitor.py --no-save
  # 저장된 파일 분석 (서버 안 띄움)
  python3 live_monitor.py --analyze data/live/me/me_live_20260519_2030.npz

Open browser (Jetson 의 경우):
  http://snup2-desktop:8050/
"""
import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stm32_source import STM32Source, CALIBRATED_SCALE  # noqa: E402

ADC_FS_LSB = 8388608           # ADS1256 24-bit signed full scale
DISPLAY_DECIM = 15             # 15 kHz raw -> 1 kHz visualization
DISPLAY_WINDOW_S = 5.0
PLOT_BUFFER_MAX = int(DISPLAY_WINDOW_S * 1500)  # generous headroom

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


# ============================================================
# Shared state (producer thread <-> Flask SSE thread)
# ============================================================
class MonitorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.plot_ring = deque(maxlen=PLOT_BUFFER_MAX)   # (t_s, v_mV)
        self.plot_seq = 0
        self.raw_chunks = []                              # raw int32 at fs_in
        self.stats = {
            "elapsed": 0.0,
            "rms_mv": 0.0,
            "peak_mv": 0.0,
            "headroom_pct": 100.0,
            "frames_ok": 0,
            "frames_bad": 0,
            "seq_drops": 0,
            "resets": 0,
            "sps": 0.0,
        }
        self.t_start = None
        self.stop_flag = False
        self.source = None

STATE = MonitorState()


# ============================================================
# Producer: STM32 -> shared buffer
# ============================================================
def producer_loop(save_enabled, duration):
    src = STATE.source
    deadline = STATE.t_start + duration if duration and duration > 0 else None

    win_sq = 0.0
    win_n = 0
    win_peak = 0
    next_stat = STATE.t_start + 0.5
    total_n = 0
    total_seen = 0  # cumulative samples for global decimation phase

    try:
        for samples in src.chunks_raw():
            now = time.time()
            n = len(samples)
            if n == 0:
                if STATE.stop_flag:
                    break
                continue

            if save_enabled:
                STATE.raw_chunks.append(samples.astype(np.int32))

            total_n += n
            win_sq += float(np.dot(samples, samples))
            win_n += n
            local_peak = int(np.max(np.abs(samples)))
            if local_peak > win_peak:
                win_peak = local_peak

            # Globally consistent decimation across chunk boundaries.
            start = (-total_seen) % DISPLAY_DECIM
            picked = samples[start::DISPLAY_DECIM]
            total_seen += n

            if len(picked):
                chunk_t_end = now - STATE.t_start
                chunk_dur = n / 15000.0
                # ts evenly spaced over the chunk's time span
                ts_picked = np.linspace(
                    chunk_t_end - chunk_dur, chunk_t_end, len(picked), endpoint=False
                ) + (chunk_dur / len(picked) / 2.0)
                vals_mv = picked.astype(np.float64) * CALIBRATED_SCALE * 1000.0
                with STATE.lock:
                    for t, v in zip(ts_picked, vals_mv):
                        STATE.plot_ring.append((float(t), float(v)))
                    STATE.plot_seq += len(picked)

            if now >= next_stat:
                rms_counts = (win_sq / max(win_n, 1)) ** 0.5
                st = src.stats()
                elapsed = now - STATE.t_start
                with STATE.lock:
                    STATE.stats.update(
                        elapsed=elapsed,
                        rms_mv=rms_counts * CALIBRATED_SCALE * 1000.0,
                        peak_mv=win_peak * CALIBRATED_SCALE * 1000.0,
                        headroom_pct=(1.0 - win_peak / ADC_FS_LSB) * 100.0,
                        frames_ok=st["frames_ok"],
                        frames_bad=st["frames_bad_crc"] + st["frames_bad_n"],
                        seq_drops=st["seq_drops"],
                        resets=st["resets"],
                        sps=total_n / max(elapsed, 1e-9),
                    )
                win_sq = 0.0
                win_n = 0
                win_peak = 0
                next_stat += 0.5

            if STATE.stop_flag:
                src.stop()
                break
            if deadline is not None and now >= deadline:
                src.stop()
                break
    except Exception as e:
        print("[producer error] {}".format(e), file=sys.stderr)
    finally:
        STATE.stop_flag = True


# ============================================================
# Flask app
# ============================================================
def build_app():
    from flask import Flask, Response, send_from_directory, jsonify, stream_with_context

    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(WEB_DIR, "live_monitor.html")

    @app.route("/state")
    def state_json():
        with STATE.lock:
            return jsonify(dict(STATE.stats))

    @app.route("/stream")
    def stream():
        @stream_with_context
        def gen():
            # Initial snapshot: all currently buffered samples
            with STATE.lock:
                init = list(STATE.plot_ring)
                last_seq = STATE.plot_seq
                stats_snapshot = dict(STATE.stats)
            yield "data: " + json.dumps(
                {"snapshot": init, "stats": stats_snapshot}
            ) + "\n\n"

            while not STATE.stop_flag:
                time.sleep(0.1)  # 10 Hz push
                with STATE.lock:
                    new_count = STATE.plot_seq - last_seq
                    if new_count <= 0:
                        new_samples = []
                    elif new_count >= len(STATE.plot_ring):
                        new_samples = list(STATE.plot_ring)
                    else:
                        # take rightmost new_count items
                        new_samples = list(STATE.plot_ring)[-new_count:]
                    last_seq = STATE.plot_seq
                    stats_snapshot = dict(STATE.stats)
                yield "data: " + json.dumps(
                    {"samples": new_samples, "stats": stats_snapshot}
                ) + "\n\n"
            yield "data: " + json.dumps({"done": True}) + "\n\n"

        return Response(gen(), mimetype="text/event-stream")

    return app


# ============================================================
# Analyze mode (sub command)
# ============================================================
def analyze(path):
    """Print diagnostic from a saved .npz. Used as first-pass feedback
    before handing the file to Claude for deeper inspection."""
    import scipy.signal as ss

    data = np.load(path, allow_pickle=False)
    raw = data["raw_int32"]
    fs_in = int(data["fs_in"])
    scale = float(data["scale_v_per_lsb"])
    pid = str(data["pid"]) if "pid" in data.files else "?"
    duration = (
        float(data["duration_s"]) if "duration_s" in data.files
        else len(raw) / max(fs_in, 1)
    )

    print()
    print("=== {} ===".format(path))
    print("pid              : {}".format(pid))
    print("duration         : {:.1f} s".format(duration))
    print("samples (raw)    : {} @ {} sps target -> actual {:.1f} sps".format(
        len(raw), fs_in, len(raw) / max(duration, 1e-9)))
    if "frames_ok" in data.files:
        ok = int(data["frames_ok"])
        bad = int(data["frames_bad"]) if "frames_bad" in data.files else 0
        drops = int(data["seq_drops"]) if "seq_drops" in data.files else 0
        resets = int(data["resets"]) if "resets" in data.files else 0
        print("frames           : ok={} bad={} drops={} resets={}".format(
            ok, bad, drops, resets))

    sig_v = raw.astype(np.float64) * scale
    rms_mv = float(np.sqrt(np.mean(sig_v ** 2))) * 1000.0
    peak_mv = float(np.max(np.abs(sig_v))) * 1000.0
    peak_counts = int(np.max(np.abs(raw)))
    headroom = (1.0 - peak_counts / ADC_FS_LSB) * 100.0
    print("RMS              : {:.3f} mV".format(rms_mv))
    print("peak             : {:.2f} mV  ({:.1f}% FS, {:.1f}% headroom)".format(
        peak_mv, 100 - headroom, headroom))

    # Footstep detection — production envelope detector
    try:
        from ads1256_source import StreamingPolyphase
        from extract import apply_envelope_detector
        rs = StreamingPolyphase(p=8, q=15)
        sig_8k = rs.feed(sig_v)
        if len(sig_8k) >= 8000:
            crops = apply_envelope_detector(sig_8k)
            print("footsteps        : {} (envelope detector, k_mad=8.0)".format(len(crops)))
            if len(crops):
                per_min = len(crops) / max(duration / 60.0, 1e-9)
                print("  rate           : {:.1f} per minute".format(per_min))
        else:
            sig_8k = sig_v[::2]  # too short, just to allow FFT below
            print("footsteps        : skipped (< 1 s of data)")
    except Exception as e:
        sig_8k = sig_v[::2]
        print("footsteps        : skipped ({})".format(e))

    # FFT — Welch's, look for AC mains pickup
    nperseg = min(8192, len(sig_8k))
    fs_fft = 8000 if len(sig_8k) > len(sig_v) // 3 else fs_in
    if nperseg < 64:
        print("[FFT skipped — too few samples]")
        return 0
    f, Pxx = ss.welch(sig_8k, fs=fs_fft, nperseg=nperseg)
    band = (f >= 5) & (f <= 500)
    if band.any():
        f_b = f[band]
        P_b = Pxx[band]
        top_idx = np.argsort(P_b)[-5:][::-1]
        print("top spectral peaks (5-500 Hz):")
        for i in top_idx:
            print("  {:6.1f} Hz   PSD={:.3e} V^2/Hz".format(f_b[i], P_b[i]))

    # AC mains pickup check
    bg_band = (f >= 30) & (f <= 80) & (np.abs(f - 50) > 2) & (np.abs(f - 60) > 2)
    if bg_band.any():
        bg = float(np.median(Pxx[bg_band]))
        idx_50 = int(np.argmin(np.abs(f - 50.0)))
        idx_60 = int(np.argmin(np.abs(f - 60.0)))
        ratio_50 = Pxx[idx_50] / max(bg, 1e-30)
        ratio_60 = Pxx[idx_60] / max(bg, 1e-30)
        print("AC mains check   : 50 Hz / bg = {:.1f}x   60 Hz / bg = {:.1f}x".format(
            ratio_50, ratio_60))
        if ratio_50 > 5 or ratio_60 > 5:
            print("  ! AC mains pickup detected — shield cable or check ground loop.")
    return 0


# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", default=None,
                    help="path to saved .npz; analyze and exit (no live server)")
    ap.add_argument("--pid", default="monitor", help="label for save file path")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--port", default=None,
                    help="STM32 VCP path (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fs-in", type=int, default=15000)
    ap.add_argument("--no-save", action="store_true",
                    help="monitor only, do not save .npz")
    ap.add_argument("--outdir", default="data/live")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (0.0.0.0 = LAN; 127.0.0.1 = local only)")
    ap.add_argument("--http-port", type=int, default=8050)
    ap.add_argument("--display-host", default=None,
                    help="hostname shown in printed URL (default: socket.gethostname())")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.analyze:
        return analyze(args.analyze)

    src = STM32Source(
        port=args.port,
        baud=args.baud,
        fs_in=args.fs_in,
        fs_out=8000,
        chunk_size=64,
        scale=1.0,
        timeout=0.5,
    )
    src.open()
    print("[connect]", src.info())
    STATE.source = src
    STATE.t_start = time.time()

    save_enabled = not args.no_save
    display_host = args.display_host or socket.gethostname()

    app = build_app()
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host=args.host, port=args.http_port,
            debug=False, use_reloader=False, threaded=True,
        ),
        daemon=True,
    )
    flask_thread.start()
    time.sleep(0.3)
    print("[serve] http://{}:{}/  (or http://localhost:{}/)".format(
        display_host, args.http_port, args.http_port))
    if args.duration > 0:
        print("[record] auto-stop after {:.0f}s — Ctrl-C to stop early".format(args.duration))
    else:
        print("[record] Ctrl-C to stop")

    def handle_sig(*_):
        STATE.stop_flag = True
        try:
            src.stop()
        except Exception:
            pass
    signal.signal(signal.SIGINT, handle_sig)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, handle_sig)
        except Exception:
            pass

    try:
        producer_loop(save_enabled, args.duration)
    finally:
        try:
            src.close()
        except Exception:
            pass

    elapsed = time.time() - STATE.t_start
    final_stats = src.stats()
    print()
    print("[result] elapsed {:.1f} s   actual {:.1f} sps".format(
        elapsed,
        sum(len(c) for c in STATE.raw_chunks) / max(elapsed, 1e-9) if STATE.raw_chunks else 0.0,
    ))
    print("  frames_ok={} bad_crc={} bad_n={} drops={} resets={}".format(
        final_stats["frames_ok"],
        final_stats["frames_bad_crc"],
        final_stats["frames_bad_n"],
        final_stats["seq_drops"],
        final_stats["resets"],
    ))

    if save_enabled and STATE.raw_chunks:
        raw = np.concatenate(STATE.raw_chunks).astype(np.int32)
        os.makedirs(os.path.join(args.outdir, args.pid), exist_ok=True)
        fname = "{}_live_{}.npz".format(
            args.pid, datetime.now().strftime("%Y%m%d_%H%M%S"))
        fpath = os.path.join(args.outdir, args.pid, fname)
        np.savez(
            fpath,
            raw_int32=raw,
            fs_in=args.fs_in,
            scale_v_per_lsb=CALIBRATED_SCALE,
            pid=args.pid,
            session=0,
            duration_s=elapsed,
            ts_start=STATE.t_start,
            frames_ok=final_stats["frames_ok"],
            frames_bad=final_stats["frames_bad_crc"] + final_stats["frames_bad_n"],
            seq_drops=final_stats["seq_drops"],
            resets=final_stats["resets"],
            n_footsteps=-1,
        )
        sz_mb = os.path.getsize(fpath) / 1e6
        print("  saved -> {}  ({:.1f} MB)".format(fpath, sz_mb))
        print()
        print("  quick analyze: python live_monitor.py --analyze {}".format(fpath))
        print("  or ask Claude: \"{} 분석해줘\"".format(fpath))
    elif save_enabled:
        print("  [no samples captured — nothing saved]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
