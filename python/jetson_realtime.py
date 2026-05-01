"""Real-time-style streaming inference on Jetson Nano.

Streaming variant of jetson_infer.py --raw:

    StreamSource          (raw 8 kHz sample chunks)
        ↓
    StreamingSmoother     (matlab_smooth(x, span=5), incremental)
        ↓
    SlidingDetector       (2800-sample windows, 40% overlap, frozen GMM,
                           consecutive event windows → 1500-sample footstep)
        ↓
    InferenceWorker       (cwt_fast → render_lut → TRT FP16 → top-1)
        ↓
    Voter                 (last K predictions → confirmed identity)

Only MatFileSource is implemented (replays a .mat as a fake stream).
ADS1256Source plugs in later as a drop-in StreamSource.

Validation:
    python3 jetson_realtime.py --replay sample/P14_1.mat \
        --plan mnv3_v2_fp16.plan --gmm gmm_params.npz

Distribution should match Stage 5b offline (P14 337/361 = 93.4%, top-5
P14/P15/P21/P92/P29). Single-sample deviation is possible from streaming
smoother edge effects in the first 4 samples (< 0.001% of signal).
"""
import argparse
import os
import queue
import threading
import time
from collections import deque
from typing import Iterator, List, Optional, Tuple

import numpy as np
import scipy.io
from scipy import signal as ss

from extract import (
    extract_features,
    gausswin,
    assign_clusters_frozen,
    load_gmm_npz,
)
# cwt_fast / render_lut / jetson_infer are imported lazily inside replay() so
# this module can be unit-tested on machines without TensorRT.


# ============================================================
# Stream sources
# ============================================================

class MatFileSource:
    """Replays a .mat file as a stream of float64 chunks."""

    def __init__(self, path, fs=8000, chunk_size=1024, realtime=False):
        self.path = path
        self.fs = fs
        self.chunk_size = chunk_size
        self.realtime = realtime
        d = scipy.io.loadmat(path)
        self.signal = d["geo_data"].ravel().astype(np.float64)
        self.n_total = len(self.signal)

    def chunks(self):
        # type: () -> Iterator[np.ndarray]
        N = self.n_total
        cs = self.chunk_size
        t0 = time.time()
        for i in range(0, N, cs):
            chunk = self.signal[i : i + cs]
            if self.realtime:
                target = (i + len(chunk)) / self.fs
                slack = target - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
            yield chunk

    def info(self):
        return "{}: {} samples ({:.1f}s)".format(
            self.path, self.n_total, self.n_total / self.fs
        )


# ============================================================
# Streaming smoother — matlab_smooth(x, span=5), incremental
# ============================================================

class StreamingSmoother:
    """matlab_smooth(x, span=5) as a streaming pipeline (bit-exact).

    Outputs are delayed by `span // 2` samples (centered moving average needs
    that lookahead). Total outputs == total inputs after flush().

    First `span // 2` outputs use left-shrinking window (y[0]=x[0],
    y[1]=mean(x[:3]), ...). Last `span // 2` outputs (emitted on flush())
    use right-shrinking window. Matches matlab_smooth byte-exactly.
    """

    def __init__(self, span=5):
        assert span % 2 == 1, "span must be odd"
        self.span = span
        self.half = span // 2
        self.kernel = np.ones(span, dtype=np.float64) / span
        self.buf = np.zeros(0, dtype=np.float64)
        self.buf_offset = 0   # absolute index of buf[0] in stream
        self.consumed = 0     # next absolute output position to emit
        self.total_seen = 0   # total raw samples received

    def __call__(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        self.buf = np.concatenate([self.buf, chunk])
        self.total_seen += len(chunk)
        p_max = self.total_seen - self.half - 1  # last position emittable now
        if p_max < self.consumed:
            return np.array([], dtype=np.float64)

        out_parts = []

        # Left boundary: positions p < half use shrinking window x[0:2p+1].
        # Only triggers on the first few calls.
        while self.consumed < self.half and self.consumed <= p_max:
            p = self.consumed
            h = p  # min(p, half) when p < half
            lo = (p - h) - self.buf_offset
            hi = (p + h + 1) - self.buf_offset
            out_parts.append(np.array([self.buf[lo:hi].mean()]))
            self.consumed += 1

        # Steady state: vectorized convolve over [consumed, p_max].
        if self.consumed <= p_max:
            seg_lo = (self.consumed - self.half) - self.buf_offset
            seg_hi = (p_max + self.half + 1) - self.buf_offset
            seg = self.buf[seg_lo:seg_hi]
            out_parts.append(np.convolve(seg, self.kernel, mode="valid"))
            self.consumed = p_max + 1

        # Trim buffer (keep last `half` samples for next call's leading edge).
        keep_from = max(self.consumed - self.half, 0)
        if keep_from > self.buf_offset:
            drop = keep_from - self.buf_offset
            self.buf = self.buf[drop:]
            self.buf_offset = keep_from

        return np.concatenate(out_parts) if out_parts else np.array([], dtype=np.float64)

    def flush(self):
        """Emit remaining [consumed, total_seen) with right-shrinking window."""
        out_parts = []
        n = self.total_seen
        while self.consumed < n:
            p = self.consumed
            h = min(p, n - 1 - p, self.half)
            lo = (p - h) - self.buf_offset
            hi = (p + h + 1) - self.buf_offset
            out_parts.append(np.array([self.buf[lo:hi].mean()]))
            self.consumed += 1
        return np.concatenate(out_parts) if out_parts else np.array([], dtype=np.float64)


# ============================================================
# Sliding detector — windowed features + frozen GMM + footstep extraction
# ============================================================

class SlidingDetector:
    """Mirrors event_detection/main.m + Event_Extract.m on a streaming buffer.

    Windows: W=2800, step=1680 (40% overlap). Per window: gausswin → 7-D
    features → assign_clusters_frozen. Consecutive event windows form a "run";
    when a non-event window closes the run, a 1500-sample peak-aligned
    footstep is emitted. End-of-stream drops single-window open runs only,
    matching the offline `j < n - 1` quirk.
    """

    def __init__(self, gmm, fs=8000, window_sec=0.35, overlap=0.40,
                 footfall_len=1500, peak_offset=400,
                 tukey_alpha=0.5, gauss_tau=1.2):
        self.gmm = gmm
        self.fs = fs
        self.W = int(window_sec * fs)
        self.step = int(np.floor((1 - overlap) * self.W))
        self.footfall_len = footfall_len
        self.peak_offset = peak_offset
        self.tukey = ss.windows.tukey(footfall_len, tukey_alpha)
        self.gauss = gausswin(self.W, gauss_tau)

        self.buf = np.zeros(0, dtype=np.float64)
        self.buf_start = 0          # absolute index of buf[0]
        self.next_window_idx = 0    # next window to emit

        self.run_start = None       # window index of run start
        self.run_end = None         # window index of run end (inclusive)

    def feed(self, smoothed_chunk):
        """Append samples; return list of (1500,) footsteps emitted."""
        self.buf = np.concatenate([self.buf, smoothed_chunk])
        out = []

        while True:
            wstart = self.next_window_idx * self.step
            wstop = wstart + self.W
            buf_end = self.buf_start + len(self.buf)
            if wstop > buf_end:
                break

            local = wstart - self.buf_start
            seg = self.buf[local : local + self.W]
            feat = extract_features(self.gauss * seg, self.fs)
            label = int(assign_clusters_frozen(
                feat[None, :],
                self.gmm["mu"], self.gmm["cov"], self.gmm["phi"],
                self.gmm["event_cluster"], self.gmm["threshold"],
            )[0])

            if label == 1:
                if self.run_start is None:
                    self.run_start = self.next_window_idx
                self.run_end = self.next_window_idx
            else:
                if self.run_start is not None:
                    fp = self._extract_run()
                    if fp is not None:
                        out.append(fp)
                    self.run_start = None
                    self.run_end = None

            self.next_window_idx += 1
            self._trim_buffer()

        return out

    def flush(self):
        """End of stream: extract any open MULTI-window run.

        Single-window open runs are dropped to mirror offline `j < n - 1`."""
        out = []
        if self.run_start is not None and self.run_end > self.run_start:
            fp = self._extract_run()
            if fp is not None:
                out.append(fp)
        self.run_start = None
        self.run_end = None
        return out

    def _extract_run(self):
        evnt_start = self.run_start * self.step
        evnt_stop = self.run_end * self.step + self.W
        ls = evnt_start - self.buf_start
        lt = evnt_stop - self.buf_start
        if ls < 0:
            return None
        seg = self.buf[ls:lt]
        if len(seg) == 0:
            return None
        peak_local = int(np.argmax(np.abs(seg)))
        mn = peak_local + evnt_start
        strt = mn - self.peak_offset
        stop = strt + self.footfall_len

        ls2 = strt - self.buf_start
        lt2 = stop - self.buf_start
        wnd = self.tukey.copy()
        if ls2 < 0:
            wnd = wnd[: lt2]
            ls2 = 0
        if lt2 > len(self.buf):
            wnd = wnd[: len(self.buf) - ls2]
            lt2 = len(self.buf)
        seg2 = self.buf[ls2:lt2]
        windowed = wnd * seg2
        out = np.zeros(self.footfall_len)
        out[: len(windowed)] = windowed
        return out

    def _trim_buffer(self):
        if self.run_start is not None:
            keep_from = self.run_start * self.step - self.peak_offset
        else:
            keep_from = self.next_window_idx * self.step - self.peak_offset
        keep_from = max(keep_from, 0)
        if keep_from > self.buf_start:
            drop = keep_from - self.buf_start
            self.buf = self.buf[drop:]
            self.buf_start += drop


# ============================================================
# Voter — sliding majority with confidence floor
# ============================================================

class Voter:
    """Multi-presence sliding-window voter (K bounded by physical reality).

    Tracks last K predictions; ANY pid with >= M confident hits in the window
    is added to the confirmed set. Multiple people can be confirmed at once.

    A pid is forgotten (eligible to re-emit enter on next reappearance) when
    it has zero hits in the current K-window — empirically, equivalent to
    about K footsteps of "this person no longer walking".

    K_MAX = 12 cap — at home, a single walking burst is rarely >10 steps.
    Larger K just mixes signals from separate walking episodes and inflates
    false-positive risk against systematic misclassification biases.

    Defaults K=10/M=3 chosen so that:
      * single-person scenarios behave identically to old K=5 voter
        (always passes threshold quickly, no transition events to miss)
      * 2 people walking simultaneously, alternating, both reach M=3 within
        ~5-8 seconds (vs old voter never confirming the secondary person)
    """
    K_MAX = 12

    def __init__(self, K=10, M=3, conf_min=0.50):
        if K > self.K_MAX:
            print("[voter] K={} clipped to K_MAX={} (home stride/space limit)".format(
                K, self.K_MAX))
            K = self.K_MAX
        if K < 1:
            K = 1
        if M > K:
            M = K
        if M < 1:
            M = 1
        self.K = K
        self.M = M
        self.conf_min = conf_min
        self.window = deque(maxlen=K)
        self.confirmed = set()       # type: set

    def update(self, top1, conf):
        # type: (int, float) -> List[int]
        """Returns list of NEWLY-confirmed pids on this footstep (often empty)."""
        self.window.append(top1 if conf >= self.conf_min else None)
        if len(self.window) < self.M:
            return []

        counts = {}
        for v in self.window:
            if v is not None:
                counts[v] = counts.get(v, 0) + 1

        # Forget anyone in confirmed who has 0 hits in current window.
        # Allows clean re-emit on reappearance after a quiet interval.
        for pid in list(self.confirmed):
            if counts.get(pid, 0) == 0:
                self.confirmed.discard(pid)

        new_confirms = []
        for pid, c in counts.items():
            if c >= self.M and pid not in self.confirmed:
                self.confirmed.add(pid)
                new_confirms.append(pid)
        return new_confirms

    def forget(self, pid):
        """Drop pid from confirmed set so re-entry will emit a fresh event."""
        self.confirmed.discard(pid)


# ============================================================
# Replay driver
# ============================================================

def replay(args):
    # Lazy imports — TensorRT / pycuda / cv2 only exist on Jetson runtime.
    from cwt_fast import cwt as fast_cwt
    from render_lut import cwt_to_rgb_direct
    from jetson_infer import TRTEngine, to_input

    src = MatFileSource(
        args.replay, fs=8000,
        chunk_size=args.chunk_size,
        realtime=args.realtime,
    )
    print("[stream] " + src.info())
    print("[stream] chunk_size={} ({:.0f} ms), realtime={}, queue_max={}".format(
        args.chunk_size, args.chunk_size / 8000 * 1000, args.realtime, args.queue_max,
    ))

    smoother = StreamingSmoother(span=5)

    print("[gmm] loading {}".format(args.gmm))
    gmm = load_gmm_npz(args.gmm)
    detector = SlidingDetector(gmm)
    print("[detector] W={}, step={}, threshold={:.2f}".format(
        detector.W, detector.step, gmm["threshold"],
    ))

    print("[infer] loading TRT plan {}".format(args.plan))
    engine = TRTEngine(args.plan)

    voter = Voter(K=args.vote_k, M=args.vote_m, conf_min=args.vote_conf)
    print("[voter] K={}, M={}, conf_min={:.2f}".format(
        args.vote_k, args.vote_m, args.vote_conf,
    ))

    # ---- Threading ----
    sample_queue = queue.Queue(maxsize=args.queue_max)
    SENTINEL = object()
    qstats = {"max_depth": 0, "full_waits": 0}

    all_preds = []      # type: List[Tuple[int, float]]
    confirmations = []  # type: List[Tuple[int, int]]

    def process_footstep(fp):
        coeffs = fast_cwt(fp)
        img = cwt_to_rgb_direct(coeffs, (224, 224))
        x = to_input(img)
        logits = engine.infer(x)[0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        top1 = int(p.argmax())
        conf = float(p.max())
        all_preds.append((top1, conf))
        for confirmed in voter.update(top1, conf):
            confirmations.append((len(all_preds), confirmed))
            if args.verbose:
                print("  -> step #{}: confirmed P{} (top1=P{}, conf={:.1f}%)".format(
                    len(all_preds), confirmed + 1, top1 + 1, conf * 100,
                ))

    def producer():
        try:
            for chunk in src.chunks():
                while True:
                    try:
                        sample_queue.put(chunk, timeout=1.0)
                        break
                    except queue.Full:
                        # Consumer fell behind — surface but keep blocking.
                        # On real ADS1256 hardware this would mean dropped
                        # samples; here producer just stalls.
                        qstats["full_waits"] += 1
        finally:
            sample_queue.put(SENTINEL)

    t_prod = threading.Thread(target=producer, name="producer", daemon=True)

    print("\n[replay] streaming (threaded)...")
    t0 = time.time()
    t_prod.start()

    chunks_total = (src.n_total + args.chunk_size - 1) // args.chunk_size
    chunks_seen = 0
    while True:
        chunk = sample_queue.get()
        if chunk is SENTINEL:
            break
        depth = sample_queue.qsize()
        if depth > qstats["max_depth"]:
            qstats["max_depth"] = depth
        chunks_seen += 1

        smoothed = smoother(chunk)
        if len(smoothed):
            for fp in detector.feed(smoothed):
                process_footstep(fp)
        if args.verbose and chunks_seen % 200 == 0:
            print("  chunk {}/{}: {} footsteps, {} conf, q={}, {:.1f}s elapsed".format(
                chunks_seen, chunks_total, len(all_preds), len(confirmations),
                depth, time.time() - t0,
            ))

    # Producer done. Flush smoother → detector.
    tail = smoother.flush()
    if len(tail):
        for fp in detector.feed(tail):
            process_footstep(fp)
    for fp in detector.flush():
        process_footstep(fp)

    t_prod.join()
    dt = time.time() - t0
    print("\n[replay] done in {:.1f}s ({:.1f}s of signal, {:.2f}x realtime)".format(
        dt, src.n_total / 8000, (src.n_total / 8000) / max(dt, 1e-9),
    ))
    print("  total footsteps: {}".format(len(all_preds)))
    print("  total confirmations: {}".format(len(confirmations)))
    print("  queue stats: max_depth={}/{}, full_waits={}".format(
        qstats["max_depth"], args.queue_max, qstats["full_waits"],
    ))
    if qstats["full_waits"] > 0:
        print("  ⚠ consumer fell behind producer ({} waits)".format(qstats["full_waits"]))

    if all_preds:
        preds = np.array([p[0] for p in all_preds])
        confs = np.array([p[1] for p in all_preds])
        vals, counts = np.unique(preds, return_counts=True)
        order = np.argsort(-counts)
        print("\n[distribution] top-5 classes:")
        for j in order[:5]:
            cls = int(vals[j])
            mask = preds == cls
            print("  P{:<3d}: {:>4d} ({:5.1f}%)  avg conf {:.1f}%".format(
                cls + 1, counts[j], 100 * counts[j] / len(preds),
                confs[mask].mean() * 100,
            ))

    if args.output:
        np.savez(
            args.output,
            preds=np.array([p[0] for p in all_preds], dtype=np.int32),
            confs=np.array([p[1] for p in all_preds], dtype=np.float32),
            confirmations=np.array(confirmations, dtype=np.int32) if confirmations
                else np.zeros((0, 2), dtype=np.int32),
        )
        print("\n[output] saved to {}".format(args.output))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True,
                    help=".mat file to replay as fake stream")
    ap.add_argument("--plan",
                    default=os.environ.get("TERRA_TRT_PLAN", "mnv3_v2_fp16.plan"))
    ap.add_argument("--gmm",
                    default=os.environ.get("TERRA_GMM", "gmm_params.npz"))
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="Samples per producer chunk (1024 = 128 ms @ 8 kHz)")
    ap.add_argument("--realtime", action="store_true",
                    help="Throttle to wall-clock 8 kHz (default: as fast as possible)")
    ap.add_argument("--queue-max", type=int, default=32,
                    help="Sample queue max chunks (32*128ms = 4s buffering)")
    ap.add_argument("--vote-k", type=int, default=10,
                    help="Voter window size, capped at Voter.K_MAX=12")
    ap.add_argument("--vote-m", type=int, default=3,
                    help="Min confident hits per pid in window to confirm")
    ap.add_argument("--vote-conf", type=float, default=0.50)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--output",
                    help="Save preds/confs/confirmations to .npz")
    args = ap.parse_args()
    replay(args)


if __name__ == "__main__":
    main()
