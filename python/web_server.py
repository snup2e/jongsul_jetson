"""Web dashboard for Terra footstep recognition.

Wraps jetson_realtime.py's pipeline (MatFileSource/Smoother/Detector/Voter)
and serves a Toss-style HTML page that shows who's home in real time.

Two run modes:

    # Real pipeline (Jetson) — replays a .mat file as fake stream
    python3 web_server.py --replay sample/P14_1.mat \
        --plan mnv3_v2_fp16.plan --gmm gmm_params.npz --realtime

    # UI dev mode (any machine, no TRT needed) — fakes events on a timer
    python3 web_server.py --demo

Open http://<host>:8000/  on any device on the same Wi-Fi.

Events stream over SSE (text/event-stream) at /events. Each line is JSON:
    {"type":"inference",  "person_id":13, "conf":0.82, "ts":...}
    {"type":"confirm",    "person_id":13, "conf":0.79, "ts":..., "step_idx":4}
    {"type":"stats",      "footsteps":N,  "queue_depth":3, "elapsed":12.3}
    {"type":"away",       "person_id":13, "ts":...}        # auto-emitted
    {"type":"hello",      "people": {...}, "present":[...]}# initial state on connect
"""
import argparse
import json
import os
import queue as queue_mod
import random
import threading
import time
from collections import deque

import numpy as np

from flask import Flask, Response, jsonify, send_from_directory, stream_with_context

# We import jetson_realtime lazily for --replay so --demo works without
# scipy/TensorRT installed.

# ============================================================
# Pub/sub broker — pipeline thread → all SSE clients
# ============================================================

class EventBroker:
    """Thread-safe fan-out of events to every connected SSE client."""

    def __init__(self, log_path=None):
        self._clients = []  # list of queue.Queue
        self._lock = threading.Lock()
        self._history = deque(maxlen=200)  # recent events for /state replay
        # Optional JSONL event log — one event per line (inference/confirm/away/stats).
        # Captures the full inference trace of a demo session.
        self._log = None
        if log_path:
            self._log = open(log_path, "a", encoding="utf-8")
            print("[broker] logging events -> {}".format(log_path))

    def subscribe(self):
        q = queue_mod.Queue(maxsize=256)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, event):
        self._history.append(event)
        if self._log is not None:
            try:
                self._log.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._log.flush()
            except Exception as e:
                print("[broker] log write failed: {}".format(e))
        with self._lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(event)
                except queue_mod.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._clients.remove(q)
                except ValueError:
                    pass

    def history(self):
        return list(self._history)


# ============================================================
# Presence tracker — translates raw events into "who's home"
# ============================================================

class PresenceTracker:
    """Tracks last-seen per person and decides who is currently 'home'.

    A person is 'present' if they were confirmed within `away_after_sec`.
    A background sweeper emits `away` events when that window expires.

    `on_away` callback is invoked with pid before the event publishes —
    used to also clear the voter's confirmed-set so a re-entry triggers
    a fresh confirm event later.
    """

    def __init__(self, broker, people_map, away_after_sec=300, on_away=None):
        self.broker = broker
        self.people_map = people_map
        self.away_after = away_after_sec
        self.on_away_cb = on_away
        self.present = {}  # pid -> {name, emoji, last_seen, conf, entered_at}
        self._lock = threading.Lock()

    def lookup(self, pid):
        # 0-indexed pid; people_map is keyed by string for JSON friendliness
        info = self.people_map.get(str(pid))
        if info is None:
            return {"name": "P{}".format(pid + 1), "emoji": "🧑"}
        return info

    def on_confirm(self, pid, conf, ts):
        info = self.lookup(pid)
        with self._lock:
            already = pid in self.present
            self.present[pid] = {
                "person_id": pid,
                "name": info["name"],
                "emoji": info["emoji"],
                "last_seen": ts,
                "conf": conf,
                "entered_at": self.present[pid]["entered_at"] if already else ts,
            }

    def on_inference(self, pid, conf, ts):
        # Keep last_seen fresh for already-present people on every footstep,
        # so the away timer doesn't fire mid-walk if the voter doesn't reconfirm.
        with self._lock:
            if pid in self.present and conf >= 0.5:
                self.present[pid]["last_seen"] = ts

    def snapshot(self):
        with self._lock:
            return list(self.present.values())

    def sweep_away(self, now):
        """Emit `away` events for anyone whose last_seen exceeded the window."""
        gone = []
        with self._lock:
            for pid, p in list(self.present.items()):
                if now - p["last_seen"] > self.away_after:
                    gone.append((pid, p))
                    del self.present[pid]
        for pid, p in gone:
            if self.on_away_cb:
                try:
                    self.on_away_cb(pid)
                except Exception as e:
                    print("[presence] on_away callback failed: {}".format(e))
            self.broker.publish({
                "type": "away",
                "person_id": pid,
                "name": p["name"],
                "emoji": p["emoji"],
                "ts": now,
            })


def sweeper_loop(presence, broker, period=2.0):
    while True:
        time.sleep(period)
        presence.sweep_away(time.time())


# ============================================================
# Pipeline runners (real / demo)
# ============================================================

def run_real_pipeline(args, broker, presence, voter):
    """Run the full Jetson pipeline against a .mat replay, fan out events.

    CRITICAL: TensorRT / pycuda must be imported and used on the SAME thread.
    pycuda.autoinit binds the CUDA primary context to whichever thread first
    imports it, and TRT engine handles created on that context are unusable
    from other threads (-> 'invalid resource handle').

    So everything CUDA-touching (jetson_infer import, TRTEngine(), engine.infer)
    is contained in `consumer()`. Producer and the Flask app run on other
    threads but never touch the GPU.

    `voter` is created in main() and shared with PresenceTracker so its
    confirmed-set stays in sync with presence on away events.
    """
    from jetson_realtime import StreamingSmoother, SlidingDetector
    from extract import load_gmm_npz

    if args.stm32:
        from stm32_source import STM32Source, CALIBRATED_SCALE
        src = STM32Source(
            port=args.stm32_port,
            chunk_size=args.chunk_size,
            scale=CALIBRATED_SCALE,
        )
    else:
        from jetson_realtime import MatFileSource
        src = MatFileSource(
            args.replay, fs=8000,
            chunk_size=args.chunk_size,
            realtime=args.realtime,
        )
    print("[stream] " + src.info())

    smoother = StreamingSmoother(span=5)
    gmm = load_gmm_npz(args.gmm)
    detector = SlidingDetector(gmm)
    # voter passed in from main()

    sample_queue = queue_mod.Queue(maxsize=args.queue_max)
    SENTINEL = object()
    stats = {"footsteps": 0, "max_depth": 0, "full_waits": 0}
    engine_ready = threading.Event()

    def producer():
        # Wait for TRT engine to finish loading before flooding the queue —
        # otherwise --realtime timing gets distorted by the load delay.
        engine_ready.wait(timeout=60.0)
        try:
            for chunk in src.chunks():
                while True:
                    try:
                        sample_queue.put(chunk, timeout=1.0)
                        break
                    except queue_mod.Full:
                        stats["full_waits"] += 1
        finally:
            sample_queue.put(SENTINEL)

    def consumer():
        # Heavy CUDA-touching imports MUST happen on this thread.
        from cwt_fast import cwt as fast_cwt
        from render_lut import cwt_to_rgb_direct
        from jetson_infer import TRTEngine, to_input

        engine = TRTEngine(args.plan)
        # Warmup: first inference after engine load triggers CUDA JIT + memory
        # allocation (~2-5s). If we let the producer start streaming first, that
        # cold-start happens mid-flow and the sample_queue floods. Burn it now.
        warmup_x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        for _ in range(3):
            _ = engine.infer(warmup_x)
        engine_ready.set()
        print("[infer] engine ready on consumer thread (warmed up)")

        def process_footstep(fp):
            coeffs = fast_cwt(fp)
            img = cwt_to_rgb_direct(coeffs, (224, 224))
            x = to_input(img)
            logits = engine.infer(x)[0]
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            top1 = int(p.argmax())
            conf = float(p.max())
            if args.test_pin_pid is not None:
                top1 = args.test_pin_pid
            ts = time.time()
            stats["footsteps"] += 1

            info = presence.lookup(top1)
            broker.publish({
                "type": "inference",
                "person_id": top1,
                "name": info["name"],
                "emoji": info["emoji"],
                "conf": conf,
                "ts": ts,
                "step_idx": stats["footsteps"],
            })
            presence.on_inference(top1, conf, ts)

            for confirmed_pid in voter.update(top1, conf):
                cinfo = presence.lookup(confirmed_pid)
                presence.on_confirm(confirmed_pid, conf, ts)
                broker.publish({
                    "type": "confirm",
                    "person_id": confirmed_pid,
                    "name": cinfo["name"],
                    "emoji": cinfo["emoji"],
                    "conf": conf,
                    "ts": ts,
                    "step_idx": stats["footsteps"],
                })

        t0 = time.time()
        last_stats_pub = 0.0
        while True:
            chunk = sample_queue.get()
            if chunk is SENTINEL:
                break
            depth = sample_queue.qsize()
            if depth > stats["max_depth"]:
                stats["max_depth"] = depth
            smoothed = smoother(chunk)
            if len(smoothed):
                for fp in detector.feed(smoothed):
                    process_footstep(fp)
            # Stats heartbeat at most once per second. The browser ticks the
            # elapsed counter locally between heartbeats so it stays smooth.
            now = time.time()
            if now - last_stats_pub >= 1.0:
                last_stats_pub = now
                broker.publish({
                    "type": "stats",
                    "footsteps": stats["footsteps"],
                    "queue_depth": depth,
                    "elapsed": now - t0,
                })
        # Flush
        tail = smoother.flush()
        if len(tail):
            for fp in detector.feed(tail):
                process_footstep(fp)
        for fp in detector.flush():
            process_footstep(fp)
        print("[pipeline] done. footsteps={} full_waits={}".format(
            stats["footsteps"], stats["full_waits"]))

    threading.Thread(target=consumer, name="consumer", daemon=True).start()
    threading.Thread(target=producer, name="producer", daemon=True).start()


def run_demo(broker, presence, period=4.0):
    """Emit fake events on a timer — for UI dev without the real pipeline."""
    print("[demo] emitting fake events every {:.1f}s".format(period))
    rng = random.Random(0)
    # Cycle through a few people defined in people.json
    candidates = [13, 49, 99, 1]   # P14, P50, P100, P2
    last_confirmed = {pid: 0.0 for pid in candidates}

    def loop():
        i = 0
        while True:
            time.sleep(period)
            pid = candidates[i % len(candidates)]
            i += 1
            ts = time.time()
            info = presence.lookup(pid)
            conf = 0.6 + rng.random() * 0.35
            broker.publish({
                "type": "inference",
                "person_id": pid, "name": info["name"], "emoji": info["emoji"],
                "conf": conf, "ts": ts, "step_idx": i,
            })
            presence.on_inference(pid, conf, ts)
            # Every 3rd fake hit → confirm (transition-only feel)
            if i % 3 == 0 and ts - last_confirmed[pid] > 10:
                last_confirmed[pid] = ts
                presence.on_confirm(pid, conf, ts)
                broker.publish({
                    "type": "confirm",
                    "person_id": pid, "name": info["name"], "emoji": info["emoji"],
                    "conf": conf, "ts": ts, "step_idx": i,
                })

    threading.Thread(target=loop, name="demo", daemon=True).start()


# ============================================================
# Flask app
# ============================================================

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def build_app(broker, presence):
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.route("/static/<path:fname>")
    def static_files(fname):
        return send_from_directory(WEB_DIR, fname)

    @app.route("/state")
    def state():
        return jsonify({
            "present": presence.snapshot(),
            "people_map": presence.people_map,
            "ts": time.time(),
        })

    @app.route("/events")
    def events():
        q = broker.subscribe()

        @stream_with_context
        def gen():
            try:
                # Initial hello with current presence
                hello = {
                    "type": "hello",
                    "ts": time.time(),
                    "present": presence.snapshot(),
                    "people_map": presence.people_map,
                }
                yield "retry: 3000\n\n"
                yield "data: " + json.dumps(hello, ensure_ascii=False) + "\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15.0)
                        yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                    except queue_mod.Empty:
                        # Comment line keeps connection alive through proxies
                        yield ": keepalive\n\n"
            finally:
                broker.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream")

    return app


# ============================================================
# Main
# ============================================================

def load_people_map():
    path = os.path.join(WEB_DIR, "people.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help=".mat file to replay (omit for --demo)")
    ap.add_argument("--demo", action="store_true",
                    help="Fake events on a timer (no TRT/scipy needed)")
    ap.add_argument("--stm32", action="store_true",
                    help="Live source: STM32 ADS1256 bridge over USB serial")
    ap.add_argument("--stm32-port", default=None,
                    help="STM32 serial port (default: auto-detect ST-Link VCP)")
    ap.add_argument("--plan",
                    default=os.environ.get("TERRA_TRT_PLAN", "mnv3_v2_fp16.plan"))
    ap.add_argument("--gmm",
                    default=os.environ.get("TERRA_GMM", "gmm_params.npz"))
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--queue-max", type=int, default=32)
    ap.add_argument("--vote-k", type=int, default=5)
    ap.add_argument("--vote-m", type=int, default=3)
    ap.add_argument("--vote-conf", type=float, default=0.50)
    ap.add_argument("--test-mode", action="store_true",
                    help="Demo recording: force-confirm every footstep (voter -> K=1/M=1/conf=0)")
    ap.add_argument("--test-pin-pid", type=int, default=None,
                    help="Demo recording: force all events to this person_id (e.g. 0). "
                         "Overrides model output so card stays consistent under domain shift.")
    ap.add_argument("--away-after", type=int, default=300,
                    help="Seconds without footsteps before person → away")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-events", default=None,
                    help="Append all events (inference/confirm/away/stats) as JSONL to this path. "
                         "Captures the full inference trace of a demo session.")
    args = ap.parse_args()

    if not (args.replay or args.demo or args.stm32):
        ap.error("specify one of: --replay <mat>, --demo, or --stm32")

    if args.test_mode:
        args.vote_k = 1
        args.vote_m = 1
        args.vote_conf = 0.0
        print("[test-mode] voter forced to K=1/M=1/conf=0 (every footstep auto-confirms)")
    if args.test_pin_pid is not None:
        print("[test-mode] pinning all events to person_id={}".format(args.test_pin_pid))

    broker = EventBroker(log_path=args.log_events)
    people = load_people_map()

    # Voter is shared between consumer thread (calls .update) and presence
    # sweeper (calls .forget on away). Created here so both have a ref.
    from jetson_realtime import Voter
    voter = Voter(K=args.vote_k, M=args.vote_m, conf_min=args.vote_conf)
    print("[voter] K={} (cap={}), M={}, conf_min={:.2f}, multi-presence ON".format(
        voter.K, Voter.K_MAX, voter.M, voter.conf_min))

    presence = PresenceTracker(broker, people, away_after_sec=args.away_after,
                               on_away=voter.forget)

    threading.Thread(target=sweeper_loop, args=(presence, broker),
                     name="sweeper", daemon=True).start()

    if args.demo:
        run_demo(broker, presence)
    else:
        run_real_pipeline(args, broker, presence, voter)

    app = build_app(broker, presence)
    print("[web] http://{}:{}/  (open this in your browser)".format(
        args.host, args.port))
    # threaded=True so SSE long-poll doesn't block other requests
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
