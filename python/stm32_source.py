"""STM32Source — drop-in replacement for MatFileSource that reads
binary frames from STM32 F411RE (ST-Link VCP) and yields 8 kHz chunks.

Pipeline:
    /dev/ttyACM0 (or COMx) bytes
      -> SYNC search (0xA5 0x5A)
      -> frame parse [SEQ u16 LE | N=64 | int24 BE x N | CRC8]
      -> int24 sign-extend -> float64
      -> StreamingPolyphase 8/15 (15000 -> 8000 Hz)
      -> chunk_size samples per yield (default 1024)

Public API mirrors MatFileSource:
    chunks() -> Iterator[np.ndarray of float64 at fs_out]
    chunks_raw() -> Iterator[np.ndarray of float64 at fs_in]
    info() -> str

Self-test (run as script):
    python stm32_source.py            # auto-detect port, 5s capture
    python stm32_source.py --port COM12 --duration 10
    python stm32_source.py --raw      # bypass polyphase, dump raw 15kHz samples
"""
import argparse
import os
import sys
import time
from typing import Iterator, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase  # already byte-validated for 8/15

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None  # type: ignore


SYNC = b"\xA5\x5A"
FRAME_SIZE = 198            # 2 SYNC + 2 SEQ + 1 N + 64*3 payload + 1 CRC
PAYLOAD_N = 64              # samples per frame
DEFAULT_FS_IN = 15000       # ADS1256 native after STM32 bring-up
DEFAULT_FS_OUT = 8000       # what the rest of the pipeline expects

# S.5 calibration (2026-05-10): VIBeID 셋업 (외부 op-amp gain 10 + sound card V) 와
# 진폭 분포 일치시키는 scale. 우리 = ADS1256 PGA=16 단독 (amp 없음).
# scale = ADS1256 LSB voltage * VIBeID amp gain
#       = (0.3125 V / 8388608) * 10
#       = 3.7253e-7 V/LSB
# 측정값 (by-peak) 3.19e-7 와 17% 안 일치. 가족 deployment fine-tune 단계에서 재조정 가능.
CALIBRATED_SCALE = 3.7253e-7


def crc8(data: bytes) -> int:
    """CRC-8, poly=0x07, init=0x00, no reflect, no xor-out.
    Mirrors STM32 firmware's crc8() byte-for-byte."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc


def _decode_int24_be(payload: bytes, n: int) -> np.ndarray:
    """Vectorized int24 BE -> int32 with sign extension."""
    arr = np.frombuffer(payload[: n * 3], dtype=np.uint8).reshape(n, 3).astype(np.int32)
    v = (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]
    # Sign-extend bit 23
    v = np.where(v & 0x800000, v - 0x1000000, v)
    return v.astype(np.int32)


def auto_detect_port() -> Optional[str]:
    """Return first ST-Link VCP COM/tty path, or None."""
    if serial is None:
        return None
    for p in list_ports.comports():
        desc = (p.description or "") + " " + (p.manufacturer or "")
        if any(s in desc for s in ("STLink", "ST-Link", "STMicroelectronics")):
            return p.device
    return None


class STM32Source:
    """Streaming source for STM32 ADS1256-bridge binary protocol.

    Args:
        port: COM/tty device path. None = auto-detect ST-Link VCP.
        baud: serial baud (must match firmware, default 921600).
        fs_in: input sample rate STM32 is producing (informational; used
               only for `info()` and rate sanity).
        fs_out: target output rate after polyphase (default 8000 Hz).
        chunk_size: yield size at fs_out (default 1024, matches MatFileSource).
        scale: float multiplier applied to int32 sample value (V/LSB or 1.0
               for raw counts). Until ADS1256 calibration we use 1.0.
        timeout: serial read timeout per call (seconds). 1.0 is safe; lower
                 makes stop() more responsive.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 921600,
        fs_in: int = DEFAULT_FS_IN,
        fs_out: int = DEFAULT_FS_OUT,
        chunk_size: int = 1024,
        scale: float = 1.0,
        timeout: float = 1.0,
    ):
        if serial is None:
            raise ImportError("pyserial not installed: pip install pyserial")
        if port is None:
            port = auto_detect_port()
        if port is None:
            raise RuntimeError("ST-Link VCP not found; pass port= explicitly.")

        self.port = port
        self.baud = baud
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.chunk_size = chunk_size
        self.scale = float(scale)
        self.timeout = timeout

        # Polyphase: 15000 -> 8000 = up 8, down 15
        # If fs_in / fs_out doesn't simplify to small integers we fall back
        # to identity (caller must resample externally).
        from math import gcd
        g = gcd(fs_out, fs_in)
        self.resample_p = fs_out // g
        self.resample_q = fs_in // g
        self._resampler = StreamingPolyphase(p=self.resample_p, q=self.resample_q)

        self._ser = None
        self._buf = bytearray()
        self._stop = False

        # Stats
        self.frames_ok = 0
        self.frames_bad_crc = 0
        self.frames_bad_n = 0
        self.bytes_dropped = 0
        self.seq_drops = 0
        self.resets = 0          # STM32 reset events (seq << last_seq, gap > RESET_GAP)
        self._last_seq = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def open(self):
        if self._ser is None:
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            self._ser.reset_input_buffer()
        return self

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def stop(self):
        """Request chunks() / chunks_raw() loop to exit on next iteration."""
        self._stop = True

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------
    # Frame parsing
    # ------------------------------------------------------------
    def _parse_one_frame(self, frame: bytes) -> Optional[np.ndarray]:
        """Validate + decode one 198-byte aligned frame. Return float64
        samples (n_samples,) or None on CRC / N mismatch."""
        seq = frame[2] | (frame[3] << 8)
        n = frame[4]
        if n != PAYLOAD_N:
            self.frames_bad_n += 1
            return None
        if crc8(frame[2:197]) != frame[197]:
            self.frames_bad_crc += 1
            return None

        # SEQ drop check (uint16 wraparound aware).
        # Standard half-wrap heuristic: if the forward gap exceeds half the
        # wraparound range, the stream most likely restarted (STM32 brownout /
        # USB re-enum / power cycle). Count that as a reset rather than tens
        # of thousands of "lost" frames. Real frame loss within continuous
        # streaming gives gap in the 1-100 range; reset gives gap in the
        # 50000+ range. (After USB re-enum, seq may already be 100-2000+
        # before host reads, so we cannot rely on seq < small.)
        if self._last_seq is not None:
            expected = (self._last_seq + 1) & 0xFFFF
            if seq != expected:
                gap = (seq - expected) & 0xFFFF
                if gap > 32768:
                    self.resets += 1
                else:
                    self.seq_drops += gap
        self._last_seq = seq

        samples = _decode_int24_be(frame[5:5 + n * 3], n)
        return samples.astype(np.float64) * self.scale

    def _read_one_frame(self) -> Optional[np.ndarray]:
        """Block until one valid frame parsed, or timeout/stop. Return
        decoded samples or None if loop should terminate."""
        ser = self._ser
        while not self._stop:
            # Top up buffer to at least FRAME_SIZE
            if len(self._buf) < FRAME_SIZE:
                need = FRAME_SIZE - len(self._buf)
                data = ser.read(need)
                if data:
                    self._buf.extend(data)
                if len(self._buf) < FRAME_SIZE:
                    # serial timed out -> let caller decide via self._stop
                    if self._stop:
                        return None
                    continue

            # Align SYNC at buffer start
            if self._buf[0] != 0xA5 or self._buf[1] != 0x5A:
                idx = self._buf.find(SYNC, 1)
                if idx < 0:
                    # No SYNC at all; preserve last byte (could be 0xA5 of next pair)
                    self.bytes_dropped += len(self._buf) - 1
                    self._buf = self._buf[-1:]
                else:
                    self.bytes_dropped += idx
                    self._buf = self._buf[idx:]
                continue

            # We have FRAME_SIZE bytes starting at SYNC
            frame = bytes(self._buf[:FRAME_SIZE])
            self._buf = self._buf[FRAME_SIZE:]
            samples = self._parse_one_frame(frame)
            if samples is None:
                # CRC or N failed. The 0xA5 0x5A may have been a coincidence
                # inside payload — but a single-byte advance + research is
                # expensive. Accept the lost frame, keep going.
                continue
            self.frames_ok += 1
            return samples
        return None

    # ------------------------------------------------------------
    # Iterators (MatFileSource-compatible)
    # ------------------------------------------------------------
    def chunks_raw(self) -> Iterator[np.ndarray]:
        """Yield decoded float64 arrays at fs_in (no resampling).
        Each yield is one frame's worth of samples (PAYLOAD_N=64)."""
        self.open()
        try:
            while not self._stop:
                samples = self._read_one_frame()
                if samples is None:
                    return
                yield samples
        finally:
            pass  # caller controls close()

    def chunks(self) -> Iterator[np.ndarray]:
        """Yield resampled float64 arrays at fs_out, sized chunk_size.
        Drop-in for MatFileSource.chunks()."""
        self.open()
        out_buf = np.empty(0, dtype=np.float64)
        try:
            while not self._stop:
                samples_in = self._read_one_frame()
                if samples_in is None:
                    break
                out = self._resampler.feed(samples_in)
                if len(out):
                    out_buf = np.concatenate([out_buf, out])
                while len(out_buf) >= self.chunk_size:
                    yield out_buf[: self.chunk_size]
                    out_buf = out_buf[self.chunk_size:]
            # Flush remainder on stop/EOF
            if len(out_buf):
                yield out_buf
        finally:
            pass

    def info(self) -> str:
        return (
            "STM32Source: {} @ {} baud, {} -> {} Hz (p={}, q={}), "
            "scale {:.3e} V/LSB, frame={} byte"
        ).format(
            self.port, self.baud, self.fs_in, self.fs_out,
            self.resample_p, self.resample_q, self.scale, FRAME_SIZE,
        )

    def stats(self) -> dict:
        return {
            "frames_ok": self.frames_ok,
            "frames_bad_crc": self.frames_bad_crc,
            "frames_bad_n": self.frames_bad_n,
            "bytes_dropped": self.bytes_dropped,
            "seq_drops": self.seq_drops,
            "resets": self.resets,
        }


# ============================================================
# Self-test CLI
# ============================================================
def _selftest(args):
    src = STM32Source(
        port=args.port,
        baud=args.baud,
        fs_in=args.fs_in,
        fs_out=args.fs_out,
        chunk_size=args.chunk_size,
        scale=args.scale,
    )
    print("[connect]", src.info())
    print("[capture] {:.1f}s {} ...".format(
        args.duration, "raw chunks_raw()" if args.raw else "resampled chunks()"
    ))

    t0 = time.time()
    deadline = t0 + args.duration
    iter_ = src.chunks_raw() if args.raw else src.chunks()
    out_samples = []
    n_yields = 0

    try:
        for chunk in iter_:
            out_samples.append(chunk.copy())
            n_yields += 1
            if time.time() >= deadline:
                src.stop()
    except KeyboardInterrupt:
        src.stop()
    finally:
        src.close()

    elapsed = time.time() - t0
    out = np.concatenate(out_samples) if out_samples else np.empty(0)
    rate = len(out) / elapsed if elapsed > 0 else 0.0
    expected_rate = args.fs_in if args.raw else args.fs_out

    s = src.stats()
    print()
    print("[result] elapsed {:.2f}s, yields {}, samples {}".format(elapsed, n_yields, len(out)))
    print("  measured rate: {:.1f} sps  (target {} sps)".format(rate, expected_rate))
    print("  frames OK: {}".format(s["frames_ok"]))
    print("  frames bad CRC: {}, bad N: {}".format(s["frames_bad_crc"], s["frames_bad_n"]))
    print("  SEQ drops: {}, resets: {}, bytes dropped (resync): {}".format(
        s["seq_drops"], s["resets"], s["bytes_dropped"]
    ))

    if len(out) >= 6:
        print("  first 3 samples: {}".format(out[:3]))
        print("  last  3 samples: {}".format(out[-3:]))
        if args.raw:
            # ramp slope check (Phase 3 fake data: +1 per sample, scale x scale)
            diffs = np.diff(out[:64])  # within first frame
            print("  ramp diff (first frame): mean={:.4f} std={:.4f} (Phase 3 expects ~scale)".format(
                float(diffs.mean()), float(diffs.std()),
            ))

    if args.save:
        np.save(args.save, out)
        print("  saved -> {}  ({:.1f} KB)".format(args.save, out.nbytes / 1024))

    pass_ = (
        s["frames_ok"] > 0
        and s["frames_bad_crc"] == 0
        and s["frames_bad_n"] == 0
        and s["seq_drops"] == 0
    )
    print()
    print("[PASS]" if pass_ else "[FAIL]")
    return 0 if pass_ else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="COMx / /dev/ttyACMx (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--duration", type=float, default=5.0, help="seconds to capture")
    ap.add_argument("--fs-in", type=int, default=DEFAULT_FS_IN)
    ap.add_argument("--fs-out", type=int, default=DEFAULT_FS_OUT)
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="yield 64-sample raw frames at fs_in")
    ap.add_argument("--save", default=None, help="save concatenated samples to .npy")
    args = ap.parse_args()
    sys.exit(_selftest(args))


if __name__ == "__main__":
    main()
