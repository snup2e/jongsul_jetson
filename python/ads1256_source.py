"""ADS1256 streaming source for Jetson Nano.

Drop-in replacement for MatFileSource in jetson_realtime.py: same
chunks() -> Iterator[np.ndarray] interface, same 8000 Hz output rate
(after polyphase upsample from native 7500 SPS).

Hardware path (Waveshare High-Precision AD/DA HAT on Jetson Nano J41):
    DRDY -> BCM 17 (J41 pin 11), input
    RST  -> BCM 18 (J41 pin 12), output
    CS   -> BCM 22 (J41 pin 15), output
    SPI0 (MOSI/MISO/SCLK on pins 19/21/23) -> /dev/spidev0.0 @ 1 MHz, mode 1

ADS1256 config: AIN0/AINCOM single-ended, gain=1x, vref=2.5V,
DRATE=7500 SPS (0xD0). 3750 SPS (0xC0) is the documented fallback.

Read pattern (RDATAC continuous mode):
    busy-poll DRDY low -> CS low -> spi.xfer2([0,0,0]) -> CS high
    -> sign-extend 24-bit two's complement -> scale to volts

Polyphase 7500 -> 8000 upsample runs inside chunks() so downstream
code (StreamingSmoother / SlidingDetector / TRT engine) sees a plain
8000 Hz stream. Filter is a Kaiser-windowed firwin lowpass with
stopband well below the 40-160 Hz signal band.

Test harness: chunks_raw() exposes native-rate chunks plus per-sample
timing metrics, used by ads1256_bench.py to validate SPI throughput.

Calibration note: the `scale` parameter converts raw signed 24-bit
counts to volts at gain=1, vref=2.5V (default 5.0/0x7FFFFF). The
training data (data/raw/*.mat geo_data) is in unknown units and was
captured with an unknown sensor sensitivity; first ADS1256 capture
should be RMS-compared to a representative geo_data slice and `scale`
adjusted to match. See ads1256_bench.py "calibrate" mode.
"""
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy import signal as ss


# ============================================================
# Streaming polyphase resampler
# ============================================================

class StreamingPolyphase:
    """Streaming p/q rational resampler with per-chunk feed().

    For input chunks of arbitrary size, produces output at rate
    (input_rate * p / q). Concatenated streaming output matches a
    block-mode reference (scipy.signal.upfirdn with the same filter)
    to numerical precision; verified by resample_for_ads1256.py.

    The prototype FIR has length L = p * taps_per_phase, designed via
    firwin with a Kaiser window. Cutoff is 1/max(p,q) of the upsampled
    fs/2 — this kills both the spectral images from upsampling (at
    multiples of fs_in/fs_up) and any content above output Nyquist
    (which would alias under the down-by-q step). Same convention as
    scipy.signal.resample_poly.

    Polyphase decomposition: subfilter k (k=0..p-1) operates on input
    phase k, with subf[k][i] = h[k + i*p]. State = last (K-1) input
    samples (zero-init). up_pos counter tracks downsample phase across
    calls.
    """

    def __init__(self, p, q, taps_per_phase=20, kaiser_beta=5.0):
        # type: (int, int, int, float) -> None
        if p <= 0 or q <= 0:
            raise ValueError("p, q must be positive integers")
        self.p = int(p)
        self.q = int(q)
        self.K = int(taps_per_phase)
        L = self.p * self.K
        # Cutoff 1/max(p,q) of upsampled fs/2 (normalized 1.0). Matches
        # scipy.signal.resample_poly convention.
        cutoff = 1.0 / float(max(self.p, self.q))
        # firwin returns h with sum(h)=1 (unity DC gain). Multiply by p
        # to compensate for zero-stuff dilution after upsampling.
        h = ss.firwin(L, cutoff, window=("kaiser", kaiser_beta)) * self.p
        self.h = h
        # Polyphase decomposition: subf[k, i] = h[k + i*p]
        self.subfilters = np.array(
            [h[k::self.p][: self.K] for k in range(self.p)],
            dtype=np.float64,
        )
        # Pre-reverse so windows @ subf_rev.T performs the convolution
        self._subf_rev = np.ascontiguousarray(self.subfilters[:, ::-1])
        self.state = np.zeros(self.K - 1, dtype=np.float64)
        self._up_pos = 0  # cumulative position in upsampled stream

    def feed(self, x):
        # type: (np.ndarray) -> np.ndarray
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        x = np.ascontiguousarray(x, dtype=np.float64)
        buf = np.concatenate([self.state, x])
        N = len(x)
        # windows[n, r] = buf[n + r], shape (N, K)
        windows = np.lib.stride_tricks.sliding_window_view(buf, self.K)
        # out_up[n, k] = sum_r subfilters[k, K-1-r] * windows[n, r]
        out_up = windows @ self._subf_rev.T  # (N, p)
        # Interleave to scalar upsampled stream y_up[n*p + k]
        out_up_flat = out_up.reshape(-1)
        # Take every q-th sample, keeping phase across calls
        first = (self.q - self._up_pos % self.q) % self.q
        out = out_up_flat[first :: self.q]
        self._up_pos += N * self.p
        self.state = buf[-(self.K - 1):].copy()
        return np.ascontiguousarray(out)


# ============================================================
# ADS1256 register / command / pin constants
# ============================================================

REG_STATUS = 0
REG_MUX    = 1
REG_ADCON  = 2
REG_DRATE  = 3

CMD_WAKEUP  = 0x00
CMD_RDATA   = 0x01
CMD_RDATAC  = 0x03
CMD_SDATAC  = 0x0F
CMD_RREG    = 0x10
CMD_WREG    = 0x50
CMD_SELFCAL = 0xF0
CMD_SYNC    = 0xFC
CMD_RESET   = 0xFE

DRATE_30000 = 0xF0
DRATE_15000 = 0xE0
DRATE_7500  = 0xD0
DRATE_3750  = 0xC0
DRATE_2000  = 0xB0
DRATE_1000  = 0xA1
DRATE_500   = 0x92
DRATE_100   = 0x82

DRATE_NOMINAL_SPS = {
    DRATE_30000: 30000,
    DRATE_15000: 15000,
    DRATE_7500:  7500,
    DRATE_3750:  3750,
    DRATE_2000:  2000,
    DRATE_1000:  1000,
    DRATE_500:   500,
    DRATE_100:   100,
}

DRATE_BY_NAME = {
    "30000": DRATE_30000, "15000": DRATE_15000, "7500": DRATE_7500,
    "3750": DRATE_3750, "2000": DRATE_2000, "1000": DRATE_1000,
    "500": DRATE_500, "100": DRATE_100,
}

# Waveshare HAT BCM pin layout — matches Jetson Nano J41 header for these pins
PIN_RST  = 18  # J41 pin 12
PIN_CS   = 22  # J41 pin 15
PIN_DRDY = 17  # J41 pin 11


# ============================================================
# ADS1256 source
# ============================================================

class ADS1256Source:
    """ADS1256 vibration capture as a chunks() generator.

    Same interface as MatFileSource (jetson_realtime.py): chunks()
    yields np.ndarray of float64 samples at output_rate Hz (8000 by
    default, after polyphase upsample from native 7500 SPS).

    Use chunks_raw() for the bench harness — yields native-rate chunks
    with per-sample timing metrics.

    Stop the producer with stop() (sets a flag the generator polls).
    Use as a context manager for SPI/GPIO cleanup, or call close().
    Re-entry not supported; create a new instance to restart.
    """

    def __init__(
        self,
        drate=DRATE_7500,
        chunk_in_size=1024,
        spi_bus=0,
        spi_device=0,
        spi_clock_hz=1_000_000,
        gain=0,                  # 0 = 1x (gain register code)
        scale=5.0 / 0x7FFFFF,    # volts/LSB at gain=1, vref=2.5V
        resample_p=16,
        resample_q=15,
        drdy_timeout_periods=5.0,
        verbose=False,
    ):
        # type: (int, int, int, int, int, int, float, int, int, float, bool) -> None
        if drate not in DRATE_NOMINAL_SPS:
            raise ValueError("Unknown DRATE code 0x{:02X}".format(drate))
        self.drate = drate
        self.nominal_sps = DRATE_NOMINAL_SPS[drate]
        self.chunk_in_size = int(chunk_in_size)
        self.gain = int(gain)
        self.scale = float(scale)
        self.resample_p = int(resample_p)
        self.resample_q = int(resample_q)
        self.output_rate = self.nominal_sps * self.resample_p / self.resample_q
        self.drdy_timeout_s = drdy_timeout_periods / self.nominal_sps
        self.verbose = bool(verbose)

        # Lazy hardware imports so this module is importable on Windows
        try:
            import spidev as _spidev
        except ImportError:
            raise ImportError("spidev not installed: pip install spidev")
        try:
            import Jetson.GPIO as _GPIO
        except ImportError:
            raise ImportError("Jetson.GPIO not installed: pip install Jetson.GPIO")
        self._spidev = _spidev
        self._GPIO = _GPIO

        self._spi = _spidev.SpiDev()
        self._spi.open(spi_bus, spi_device)
        self._spi.max_speed_hz = int(spi_clock_hz)
        self._spi.mode = 0b01  # CPOL=0, CPHA=1 — ADS1256 datasheet

        _GPIO.setmode(_GPIO.BCM)
        _GPIO.setwarnings(False)
        _GPIO.setup(PIN_RST,  _GPIO.OUT, initial=_GPIO.HIGH)
        _GPIO.setup(PIN_CS,   _GPIO.OUT, initial=_GPIO.HIGH)
        _GPIO.setup(PIN_DRDY, _GPIO.IN)

        self._stop_flag = False
        self._closed = False
        self._resampler = StreamingPolyphase(self.resample_p, self.resample_q)

        if self.verbose:
            print("[ads1256] SPI {} Hz mode 1, DRATE 0x{:02X} ({} SPS), output {} Hz".format(
                spi_clock_hz, drate, self.nominal_sps, self.output_rate,
            ))

        self._init_chip()

    # ---------- low-level helpers ----------

    def _cs(self, level):
        # type: (int) -> None
        self._GPIO.output(PIN_CS, level)

    def _write_cmd(self, cmd):
        # type: (int) -> None
        self._cs(0)
        self._spi.xfer2([cmd])
        self._cs(1)

    def _write_reg(self, reg, value):
        # type: (int, int) -> None
        self._cs(0)
        self._spi.xfer2([CMD_WREG | reg, 0x00, value])
        self._cs(1)

    def _read_reg(self, reg):
        # type: (int) -> int
        self._cs(0)
        self._spi.xfer2([CMD_RREG | reg, 0x00])
        # ADS1256 datasheet: 4 t_clkin (~6.5 us at 7.68 MHz fCLKIN) before
        # device responds. spidev framing usually covers this; add a small
        # delay to be safe at higher SPI clocks.
        time.sleep(10e-6)
        result = self._spi.xfer2([0x00])[0]
        self._cs(1)
        return result

    def _wait_drdy(self):
        # type: () -> float
        """Busy-poll DRDY low. Returns wait time in seconds.

        Tight inner loop; only checks wall clock every ~1000 iterations
        to avoid syscall overhead in the hot path.
        """
        t0 = time.perf_counter()
        deadline = t0 + self.drdy_timeout_s
        gpio_input = self._GPIO.input
        pin = PIN_DRDY
        i = 0
        while gpio_input(pin) != 0:
            i += 1
            if i >= 1000:
                i = 0
                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        "DRDY timeout after {:.3f} ms (DRATE=0x{:02X}, "
                        "expected period {:.1f} us)".format(
                            self.drdy_timeout_s * 1000, self.drate,
                            1e6 / self.nominal_sps,
                        )
                    )
        return time.perf_counter() - t0

    @staticmethod
    def _parse_24bit(b0, b1, b2):
        # type: (int, int, int) -> int
        v = (b0 << 16) | (b1 << 8) | b2
        if v & 0x800000:
            v -= 0x1000000
        return v

    # ---------- init / config ----------

    def _init_chip(self):
        # Hardware reset pulse on RST pin
        self._GPIO.output(PIN_RST, 0)
        time.sleep(0.001)
        self._GPIO.output(PIN_RST, 1)
        time.sleep(0.005)

        # Software reset
        self._write_cmd(CMD_RESET)
        time.sleep(0.005)
        # After RESET, the chip auto-calibrates and DRDY pulses when ready
        self._wait_drdy()

        # Make sure we are NOT in RDATAC before doing register I/O
        self._write_cmd(CMD_SDATAC)
        time.sleep(100e-6)

        # Verify chip ID (high nibble of STATUS register == 0x03)
        status = self._read_reg(REG_STATUS)
        chip_id = (status >> 4) & 0x0F
        if chip_id != 0x03:
            raise RuntimeError(
                "ADS1256 chip ID mismatch: expected 0x03, got 0x{:02X} "
                "(STATUS=0x{:02X}). Check SPI wiring / vref / power.".format(
                    chip_id, status,
                )
            )
        if self.verbose:
            print("[ads1256] chip ID OK (STATUS=0x{:02X})".format(status))

        # Write STATUS / MUX / ADCON / DRATE in one WREG transaction (count=3)
        # STATUS = 0x04: ORDER=0 (MSB first), ACAL=1 (auto-cal on register changes), BUFEN=0
        # MUX    = 0x08: AIN0 / AINCOM (single-ended channel 0)
        # ADCON  = gain << 0: clock-out off, sensor-detect off, gain code in bits 2:0
        # DRATE  = self.drate
        status_val = (0 << 3) | (1 << 2) | (0 << 1)
        mux_val    = 0x08
        adcon_val  = (0 << 5) | (0 << 3) | (self.gain & 0x07)
        self._cs(0)
        self._spi.xfer2([CMD_WREG | REG_STATUS, 0x03,
                         status_val, mux_val, adcon_val, self.drate])
        self._cs(1)
        time.sleep(0.001)

        # Self-calibrate (offset + gain) — duration depends on DRATE
        self._write_cmd(CMD_SELFCAL)
        self._wait_drdy()

        # Enter continuous read mode — subsequent reads are 3 bytes only
        self._write_cmd(CMD_RDATAC)
        time.sleep(10e-6)
        if self.verbose:
            print("[ads1256] init complete, RDATAC armed at DRATE 0x{:02X}".format(
                self.drate,
            ))

    # ---------- public API ----------

    def stop(self):
        self._stop_flag = True

    def close(self):
        if self._closed:
            return
        try:
            self._write_cmd(CMD_SDATAC)
        except Exception:
            pass
        try:
            self._spi.close()
        except Exception:
            pass
        try:
            self._GPIO.cleanup([PIN_RST, PIN_CS, PIN_DRDY])
        except Exception:
            pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def chunks_raw(self):
        # type: () -> Iterator[Tuple[np.ndarray, dict]]
        """Yield (chunk, metrics) at native data rate, no resampling.

        chunk: shape (chunk_in_size,) float64, samples in volts (scaled
            by self.scale). Final chunk may be shorter on stop().
        metrics: dict with arrays of per-sample timings (microseconds):
            drdy_wait_us, spi_xfer_us, inter_sample_us
            and scalar fields: t_first, t_last, n_samples.
        """
        N = self.chunk_in_size
        buf = np.empty(N, dtype=np.float64)
        drdy_us = np.empty(N, dtype=np.float64)
        xfer_us = np.empty(N, dtype=np.float64)
        inter_us = np.empty(N, dtype=np.float64)

        t_prev_sample = None
        i = 0
        chunk_t_first = None

        # Local refs in tight loop
        wait_drdy = self._wait_drdy
        spi_xfer = self._spi.xfer2
        cs = self._cs
        parse = self._parse_24bit
        scale = self.scale
        perf = time.perf_counter

        while not self._stop_flag:
            t_drdy_start = perf()
            wait_drdy()
            t_drdy_done = perf()
            cs(0)
            t_xfer_start = perf()
            b = spi_xfer([0, 0, 0])
            t_xfer_done = perf()
            cs(1)
            value = parse(b[0], b[1], b[2]) * scale

            buf[i] = value
            drdy_us[i] = (t_drdy_done - t_drdy_start) * 1e6
            xfer_us[i] = (t_xfer_done - t_xfer_start) * 1e6
            if t_prev_sample is None:
                inter_us[i] = 0.0
                chunk_t_first = t_xfer_done
            else:
                inter_us[i] = (t_xfer_done - t_prev_sample) * 1e6
            t_prev_sample = t_xfer_done

            i += 1
            if i >= N:
                metrics = {
                    "n_samples": N,
                    "t_first": chunk_t_first,
                    "t_last": t_xfer_done,
                    "drdy_wait_us": drdy_us.copy(),
                    "spi_xfer_us": xfer_us.copy(),
                    "inter_sample_us": inter_us.copy(),
                }
                yield buf.copy(), metrics
                i = 0
                chunk_t_first = None

        # Final partial chunk on stop()
        if i > 0:
            metrics = {
                "n_samples": i,
                "t_first": chunk_t_first,
                "t_last": t_prev_sample,
                "drdy_wait_us": drdy_us[:i].copy(),
                "spi_xfer_us": xfer_us[:i].copy(),
                "inter_sample_us": inter_us[:i].copy(),
            }
            yield buf[:i].copy(), metrics

    def chunks(self):
        # type: () -> Iterator[np.ndarray]
        """Yield resampled chunks at output_rate (default 8000 Hz).

        Drop-in for MatFileSource.chunks(). Polyphase upsample runs
        per chunk; chunk size at output is roughly chunk_in_size *
        resample_p / resample_q (slight variation due to phase).
        """
        for chunk_in, _metrics in self.chunks_raw():
            out = self._resampler.feed(chunk_in)
            if len(out):
                yield out

    def info(self):
        # type: () -> str
        return ("ADS1256: DRATE 0x{:02X} ({} SPS) -> {:.0f} Hz output "
                "(p={}, q={}), SPI {} Hz, scale {:.3e} V/LSB").format(
            self.drate, self.nominal_sps, self.output_rate,
            self.resample_p, self.resample_q,
            self._spi.max_speed_hz, self.scale,
        )
