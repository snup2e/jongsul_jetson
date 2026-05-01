"""ads1256_bench.py: SPI throughput verification ladder for Jetson Nano.

Runs ADS1256Source at one or more DRATE codes, captures per-sample
timing metrics for `duration` seconds each, prints a summary table,
and applies the Plan B fallback trigger documented in CLAUDE.md.

Default ladder: 1000 -> 3750 -> 7500 SPS, 60 s each.

Per stage we report:
  - actual_sps        (count / wall-clock seconds)
  - missed_samples    (inter_sample_us > 1.5 * expected_period)
  - jitter_warnings   (1.2 * expected < inter_sample_us <= 1.5 * expected)
  - drdy_wait_us      p50 / p99 / max
  - spi_xfer_us       p50 / p99 / max
  - inter_sample_us   p50 / p99 / max

Plan B trigger (any one fires recommendation to fall back to 3750):
  T1: missed_samples > 0
  T2: |actual_sps - nominal| / nominal > 0.005
  T3: inter_sample_us p99 > 1.3 * expected_period_us

Run on Jetson Nano:
    python3 ads1256_bench.py
    python3 ads1256_bench.py --rates 7500 --duration 120
    python3 ads1256_bench.py --rates 1000 3750 7500 --duration 60 \
        --output bench_$(date +%Y%m%d_%H%M).npz

NOT runnable on Windows (requires Jetson.GPIO + spidev).
"""
import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import (
    ADS1256Source,
    DRATE_BY_NAME,
    DRATE_NOMINAL_SPS,
    DRATE_7500,
    DRATE_3750,
    DRATE_1000,
)


# ============================================================
# Stage runner
# ============================================================

def run_stage(drate, duration_s, chunk_in_size=512, spi_clock_hz=1_000_000,
              verbose=False):
    # type: (int, float, int, int, bool) -> Dict
    """Run ADS1256 at given DRATE for duration_s seconds, collect metrics."""
    nominal_sps = DRATE_NOMINAL_SPS[drate]
    expected_period_us = 1e6 / nominal_sps

    print("\n" + "=" * 60)
    print("STAGE  DRATE=0x{:02X}  nominal {} SPS  duration {:.0f}s".format(
        drate, nominal_sps, duration_s,
    ))
    print("=" * 60)

    # All per-sample arrays
    drdy_us_all = []
    xfer_us_all = []
    inter_us_all = []
    n_samples = 0

    src = ADS1256Source(
        drate=drate,
        chunk_in_size=chunk_in_size,
        spi_clock_hz=spi_clock_hz,
        verbose=verbose,
    )
    print("[bench] " + src.info())

    # Warm-up: discard first chunk so init transients (calibration tail,
    # cold caches) don't bias the metrics.
    chunks_iter = src.chunks_raw()
    try:
        warm_chunk, _ = next(chunks_iter)
    except StopIteration:
        warm_chunk = None
    print("[bench] warm-up: discarded {} samples".format(
        len(warm_chunk) if warm_chunk is not None else 0,
    ))

    t_start = time.perf_counter()
    deadline = t_start + duration_s
    t_first_sample = None
    t_last_sample = None

    try:
        for chunk, m in chunks_iter:
            if t_first_sample is None:
                t_first_sample = m["t_first"]
            t_last_sample = m["t_last"]
            n_samples += m["n_samples"]
            drdy_us_all.append(m["drdy_wait_us"])
            xfer_us_all.append(m["spi_xfer_us"])
            # Skip the first inter_sample of each chunk (= 0 placeholder
            # for the per-chunk first sample); take only true diffs.
            if m["n_samples"] > 1:
                inter_us_all.append(m["inter_sample_us"][1:])
            if time.perf_counter() >= deadline:
                src.stop()
    except KeyboardInterrupt:
        print("\n[bench] interrupted by user")
        src.stop()
    finally:
        src.close()

    elapsed = (t_last_sample or t_start) - (t_first_sample or t_start)
    if elapsed <= 0:
        elapsed = max(time.perf_counter() - t_start, 1e-9)

    drdy = np.concatenate(drdy_us_all) if drdy_us_all else np.array([])
    xfer = np.concatenate(xfer_us_all) if xfer_us_all else np.array([])
    inter = np.concatenate(inter_us_all) if inter_us_all else np.array([])

    actual_sps = n_samples / elapsed
    sps_dev = abs(actual_sps - nominal_sps) / nominal_sps

    # Inter-sample anomaly counts
    missed = int(np.sum(inter > 1.5 * expected_period_us)) if len(inter) else 0
    jitter = int(np.sum((inter > 1.2 * expected_period_us) &
                        (inter <= 1.5 * expected_period_us))) if len(inter) else 0

    def pct(arr, p):
        return float(np.percentile(arr, p)) if len(arr) else float("nan")

    result = {
        "drate":           drate,
        "nominal_sps":     nominal_sps,
        "expected_us":     expected_period_us,
        "duration_s":      elapsed,
        "n_samples":       n_samples,
        "actual_sps":      actual_sps,
        "sps_dev":         sps_dev,
        "missed_samples":  missed,
        "jitter_warnings": jitter,
        "drdy_p50":        pct(drdy, 50),
        "drdy_p99":        pct(drdy, 99),
        "drdy_max":        float(drdy.max()) if len(drdy) else float("nan"),
        "xfer_p50":        pct(xfer, 50),
        "xfer_p99":        pct(xfer, 99),
        "xfer_max":        float(xfer.max()) if len(xfer) else float("nan"),
        "inter_p50":       pct(inter, 50),
        "inter_p99":       pct(inter, 99),
        "inter_max":       float(inter.max()) if len(inter) else float("nan"),
        "_drdy":           drdy,
        "_xfer":           xfer,
        "_inter":          inter,
    }
    _print_stage_summary(result)
    return result


def _print_stage_summary(r):
    print("\n[result]  DRATE=0x{:02X} ({} SPS), {:.1f}s, {} samples".format(
        r["drate"], r["nominal_sps"], r["duration_s"], r["n_samples"],
    ))
    print("  actual_sps     = {:.2f}  (nominal {}, dev {:+.3f}%)".format(
        r["actual_sps"], r["nominal_sps"], r["sps_dev"] * 100 *
        np.sign(r["actual_sps"] - r["nominal_sps"]),
    ))
    print("  missed_samples = {}  (inter > 1.5 * {:.1f}us)".format(
        r["missed_samples"], r["expected_us"],
    ))
    print("  jitter_warnings= {}  (1.2 * exp < inter <= 1.5 * exp)".format(
        r["jitter_warnings"],
    ))
    print("  drdy_wait_us   p50={:.1f}  p99={:.1f}  max={:.1f}".format(
        r["drdy_p50"], r["drdy_p99"], r["drdy_max"],
    ))
    print("  spi_xfer_us    p50={:.1f}  p99={:.1f}  max={:.1f}".format(
        r["xfer_p50"], r["xfer_p99"], r["xfer_max"],
    ))
    print("  inter_sample_us p50={:.1f}  p99={:.1f}  max={:.1f}  (target {:.1f})".format(
        r["inter_p50"], r["inter_p99"], r["inter_max"], r["expected_us"],
    ))


# ============================================================
# Plan B trigger
# ============================================================

def evaluate_plan_b(result):
    # type: (Dict) -> Tuple[bool, List[str]]
    """Return (fallback_triggered, reasons[]) per CLAUDE.md D5."""
    reasons = []
    if result["missed_samples"] > 0:
        reasons.append("T1: missed_samples = {} > 0".format(result["missed_samples"]))
    if result["sps_dev"] > 0.005:
        reasons.append("T2: sps_dev = {:.3f}% > 0.5%".format(result["sps_dev"] * 100))
    if (not np.isnan(result["inter_p99"]) and
            result["inter_p99"] > 1.3 * result["expected_us"]):
        reasons.append("T3: inter p99 = {:.1f}us > 1.3 * {:.1f}us".format(
            result["inter_p99"], result["expected_us"],
        ))
    return (len(reasons) > 0), reasons


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", default=["1000", "3750", "7500"],
                    help="DRATE names to run in sequence (default: 1000 3750 7500)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Seconds per stage (default 60)")
    ap.add_argument("--chunk-in-size", type=int, default=512,
                    help="Native-rate chunk size (default 512)")
    ap.add_argument("--spi-clock", type=int, default=1_000_000,
                    help="SPI clock Hz (default 1e6)")
    ap.add_argument("--output",
                    help="Save raw arrays to .npz (one stage per key)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    drates = []
    for name in args.rates:
        if name not in DRATE_BY_NAME:
            print("Unknown rate name: {!r}. Choices: {}".format(
                name, list(DRATE_BY_NAME.keys()),
            ))
            sys.exit(2)
        drates.append(DRATE_BY_NAME[name])

    print("[bench] ladder: {}  duration {:.0f}s/stage  spi_clock {} Hz".format(
        " -> ".join(args.rates), args.duration, args.spi_clock,
    ))

    results = []
    for drate in drates:
        try:
            r = run_stage(
                drate, args.duration,
                chunk_in_size=args.chunk_in_size,
                spi_clock_hz=args.spi_clock,
                verbose=args.verbose,
            )
            results.append(r)
        except Exception as e:
            print("\n[bench] STAGE FAILED at DRATE 0x{:02X}: {}".format(drate, e))
            import traceback
            traceback.print_exc()
            results.append({"drate": drate, "error": str(e)})

    # Final summary table
    print("\n" + "=" * 60)
    print("LADDER SUMMARY")
    print("=" * 60)
    print("{:>8s} {:>10s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "rate", "actual_sps", "missed", "jitter", "inter_p99", "verdict",
    ))
    fallback_chosen = None
    for r in results:
        if "error" in r:
            print("{:>8d} {:>10s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
                r["drate"], "ERROR", "-", "-", "-", "FAIL",
            ))
            continue
        triggered, reasons = evaluate_plan_b(r)
        verdict = "FALLBACK" if triggered else "OK"
        print("{:>8d} {:>10.1f} {:>9d} {:>9d} {:>9.1f} {:>9s}".format(
            r["nominal_sps"], r["actual_sps"], r["missed_samples"],
            r["jitter_warnings"], r["inter_p99"] if not np.isnan(r["inter_p99"]) else 0,
            verdict,
        ))
        if r["drate"] == DRATE_7500 and triggered:
            fallback_chosen = DRATE_3750
            print("\n[plan-b]  7500 SPS triggered fallback. Reasons:")
            for reason in reasons:
                print("  - {}".format(reason))
            print("[plan-b]  RECOMMENDATION: production rate -> 3750 SPS")

    if any("error" not in r for r in results):
        # Find best stable rate
        ok_rates = [r["nominal_sps"] for r in results
                    if "error" not in r and not evaluate_plan_b(r)[0]]
        if ok_rates:
            best = max(ok_rates)
            print("\n[plan-b]  highest stable rate observed: {} SPS".format(best))
            if best == 7500:
                print("[plan-b]  PASS — production rate stays at 7500 SPS")
            elif best == 3750:
                print("[plan-b]  PASS at 3750 SPS — use this in production")
            else:
                print("[plan-b]  WARN  no rate at or above 3750 was stable")
        else:
            print("\n[plan-b]  WARN  no stage passed the trigger thresholds")

    if args.output:
        out = {}
        for r in results:
            if "error" in r:
                continue
            tag = "rate_{}".format(r["nominal_sps"])
            for key in ("drdy", "xfer", "inter"):
                arr = r.get("_" + key)
                if arr is not None:
                    out["{}_{}_us".format(tag, key)] = arr
            out["{}_meta".format(tag)] = np.array([
                r["drate"], r["nominal_sps"], r["duration_s"], r["n_samples"],
                r["actual_sps"], r["missed_samples"], r["jitter_warnings"],
            ], dtype=np.float64)
        np.savez(args.output, **out)
        print("\n[output] saved {} arrays to {}".format(len(out), args.output))


if __name__ == "__main__":
    main()
