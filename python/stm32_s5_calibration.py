"""S.5 calibration - STM32 raw 30초 캡처 (15 kHz LSB) vs P14_1.mat 첫 30초 (8 kHz V) 비교.

목적:
    1) STM32 -> polyphase 8 kHz -> P14 와 동일 도메인에서 RMS/peak 비교
    2) LSB -> V scale factor 결정 (ML 모델 입력 분포 일치용)
    3) 노이즈 floor / 신호 수준 sanity check

사용법:
    python python/stm32_s5_calibration.py
    python python/stm32_s5_calibration.py --stm32 data/stm32_30s_raw.npy --p14 data/raw/P14/P14_1.mat
"""
import argparse
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase


def stats(name: str, x: np.ndarray, fs: int) -> dict:
    ac = x - x.mean()
    rms = float(np.sqrt(np.mean(ac ** 2)))
    peak = float(np.abs(ac).max())
    p50 = float(np.percentile(np.abs(ac), 50))
    p95 = float(np.percentile(np.abs(ac), 95))
    p99 = float(np.percentile(np.abs(ac), 99))
    print(f"\n{name}  ({len(x)} samples @ {fs} Hz = {len(x)/fs:.2f} s)")
    print(f"  mean (DC)  = {x.mean():.4g}")
    print(f"  RMS  (AC)  = {rms:.4g}")
    print(f"  |x| peak   = {peak:.4g}")
    print(f"  |x| p50    = {p50:.4g}")
    print(f"  |x| p95    = {p95:.4g}")
    print(f"  |x| p99    = {p99:.4g}")
    return {"rms": rms, "peak": peak, "p50": p50, "p95": p95, "p99": p99}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stm32", default="data/stm32_30s_raw.npy")
    ap.add_argument("--p14",   default="data/raw/P14/P14_1.mat")
    ap.add_argument("--p14-key", default="geo_data")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    # === STM32 raw 15 kHz -> polyphase 8 kHz ===
    stm32_raw = np.load(args.stm32).astype(np.float64)
    print(f"[Load] STM32 raw: {args.stm32}  ({len(stm32_raw)} samples)")

    poly = StreamingPolyphase(p=8, q=15)
    stm32_8k = poly.feed(stm32_raw)
    # 가능하면 같은 길이로 자르기
    n_target = int(args.seconds * 8000)
    if len(stm32_8k) > n_target:
        stm32_8k = stm32_8k[:n_target]

    # === P14 첫 30초 ===
    mat = sio.loadmat(args.p14)
    p14 = np.asarray(mat[args.p14_key]).flatten().astype(np.float64)
    print(f"[Load] P14: {args.p14}  ({len(p14)} samples = {len(p14)/8000:.1f} s)")
    p14_30 = p14[:n_target]

    # === 통계 ===
    s_stm = stats("STM32 8kHz (LSB)", stm32_8k, 8000)
    s_p14 = stats("P14 first 30s (V)", p14_30, 8000)

    # === Scale 후보 ===
    print("\n" + "=" * 60)
    print("Scale candidates (V/LSB) - LSB 값에 곱해서 V 단위로 환산")
    print("=" * 60)
    for metric in ("rms", "peak", "p95", "p99"):
        if s_stm[metric] > 0:
            scale = s_p14[metric] / s_stm[metric]
            print(f"  by {metric:5s}: {scale:.4e}")
    print()

    # === 진단 ===
    print("[진단]")
    # ADS1256 PGA=16: 1 LSB = 312.5 mV / 8388608 = 3.726e-8 V
    lsb_v_pga16 = 0.3125 / 8388608.0
    print(f"  ADS1256 PGA=16 nominal V/LSB = {lsb_v_pga16:.4e}  (외부 amp 없음 가정)")

    rms_ratio = s_p14["rms"] / s_stm["rms"] if s_stm["rms"] > 0 else 0
    print(f"  P14/STM32 RMS 비율          = {rms_ratio:.2f}x")
    print(f"  (VIBeID: 외부 op-amp gain 10 거친 후 sound card. 우리는 PGA=16 만)")

    # 활동 추정
    if s_stm["p99"] / s_stm["rms"] > 5:
        print("  [OK] STM32 캡처에 발걸음/임팩트 활동 있음 (p99/rms > 5)")
    else:
        print("  [!] STM32 캡처가 거의 idle 만. 발걸음 캡처 후 재실행 권장:")
        print(f"    python python/stm32_source.py --raw --duration 30 --save {args.stm32}")
        print("    캡처 동안 1m 거리에서 보통 속도로 걸어다니기")

    if s_p14["p99"] / s_p14["rms"] > 5:
        print("  [OK] P14 첫 30초에도 활동 있음")
    else:
        print("  [!] P14 첫 30초가 idle 일 수 있음 - 다른 구간으로 비교 검토")


if __name__ == "__main__":
    main()
