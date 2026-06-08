# -*- coding: utf-8 -*-
"""bench_jetson_latency.py — Jetson 단말 per-footstep end-to-end 지연 실측 (센서 없이 replay).

센서 체인(geophone/ADS1256/STM32) 해체 후, **내가 수집한 녹음**
(`data/recordings/<pid>/<pid>_s<session>_*.npz`, key `raw_int32` @ 15 kHz)을
라이브와 동일한 코드 경로로 흘려 Jetson에서 단계별 지연을 잰다.
라이브 시연과 유일한 차이 = live sensor 대신 녹음 입력 (compute path 동일).

측정 단계 (footstep 1개당):
    CWT        cwt_fast.cwt
    render     render_lut.cwt_to_rgb_direct   (LUT + cv2.resize)
    normalize  to_input                        (ImageNet 정규화)
    TRT        TRTEngine.infer                  (H2D + execute + D2H + sync)
    --------------------------------------------------------------
    TOTAL / footstep  (+ envelope 추출은 윈도우당 1회 → footstep당 amortize 별도 보고)

목적: 기말 덱의 `≈250 ms` **추정치**를 Jetson 실측으로 대체 (CLAUDE.md latency 섹션 참조).
마지막 실측은 최종 CWT 벡터화 *이전* 값(~400 ms, CLAUDE_ARCHIVE) 뿐이라 재측정이 필요함.

런타임 (Jetson ~/terra/, JetPack 4.6.6 / py3.6 / TRT 8.2.1 / pycuda):
    python3 bench_jetson_latency.py --pid 김건형 --session 1 \
        --plan mnv3_family_v1_fp16.plan
또는 파일 직접 지정:
    python3 bench_jetson_latency.py --rec data/recordings/김건형/김건형_s1_XXXX.npz \
        --plan mnv3_family_v1_fp16.plan

녹음이 Jetson에 없으면 먼저 scp (CLAUDE.md 런북 참조).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# replay_recording.py 와 동일 import 경로 (Jetson 검증됨)
from resample_for_ads1256 import StreamingPolyphase  # noqa: E402
from extract import apply_envelope_detector          # noqa: E402
from cwt_fast import cwt as fast_cwt                  # noqa: E402
from render_lut import cwt_to_rgb_direct              # noqa: E402
from jetson_infer import TRTEngine, to_input          # noqa: E402

try:
    from stm32_source import CALIBRATED_SCALE
except Exception:                       # pyserial 미설치 등 — frozen 상수로 폴백 (S.5)
    CALIBRATED_SCALE = 3.7253e-7        # V/LSB


def med(a):
    return float(np.median(a) * 1000.0)


def p90(a):
    return float(np.percentile(a, 90) * 1000.0)


def find_rec(recordings, pid, session):
    pat = os.path.join(recordings, pid, "{}_s{}_*.npz".format(pid, session))
    cands = [c for c in sorted(glob.glob(pat)) if not c.endswith(".bak")]
    return cands[-1] if cands else None


def load_geo8k(rec_path, offset_s, dur_s, fs_in=15000):
    d = np.load(rec_path, allow_pickle=True)
    raw = d["raw_int32"].astype(np.float64)
    i0 = int(offset_s * fs_in)
    i1 = len(raw) if not dur_s else int((offset_s + dur_s) * fs_in)
    seg = raw[i0:i1] * CALIBRATED_SCALE           # int24 counts -> volts
    rs = StreamingPolyphase(p=8, q=15)            # 15 kHz -> 8 kHz (라이브 동일)
    return rs.feed(seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=None, help="녹음 .npz 직접 지정 (raw_int32 @15kHz)")
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--pid", default="김건형", help="--rec 없을 때 탐색할 pid 폴더명")
    ap.add_argument("--session", default="1")
    ap.add_argument("--offset", type=float, default=30.0, help="녹음 내 시작(s)")
    ap.add_argument("--dur", type=float, default=120.0, help="윈도우 길이(s), 0=끝까지")
    ap.add_argument("--plan", default=os.environ.get("TERRA_TRT_PLAN", "mnv3_family_v1_fp16.plan"))
    ap.add_argument("--max-footsteps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=5, help="TRT 콜드스타트 워밍업 횟수")
    args = ap.parse_args()

    rec = args.rec or find_rec(args.recordings, args.pid, args.session)
    if not rec or not os.path.exists(rec):
        print("ERROR: 녹음을 찾지 못함. --rec PATH 또는 --pid/--session 확인.")
        avail = sorted(glob.glob(os.path.join(args.recordings, "*", "*.npz")))
        if avail:
            print("사용 가능한 녹음 (예):")
            for a in avail[:10]:
                print("  ", a)
        sys.exit(1)

    print("=" * 64)
    print("Jetson per-footstep latency  (replay, 센서 없음)")
    print("  recording :", rec)
    print("  plan      :", args.plan)
    print("  window    : [{:.0f}s, +{:.0f}s]".format(args.offset, args.dur))
    print("=" * 64)

    t0 = time.perf_counter()
    geo8k = load_geo8k(rec, args.offset, args.dur)
    t_load = time.perf_counter() - t0
    print("geo @8kHz : {} samples = {:.1f}s  (load+resample {:.0f} ms)".format(
        geo8k.size, geo8k.size / 8000.0, t_load * 1000))

    t0 = time.perf_counter()
    footsteps = apply_envelope_detector(geo8k)
    t_extract = time.perf_counter() - t0
    n = len(footsteps)
    if n == 0:
        print("ERROR: 0 footsteps — --offset/--dur 조정 (조용한 구간일 수 있음).")
        sys.exit(1)
    ext_amort = 1000.0 * t_extract / n
    print("envelope  : {} footsteps in {:.0f} ms  ({:.2f} ms/footstep amortized)".format(
        n, t_extract * 1000, ext_amort))

    M = min(n, args.max_footsteps)
    print("TRT engine load + warmup {}x ...".format(args.warmup))
    engine = TRTEngine(args.plan)
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    for _ in range(args.warmup):
        engine.infer(dummy)

    cwt_t, ren_t, nrm_t, trt_t, tot_t = [], [], [], [], []
    for i in range(M):
        sig = footsteps[i]
        s = time.perf_counter()
        coeffs = fast_cwt(sig)
        t1 = time.perf_counter()
        img = cwt_to_rgb_direct(coeffs, (224, 224))
        t2 = time.perf_counter()
        x = to_input(img)
        t3 = time.perf_counter()
        engine.infer(x)
        t4 = time.perf_counter()
        cwt_t.append(t1 - s); ren_t.append(t2 - t1); nrm_t.append(t3 - t2)
        trt_t.append(t4 - t3); tot_t.append(t4 - s)

    def line(name, a):
        print("  {:24s}: median {:6.1f} ms | p90 {:6.1f} ms".format(name, med(a), p90(a)))

    print("\nper-footstep over {} footsteps (Jetson, TRT FP16):".format(M))
    line("CWT (cwt_fast)", cwt_t)
    line("render (LUT + resize)", ren_t)
    line("normalize", nrm_t)
    line("TRT infer", trt_t)
    print("  " + "-" * 50)
    line("compute TOTAL / footstep", tot_t)
    print("  {:24s}: {:6.2f} ms (window당 1회 분할)".format("envelope (amortized)", ext_amort))
    e2e = med(tot_t) + ext_amort
    print("  " + "-" * 50)
    print("  {:24s}: {:6.1f} ms".format("END-TO-END / footstep", e2e))
    print("\n  보행 간격 640~970 ms 대비 여유: {:.1f}x ~ {:.1f}x".format(640.0 / e2e, 970.0 / e2e))
    print("\n  ※ 이 값으로 기말/최종 덱의 ≈250 ms(추정)를 대체. 슬라이드 19 + CLAUDE.md 갱신.")


if __name__ == "__main__":
    main()
