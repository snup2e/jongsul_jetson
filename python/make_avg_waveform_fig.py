# -*- coding: utf-8 -*-
"""평균 발걸음 파형 figure (신호지문 파트 도입 — CWT 들어가기 전 직관용).

(a) 전체 등록 사용자 grand-average 파형 (peak 정렬, ±1σ 밴드 + Hilbert 포락선)
(b) 사용자별 평균 파형 overlay  → "사람마다 모양/세기가 다르다" → CWT 동기

출력: 기말발표/figures/fig_avg_waveform.png
"""
import glob
import os
import sys

import numpy as np
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads1256_source import StreamingPolyphase
from extract import matlab_smooth

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.dpi"] = 200

FS = 8000
PEAK_OFF = 400          # peak 가 crop 의 sample 400 에 정렬
LEN = 1500
INK = "#191F28"; GRAY = "#8B95A1"; GRAYL = "#E5E8EB"

# (실명 → 발표용 익명 P1~P4, EDA figure 와 색 통일)
PEOPLE = [("김건형", "P1", "#3182F6"),
          ("김범수", "P2", "#15B86B"),
          ("성재용", "P3", "#FF9F43"),
          ("박지훈", "P4", "#A055FF")]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "data", "recordings")
OUT = os.path.join(ROOT, "기말발표", "figures")
os.makedirs(OUT, exist_ok=True)


def load_8k(path):
    d = np.load(path)
    sig = d["raw_int32"].astype(np.float64) * float(d["scale_v_per_lsb"])
    if int(d["fs_in"]) == 8000:
        return sig
    return StreamingPolyphase(p=8, q=15).feed(sig)


def raw_crops(sig, k_mad=8.0):
    """peak 정렬 RAW crop (tukey 없음). 가장자리 crop 은 평균 왜곡 방지로 제외."""
    g = matlab_smooth(sig, span=5).ravel(); L = len(g)
    env = np.abs(ss.hilbert(g))
    n = max(1, int(0.030 * FS))
    env = np.convolve(env, np.ones(n) / n, mode="same")
    med = float(np.median(env)); mad = float(np.median(np.abs(env - med)))
    thr = med + k_mad * mad * 1.4826
    peaks, _ = ss.find_peaks(env, height=thr, distance=max(1, int(0.120 * FS)))
    radius = max(1, int(0.030 * FS)); ag = np.abs(g)
    out = []
    for p in peaks:
        lo = max(0, p - radius); hi = min(L, p + radius)
        mn = int(np.argmax(ag[lo:hi])) + lo
        s = mn - PEAK_OFF; e = s + LEN
        if s < 0 or e > L:
            continue
        out.append(g[s:e])
    return np.asarray(out)


def collect(pid):
    files = sorted(glob.glob(os.path.join(REC, pid, "{}_s*.npz".format(pid))))
    cs = [raw_crops(load_8k(f)) for f in files]
    cs = [c for c in cs if len(c)]
    return np.concatenate(cs, axis=0) if cs else np.empty((0, LEN))


# ---- 수집 ----
per = {}
for name, lab, col in PEOPLE:
    C = collect(name)
    per[lab] = C
    print("{} ({}): {} crops".format(lab, name, len(C)))

# 공통 시간축: peak = 0 ms
t = (np.arange(LEN) - PEAK_OFF) / FS * 1000.0   # ms

# 전체 결합 (signed — 극성 일관)
allC = np.concatenate([per[l] for _, l, _ in PEOPLE], axis=0) * 1000.0  # mV
mu = allC.mean(0)
sd = allC.std(0)
env_mu = np.abs(ss.hilbert(allC, axis=1)).mean(0)   # allC already in mV

# ---- plot ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.2))

# (a) grand average
axA.fill_between(t, mu - sd, mu + sd, color="#3182F6", alpha=0.13,
                 lw=0, label="±1σ (개별 발걸음 분산)")
axA.plot(t, mu, color="#1B64DA", lw=2.2, label="평균 파형 (signed)")
axA.plot(t, env_mu, color="#F0506E", lw=1.6, ls="--", label="평균 포락선 (Hilbert)")
axA.plot(t, -env_mu, color="#F0506E", lw=1.6, ls="--")
axA.axvline(0, color=GRAY, lw=0.8, ls=":")
axA.axhline(0, color=GRAYL, lw=0.8)
axA.set_xlim(-30, 120)
axA.set_xlabel("시간 (ms, peak = 0)", fontsize=11.5)
axA.set_ylabel("지오폰 전압 (mV)", fontsize=11.5)
axA.set_title("(a) 전체 평균 발걸음 파형  (N = {:,} steps)".format(len(allC)),
              fontsize=13, fontweight="bold")
axA.legend(fontsize=10, framealpha=0.92, loc="upper right")
axA.grid(color=GRAYL, lw=0.7); axA.set_axisbelow(True)
axA.text(0.015, 0.04,
         "임펄스 → 감쇠 진동\n(1m 이내, 990Ω 댐핑)",
         transform=axA.transAxes, fontsize=9.5, color=GRAY, va="bottom")

# (b) per-person overlay (signed — 4명 모두 음극성으로 일관, 모양/세기 비교)
for name, lab, col in PEOPLE:
    C = per[lab] * 1000.0
    m = C.mean(0)
    axB.plot(t, m, color=col, lw=2.0,
             label="{}  (n={:,})".format(lab, len(C)))
axB.axvline(0, color=GRAY, lw=0.8, ls=":")
axB.axhline(0, color=GRAYL, lw=0.8)
axB.set_xlim(-30, 120)
axB.set_xlabel("시간 (ms, peak = 0)", fontsize=11.5)
axB.set_ylabel("지오폰 전압 (mV)", fontsize=11.5)
axB.set_title("(b) 등록 사용자별 평균 파형", fontsize=13, fontweight="bold")
axB.legend(fontsize=10, framealpha=0.92, loc="upper right")
axB.grid(color=GRAYL, lw=0.7); axB.set_axisbelow(True)

fig.suptitle("발걸음 = 임펄스성 지진파 — 시간영역 평균  (8 kHz, calibrated mV)",
             fontsize=14.5, fontweight="bold", y=1.01)
fig.tight_layout()
out = os.path.join(OUT, "fig_avg_waveform.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.2)
print("saved ->", out)
