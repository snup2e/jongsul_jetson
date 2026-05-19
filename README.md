# Terra — Footstep Recognition Edge Deployment

Real-time, person-identifying footstep recognition for an edge device (NVIDIA Jetson Nano), trained on the [VIBeID](https://arxiv.org/abs/2306.14640) structural-vibration dataset and adaptable to a household via transfer learning.

This repository covers the full stack: MATLAB → Python pipeline port, model training (MobileNet V3-Large on CWT spectrograms), STM32-based geophone bridge firmware, Jetson real-time inference with TensorRT FP16, and a Toss-style web dashboard.

---

## Pipeline

```
Geophone (SM-24, 28.8 V/m/s)
  └─ 990 Ω damping shunt (≈ 1 kΩ)
       │
       ▼
ADS1256 24-bit ADC  (PGA = 16, 15 kSPS, BUFEN = 0)
       │  SPI 1.31 MHz
       ▼
STM32 NUCLEO-F411RE  (RDATAC, EXTI on DRDY, ring buffer)
       │  USART2 921 600 8-N-1 over ST-Link VCP
       │  binary frame: [SYNC | SEQ u16 | N=64 | int24 BE×N | CRC8]
       ▼
Jetson  /dev/ttyACM0
  ├─ stm32_source.py  (frame parser + sign-extension)
  ├─ polyphase 8/15  (15 kHz → 8 kHz)
  ├─ Hilbert envelope detector  (peak align, MAD threshold)
  ├─ cwt_fast.py  (FFT + cached freq-domain Morlet, 256 scales)
  ├─ JET LUT  (224 × 224 RGB)
  ├─ MobileNetV3-Large  (TensorRT FP16, single ONNX)
  ├─ multi-presence voter  (K = 10 / M = 3)
  └─ Flask + SSE  → web dashboard (Toss radar)
```

End-to-end per-footstep latency on Jetson Nano: **~265 ms** (TRT 23 ms + CWT/render ~200 ms).

---

## Status

- **MATLAB → Python port**: complete, bit-equivalent to original VIBeID pipeline (Event_Extract, GMM-EM, smooth, gausswin), then replaced by Hilbert-envelope detector in production (P14 accuracy 84.89% vs 83.38% with frozen GMM, +12.7% recall).
- **Model**: MobileNetV3-Large, ImageNet-pretrained, fine-tuned on VIBeID. v2 (100-class P14/P50/P100/P2) val acc 86.76%, P14 sub-dataset 93%+. v3 (170-class) val 70.29%.
- **Real-time CWT**: `cwt_fast.py` is **7.6×** faster than `pywt.cwt`, bit-identical on 50 random samples × 3 sub-datasets (P14, P50, P100). Vectorized over 256 scales via cached frequency-domain wavelets.
- **STM32 firmware**: ADS1256 bring-up + RDATAC + 15 kSPS sustained over USB CDC at 921 600 baud, frame loss 0 over 5-min capture. ISR uses direct SPI register access (HAL overhead is too high for the 66.67 µs DRDY window).
- **Jetson live integration**: STM32 → polyphase → CWT → TRT → voter → SSE web dashboard, end-to-end live confirmed with operator's own footsteps.
- **Family transfer learning**: data collection pending (see `family_collection_runbook.md`).

---

## Hardware

| Component | Part | Notes |
|---|---|---|
| Geophone | SM-24 | 28.8 V/m/s, DCR 375 Ω, 10 Hz natural freq |
| Damping shunt | 990 Ω (3 × 330 Ω series) | ratio ≈ 0.55, flattens 10 Hz peak |
| ADC | ADS1256IDB on vctec 8-ch board | 24-bit, PGA=16, AVDD/DVDD LDO, 7.68 MHz xtal |
| MCU | NUCLEO-F411RE | SPI1 @ 1.31 MHz, USART2 921 600 over ST-Link VCP |
| Edge | Jetson Nano 4 GB | TensorRT FP16, custom polyphase + CWT in Python |

No external pre-amplifier (ADS1256 PGA replaces it). Switched-cap self-bias keeps the geophone mid-rail with BUFEN = 0.

---

## Repository structure

```
.
├── python/
│   ├── extract.py                # MATLAB → Python (smooth, gausswin, Hilbert envelope detector)
│   ├── cwt_fast.py               # FFT + cached freq-domain Morlet (7.6× over pywt)
│   ├── render_lut.py             # CWT → 224×224 RGB via JET LUT
│   ├── jetson_realtime.py        # streaming pipeline (Source/Smoother/Detector/Voter)
│   ├── jetson_infer.py           # TensorRT FP16 engine wrapper
│   ├── web_server.py             # Flask + SSE web dashboard
│   ├── stm32_source.py           # STM32 binary frame parser (Jetson side)
│   ├── ads1256_source.py         # StreamingPolyphase + Jetson-direct ADS1256 fallback
│   ├── record_session.py         # 5-min sustained capture → .npz
│   ├── live_monitor.py           # real-time waveform viewer (browser) + .npz save + --analyze
│   ├── prep_transfer_dataset.py  # family .npz → transfer-learning dataset
│   ├── export_onnx.py            # .pth → single-file ONNX (opset 13)
│   └── web/                      # HTML/SSE assets (index.html, live_monitor.html, people.json)
├── notebook/                     # Colab training notebooks (v2, v3)
├── weights/                      # MobileNet checkpoints + JET LUT (.npy)
├── event_detection/              # original VIBeID MATLAB scripts (reference)
├── paper/                        # VIBeID paper + our final report
├── CLAUDE.md                     # working notes (Korean) — currently authoritative
├── CLAUDE_ARCHIVE.md             # completed-stage verification numbers, full pipeline specs
└── family_collection_runbook.md  # day-of data collection runbook
```

Data (`data/raw/`, `data/interim/`, all `*.mat` / `*.npz`) are git-ignored and must be re-fetched from the VIBeID project page or re-recorded.

---

## Quickstart

### 1. Train (Colab or local CUDA)

Open `notebook/colab_v3_train.ipynb`. Mount the VIBeID interim `.mat` files (footstep_feat). Training renders CWT-LUT spectrograms on the fly with our pipeline — **never** consume the OSF `processed/*.zip` PNGs (they re-introduce matplotlib brittleness; see CLAUDE.md "절대 원칙 #1").

### 2. Export single-file ONNX

```bash
python python/export_onnx.py --ckpt weights/mobilenet_v3_large_v3.pth \
    --out weights/mobilenet_v3_large_v3.onnx
```

### 3. Build TensorRT plan on Jetson

```bash
scp weights/mobilenet_v3_large_v3.onnx snup2@snup2-desktop:~/terra/
ssh snup2@snup2-desktop "cd ~/terra && \
    trtexec --onnx=mobilenet_v3_large_v3.onnx --fp16 \
            --saveEngine=mnv3_v3_fp16.plan --workspace=512"
```

### 4. Run live dashboard

```bash
ssh snup2@snup2-desktop
cd ~/terra
python3 web_server.py --stm32 --plan mnv3_v3_fp16.plan
# open http://snup2-desktop:8000/  in any browser
```

### 5. (Optional) Real-time waveform monitor

While iterating on hardware or before a capture session:

```bash
python3 live_monitor.py --pid sanity --duration 60
# open http://snup2-desktop:8050/  → live mV waveform + RMS / peak / drops
# Ctrl-C  → saves .npz, prints quick-analyze command
```

The same script analyzes any saved `.npz`:

```bash
python3 live_monitor.py --analyze data/live/sanity/sanity_live_*.npz
# prints: footstep count (envelope detector), Welch FFT top peaks,
#         AC-mains 50/60 Hz pickup ratio, RMS, headroom
```

### 6. Capture a 5-min session

```bash
python3 record_session.py --pid <name> --session 1 --duration 300 \
    --outdir data/recordings
```

---

## Documentation

- **`CLAUDE.md`** (Korean): authoritative working doc. Contains current pipeline contract, calibration constants, hardware quirks, "절대 원칙" (rules that must not be broken on re-training / re-deployment), and the next-steps plan.
- **`CLAUDE_ARCHIVE.md`** (Korean): completed-stage verification numbers and MATLAB-pipeline precise specs (kept out of `CLAUDE.md` for brevity).
- **`family_collection_runbook.md`** (Korean): day-of runbook for the family-transfer data collection — Phase 0 (sanity), Phase 1 (setup), Phase 2 (capture). PASS-criteria tables, NG response tables, command cheat sheet.
- **`python/JETSON_SETUP.md`**: one-shot Jetson bring-up notes (CUDA / pyserial / pycuda / glibc quirks).

---

## Key design constants (do not break)

These are documented in `CLAUDE.md` § 절대 원칙 — listed here for visibility:

- **Training inputs are always rendered from interim `.mat`** via our LUT, not from OSF `processed/*.zip` PNGs. The latter were rendered with the authors' matplotlib JET and re-introduce the v1-era brittleness (P50 local accuracy dropped from 7% to 86% once we switched to our LUT).
- **`data/raw/P1/` is actually P2** (1-indexed ↔ 0-indexed off-by-one in the upstream files; verified by 100/100 corr 0.999 vs `a1.mat` label 2). `data/raw/P14`, `P50`, `P100` are not affected.
- **ONNX export must be a single file** (opset 13). PyTorch ≥ 2.x default `external_data=True` splits the weights and TRT build fails.
- **CWT engine is `cwt_fast.cwt`**, not `pywt.cwt`. Bit-identical on the standard 150-sample regression set.

---

## Acknowledgments

- VIBeID dataset and original MATLAB pipeline: Tewari & Bhattacharya, *VIBeID: A Structural-Vibration-Based Dataset for Footstep Recognition*, 2023 — see `paper/10133_VIBEID_A_STRUCTURAL_VIBR.pdf`.
- Similar hardware setups consulted during ADC selection: [seismometer.info SM-24 + ADS1115](https://www.seismometer.info/beta-version-v0-1-of-seismometer-based-on-raspberry-pi-ads1115-and-sm-24/), [Seisberry 3-axis ADS1256](https://erellaz.com/blog/seisberry/seisberry-install/), [LiU vibration monitoring with ADS1256 PGA=32](https://liu.diva-portal.org/smash/get/diva2:1784892/FULLTEXT01.pdf), [CEDAS_geo (PGA=16)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11243846/).
- Reference ADC schematic: [Waveshare High-Precision AD/DA](https://www.waveshare.com/w/upload/0/03/High-Precision-AD-DA--Schematic.pdf).
