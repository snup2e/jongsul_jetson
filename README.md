# 발걸음 진동 기반 엣지 AI 신원 식별
### Edge-AI Person Identification from Footstep-induced Floor Vibration

> 카메라·웨어러블 없이 **바닥 진동만으로** 누가 지나가는지 실시간으로 알아내는 엣지 디바이스.
> 성균관대학교 전자전기공학부 종합설계프로젝트.

바닥 밑 지오폰(geophone)이 발걸음이 만든 미세 진동을 감지하면, STM32가 이를 NVIDIA Jetson Nano로 흘려보내고, Jetson이 전체 추론 파이프라인(발걸음 검출 → CWT 스펙트로그램 → MobileNetV3 → 다수결)을 **실시간으로** 돌려 누구인지 웹 대시보드에 표시합니다. 카메라가 없어 프라이버시 친화적이고, 사용자가 아무것도 착용/조작하지 않아도 **그냥 지나가기만 하면** 식별됩니다.

[VIBeID](https://arxiv.org/abs/2306.14640) 구조진동 데이터셋으로 백본을 사전학습한 뒤, 가정 환경의 **등록 사용자 4명(P1–P4)** 에게 전이학습(transfer learning)으로 적응시켰습니다. 등록되지 않은 사람·비-발걸음 충격은 `unknown` 클래스로 거부합니다.

---

## 결과 (Results)

| 지표 | 값 |
|---|---|
| 걸음 단위 정확도 — **Test** (held-out) | **84.5 %**  (Validation 87.4 %) |
| 사람 단위 정확도 — K=5 다수결 voter | **≈ 92 %** |
| 미등록 사용자(unknown) 거부율 | **≈ 98 %** |
| Jetson Nano end-to-end 지연 | **≈ 264 ms / 걸음**  (TensorRT FP16, 실측) |
| 실시간 여유 | **2.4 – 3.7×**  (보행 간격 640–970 ms 대비) |

- 5-class(등록 4명 + unknown), **집 wood floor**, 사람당 3가지 페이스(평상/느림/빠름) × 5분.
- **Train / Val / Test 분할은 녹음별 시간 분할 (70 / 15 / 15).** 각 페이스 녹음을 시간순으로 잘라 test 구간이 학습 구간과 시간적으로 분리되므로, 무작위 분할에서 생기는 *인접 발걸음 누수*가 없습니다. test는 마지막에 1회만 평가하고, validation은 모델·에폭 선택에만 사용합니다.
- 단일 걸음 오차는 K=5 다수결(M=3)로 평균화되어 사람 단위 신뢰도가 올라갑니다.

---

## 파이프라인 (Pipeline)

```
지오폰 Geophone (SM-24, 28.8 V/m/s)
  └─ 990 Ω 댐핑 션트 (≈ 1 kΩ)
       │
       ▼
ADS1256 24-bit ADC  (PGA = 16, 15 kSPS, BUFEN = 0)
       │  SPI 1.31 MHz
       ▼
STM32 NUCLEO-F411RE  (RDATAC, DRDY EXTI, ring buffer)
       │  USART2 921 600 8-N-1 (ST-Link VCP)
       │  binary frame: [SYNC | SEQ u16 | N=64 | int24 BE×N | CRC8]   — 30초 240,640 sample, CRC 0, drop 0
       ▼
Jetson Nano  /dev/ttyACM0
  ├─ stm32_source.py     프레임 파서 + 부호확장
  ├─ polyphase 8/15      15 kHz → 8 kHz
  ├─ Hilbert envelope    발걸음 검출 (peak align, MAD threshold)
  ├─ cwt_fast.py         FFT + 캐시된 주파수영역 Morlet, 256 scale
  ├─ JET LUT             224 × 224 RGB
  ├─ MobileNetV3-Large   TensorRT FP16 (단일 ONNX)
  ├─ multi-presence voter  K = 5 / M = 3
  └─ Flask + SSE         → 웹 대시보드 (Toss 레이더 UI)
```

Jetson Nano 걸음당 지연(실측, TRT FP16): **≈ 264 ms** — CWT ~181 + 렌더 ~46 + TRT ~25 + 정규화 ~4 + envelope ~6 ms. 보행 간격 640–970 ms 대비 2.4–3.7× 여유.

---

## 시스템 상태 (완료)

- **MATLAB → Python 포팅**: VIBeID 원본 파이프라인과 bit-equivalent 검증 후, production 검출기는 **Hilbert envelope**(peak align + MAD threshold)로 교체 — P14 84.89 % vs frozen GMM 83.38 %, recall +12.7 %.
- **모델**: MobileNetV3-Large, ImageNet 사전학습 → VIBeID 백본 → **등록 사용자 4명 전이학습(5-class)**. Test 84.5 % (held-out), unknown recall 98 %.
- **실시간 CWT**: `cwt_fast.py` 가 `pywt.cwt` 대비 수 배 빠르고 bit-identical (256 scale 벡터화, 캐시된 주파수영역 wavelet).
- **STM32 펌웨어**: ADS1256 bring-up + RDATAC + 15 kSPS sustained over USB CDC @ 921 600, 5분 캡처 프레임 손실 0. ISR은 직접 SPI 레지스터 접근(HAL은 66.67 µs DRDY 윈도우에 과부하).
- **Jetson 라이브 통합**: 센서 → polyphase → CWT → TRT → voter → SSE 대시보드 end-to-end 실작동, **라이브 시연 + 영상 촬영 완료**.

---

## 하드웨어 (Hardware)

| 구성 | 부품 | 비고 |
|---|---|---|
| 지오폰 | SM-24 | 28.8 V/m/s, DCR 375 Ω, 자연주파수 10 Hz |
| 댐핑 션트 | 990 Ω (330 Ω × 3 직렬) | ratio ≈ 0.55, 10 Hz 피크 평탄화 |
| ADC | ADS1256IDB (vctec 8-ch 보드) | 24-bit, PGA=16, AVDD/DVDD LDO, 7.68 MHz xtal |
| MCU | NUCLEO-F411RE | SPI1 @ 1.31 MHz, USART2 921 600 (ST-Link VCP) |
| 엣지 | Jetson Nano 4 GB | TensorRT FP16, polyphase + CWT를 Python으로 |

외부 프리앰프 없음 (ADS1256 PGA가 대체). BUFEN=0 + switched-cap self-bias로 지오폰을 mid-rail 유지.

---

## 저장소 구조 (Repository structure)

```
.
├── python/
│   ├── extract.py                # MATLAB→Python (smooth, gausswin, Hilbert envelope 검출기)
│   ├── cwt_fast.py               # FFT + 캐시 주파수영역 Morlet
│   ├── render_lut.py             # CWT → 224×224 RGB (JET LUT)
│   ├── jetson_realtime.py        # 스트리밍 파이프라인 (Source/Smoother/Detector/Voter)
│   ├── jetson_infer.py           # TensorRT FP16 엔진 래퍼
│   ├── web_server.py             # Flask + SSE 웹 대시보드
│   ├── stm32_source.py           # STM32 binary frame 파서 (Jetson 측)
│   ├── ads1256_source.py         # StreamingPolyphase + Jetson-direct ADS1256 폴백
│   ├── record_session.py         # 5분 sustained capture → .npz
│   ├── prep_transfer_dataset.py  # 녹음 → 전이학습 데이터셋 (시간분할 train/val/test)
│   ├── replay_recording.py       # 녹음을 전체 파이프라인으로 오프라인 재생 (정합성 검증)
│   ├── bench_jetson_latency.py   # Jetson 걸음당 end-to-end 지연 실측 (단계별 median/p90)
│   ├── run_family_demo.sh        # 라이브 시연 런처 (web_server)
│   ├── export_onnx.py            # .pth → 단일 ONNX (opset 13)
│   └── web/                      # HTML/SSE 자산 (index.html, people.json)
├── notebooks/                    # 전이학습 노트북 (Colab; train/val/test + 최종 test 평가)
├── notebook/                     # 백본 학습 노트북 (VIBeID v2/v3)
├── weights/                      # MobileNet 백본 체크포인트 + JET LUT (.npy)
├── event_detection/              # VIBeID 원본 MATLAB 스크립트 (참조)
├── paper/                        # VIBeID 논문 + 본 프로젝트 보고서
├── CLAUDE.md / CLAUDE_ARCHIVE.md # 작업 노트 (한국어) — 파이프라인 계약·검증 수치
└── family_collection_runbook.md  # 데이터 수집 런북
```

데이터(`data/`, 모든 `*.mat` / `*.npz`), TRT plan(`*.plan`), 발표 자료·영상은 git-ignore — VIBeID 페이지에서 재취득하거나 재녹음.

---

## 빠른 시작 (Quickstart)

```bash
# 1) 5분 세션 수집 (사람·페이스별)
python3 python/record_session.py --pid P1 --session 1 --duration 300 --outdir data/recordings

# 2) 전이학습 데이터셋 빌드 (녹음별 시간분할 train/val/test)
python python/prep_transfer_dataset.py --pids P1 P2 P3 P4 --unknown-source P14

# 3) 전이학습 (Colab): notebooks/transfer_family_v1.ipynb
#    백본 freeze + head 교체 → train, val로 선택, 마지막에 test 1회 → 단일 ONNX export

# 4) Jetson에 TRT 빌드 + 라이브 시연
scp mobilenet_v3_family_v1.onnx <jetson>:~/terra/
ssh <jetson> "cd ~/terra && trtexec --onnx=mobilenet_v3_family_v1.onnx --fp16 \
    --saveEngine=mnv3_family_v1_fp16.plan --workspace=512"
ssh <jetson> "cd ~/terra && bash run_family_demo.sh"   # web_server, K=5/M=3
# 브라우저로 대시보드 접속

# (선택) 오프라인 검증 / 지연 실측
python python/replay_recording.py                       # 녹음 → 파이프라인 재생, 정답 확인
python3 bench_jetson_latency.py --pid P1 --session 1 --plan mnv3_family_v1_fp16.plan
```

---

## 핵심 설계 원칙 (깨지 말 것)

- **학습 입력은 항상 interim `.mat`에서 우리 LUT로 렌더** — OSF `processed/*.zip` PNG(저자 matplotlib JET)는 v1 brittleness를 재유발하므로 사용 금지.
- **ONNX는 단일 파일(opset 13)** — PyTorch ≥ 2.x 기본 external-data 분할 시 TRT 빌드 실패.
- **CWT 엔진은 `cwt_fast.cwt`** (pywt.cwt 아님). 회귀셋에서 bit-identical.
- **정확도 분할은 녹음별 시간분할** (무작위 per-footstep 금지 — 인접 발걸음 누수).
- `data/raw/P1/`은 실제로 P2 데이터 (upstream off-by-one); P14/P50/P100은 영향 없음.

---

## Acknowledgments

- VIBeID 데이터셋·원본 MATLAB 파이프라인: Tewari & Bhattacharya, *VIBeID: A Structural-Vibration-Based Dataset for Footstep Recognition*, 2023 — `paper/10133_VIBEID_A_STRUCTURAL_VIBR.pdf`.
- ADC 선정 참고: [seismometer.info SM-24 + ADS1115](https://www.seismometer.info/beta-version-v0-1-of-seismometer-based-on-raspberry-pi-ads1115-and-sm-24/), [Seisberry ADS1256](https://erellaz.com/blog/seisberry/seisberry-install/), [Waveshare High-Precision AD/DA 회로도](https://www.waveshare.com/w/upload/0/03/High-Precision-AD-DA--Schematic.pdf).
