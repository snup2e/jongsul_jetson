# Terra — Footstep Recognition Edge Deployment

> **Scope of this file**: 앞으로 할 일 + 절대 깨면 안 되는 원칙만. 완료된 단계의 검증 수치 / VIBeID 데이터셋 전체 구조 / MATLAB 파이프라인 정밀 사양 / 트랙 A·B 결과 상세는 → **`CLAUDE_ARCHIVE.md`** 참조.

## 현재 상태 (2026-05-03)
- Stage 1~5d + 다중 인원 voter (5e) + 웹 대시보드 (7) + Track A (v3 backbone) + Track B (multi-presence 검증) **모두 PASS**
- raw .mat → GMM → CWT(cwt_fast) → LUT → TRT FP16 → top-1 → multi-presence voter → SSE → Toss-style 웹페이지 Jetson Nano 에서 end-to-end 작동
- v2 검증 P14/P50/P100/P2 = 93~97%, v3 P14 = 86.7% (170-class trade-off), per-footstep 386 ms
- **블로커**: ADS1256 모듈 도착 대기. 마감 2026-06.
- GitHub: `snup2e/jongsul_jetson` (PUBLIC, weights+paper 포함, data/.mat 제외)

## 다음에 할 일 (우선순위 순)

### 센서 도착 전 — 컴퓨팅 안 쓰는 트랙
1. **트랙 C: STM32 펌웨어 골격** (NUCLEO-F411RE 단독 bring-up, ADS1256 무관)
   - C.1: CubeIDE 프로젝트 생성 + CubeMX 설정 (아래 "STM32 결정사항" 참조)
   - C.2: USART2 1초 tick → `/dev/ttyACM0` 수신 확인
   - C.3: PB1 (RESET) GPIO 토글 멀티미터 확인
   - C.4: `python/stm32_source.py` 골격 — frame parser (SYNC/SEQ/CRC8) + int24 BE decoder
   - C.5: `resample_for_ads1256.py` 에 폴리페이즈 8/15 모드 추가 + scipy reference 와 max\|d\| < 1e-15 검증
2. **트랙 D: 가족 데이터 수집 protocol** (수집은 sensor 후, 문서/script 골격은 미리)
   - D.1: `docs/COLLECTION_PROTOCOL.md` — 가족 N명, 5분 × 3 세션 (신발/시간대), 라벨링 룰
   - D.2: `python/record_session.py` 골격 — STM32Source → 30초/5분 파일 저장 + 라이브 RMS
   - D.3: `docs/TRANSFER_PLAN.md` — head 교체 (1280→N+1, +1=unknown) vs embedding+centroid 비교 plan
   - D.4: Unknown class 임계 디자인 (max-prob ≥0.7 또는 centroid 거리)

### 센서 도착 후 — 순차 (S.1~S.10)
| # | 작업 | 의존 |
|---|---|---|
| S.1 | ADS1256 모듈 식별 (Waveshare HAT vs generic) → 핀헤더 확정 | 도착 |
| S.2 | STM32 펌웨어에 ADS1256 통신 추가 (RREG STATUS / Self-cal / RDATAC) | C.1~C.4 + S.1 |
| S.3 | 15000 SPS sustained 30초 → frame 카운트 = 7031 ± 1 | S.2 |
| S.4 | `stm32_source.py` 완성 + `stm32_validate.py` (scipy 8/15 reference 비교) | C.4 + S.3 |
| S.5 | calibration: 30초 캡처 RMS vs `P14_1.mat` 첫 30초 RMS → scale 결정 | S.4 |
| S.6 | `web_server.py` 의 `MatFileSource` → `STM32Source` 교체 (한 줄) | S.5 |
| S.7 | 본인 발걸음 raw → top-1 분포 spot-check (v2/v3 둘 다) | S.6 |
| S.8 | 가족 데이터 수집 (D.1, D.2 protocol) — 약 1주 | D.* + S.6 |
| S.9 | Transfer learning (head 교체 vs embedding) → 가정 deployment 정확도 평가 | D.3 + S.8 |
| S.10 | 종설 발표 자료 (시연 영상 + accuracy table + 시스템 다이어그램) | 위 모두 |

### 폴백
- STM32 루트 막힘 시 → Jetson-direct ADS1256 (`python/ads1256_source.py`, `ads1256_bench.py`, `resample_for_ads1256.py` 16/15) 자산 그대로 활용

---

## ⚠️ 절대 원칙 (재학습/재배포 시 반드시 준수)

### 1. 학습 데이터는 항상 OSF `interim/*.mat` 에서 시작, 우리 LUT 로 직접 렌더
- OSF `processed/*.zip` (PNG) 은 **절대 학습에 쓰지 않음** — 저자 환경의 matplotlib jet 으로 렌더된 것이라 v1 brittleness 함정 그대로 재현됨 (v1 로컬 P50 7%, P100 8% 처참 사태 원인).
- 파이프라인:
  ```
  interim .mat (footstep_feat, N×1501)
    → pywt.cwt(scales=1..256, 'morl') → (256, 1500)
    → JET LUT (matplotlib cm.get_cmap("jet") 256-entry, 또는 weights/jet_lut_v2.npy)
    → cv2.resize → (224, 224, 3) uint8
    → MobileNetV3-Large (ImageNet pretrained → classifier[3] 교체)
  ```
- Jetson 추론도 동일 LUT 파이프라인 (`python/render_lut.py` + `python/cwt_fast.py`) → 학습-추론 분포 일치 보장.

### 2. P1 폴더 = 실제 P2 라벨
- `data/raw/P1/` 의 `P1_1..4.mat` 은 1-indexed↔0-indexed off-by-one 으로 **실제는 P2 데이터**. 검증: P1 raw 추출 1363개 footsteps 가 100/100 a1.mat label==2 와 corr 0.999.
- 폴더명은 그대로 둠. raw "P1" 추론 검증 시 정답 라벨 P2 로 사용.
- P14/P50/P100 은 정상 (자기 라벨과 99-100% 매칭).

### 3. ONNX export 는 단일 파일로
- PyTorch ≥2.x 기본 export 가 external data (`.onnx` + `.onnx.data`) 분리하면 TRT 빌드 실패. `python/export_onnx.py` 가 opset 13 단일 파일로 export. size < 10 MB 면 weight 누락 의심.

---

## ⚠️ Jetson 환경 quirks

### 환경변수 (`~/.bashrc` 박아둠)
```bash
export OPENBLAS_CORETYPE=ARMV8       # numpy "Illegal instruction" 픽스 (Cortex-A57)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
- 없으면 numpy 1.19.5 aarch64 wheel 이 ARMv8.2 명령어 사용 → SIGILL.

### pycuda 설치 — 반드시 `--no-build-isolation`
```bash
python3 -m pip install --user --no-build-isolation pycuda
```
- isolation 모드면 numpy 1.12.1 옛날 버전 새로 빌드 시도 → `xlocale.h` 없음 실패.

### Python 3.6 호환 패치 (이미 적용됨)
- `from __future__ import annotations` 제거, PEP 604 `int | None` → `Optional[int]`, PEP 585 `tuple[X, Y]` → `Tuple[X, Y]`, `argparse.add_subparsers(required=...)` 안 씀, `np.quantile(method='hazen')` 안 됨 → `extract.py` 수동 fallback.

### Claude Code / VS Code Remote-SSH **둘 다 안 됨** (glibc 2.27)
- 워크플로우: Windows 쪽 Claude Code + SSH 터미널 + scp. `JETSON_SETUP.md` 의 Claude Code 섹션 무시.

### CUDA 스레드 binding (web_server.py)
- `pycuda.autoinit` 가 import 스레드에 context 묶음 → consumer 스레드에서 `from jetson_infer import TRTEngine` + `TRTEngine()` + `engine.infer()` 모두 처리해야 함. main 스레드 import 시 `invalid resource handle`.

---

## 핵심 파일 경로

### Windows `E:\Terra\`
```
weights/
├── mobilenet_v3_large_v2_best.pth      # v2 production (val 86.76%, 100-class)
├── mobilenet_v3_large_v2.onnx          # v2 ONNX
├── mobilenet_v3_large_v3.onnx          # v3 (170-class, val 70.29%) — 2026-05-02
├── jet_lut_v2.npy                      # 결정론 jet LUT (sha256 fc594b04...)
└── mobilenetv3_vibeid_a1_best.pth      # v1 (legacy, brittle)

python/
├── extract.py              # MATLAB → Python (smooth/gausswin/GMM-EM/Event_Extract, Hazen fallback)
├── cwt_fast.py             # FFT + cached freq-domain wavelets (7.6× over pywt)
├── render_lut.py           # CWT → 224×224 RGB (npy LUT) — 경로 자동 탐색
├── jetson_realtime.py      # 실시간 streaming (Source/Smoother/Detector/Voter multi-presence)
├── web_server.py           # Flask + SSE 웹 대시보드
├── web/index.html          # Toss-style 대시보드
├── web/people.json         # pid → {name, emoji} (가족 시연 시 편집)
├── gmm_params.npz          # frozen GMM (P14 100s seed=0)
├── export_onnx.py          # .pth → 단일 .onnx
├── ads1256_*.py            # Jetson-direct 폴백 자산
└── resample_for_ads1256.py # StreamingPolyphase (16/15 검증됨, 8/15 추가 예정)

stm32/ads1256_bridge/       # CubeIDE 프로젝트 (작성 예정)
```

### Jetson `~/terra/` (flat 구조 — `python/` + `weights/` 평탄화)
```
~/terra/
├── extract.py / cwt_fast.py / render_lut.py / jetson_infer.py / jetson_realtime.py / web_server.py / stream.py
├── gmm_params.npz / jet_lut_v2.npy
├── mobilenet_v3_large_v2.onnx / mnv3_v2_fp16.plan       # v2 production
├── mobilenet_v3_large_v3.onnx / mnv3_v3_fp16.plan       # v3 (작성 예정 trtexec)
├── sample/         # 시연용 raw .mat
└── web/            # SSE static
```
- scp 시 `~/terra/` flat 에 직접. 하위 폴더 만들지 말 것 (스크립트 cwd 기준 상대 경로).
- v3 plan 빌드:
  ```bash
  scp E:/Terra/weights/mobilenet_v3_large_v3.onnx snup2@121.254.39.88:~/terra/
  ssh snup2@121.254.39.88 "cd ~/terra && trtexec --onnx=mobilenet_v3_large_v3.onnx --fp16 --saveEngine=mnv3_v3_fp16.plan --workspace=512"
  ```

---

## 추론 파이프라인 (확정, 깨지 말 것)
```
raw .mat (8 kHz)
  → extract.apply_frozen_gmm(geo, gmm_params.npz)   # P14 학습 frozen GMM
  → footsteps (M, 1500) float64
  → cwt_fast.cwt(sig)                               # FFT + cached freq-domain wavelets
  → render_lut.cwt_to_rgb_direct(coeffs, (224,224)) # npy LUT + cv2.resize
  → MobileNetV3-Large (TRT FP16 plan)
  → top-1 → multi-presence Voter (K=10/M=3, K_MAX=12)
  → SSE → 웹 대시보드
```
- per-footstep budget: **700 ms** (보통 걸음 1.5 Hz → 660 ms 간격, 큐 누적 방지). 현재 약 400 ms (추출 42 + 렌더 336 + TRT 23) — 1.7× 마진.

---

## STM32 결정사항 (CubeMX 설정)

- **보드**: NUCLEO-F411RE
- **펌웨어 스타일**: HAL (CubeIDE — 임베디드시스템설계 수업 기반)
- **전송 경로**: ST-Link VCP (UART2 → ST-Link MCU 자동 USB CDC) — 케이블 1개로 빌드/플래시/통신/전원 전부.
- **활성**: SPI1 (full-duplex master, 8-bit MSB, CPOL=0/CPHA=2 Edge, NSS Disable, prescaler /64), GPIO (PB6=CS, PB1=RESET), EXTI0 (PB0 falling, no pull), USART2 (Async, 921600 8-N-1, NVIC priority 5), NVIC (EXTI0 priority 0)
- **비활성**: USB_OTG_FS / USB_DEVICE / CDC class 미들웨어 모두 깔지 않음
- **데이터 경로**:
  ```
  Geophone → AIN0/AIN1 (BUFEN=1) → ADS1256 (15000 SPS RDATAC, DRDY 66.67 µs)
    → STM32 SPI1 ~1.5 MHz, EXTI0 (DRDY), 링버퍼 SPSC
    → USART2 921600 binary frame [SYNC 0xA5 0x5A | SEQ uint16 LE | N=64 | int24 BE × N | CRC8]
    → Jetson /dev/ttyACM0 → STM32Source.chunks_raw()
    → int24 decode → 폴리페이즈 8/15 → 8 kHz float32
    → (이하 기존 파이프라인 동일)
  ```
- **샘플레이트**: 15000 SPS native (DRATE=0xE0). Plan B = 7500 SPS (폴리페이즈 16/15 로 스왑, 학습 자산 영향 0).

---

## 사용자 정보 / 협업 스타일
- 한국어 소통
- 코드 작성 전 설명/타당성 분석 선호 (큰 결정 미리 합의)
- Claude Code CLI (로컬 파일 직접 접근)
- 백본/Colab/Jetson 사용자 직접. Claude 는 코드 작성 + 로컬 검증.
- Windows 11 + bash + Python 3.13.5 (h5py, scipy, numpy, pywavelets, matplotlib, pillow)
- ⚠️ Windows 콘솔 cp949 → 한글/이모지 출력 시 `PYTHONIOENCODING=utf-8`
