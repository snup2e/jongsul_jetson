# Terra — Footstep Recognition Edge Deployment

> **Scope of this file**: 앞으로 할 일 + 절대 깨면 안 되는 원칙만. 완료된 단계의 검증 수치 / VIBeID 데이터셋 전체 구조 / MATLAB 파이프라인 정밀 사양 / 트랙 A·B 결과 상세는 → **`CLAUDE_ARCHIVE.md`** 참조.

## 현재 상태 (2026-05-03)
- Stage 1~5d + 다중 인원 voter (5e) + 웹 대시보드 (7) + Track A (v3 backbone) + Track B (multi-presence 검증) + **Track C (STM32 bridge skeleton)** **모두 PASS**
- raw .mat → GMM → CWT(cwt_fast) → LUT → TRT FP16 → top-1 → multi-presence voter → SSE → Toss-style 웹페이지 Jetson Nano 에서 end-to-end 작동
- v2 검증 P14/P50/P100/P2 = 93~97%, v3 P14 = 86.7% (170-class trade-off), per-footstep 386 ms
- **블로커**: ADS1256 모듈 도착 대기. 마감 2026-06.
- GitHub: `snup2e/jongsul_jetson` (PUBLIC, weights+paper 포함, data/.mat 제외)

### 하드웨어 식별
- **ADC 모듈**: vctec.co.kr 8채널 24비트 ADC 보드 (상품코드 P000BATS) — 칩 ADS1256IDB (TI 정품 SSOP-28), 7.68 MHz 크리스탈, 정밀 voltage reference IC 내장 (SOIC-8), AVDD/DVDD LDO 분리 (SOT-223). PCB는 Waveshare High-Precision AD/DA Board ADC 부분과 회로 거의 동일 (회로도 비교용 → References).
- **핀헤더 배치**: 좌측 SPI (5V / GND / SCLK / DIN / DOUT / DRDY / CS / PDWN), 우측 아날로그 (AIN0~AIN7 각 채널마다 GND 페어).
- **지오폰**: SM-24 (28.8 V/m/s, DCR 375 Ω) — 외부 op-amp 없음 (옵션 B, 아래 STM32 결정사항 참조).

### Track C 완료 (2026-05-03, commit f8fc46b)
NUCLEO-F411RE 단독 bring-up Phase 0~4 모두 PASS:
- Phase 0~1: CubeIDE 프로젝트 + USART2 tick (115200 → 921600 전환 확인)
- Phase 2: PB1/PB6 GPIO_Output 핀 추가 (코드 토글 생략, Phase 5 SPI 통신 시 자동 검증 예정)
- Phase 3: Binary frame format 송신 (SYNC 0xA5 0x5A | SEQ u16 LE | N=64 | int24 BE × N | CRC8 poly 0x07) — 가짜 ramp 데이터, 10 fps. 라이브 검증: SYNC 간격 198 byte, CRC fail 0, SEQ drops 0, ramp diff = 1.0 ± 0.0.
- Phase 4: `python/stm32_source.py` (streaming source, MatFileSource API 호환) + `python/stm32_phase3_check.py` (one-shot frame 검증) + `resample_for_ads1256.py` 8/15 모드 추가 (max\|d\|=3.331e-16 vs scipy block reference)
- 의의: ADS1256 도착 후 펌웨어에 SPI/EXTI 핸들러만 추가 + `web_server.py` 의 `MatFileSource → STM32Source` 한 줄 교체로 본 deployment 진입.

## 다음에 할 일 (우선순위 순)

### 센서 도착 전 — 컴퓨팅 안 쓰는 트랙
1. **트랙 D: 가족 데이터 수집 protocol** (수집은 sensor 후, 문서/script 골격은 미리, 약 1일)
   - D.1: `docs/COLLECTION_PROTOCOL.md` — 가족 N명, 5분 × 3 세션 (신발/시간대), 라벨링 룰
   - D.2: `python/record_session.py` 골격 — STM32Source → 30초/5분 파일 저장 + 라이브 RMS
   - D.3: `docs/TRANSFER_PLAN.md` — head 교체 (1280→N+1, +1=unknown) vs embedding+centroid 비교 plan
   - D.4: Unknown class 임계 디자인 (max-prob ≥0.7 또는 centroid 거리)

### 센서 도착 후 — 순차 (S.1~S.10)
| # | 작업 | 의존 |
|---|---|---|
| S.1 | STM32 ↔ ADS1256 bring-up (substep 1a~1f, 아래) | 도착 |
| S.2 | STM32 펌웨어에 ADS1256 통신 정식 통합 (Self-cal / RDATAC + EXTI) | S.1 |
| S.3 | 15000 SPS sustained 30초 → frame 카운트 = 7031 ± 1 | S.2 |
| S.4 | `stm32_source.py` 완성 + `stm32_validate.py` (scipy 8/15 reference 비교) | C.4 + S.3 |
| S.5 | calibration: 30초 캡처 RMS vs `P14_1.mat` 첫 30초 RMS → scale 결정 + PGA 튜닝 | S.4 |
| S.6 | `web_server.py` 의 `MatFileSource` → `STM32Source` 교체 (한 줄) | S.5 |
| S.7 | 본인 발걸음 raw → top-1 분포 spot-check (v2/v3 둘 다) | S.6 |
| S.8 | 가족 데이터 수집 (D.1, D.2 protocol) — 약 1주 | D.* + S.6 |
| S.9 | Transfer learning (head 교체 vs embedding) → 가정 deployment 정확도 평가 | D.3 + S.8 |
| S.10 | 종설 발표 자료 (시연 영상 + accuracy table + 시스템 다이어그램) | 위 모두 |

#### S.1 substep (ADS1256 bring-up)
1. **1a**: 모듈에 5V/3.3V/GND + SPI 5선 (SCLK/DIN/DOUT/DRDY/CS) + RESET (PB1) + 1kΩ 댐핑 션트 1개 (코일 양단) 배선. 양쪽 1kΩ 직렬 ESD 보호는 선택 (5mA cap).
2. **1b**: STM32 펌웨어에 SPI write/read 헬퍼만 먼저. RESET pin pulse 후 RREG STATUS (0x00) 읽기 → factory ID bits 7-4 = 0x3X 패턴 확인 (모듈 sanity check).
3. **1c**: Init 시퀀스 작성 — RESET pin → WREG STATUS (BUFEN=0, ACAL=1) → WREG MUX (AIN0/AIN1) → WREG ADCON (PGA=8) → WREG DRATE (0xE0=15000 SPS) → SELFCAL → DRDY low 대기 → RDATAC.
4. **1d**: USART2 921600으로 raw 24-bit 샘플 노트북 출력 (Track C frame format 그대로 사용 가능). PuTTY/SerialPlot 또는 임시 Python script로 라이브 waveform 모니터링.
5. **1e**: 발 굴러보면서 saturation 확인 → PGA 튜닝. 시작 PGA=8 (±625 mV FS). 작으면 PGA=16 (±312.5 mV FS). PGA=32 (±156 mV FS) 이상은 큰 발걸음 impact 클리핑 위험 — 야외 같은 weak signal 환경에서만.
6. **1f**: 일관된 신호 캡처되면 EXTI0 (PB0 falling, DRDY) 핸들러로 전환 → S.2 정식 통합.

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
├── stm32_source.py         # STM32 binary frame streaming source (MatFileSource API 호환) — 2026-05-03
├── stm32_phase3_check.py   # Phase 3 frame format one-shot 검증 — 2026-05-03
├── ads1256_*.py            # Jetson-direct 폴백 자산
└── resample_for_ads1256.py # StreamingPolyphase (16/15 + 8/15 둘 다 검증 완료)

stm32/ads1256_bridge/       # CubeIDE 프로젝트 — Phase 0~4 PASS, ADS1256 통신 코드는 Phase 5~7 에서
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
- **활성**: SPI1 (full-duplex master, 8-bit MSB, CPOL=0/CPHA=2 Edge, NSS Disable, prescaler /64 → ~1.5625 MHz), GPIO (PB6=CS, PB1=RESET), EXTI0 (PB0 falling, no pull), USART2 (Async, 921600 8-N-1, NVIC priority 5), NVIC (EXTI0 priority 0)
- **비활성**: USB_OTG_FS / USB_DEVICE / CDC class 미들웨어 모두 깔지 않음
- **데이터 경로**:
  ```
  Geophone (SM-24, 28.8 V/m/s, DCR 375 Ω) ─┬─[1kΩ 댐핑 션트]─┬─ AIN0/AIN1 차분
                                              (양쪽 1kΩ 직렬 ESD 선택)
    → ADS1256 (BUFEN=0, PGA=8, 15000 SPS RDATAC, DRDY 66.67 µs)
    → STM32 SPI1 ~1.5 MHz, EXTI0 (DRDY), 링버퍼 SPSC
    → USART2 921600 binary frame [SYNC 0xA5 0x5A | SEQ uint16 LE | N=64 | int24 BE × N | CRC8]
    → Jetson /dev/ttyACM0 → STM32Source.chunks_raw()
    → int24 decode → 폴리페이즈 8/15 → 8 kHz float32
    → (이하 기존 파이프라인 동일)
  ```
- **ADS1256 init 시퀀스** (S.1c 펌웨어):
  ```
  1. RESET pin low → high (auto self-cal 트리거)
  2. WREG STATUS  (BUFEN=0, ACAL=1)
  3. WREG MUX     (AIN0/AIN1 differential)
  4. WREG ADCON   (PGA=8, SDCS=00)
  5. WREG DRATE   (0xE0 = 15000 SPS)
  6. SELFCAL      (config 변경 후 필수)
  7. DRDY low 대기
  8. RDATAC       (continuous mode 진입)
  ```
  S.1b sanity check: RREG STATUS (0x00) → factory ID bits 7-4 = 0x3X 확인.
- **외부 회로 = 1kΩ 댐핑 션트 1개 표준** (옵션 B = 외부 op-amp 없음, ADS1256 PGA로만 처리):
  - **댐핑 1kΩ (코일 양단)**: SM-24 open-circuit 댐핑 0.27 → ~0.55, 10 Hz 공진 ringing 억제 + 주파수 응답 평탄화. VIBeID 셋업 (외부 op-amp 거친 신호로 학습) 과의 ringing 분포 미스매치 방지.
  - **양쪽 1kΩ 직렬 보호 (선택)**: ESD/큰 전압 입력 시 ADS1256 입력 전류 5 mA 이하로 제한. 1kΩ × ~50 pF = 50 ns RC, 15 kHz 샘플 settling 충분.
  - **DC 바이어스 회로 없음**: switched-cap self-bias (BUFEN=0, Zeff ~33 kΩ @ PGA=8) + 코일 DCR 375 Ω 로 AIN0/AIN1 자연스럽게 mid-rail 안착 (seismometer.info, Seisberry 동일 접근).
  - S.5 에서 P14_1.mat RMS+스펙트럼 비교 후 이상 시 단계적 강화 옵션: AINCOM→2.5V 분압 강제 bias.
- **PGA 튜닝 정책** (옵션 B 핵심): 외부 op-amp 없이 ADS1256 PGA로만 게인 처리.
  - PGA=8 → ±625 mV FS (differential FSR = ±2·Vref/PGA = ±5V/PGA, Vref=2.5V).
  - 시작 PGA=8 → 신호 너무 작으면 PGA=16 (±312.5 mV FS) 까지. PGA=32 (±156 mV FS) 이상은 큰 발걸음 impact 클리핑 위험 — 야외 weak signal 환경에서만 검토.
  - 근거: ADS1256 24-bit + PGA 가 ADS1115 16-bit + 외부 op-amp 보다 noise-free bits 동등 이상. seismometer.info / Seisberry 모두 외부 amp 없이 운용. Fine-tuning 으로 분포 차이 흡수.
- **VIBeID 셋업과의 차이** (논문 6.2.1 기준, 펌웨어/모델 설계에 영향):
  - VIBeID: SM-24 + 외부 pre-amp gain 10 + 16-bit Logic Sound Card HAT (8 kHz, RPi 3B+).
  - VIBeID raw waveform 진폭 (외부 amp 거친 후): Carpet ±0.5 V, Cement ±0.2 V, Outdoor ±0.04 V → raw geophone 출력 추정 ±50 mV / ±20 mV / ±4 mV.
  - 우리: 외부 amp 없음 → ADS1256 PGA 가 그 자리를 대신. PGA=8 에서 carpet 50 mV / 625 mV = 8% FS, cement 3.2% FS. PGA=16 으로 올리면 carpet 16% FS, cement 6.4% FS.
- **SPI 타이밍 제약** (ADS1256 datasheet):
  - CLKIN = 7.68 MHz (확인됨), SCLK 최대 = 4 / fCLKIN = **1.92 MHz**.
  - STM32 SPI prescaler /64 (PCLK2 100 MHz → 1.5625 MHz) **안전**. /32 (3.125 MHz) 는 datasheet 위반 — 사용 금지.
  - DRDY low 후 32 DRDY 주기 (= 2.13 ms @ 15000 SPS) 안에 SCLK 시작해야 RDATAC 데이터 유효 — 다른 ISR 이 EXTI0 막으면 안 됨 (EXTI0 priority 0 이미 설정).
- **샘플레이트**: 15000 SPS native (DRATE=0xE0). Plan B = 7500 SPS (폴리페이즈 16/15 로 스왑, 학습 자산 영향 0).

---

## References (외부 자료)

- **VIBeID 논문 hardware section**: 6.2.1 (geophone 28.8 V/m/s + 외부 op-amp gain 10 + 16-bit sound card HAT 8 kHz, RPi 3B+ BCM2837B0).
- **유사 프로젝트** (ADS1256 PGA-only / 외부 op-amp 없는 셋업):
  - seismometer.info — SM-24 + ADS1115 셋업: <https://www.seismometer.info/beta-version-v0-1-of-seismometer-based-on-raspberry-pi-ads1115-and-sm-24/>
  - Seisberry — 3-axis seismograph + ADS1256 Waveshare HAT: <https://erellaz.com/blog/seisberry/seisberry-install/>
  - LiU 코끼리 진동 모니터링 (PGA=32 사례): <https://liu.diva-portal.org/smash/get/diva2:1784892/FULLTEXT01.pdf>
  - CEDAS_geo (PGA=16 학술 사례): <https://pmc.ncbi.nlm.nih.gov/articles/PMC11243846/>
- **Waveshare High-Precision AD/DA 회로도** (vctec P000BATS PCB와 ADC 부분 거의 동일): <https://www.waveshare.com/w/upload/0/03/High-Precision-AD-DA--Schematic.pdf>

---

## 사용자 정보 / 협업 스타일
- 한국어 소통
- 코드 작성 전 설명/타당성 분석 선호 (큰 결정 미리 합의)
- Claude Code CLI (로컬 파일 직접 접근)
- 백본/Colab/Jetson 사용자 직접. Claude 는 코드 작성 + 로컬 검증.
- Windows 11 + bash + Python 3.13.5 (h5py, scipy, numpy, pywavelets, matplotlib, pillow)
- ⚠️ Windows 콘솔 cp949 → 한글/이모지 출력 시 `PYTHONIOENCODING=utf-8`
