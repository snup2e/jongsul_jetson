# Terra — Footstep Recognition Edge Deployment

> **Scope of this file**: 앞으로 할 일 + 절대 깨면 안 되는 원칙만. 완료된 단계의 검증 수치 / VIBeID 데이터셋 전체 구조 / MATLAB 파이프라인 정밀 사양 / 트랙 A·B 결과 상세는 → **`CLAUDE_ARCHIVE.md`** 참조.

## 현재 상태 (2026-05-11)
- **S.1 ~ S.6 모두 PASS** (S.6 SEQ drop 이슈 해결 — STM32 USB 포트 변경)
- **D.2 PASS** — `record_session.py` 5분 sustained capture 검증: 4500352 samples @ 15000.4 sps, frames_ok=70318, drops/resets/CRC/N 모두 0, 18 MB/세션. 본인 dry-run 1세션 완료 (footsteps detected: 71).
- Stage 1~5d + 다중 인원 voter (5e) + 웹 대시보드 (7) + Track A (v3 backbone) + Track B (multi-presence 검증) + Track C (STM32 bridge skeleton) **모두 PASS**
- raw geophone → STM32 ADS1256 → polyphase 8 kHz → CWT → TRT FP16 → top-1 → voter → SSE → 웹 대시보드 **end-to-end 라이브 작동 확인** (본인 발걸음 → count 증가).
- v2 검증 P14/P50/P100/P2 = 93~97%, v3 P14 = 86.7% (170-class trade-off), per-footstep 386 ms
- 마감 2026-06. 남은 트랙: **2026-05-11 가족 4명 수집** (S.8) → S.9 transfer → S.10 종설 발표.
- GitHub: `snup2e/jongsul_jetson` (PUBLIC, weights+paper 포함, data/.mat 제외)

### 하드웨어 식별 (확정)
- **ADC 모듈**: vctec.co.kr 8채널 24비트 ADC 보드 (상품코드 P000BATS) — 칩 ADS1256IDB (TI 정품 SSOP-28), 7.68 MHz 크리스탈, 정밀 voltage reference IC 내장 (SOIC-8), AVDD/DVDD LDO 분리 (SOT-223). PCB는 Waveshare High-Precision AD/DA Board ADC 부분과 회로 거의 동일.
- **핀헤더 배치**: 좌측 SPI (5V / GND / SCLK / DIN / DOUT / DRDY / CS / PDWN), 우측 아날로그 (AIN0~AIN7 각 채널마다 GND 페어).
- **지오폰**: SM-24 (28.8 V/m/s, DCR 375 Ω) — 외부 op-amp 없음.
- **션트**: 330Ω × 3 직렬 = 990Ω (≈ 1kΩ 댐핑 표준, ratio 0.55), 코일 양단.
- **STM32**: NUCLEO-F411RE.

### S.1 ~ S.6 확정 결과 (2026-05-10)
**S.1 (ADS1256 bring-up):** 1a~1f 모두 PASS.
- STATUS = 0x34 (factory ID 0x3, ACAL=1)
- MUX/ADCON/DRATE 모두 expect 일치
- **PGA = 16 확정** (ADCON = 0x04). 보통 발걸음 2~7% FS, hard stomp ~11% FS, idle 노이즈 ~0.78% FS. PGA=8 너무 작아 PGA=16 채택. PGA=32 안 가는 이유: 노이즈 floor 가 PGA 따라 안 줄어 SNR 동일.
- EXTI 전환: PA9 falling + pull-up + EXTI9_5_IRQn priority 0.

**S.2/S.3 (RDATAC + 30초 sustained):** 14998 SPS sustained, 234.3 fps, CRC fail 0, SEQ 누락 0 (Windows 측). **ISR 안 SPI 는 반드시 직접 레지스터 접근** (`SPI1->DR`/`SPI1->SR`) — `HAL_SPI_TransmitReceive` overhead 가 RDATAC 의 66.67µs 윈도우에 너무 커서 13.4 kSPS 로 떨어짐.

**S.4 (Python 통합):** stm32_source.py 가 raw mode 14992.7 sps + polyphase mode 7985.7 sps (target 8000) 둘 다 PASS (Windows). Track C 의 stm32_source.py 가 실 ADC frame 에 그대로 동작.

**S.5 (Calibration):** scale = **3.7253e-7 V/LSB** 확정 (이론값: ADS1256 PGA=16 LSB 전압 × VIBeID amp gain 10). 측정값 by-peak 3.19e-7 와 17% 일치. `stm32_source.py` 에 `CALIBRATED_SCALE` 상수로 박힘. S.8 transfer learning 후 재조정 가능.

**S.6 (Jetson 라이브 통합):** `web_server.py` 에 `--stm32` / `--stm32-port` 플래그 추가, `MatFileSource` ↔ `STM32Source` 분기. 본인 발걸음 → `inference` 이벤트 발생, 발걸음 count 증가 확인. person_id 는 v2 (VIBeID 100명) 학습이라 본인 없으므로 의미없게 뜨는 게 정상.

**S.6 SEQ drop 이슈 — 해결됨 (2026-05-10):** 30초 capture 에서 SEQ drops 0, 7997.5 sps. 원인은 STM32 USB 가 SSD 와 같은 Jetson 내부 hub 다리에 꽂혀 있어 SSD 전류 스파이크 시 STM32 brownout reset. **운영 룰**: STM32 USB 는 항상 SSD 와 다른 물리 포트에. 배럴잭 5V/4A 자체 용량은 충분.

## 다음에 할 일 — **2026-05-11 가족 4명 모이는 날 plan** (총 ~2.5~3시간)

### Phase 1 — 셋업 (~10분)
1. Jetson 부팅, STM32 USB 연결 (SSD 와 다른 포트 — 운영 룰).
2. 30초 sanity check: `python3 record_session.py --pid me --session 0 --duration 30 --outdir data/recordings`. PASS (drops=0, resets≤1, footsteps>0) 확인.
3. 가족 protocol 짧게 설명: 지오폰 위/근처 5분 자연스럽게 걷기. 세션마다 신발/페이스 약간 다르게.

### Phase 2 — 수집 (~1시간 30분, 인당 ~19분)
인원: me, mom, dad, sis (사용자 가족 구성에 따라 pid 조정).
세션: 1, 2, 3 (총 12 세션 = 60분 녹음 + 사이 휴식 ~16분).
```bash
# 예시 (각 인원 × 3 세션)
python3 record_session.py --pid mom --session 1 --duration 300 --outdir data/recordings
python3 record_session.py --pid mom --session 2 --duration 300 --outdir data/recordings
python3 record_session.py --pid mom --session 3 --duration 300 --outdir data/recordings
# (다음 사람으로 교체 후 반복)
```
세션마다 footsteps detected 확인 (50~250 범위면 OK). 일부 세션이 너무 적으면 (< 30) 그 세션만 재수집.

### Phase 3 — 데이터 prep + transfer 학습 (~20분)
**아직 작성 안 된 파일 — 가족 모이기 전 이른 아침에 미리 작성하면 좋음**:
- **D.3 `python/prep_transfer_dataset.py`** (Windows 측):
  - `data/recordings/<pid>/*.npz` 의 raw_int32 → `apply_frozen_gmm(geo, gmm_params.npz)` → footsteps (M, 1500)
  - `cwt_fast.cwt` → `render_lut.cwt_to_rgb_direct` → (M, 224, 224, 3) uint8
  - VIBeID P14 footsteps 도 동일 파이프라인으로 unknown 클래스 (라벨 N) 추가
  - 출력: `data/transfer/family_v1.npz` with `X (Total, 224, 224, 3) uint8`, `y (Total,) int`, `pid_map (dict)` (예: {0:'me', 1:'mom', 2:'dad', 3:'sis', 4:'unknown'})
  - 80/20 train/val split (per-pid stratified)
- **D.4 Colab 노트북** `notebooks/transfer_family_v1.ipynb`:
  - v3 backbone (`mobilenet_v3_large_v3.onnx` 의 PyTorch 버전 또는 v3 best.pth) 로드
  - backbone 전체 freeze, classifier[3] 1280→5 로 교체 (4 family + 1 unknown)
  - Adam LR=1e-3, 10 epoch, batch 64, label smoothing 0.05
  - val accuracy + per-class confusion matrix
  - ONNX export (opset 13 단일 파일)

### Phase 4 — 배포 + 라이브 검증 (~30분)
1. `weights/mobilenet_v3_family_v1.onnx` Jetson 으로 scp (IP 그날 확인 — DHCP).
2. Jetson 에서 `trtexec --onnx=mobilenet_v3_family_v1.onnx --fp16 --saveEngine=mnv3_family_v1_fp16.plan --workspace=512`.
3. `python/web/people.json` 편집: `{0: {name: "나", emoji: "🧑"}, 1: {name: "엄마", emoji: "👩"}, ...}`.
4. `web_server.py` model 경로 갱신 (또는 CLI flag), unknown threshold 적용 (max-prob < 0.7 → "unknown" 표시).
5. 가족 한 명씩 라이브 시연.

### 미리 작성하면 좋은 것 (오늘 밤 / 내일 아침)
- `python/prep_transfer_dataset.py` 골격
- `notebooks/transfer_family_v1.ipynb` 골격
- `python/web/people.json` 가족 이름 채워두기

### S.7 ~ S.10
| # | 작업 | 의존 |
|---|---|---|
| S.7 (선택) | 본인 발걸음 raw → top-1 분포 spot-check (v2/v3 둘 다) | S.6 PASS |
| S.8 | 가족 데이터 수집 — Phase 1+2 | 위 phase |
| S.9 | Transfer learning + 라이브 검증 — Phase 3+4 | S.8 |
| S.10 | 종설 발표 자료 (시연 영상 + accuracy table + 시스템 다이어그램) | 위 모두 |

### 폴백
- 만약 향후 USB/전원 이슈 재발 → Jetson-direct ADS1256 (`python/ads1256_source.py`, `ads1256_bench.py`, `resample_for_ads1256.py` 16/15) 자산 그대로 활용 (USB 안 거치니 전원 이슈 무관)

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
├── web_server.py           # Flask + SSE 웹 대시보드 — `--stm32` / `--stm32-port` 플래그 추가됨 (S.6)
├── web/index.html          # Toss-style 대시보드
├── web/people.json         # pid → {name, emoji} (가족 시연 시 편집)
├── gmm_params.npz          # frozen GMM (P14 100s seed=0)
├── export_onnx.py          # .pth → 단일 .onnx
├── stm32_source.py         # STM32 binary frame streaming source — CALIBRATED_SCALE=3.7253e-7 V/LSB (S.5)
├── stm32_phase3_check.py   # Phase 3 (가짜 ramp) frame 검증 — Track C 자산
├── stm32_s2_check.py       # S.2/S.3 검증 (실 ADC, anchor+stride parser, frame rate sanity) — 2026-05-10
├── stm32_s5_calibration.py # P14_1.mat vs STM32 30s capture RMS 비교 + scale 후보 출력 — 2026-05-10
├── ads1256_source.py       # StreamingPolyphase + Jetson-direct 폴백 — sliding_window_view fallback 추가 (numpy 1.19.5)
└── resample_for_ads1256.py # StreamingPolyphase (16/15 + 8/15 둘 다 검증 완료)

E:\STM32CubeIDE\workspace\jetson_bridge\    # 실제 CubeIDE 프로젝트 위치 (E:\Terra\stm32\ 아님!)
├── Core/Src/main.c          # USER CODE BEGIN 2 에 RDATAC + frame builder 루프
├── Core/Src/ads1256.c       # SPI 헬퍼 + 링버퍼 + ISR 직접 레지스터 SPI 읽기
└── Core/Inc/ads1256.h       # API: ads_init_15ksps, ads_start_rdatac, ads_pop, ads_get_drops, ads_sanity_check 외
```

### Jetson `~/terra/` (flat 구조 — `python/` + `weights/` 평탄화)
```
~/terra/
├── extract.py / cwt_fast.py / render_lut.py / jetson_infer.py / jetson_realtime.py / web_server.py / stream.py
├── stm32_source.py / ads1256_source.py / resample_for_ads1256.py    # S.6 부터 deployment 라인
├── gmm_params.npz / jet_lut_v2.npy
├── mobilenet_v3_large_v2.onnx / mnv3_v2_fp16.plan       # v2 production
├── mobilenet_v3_large_v3.onnx / mnv3_v3_fp16.plan       # v3 (작성 예정 trtexec)
├── sample/         # 시연용 raw .mat
└── web/            # SSE static
```
- scp 시 `~/terra/` flat 에 직접. 하위 폴더 만들지 말 것 (스크립트 cwd 기준 상대 경로).
- **Jetson 사용자 dialout 그룹 추가 필수** (S.6 에서 발견): `sudo usermod -aG dialout $USER` 후 재로그인 — 안 하면 `/dev/ttyACM0` PermissionError.
- **pyserial 설치**: `pip3 install --user pyserial`
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

## STM32 결정사항 (CubeMX 설정 — S.6 시점 확정)

- **보드**: NUCLEO-F411RE
- **CubeIDE 워크스페이스**: `E:\STM32CubeIDE\workspace\jetson_bridge\` (CLAUDE.md 의 `stm32/ads1256_bridge/` 는 부정확, 실제 위치는 여기)
- **펌웨어 스타일**: HAL (CubeIDE)
- **전송 경로**: ST-Link VCP (UART2 → ST-Link MCU 자동 USB CDC) — 케이블 1개로 빌드/플래시/통신/전원
- **활성 peripheral**:
  - **SPI1** (PA5/PA6/PA7) — Full-Duplex Master, 8-bit MSB, CPOL=Low, **CPHA = 2 Edge** (= ADS1256 mode 1), NSS Soft, **prescaler /64** (84 MHz APB2 → 1.3125 MHz, ADS1256 한계 1.92 MHz 안전)
  - **GPIO**: PB6 = CS (Output, default high), PB1 = unused (계획 변경: ADS1256 RESET 안 씀)
  - **EXTI**: **PA9 = DRDY** (External Interrupt, Falling edge, **Pull-up**), `EXTI9_5_IRQn` priority 0
  - **USART2** (PA2/PA3) — Async, **921600 8-N-1** (S.1 디버그 동안만 115200, S.2~ 부터 921600)
- **비활성**: USB_OTG_FS / USB_DEVICE / CDC class 미들웨어 모두 깔지 않음
- **핀맵 변경 사항** (CLAUDE.md 원본 plan 과 다른 점):
  - DRDY: PB0/EXTI0 → **PA9/EXTI9** (사용자 선택)
  - ADS1256 RESET: PB1 GPIO pulse → **미연결, SPI 0xFE 소프트 리셋**
- **데이터 경로**:
  ```
  Geophone (SM-24, 28.8 V/m/s, DCR 375 Ω) ─┬─[330Ω×3 직렬 = 990Ω 댐핑]─┬─ AIN0/AIN1 차분
    → ADS1256 (BUFEN=0, PGA=16, 15000 SPS RDATAC, DRDY 66.67 µs)
    → STM32 SPI1 1.3125 MHz, EXTI9 (DRDY), ISR 안 직접 레지스터 SPI 읽기
    → 256-entry SPSC 링버퍼
    → USART2 921600 binary frame [SYNC 0xA5 0x5A | SEQ uint16 LE | N=64 | int24 BE × N | CRC8]
    → Jetson /dev/ttyACM0 → STM32Source.chunks() (CALIBRATED_SCALE = 3.7253e-7 V/LSB)
    → int24 decode → 폴리페이즈 8/15 → 8 kHz float64
    → (이하 기존 파이프라인 동일)
  ```
- **⚠️ ISR 안 SPI 는 반드시 직접 레지스터 접근**: HAL_SPI_TransmitReceive 의 ~10µs overhead 가 RDATAC 의 66.67µs 윈도우에 너무 커서 13.4 kSPS 로 떨어짐. 직접 레지스터 (`SPI1->DR` / `SPI1->SR`) 로 ISR ~30µs → ~20µs → 15000 SPS sustained.
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
- **PGA 결정** (S.1e 실측 후 확정): **PGA=16** (ADCON = 0x04, ±312.5 mV FS).
  - PGA=8 일 땐 보통 발걸음 2~8% FS 로 ADC 다이나믹 레인지 너무 적게 사용
  - PGA=16 으로 올린 후: 보통 발걸음 2~7%, hard stomp 5~11%, idle 0.78% — 80%+ 헤드룸 안전.
  - PGA=32 은 안 감. 이유: 노이즈 floor 가 PGA 따라 안 줄어 (idle ~65k LSB 동일) SNR 동일, 단지 클립 마진만 깎임.
  - 근거: ADS1256 24-bit + PGA 가 ADS1115 16-bit + 외부 op-amp 보다 noise-free bits 동등 이상. seismometer.info / Seisberry 모두 외부 amp 없이 운용.
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
