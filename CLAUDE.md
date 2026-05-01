# Terra — Footstep Recognition Edge Deployment Project

## 프로젝트 목표
원본 Terra 레포지토리 (MATLAB 기반 발걸음 분류 연구 코드)를 **Python으로 포팅**하여 **Jetson Nano P3450**에서 **실시간 추론**을 수행한다. 원작자의 데이터 처리 파이프라인을 정확히 재현하는 것이 1차 목표.

## 현재 단계: 센서 도착 대기 / 병행 작업 진행 (2026-05-01 갱신)
**Stage 1~5d + 다중 인원 voter (5e) + 웹 대시보드 (7) 모두 PASS.** raw .mat → GMM → CWT(cwt_fast) → LUT → TRT FP16 → top-1 → multi-presence voter → SSE → Toss-style 웹페이지가 Jetson Nano에서 end-to-end 작동. P14/P50/P100/P2 정확도 93~97%, per-footstep 386ms, `--realtime` wall-clock 8 kHz throttle 통과, producer/consumer threading 검증, 실시간 시연 가능.

**ADS1256 + STM32 F411RE bridge 아키텍처 확정**: ADS1256 → STM32 F411RE Nucleo (HAL, ST-Link VCP) → UART2 921600 → Jetson `/dev/ttyACM0` → polyphase 8/15 (15000→8000). **마감 2026-06.**

**현재 블로커 = ADS1256 모듈 식별 + STM32 펌웨어 작성**. 그동안 병행 가능한 작업 다수 — VIBeID A2/A3 통합 backbone 재학습 (A100), 다중 인원 합성 검증, 가족 데이터 수집 protocol 설계, STM32 CubeIDE 프로젝트 골격. 자세한 task 트리는 본 문서 하단 `## ⏭️ 다음 단계` 참조.

### v2 핵심 변경 (2026-04-30)
- v1 (`mobilenetv3_vibeid_a1_best.pth`, 84.69%)은 Colab matplotlib 픽셀에 brittle — 로컬 LUT 입력 시 P50 7%, P100 8% 처참.
- v2 (`mobilenet_v3_large_v2_best.pth`, val **86.76%**) — augmentation 강화 (ColorJitter↑/RandomErasing/MixUp/label smoothing 0.15) + **LUT 학습** + per-class stratified 80/20 split (seed=42).
- 로컬 검증: P14 93%, P50 98%, P100 95% (raw → 추출 → LUT). brittleness 완전 해결.
- **JET LUT 결정론화**: Colab matplotlib 3.7 jet → `weights/jet_lut_v2.npy` (256×3 uint8, sha256 `fc594b04...`). `render_lut.py`가 이 .npy를 우선 로드.

### ⚠️ 데이터 라벨링 이슈 (2026-04-30 발견)
- `data/raw/P1/` 폴더의 `P1_1..4.mat`은 **실제로는 P2 (a1.mat label==2) 데이터**.
- 사용자가 1-indexed↔0-indexed 변환할 때 off-by-one 발생.
- 검증: `python/diagnose_p1_label.py` — raw P1 추출 1363개 footsteps이 100/100 a1.mat label==2와 corr 0.999로 매칭.
- P14/P50/P100은 정상 (자기 라벨과 99-100% 매칭).
- **폴더 안 건드림** (옵션 B). 추론 검증 시 raw "P1" → 정답 라벨 P2로 사용.

### 추론 파이프라인 (확정)
```
raw .mat (8 kHz)
  → extract.apply_frozen_gmm(geo, gmm_params.npz)   # P14로 학습한 frozen GMM
  → footsteps (M, 1500) float64
  → cwt_fast.cwt(sig)                               # FFT + cached freq-domain wavelets (7.6× over pywt)
  → render_lut.cwt_to_rgb_direct(coeffs, (224,224)) # npy LUT + cv2.resize
  → MobileNetV3-Large(weights/mobilenet_v3_large_v2_best.pth)
  → top-1 class
```

---

## 하드웨어 / 환경 제약

### 추론 타겟: Jetson Nano P3450 (원본, Orin 아님)
- Quad Cortex-A57 @ 1.43 GHz, Maxwell 128-core GPU, 4 GB LPDDR4
- **JetPack 4.6.6-b24 (EOL), CUDA 10.2, cuDNN 8.2.1, TensorRT 8.2.1, PyTorch 1.10.0**
- ADC: **ADS1256 구매 완료** (SPI, 24-bit, 30 kSPS 가능)
- 진동 신호 샘플링: **Fs = 8 kHz**

### 학습 환경 (완료)
- 백본: **MobileNetV3-Large** (ImageNet pretrained → A1 100-class fine-tune)
- 입력: PNG는 디스크에 84×496 RGBA, **학습/추론은 224×224 RGB로 리사이즈 + ImageNet normalize**
- **Best val acc: 84.69%** (15 epochs, AdamW lr=3e-4, label smoothing 0.1, AMP, cosine)
- 가중치 17 MB `.pth` Drive 백업: `/content/drive/MyDrive/vibeid_capstone/weights/mobilenetv3_vibeid_a1_best.pth`
- Worst classes: P72 58%, P67 59% / Best: P4 100%, P55 99.65%

---

## 데이터셋 (현재 로컬 부분집합)

`E:\Terra\data\` 하위:
```
data/
├── raw/         # P1 (6 mat), P14 (6 mat), P50 (4 mat), P100 (4 mat)
│                #   변수명: geo_data, shape=(2400000, 1) per 5-min file
├── interim/
│   └── a1.mat   # 1.5 GB, scipy.io.loadmat (v7, HDF5 아님)
│                #   footstep_feat: (144_371, 1_501) float64
│                #   cols [0:1500]=발걸음 waveform (zero-padded), col[1500]=label (1~100)
└── processed/   # P1=1639, P14=1751, P50=510, P100=839 PNG (84×496 RGBA)
                 #   파일명: cwt_image_{person0idx}_{footstep_idx}.png
                 #   예: P50의 첫 발걸음 → cwt_image_49_0.png
```

**확정된 사실**: a1.mat row #i (label==pid 필터링) ↔ `processed/Pn/cwt_image_{pid-1}_{i}.png` 가 1:1 매칭. 카운트 정확히 일치.

---

## VIBeID 데이터셋 전체 구조 (논문 ICLR 2025 under-review)

VIBeID 는 **A1~A4 네 개의 서브셋** 으로 구성됨. 모두 동일 센서/샘플레이트 (geophone 2.88 V/m/sec, gain 10, Logic sound card hat 16-bit ADC, **Fs = 8 kHz**) 사용. 본 프로젝트에서 현재 사용 중인 a1.mat 은 그중 A1 만.

| 서브셋 | 인원 | 조건 변동 | 길이 | 샘플 수 (footsteps) | 용도 |
|---|---|---|---|---|---|
| **A1** | 100 | 단일 floor, 단일 거리(2.5-4.0m) | 33.66 h | ~187,500 (논문) / **144,371 (a1.mat)** | Person Identification (현재 사용) |
| **A2.1** | 30 | 거리 1.5 m, **cement** floor | (15min×30 each) | 33,151 | Multi-Distance |
| **A2.2** | 30 | 거리 2.5 m, cement | | 19,494 | |
| **A2.3** | 30 | 거리 4.0 m, cement | total 22.5 h | 23,030 | |
| **A3.1** | 40 | 거리 2.5-4.0m, **wooden** floor | | 63,394 | Multi-Structure |
| **A3.2** | 40 | 거리 2.5-4.0m, **carpet** | total 30 h | 54,840 | (실배포 robustness 핵심) |
| **A3.3** | 40 | 거리 2.5-4.0m, **cement** | | 34,328 | |
| **A4.1** | 15 | **outdoor** + 카메라 동시 | | 2,171 | Multi-Modal (geophone) |
| **A4.2a/b** | 15 | 위 동일, dual cameras | total 2.5 h | 2,670 / 2,074 | Multi-Modal (vision) |

**중요 사실**:
- A1/A2/A3 의 **피험자 집합은 별개** (overlap 가능성은 있으나 논문에서 별도 명시 X). 즉 A1 100명 + A2 30명 + A3 40명 = 단순 합치면 ~170 클래스 (overlap 미고려).
- 논문이 정의한 cross-domain task: A2.1↔A2.2↔A2.3 / A3.1↔A3.2↔A3.3 (총 6×2=12개 transfer task) — domain adaptation 평가 protocol.
- **A4 만 outdoor**, A1~A3 은 indoor.
- A1 의 floor type 은 supplementary 에만 명시 (paper 본문 미기재).
- 논문 a1.mat 카운트 187,500 vs 우리 a1.mat 144,371 차이 — 논문은 raw event extraction, 우리 a1.mat 은 후처리/필터링된 publication 버전인 듯 (확정 X).

**프로젝트 페이지**: vibeidiclr.github.io (License CC BY-NC-SA 4.0)

**우리 사용 현황**:
- 현재: A1 일부 (P1, P14, P50, P100 raw + 전체 a1.mat) 만 다운로드
- v2 backbone 학습: A1 100-class 만 사용 → val 86.76%
- A2/A3 미사용 → **실배포 시 floor/distance 차이로 인한 일반화 성능 손실 예상지점**

---

## MATLAB 파이프라인 — 정밀 사양 (코드 정독 결과)

### `event_detection/NewDatasetCreation2.m` — Pn_*.mat 통합
- 항상 `Pn_1.mat` ~ `Pn_4.mat` (4개)만 vstack → `Pn_all.mat` 저장
- ⚠️ **a1.mat 출판본은 P1을 5 파일로 처리한 것으로 추정** (코드 4 파일과 불일치, Stage 1b 카운트 비교 결과)

### `event_detection/main.m` — GMM 학습 + 발걸음 추출
| 파라미터 | 값 |
|---|---|
| Fs | 8000 Hz |
| GMM 학습 입력 | 첫 100초 (`n=100*Fs`)의 한 사람. 코드는 P1 하드코딩, **논문은 P14 명시** |
| 스무딩 | `smooth(geo_data, 5)` — MATLAB moving avg (endpoint adaptive) |
| 윈도우 | **0.35 sec = 2800 샘플** (코드 주석 "35ms"는 오기) |
| 오버랩 | 40% |
| 윈도우 가중 | `gausswin(N, 1.2)` (tau=1.2) |
| 분류 임계 | 사후확률 < 0.90 → noise (label 0) |

### `Events_Features_Extraction.m` — 윈도우당 7-D 특징
1. `std(sig)` (ddof=1)
2. `kurtosis(sig)` (non-Fisher, biased)
3. `rms(sig)` = `sqrt(mean(sig²))`
4. `quantile(sig, 0.25)` (Hyndman-Fan type 5 / Hazen)
5. `||FFT||²` in 40-80 Hz (NFFT = 8 * nextpow2(L) = 32768 for L=2800)
6. `||FFT||²` in 80-120 Hz
7. `||FFT||²` in 120-160 Hz

→ 논문의 "134 features"는 ML 베이스라인용 별개 toolkit, 이 레포 코드와 무관.

### `GMM_EM.m` — 자체 EM 구현
- 2 클러스터, `randperm`로 행 인덱스 2개 뽑아 평균 초기화 ⚠️ 비결정성
- 공분산 초기화: 전체 데이터 cov (두 클러스터 동일)
- 정규화: `+ 0.0001 * I` (특이행렬 방지)
- 수렴: `||μ_new||₂ - ||μ_prev||₂ < 1e-12` (operator 2-norm)
- |Σ_event| > |Σ_noise| 인 쪽이 footstep 클러스터

### `Event_Extract.m` — 1500-샘플 발걸음 추출
- 연속된 "event" 윈도우 그룹화
- 각 그룹에서 `argmax(|signal|)` 위치 = 피크
- 시작 = 피크 − 400, 길이 = 1500 샘플
- Tukey window (alpha=0.5) 적용 후 저장
- 경계 처리: 신호 끝/시작 넘어가면 truncate + zero-pad

### `specmeaker.py` (Python, 원본) — CWT 이미지 생성
```python
coefficients, _ = pywt.cwt(sig_1500, np.arange(1, 257), 'morl')  # (256, 1500)
plt.imshow(coefficients, cmap='jet', aspect='auto')
plt.axis('off')
plt.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
```
- ⚠️ figsize 명시 안 됨 — 출판본은 환경 rcParams로 84×496 RGBA. 우리는 `figsize=(4.96, 0.84), dpi=100` 명시로 매칭.

---

## ✅ 검증 결과 (Stage 1~3 모두 PASS)

### Stage 1a — a1.mat 카운트 sanity → PASS
- P1=1639, P14=1751, P50=510, P100=839 모두 PNG 카운트와 정확 일치

### Stage 1b — Python 포팅 vs a1.mat → PASS (caveat 있음)
GMM 학습: P14 첫 100초, seed=0, EM 113 iter 수렴

| Person | a1.mat | 추출 | Δ% | med corr | 비고 |
|---|---|---|---|---|---|
| P1 (4 files) | 1639 | 1363 | -16.8% | 0.9329 | corr 한계 = intra-class p95 (0.9364) |
| P1 (5 files) | 1639 | 1703 | **+4%** | 0.9316 | 카운트는 5 파일로 일치 → 출판본은 5 파일 |
| P14 | 1751 | 1488 | -15% | 0.9992 | **byte-exact 매칭** |
| P50 | 510 | 267 | -47.6% | 0.9984 | byte-exact |
| P100 | 839 | 629 | -25% | 0.9985 | byte-exact |

**해석**:
- 3/4 사람 corr 0.999 → 추출 waveform이 a1.mat 행과 사실상 byte-equal
- P1은 발걸음 자체가 변동성 큼 (intra-class median=0.63, p95=0.94) → 우리 0.93은 **ceiling 도달**
- 카운트 미달은 GMM 보수적 분류 결과 — 시드/임계값 변경해도 5% 이내 변동 (튜닝 영향 미미)
- **Jetson 추론 시 a1.mat 매칭 불필요** — 일관된 추출만 하면 됨

### Stage 2 — CWT 변환 → PASS
`pywt.cwt(scales=1..256, morl)` 정상, NaN 없음, 출력 (256, 1500) shape.

### Stage 3 — PNG 재현 → PASS
| 비교 샘플 | shape | max\|Δ\| | mean\|Δ\| | RMSE |
|---|---|---|---|---|
| P1, P14, P50, P100 (6 샘플) | (84, 496, 4) | 24-33 | 1.08-1.45 | 2.37-3.25 |

mean RMSE = **2.87/255 = 1.1%** << acceptance 5/255. 픽셀 차이는 matplotlib LUT 버전/안티에일리어싱 정도.

### Stage 5 — Jetson 실배포 (offline 정확도) → PASS
**환경**: JetPack 4.6.6 / Python 3.6.9 / numpy 1.19.5 / pywt 1.1.1 / TRT 8.2.1.8 / pycuda 2022.1

**TRT FP16 plan** (`mnv3_v2_fp16.plan`, 9.3 MB):
- median latency 11.43 ms, p99 11.51 ms, throughput ~88 QPS (single batch)
- verify (random N=16): max\|Δ\| logits 4.48e-2, top-1 16/16 일치 → FP16 양자화 정상

**End-to-end** (sample/P14_1.mat, 5분 8kHz 신호, Stage 5 baseline):
- 추출: 361 footsteps in 15.1s (10.6 ms/window)
- CWT+LUT 렌더: 548.1s (**1518 ms/footstep ← 실시간 병목**)
- TRT 추론: 23.3 ms/sample (Python loop 포함)
- 분류: **P14 93.4%** (337/361), 평균 conf 74.1%. 오분류 24개 모두 conf 35-41% (잘 calibrate)
- ⇒ **Colab/Windows 로컬 93%와 일치** — pywt 1.8 → 1.1 차이도 v2 augmentation margin 안

### Stage 5b — CWT 최적화 (`cwt_fast.py`) → PASS
**병목 분석** (`jetson_cwt_bench.py`로 측정, scales 1..256, morl, 1500-sample):
| variant | ms/footstep | speedup | pixel diff |
|---|---|---|---|
| pywt method='conv' (Stage 5) | 1486 | 1.0× | reference |
| pywt method='fft' | 262 | 5.7× | 0 |
| **cwt_fast (FFT + cached wavelets)** | **195** | **7.6×** | **0** |

`cwt_fast.cwt`는 morl integrated wavelet의 FFT 표현 256개를 module-level 캐싱 → 매 호출 1× FFT(signal) + 256-batch IFFT만. pywt 1.1.1의 `method='fft'`는 매 scale마다 FFT(int_psi)를 다시 계산하는 게 손해.

**End-to-end 재측정** (sample/P14_1.mat):
- 추출: 15.4s (이전과 동일)
- 렌더 (CWT + LUT + resize): **121.4s = 336 ms/footstep** (이전 1518의 4.5×). CWT 단독은 195 ms이고 나머지 ~141 ms는 LUT lookup + cv2.resize (안 건드림).
- 추론: 23.2 ms/sample (이전과 동일)
- 분류: **P14 93.4% (337/361) — 분포 bit-identical** (P14/P15/P21/P92/P29 카운트, conf 모두 일치). 픽셀 diff 0이 정확히 입증됨.

**budget 체크**:
```
per-footstep 처리 = 추출 42ms + 렌더 336ms + TRT 23ms ≈ 400ms
목표 budget 700ms (보통 걸음 1.5 Hz = 660ms 간격) → 1.7× 마진 통과
```

ADS1256 통합 시 single-step latency 충분, 3-step voting까지 해도 end-to-end 1.5~2초 안에 들어옴.

### Stage 5c — 실시간 fake stream 파이프라인 (`jetson_realtime.py`) → PASS
**구성**: `MatFileSource` (1024-sample chunk 단위로 .mat → stream) → `StreamingSmoother` (matlab_smooth byte-exact, 2-sample latency) → `SlidingDetector` (2800/40% window + frozen GMM + run grouping → 1500 footstep) → `cwt_fast` → `render_lut` → TRTEngine → `Voter` (K=5/M=3 majority, transition-only emit).

**Windows 사전 검증** (`validate_realtime.py`, TRT 없이):
- StreamingSmoother (chunk size 1/137/1024/9999): max\|diff\| = **0**
- SlidingDetector (3 chunk sizes, pre-smoothed): 361 footsteps, max\|diff\| = **0**
- 전체 파이프라인 (raw → smoother → detector): 361 footsteps, max\|diff\| = **0** vs offline

**Jetson 실측** (sample/P14_1.mat):
- 361 footsteps in 139.3s = **2.15× realtime**, 386 ms/footstep
- 분포: P14 337 (93.4%, conf 74.1%) / P15 6 / P21 6 / P92 3 / P29 2 → **Stage 5b와 비트 단위 일치**
- voter: P14 confirm 1회 (#4 footstep, conf 79.8%) — 동일 클래스 transition-only emit이라 정상

**Cross-person spot check** (raw .mat 3개 추가 검증):
| 파일 | top-1 결과 | 정확도 | conf | realtime |
|---|---|---|---|---|
| P50_1.mat | P50 58/61 | 95.1% | 75.7% | 6.64× |
| P100_1.mat | P100 145/154 | 94.2% | 66.0% | 3.91× |
| P1_1.mat | **P2** 343/352 | 97.4% | 79.9% | 2.10× |

- P1 폴더 데이터의 진짜 라벨이 P2임이 추가 확인됨 (raw → top-1 P2 강하게 나옴, 라벨링 이슈 검증).
- frozen GMM (P14 100s 학습)이 P50/P100/P2 신호에서도 정상 발걸음 검출 (일반화 확인).
- 모든 오분류 conf 21-46% → voter (conf_min=0.5) 통과 못 함, 실시간 false-positive 위험 0.

**의의**: streaming smoother + sliding detector 로직이 offline과 byte-equal. ADS1256 통합 시 파이프라인 코드는 의심 0 — `MatFileSource`를 `ADS1256Source`로 갈아끼우면 끝. 8 kHz 입력 (=125 µs/sample)에 대해 큐 누적 0 (worst case 2.15× 마진).

### Stage 5d — `--realtime` wall-clock throttle + producer/consumer threading → PASS
**A. `--realtime` 베이스라인 (single-thread, P14_1.mat)**: 300.0s 정확 (1.00× realtime), 분포 P14 337/361 비트 단위 일치. single-thread도 strict 8 kHz 따라잡음 (footstep burst가 chunk 도착 간격 초과해도 평균적 idle로 catch-up).

**B. Producer/Consumer threading 리팩토링** (`jetson_realtime.py`):
- `queue.Queue(maxsize=32)` 분리, producer 스레드는 `src.chunks()` → put, consumer (메인 스레드)는 get → smoother → detector → infer.
- 종료 신호 SENTINEL 패턴, 큐 만석 시 `full_waits` 카운트.
- `--realtime` P14_1.mat 재실행: 300.0s, 분포 동일, **max_depth=16/32, full_waits=0**.
- max_depth 16은 footstep burst (4~5 연속) 동안 producer가 consumer 앞서가는 결정적 증거 — single-thread였으면 0~1.

**의의**: ADS1256 strict timing (catch-up 불가, 늦으면 sample drop) 대응 완료. ADS1256 producer 스레드는 SPI 전담, queue 16 chunks 여유. `queue_max=32`는 여유 충분. consumer 지연 발생 시 `full_waits` > 0으로 surface.

### Stage 5e — 다중 인원 voter (`Voter` 멀티 presence) → PASS (2026-05-01)
**동기**: 단일 채널 지오폰으로도 발걸음이 시간상 안 겹치는 ~74% 케이스 + CWT 선형성으로 겹치는 26% 일부도 식별 가능. 기존 voter (single-track + transition-only emit)는 두 사람 alternating 시 항상 한 명만 confirm, 두 번째 영원히 누락 → multi-presence 로 전환.

**새 `Voter` 구조** (`python/jetson_realtime.py`):
- 각 pid 가 K-window 안에 ≥M confident hits 쌓이면 독립적으로 confirm. 다수 pid 동시 confirm 가능.
- 한 pid 가 K-window 안에 0 hits → forget (재진입 시 fresh enter event).
- **K_MAX = 12 하드 캡** (가정 보행 burst ≤ 10 step 물리 제약). 그 이상 인자 들어오면 강제 클립 + 경고.
- 기본 K=10 / M=3 / conf_min=0.5. (이전 K=5/M=3에서 K만 증가).
- API 변경: `update()` 가 `Optional[int]` → `List[int]` (newly-confirmed pids). caller (jetson_realtime.replay, web_server.process_footstep) 모두 list iterate 로 갱신.

**검증 (단위 테스트, 5 케이스)**:
| 시나리오 | 출력 | 검증 의도 |
|---|---|---|
| P14 단독 ×20 | `[13]` 한 번 | 단일 인원 거동 보존 (Stage 5c 분포 영향 0) |
| P14 + sparse P15 (1.7%) | `[13]` | 실측 노이즈 비율에서 false positive 0 |
| P14↔P50 alternate | `[13, 49]` step 5,6 | 다중 인원 둘 다 confirm |
| K=20 입력 | clip → 12 + 경고 | 하드 캡 작동 |
| P14→P50→P14 | `[13, 49, 13]` | forget/re-emit 작동 |

**presence ↔ voter 동기화** (`python/web_server.py`):
- `Voter` 를 `main()` 에서 생성 → `PresenceTracker(on_away=voter.forget)` 콜백으로 sweep_away 발생 시 voter confirmed-set 정리.
- voter 가 confirmed 유지 중인데 presence 가 시간 경과로 away 처리한 stale state 방지.

**Stage 5c 회귀 0**: 단일 인원 P14_1.mat 시연 시 P14 1회 confirm 결과 동일.

**다중 인원 합성 검증 → PASS** (`validate_multi_presence.py`, 2026-05-02). P14+P50 (P50 sparse) interleave 45s 블록 + P14+P100 (둘 다 dense) interleave 15s/30s 블록 → 6/6 케이스에서 두 사람 모두 confirm. 30s 블록 케이스에서 forget/re-emit 동작 실증 (P14 confirm → P100 phase → P14 phase 복귀 시 P14 재confirm). OOD sample-level 신호 합산 (`sum 1:1`) 은 분류기 학습 분포 밖이라 P41 같은 임의 클래스 출력 — 단일 채널 근본 한계, 종설 narrative 에 명시적 한계로 기록.

### Stage 7 — 실시간 웹 대시보드 (Flask + SSE) → PASS (2026-05-01)
**동기**: Jetson 출력 → 노트북/폰 브라우저로 "지금 집에 누가 있는지" 시각화. ADS1256 도착 후 본인 / 가족 시연 즉시 가능하도록 dashboard 미리 구축.

**스택**:
- **백엔드**: Flask 2.0.3 (Python 3.6 호환) + Server-Sent Events (text/event-stream).
- **파이프라인 wrapper** (`python/web_server.py`): `jetson_realtime` 컴포넌트 (MatFileSource/Smoother/Detector/Voter) 그대로 import. `/events` SSE, `/state` JSON snapshot, `/` static HTML.
- **CUDA 스레드 binding**: `pycuda.autoinit` 가 import 스레드에 context 묶음 → consumer 스레드에서 `from jetson_infer import TRTEngine` + `TRTEngine()` + `engine.infer()` 모두 처리 (main 스레드에서 import 시 `invalid resource handle` 발생 → 한 번 디버그함).
- **프론트엔드** (`python/web/index.html`): Pretendard 폰트 + 토스블루(#3182F6) + 라운드 큰 카드. 데스크탑 760px+ 에선 2단 그리드 (좌: 레이더+홈 그리드 / 우: 최근 발자취+stats), 모바일은 자동 1단.
- **레이더 SVG**: 회전 sweep cone (3.6s 주기) + center ping (2.2s) + voter-confirmed 사람마다 영구 점 (이름 라벨, 발걸음 들어올 때 부드러운 glow). raw inference 가 아니라 **시스템 결정 (voter confirm) 만** 시각화 → 오분류 노이즈 화면 안 들어옴.

**이벤트 흐름** (SSE JSON):
- `hello` (연결 시 초기 스냅샷 + 사람 매핑)
- `inference` (매 footstep, 라이브 ticker용)
- `confirm` (voter 새 entry, 토스트 + 홈 그리드 + 레이더 점 추가)
- `stats` (1 Hz, footsteps/queue_depth/elapsed)
- `away` (sweep_after 만료, 토스트 + 점 fade out)

**사람 매핑** (`python/web/people.json`): pid → {name, emoji}. 가족 데이터 수집 후 P1~P100 → "아빠/엄마/형/동생" 등으로 자유 편집. 미매핑 pid 는 fallback "P{N+1}" + 🧑 이모지.

**런타임 두 모드**:
- `python3 web_server.py --replay sample/P14_1.mat --plan ... --gmm ... --realtime` (Jetson 진짜 파이프라인)
- `python web_server.py --demo` (Windows, TRT 없이 가짜 이벤트로 UI 디자인 iter)

**의의**: ADS1256 도착 시 `MatFileSource` → `STM32Source` 한 줄 교체로 실시간 시연 즉시 가능. 가족이 와이파이 안에서 폰으로 `http://<jetson-ip>:8000` 접속 → 누가 들어왔는지 실시간 알림.

---

## 핵심 기술 결정 (v2 기준 업데이트)

1. **백본: MobileNetV3-Large + augmentation 강화 (v2).** v1 (84.69%)은 환경 brittle해서 폐기. v2 `mobilenet_v3_large_v2_best.pth` (val 86.76%)이 production. 입력 224×224 RGB + ImageNet normalize.
2. **CWT 유지** — pywt 1.8.0 (학습), 1.1.1 (Jetson). Jetson Nano CPU에서 1500 샘플 ms 단위.
3. **추론 시 matplotlib 제거** — `weights/jet_lut_v2.npy` (Colab matplotlib 3.7 jet LUT을 .npy로 박은 것). `render_lut.py`가 이걸 우선 로드, 없으면 fallback (loud warning).
4. **frozen GMM** — `python/gmm_params.npz` (P14 100s seed=0 학습 결과). 추론 시 학습 단계 스킵.
5. **학습-추론 픽셀 일치 보장 안 함** — v2 augmentation으로 robustness margin 충분히 확보됨. Stage 4에서 RMSE 1.26%, v2 모델은 그 이상도 견딤 (Path C 95%+).
6. **CWT 병목 해소 = `cwt_fast.py`** (Stage 5b). pywt.cwt → manual FFT + cached freq-domain wavelets (7.6× speedup, pixel-equivalent). 재학습 0. GPU CWT / scales 축소는 가지 않음 (Maxwell GPU 약함, scales 축소는 재학습 필요). 새 병목은 LUT+cv2.resize ~141 ms이지만 budget 안이라 안 건드림.
7. **per-footstep budget = 700ms** (Stage 5b에서 결정). 이유: 보통 걸음 1.5 Hz → 660ms/step, 처리가 이보다 오래 걸리면 큐 누적. 사람 체감(UX) budget은 별개로 2~3초 (foot-land → 앱 표시), 3-5 step voting으로 99% 정확도 달성하므로 single-step latency를 더 짤 필요 없음.

---

## ⚠️ Jetson 환경 quirks (2026-04-30 세션 발견)

이 환경에서 **하지 말아야 할 것**과 **반드시 해야 할 것**:

### 환경변수 (`~/.bashrc`에 박아둠)
```bash
export OPENBLAS_CORETYPE=ARMV8       # numpy "Illegal instruction" 픽스 (Cortex-A57)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
- `OPENBLAS_CORETYPE=ARMV8` 없으면 numpy 1.19.5 aarch64 wheel이 ARMv8.2 명령어 사용 → SIGILL.

### pycuda 설치 — 반드시 `--no-build-isolation`
```bash
python3 -m pip install --user --no-build-isolation pycuda
```
- isolation 모드면 numpy 1.12.1 옛날 버전을 새로 빌드하려 함 → `xlocale.h` 없어서 실패.

### Python 3.6 호환 패치 (이미 적용됨)
모든 .py 파일에서:
- `from __future__ import annotations` 제거 (3.7+ 기능)
- PEP 604 `int | None` → `Optional[int]`
- PEP 585 `tuple[X, Y]` → `Tuple[X, Y]`
- `argparse.add_subparsers(required=...)` 사용 금지 (3.7+)
- `np.quantile(method='hazen')` 안 됨 → `extract.py`에 수동 Hazen fallback (numpy 1.19.5)

### Claude Code / VS Code Remote-SSH **둘 다 안 됨**
- glibc 2.27 (Ubuntu 18.04) — Node 18+ / VS Code Server는 glibc 2.28+ 요구
- **워크플로우**: Windows 쪽에서 Claude Code 띄우고 SSH 터미널 + scp로 Jetson 다룸. 출력은 복붙으로 공유.
- `JETSON_SETUP.md`의 Claude Code 설치 섹션은 **무시** (deprecated).

### ONNX export — 반드시 단일 파일로
- 현 PyTorch (≥2.x) 기본 export가 external data 형식이라 `.onnx` (graph) + `.onnx.data` (weights) 분리됨.
- `.data` 파일 빠뜨리면 TRT가 `Failed to open file: ...onnx.data` 로 빌드 실패 (실제로 한 번 발생).
- **재 export 스크립트**: `python/export_onnx.py` — torchvision MobileNetV3-Large 빌드 + `.pth` 로드 + opset 13 단일 파일 export. 16.5 MB 단일 .onnx 생성 (size < 10 MB면 weight 누락 의심).

### LUT 경로 자동 탐색 (`render_lut.py`)
- `$TERRA_JET_LUT` env var 먼저, 없으면 candidate 순회:
  1. `Path(__file__).parent / "jet_lut_v2.npy"` (Jetson: `~/terra/`)
  2. `Path(__file__).parent.parent / "weights" / "jet_lut_v2.npy"` (Windows: `E:/Terra/weights/`)
- 양쪽 환경에서 코드 수정 없이 작동.

---

## 디렉토리 구조

```
E:\Terra\
├── CLAUDE.md                    # ← 이 파일
├── README.md                    # 원본 README
├── requirements.txt             # 원본
├── specmeaker.py                # 원본 CWT 생성 (참고용)
├── supplimentary_files.pdf      # 원본 보충자료
├── event_detection/             # 원본 MATLAB 코드 (포팅 원본)
├── Demographic Details/         # 피험자 메타데이터 CSV
├── paper/                       # 논문 PDF
├── notebook/                    # 백본 학습 ipynb (val 84.69% 완료)
├── stm32/                       # STM32 F411RE Nucleo 펌웨어 (CubeIDE) — 2026-04-30 후반 신규
│   └── ads1256_bridge/          # SPI bridge 프로젝트: ADS1256 → USB CDC
├── data/                        # 부분 데이터셋
│   ├── raw/                     # P1, P14, P50, P100
│   ├── interim/a1.mat
│   └── processed/
├── weights/                     # 모델 + LUT
│   ├── mobilenet_v3_large_v2_best.pth   # v2 (production, val 86.76%)
│   ├── mobilenet_v3_large_v2.onnx       # ONNX (Jetson용)
│   ├── jet_lut_v2.npy                   # 결정론적 jet LUT (Colab matplotlib 3.7)
│   └── mobilenetv3_vibeid_a1_best.pth   # v1 (legacy, brittle, 비교용 보존)
└── python/                      # 포팅 + 추론 작업물
    ├── extract.py                       # MATLAB → Python (smooth/gausswin/GMM-EM/Event_Extract) — Hazen fallback 포함
    ├── cwt_fast.py                      # FFT + cached freq-domain wavelets (7.6× over pywt) — Stage 5b production
    ├── render_lut.py                    # CWT → 224×224 RGB (npy LUT) — 경로 자동 탐색, cwt_fast 호출
    ├── stream.py                        # raw → footsteps → inputs 묶음 (cwt_fast 호출, batch mode)
    ├── jetson_realtime.py               # 실시간 streaming (MatFileSource/Smoother/Detector/Voter 멀티-presence) — Stage 5c+5e PASS
    ├── validate_realtime.py             # Windows에서 jetson_realtime 컴포넌트 byte-exact 검증
    ├── web_server.py                    # Flask + SSE 웹 대시보드 (jetson_realtime wrap) — Stage 7 PASS
    ├── web/
    │   ├── index.html                   # Toss-style 대시보드 (Pretendard, 레이더, 2단 레이아웃)
    │   └── people.json                  # pid → {name, emoji} 매핑 (가족 시연 시 편집)
    ├── validate_multi_presence.py       # P14+P50/P100 temporal interleave → 다중 voter PASS (2026-05-02)
    ├── record_session.py                # [작성 예정] 가족 발걸음 데이터 수집 스크립트
    ├── ads1256_source.py                # [LEGACY/폴백] Jetson-direct RDATAC + 폴리페이즈 16/15 (7500→8000)
    ├── ads1256_bench.py                 # [LEGACY/폴백] Jetson-direct SPI ladder + Plan B trigger
    ├── resample_for_ads1256.py          # StreamingPolyphase 검증 (16/15 검증됨, 8/15 모드 추가 예정)
    ├── stm32_source.py                  # [작성 예정] USB CDC reader + int24 decode + 폴리페이즈 8/15 (15000→8000)
    ├── stm32_validate.py                # [작성 예정] STM32 캡처 vs scipy reference byte-equal 검증
    ├── infer.py                         # 로컬 PyTorch 추론 (Path A vs Path C 검증)
    ├── jetson_infer.py                  # TRT inference + verify (Jetson용, Python 3.6 호환)
    ├── jetson_cwt_bench.py              # CWT variants 벤치마크 (Stage 5b에 사용, 재실행 가능)
    ├── export_onnx.py                   # .pth → 단일 .onnx 재 export (external data 회피)
    ├── gmm_params.npz                   # frozen GMM (P14 100s seed=0)
    ├── JETSON_SETUP.md                  # Jetson 환경 셋업 가이드 (Claude Code 섹션은 deprecated)
    ├── validate_stage{1,2_3,4}.py       # Stage 1~4 검증 스크립트
    ├── diagnose_*.py                    # 진단 (path_c, render, versions, p1_label)
    └── _out_stage23/                    # 렌더 PNG (참조 비교용)
```

---

## ⏭️ 다음 단계 — 실시간 최적화 + ADS1256 통합

### 완료 (2026-05-01 기준)
- [x] MATLAB → Python 포팅 (`extract.py`, Stage 1~3 검증)
- [x] LUT 결정론화 (`weights/jet_lut_v2.npy`, sha256 fc594b04...)
- [x] frozen GMM (`python/gmm_params.npz`, P14 100s seed=0)
- [x] v2 학습 (Colab A100, val 86.76%)
- [x] ONNX 단일 파일 export (`weights/mobilenet_v3_large_v2.onnx`, 16.5 MB, opset 13)
- [x] **Jetson Nano TRT FP16 plan** (`mnv3_v2_fp16.plan`, 9.3 MB, 11.4 ms latency)
- [x] **Jetson offline 정확도 검증** (P14 93.4% — Stage 5 PASS)
- [x] Python 3.6 호환 패치 (4개 .py 파일)
- [x] Hazen quantile / LUT 경로 plumbing (`extract.py`, `render_lut.py`)
- [x] **CWT 병목 해소** (`cwt_fast.py`, Stage 5b PASS, 7.6× CWT / 4.5× end-to-end, P14 93.4% bit-identical)
- [x] **실시간 fake stream 파이프라인** (`jetson_realtime.py` + `validate_realtime.py`, Stage 5c PASS, P14 분포 비트 단위 일치, 2.15× realtime)
- [x] **Cross-person spot check** (P50 95.1% / P100 94.2% / P1=P2 97.4%, Stage 5c)
- [x] **`--realtime` wall-clock throttle + producer/consumer threading** (Stage 5d PASS, max_depth 16/32, full_waits 0)
- [x] **ADS1256 Jetson-direct producer 코드 + StreamingPolyphase 16/15 검증** (Stage 6a PASS — `ads1256_source.py` / `ads1256_bench.py` / `resample_for_ads1256.py`, streaming = block @ 1e-16). **STM32 bridge 채택으로 폴백용 보존**.
- [x] **다중 인원 voter** (`Voter` multi-presence, K_MAX=12 캡, on_away 콜백 sync — Stage 5e PASS 2026-05-01)
- [x] **실시간 웹 대시보드** (`web_server.py` + `web/index.html`, Flask SSE + Toss-style 레이더 UI — Stage 7 PASS 2026-05-01)
- [x] **VIBeID A1~A4 전체 데이터셋 구조 정리** (paper 정독 → CLAUDE.md 본문 추가, A2/A3 학습 plan 도출)

### Stage 6 — ADS1256 통합: STM32 F411RE bridge 아키텍처 (2026-04-30 후반 갱신)

#### 아키텍처 변경 배경
초기 plan = Jetson Nano direct SPI (7500 SPS RDATAC + busy-wait DRDY polling). Stage 6a에서 producer 코드 + 폴리페이즈 16/15 검증까지 완료된 상태. **2026-04-30 후반 재논의에서 STM32 F411RE bridge로 변경**:
- **학습 가치**: 사용자가 임베디드시스템설계 수강 중 + 종합설계프로젝트 통합 점수에서 heterogeneous compute (MCU 하드 실시간 + edge GPU 추론) 가 단순 Jetson SPI 대비 narrative 우월.
- **타이밍 risk 제거**: Linux scheduler / Python GIL / spidev xfer 지터 모두 무관해짐. STM32 베어메탈 EXTI는 <100 ns 응답.
- **상한 해소**: Jetson direct는 7500 SPS가 사실상 한계지만 (Waveshare "analog SPI" 경고), STM32는 15000 SPS도 여유 (M4 100 MHz, sample 당 6666 cycle).
- **마감 여유**: 2026-06까지 시간 충분, CubeIDE 사용자 익숙.

#### 사전 조사 결과 — Jetson-direct 시 발견 (배경 참고)
1. **ADS1256는 8000 SPS를 직접 지원 안 함**. data rate 메뉴 16개 고정값 (2.5/5/.../7500/15000/30000 SPS). 폴리페이즈 필요 — STM32 루트에서도 동일.
2. **단일 채널 RDATAC 모드면 DRDY 주기 = 정확히 1/data_rate**. 채널 cycling은 우리 시나리오 아님 (진동 1채널 AIN0).
3. **Python Jetson.GPIO callback latency 한계 ~150 Hz**. STM32 EXTI는 <100 ns로 무관.
4. ADS1256 SCLK 한계 ~1.92 MHz (fCLKIN/4 = 7.68 MHz/4). 24-bit read = 12.5 µs at 1.92 MHz.
5. ADS1256 칩 자체 BUFEN bit (STATUS 레지스터): 고임피던스 소스(geophone) 입력 시 ON 권장. 노이즈 floor 약간 증가하지만 source impedance 영향 회피.

#### 새 결정 사항 (D1'~D5')
- **D1' 샘플 레이트**: **15000 SPS native** (DRATE = 0xE0). STM32에서 7500 SPS 한계 없음 + anti-alias 마진 ×2.
- **D2' 학습 데이터**: 옵션 (a) 유지. STM32 raw bridge → Jetson 폴리페이즈 8/15 (15000 → 8000). 학습 자산 (frozen GMM, v2 모델, Stage 1~5 검증) 100% 보존.
- **D3' STM32 펌웨어 분담**: transparent bridge. **STM32는 SPI/EXTI/USB CDC만**, DSP는 Jetson. 펌웨어 단순화 + 기존 검증된 StreamingPolyphase 재사용.
- **D4' 검증**: (a) STM32 출력 byte stream → numpy raw int24 디코딩, (b) 30초 캡처 RMS vs `data/raw/P14/P14_1.mat` 첫 30초 RMS = scale calibration, (c) 폴리페이즈 8/15 검증 (`resample_for_ads1256.py`에 8/15 모드 추가, max|d| < 1e-15 확인), (d) bench 60초 — missed sample 0 / drift < 0.1% / USB frame CRC error 0.
- **D5' Plan B**: STM32 → 7500 SPS로 강하 (학습 자산 영향 0, 폴리페이즈 16/15로 스왑). 그래도 안 되면 → Jetson direct 폴백 (Stage 6a 자산 그대로 활용).

#### 데이터 경로 요약
```
Geophone (passive coil)
  → AIN0/AIN1 (differential, BUFEN=1)
  → ADS1256 (15000 SPS RDATAC, DRDY pulses every 66.67 µs)
  → STM32 F411RE Nucleo (HAL, CubeIDE)
       SPI1 @ ~1.5 MHz, mode 1 (CPOL=0/CPHA=1), MSB-first
       DRDY → EXTI (PB0 falling edge)
       ring buffer (uint8[3] × 64 = 192 byte, lock-free SPSC)
       USART2 (PA2/PA3) @ 921600 baud → ST-Link MCU 가 USB CDC로 자동 변환
       binary frame: SYNC(0xA5 0x5A) / SEQ uint16 LE / N=64 / int24 BE × N / CRC8
  → Jetson Nano (ST-Link USB 케이블 1개 = 전원 + 플래시 + 통신)
       /dev/ttyACM0 read → STM32Source.chunks_raw()
       int24 decode → polyphase 8/15 → 8 kHz float32
       (이후 StreamingSmoother/Detector/TRT 동일)
```

#### 전송 경로 / 펌웨어 스타일 결정 (2026-05-01)
- **펌웨어 스타일 = HAL** (CubeIDE). 사용자 임베디드시스템설계 수업이 HAL 기반.
- **전송 경로 = ST-Link VCP (Virtual COM Port)**, 진짜 USB OTG CDC 아님. 이유: NUCLEO-F411RE 보드에 user USB 커넥터 없음 (ST-Link CN1 USB 한 개뿐). PA11/PA12 USB OTG 핀에 외부 마이크로 USB 커넥터 빼는 것보다, **UART2 (PA2/PA3) → ST-Link MCU 자동 USB CDC 변환** 경로가 케이블 1개로 끝나서 압도적 간편.
- **케이블 워크플로우**: Nucleo CN1 (보드측 micro-B 또는 mini-B, 보드 리비전 따라) ↔ USB-A (Jetson). 노트북에 꽂으면 CubeIDE 빌드/플래시, 그대로 뽑아 Jetson에 꽂으면 통신 + 전원 (JP1 점퍼 기본 U5V) 동시 해결. CubeIDE 디버그도 같은 케이블.
- **Baud rate = 921600** (8-N-1, no flow control). 우리 페이로드 360 kbps + frame overhead → 약 470 kbps. 921 kbps의 51% 사용, 여유 ×2.
- **CubeMX 설정 결정사항**:
  - **활성**: SPI1 (full-duplex master, 8-bit MSB, CPOL=0/CPHA=2 Edge, NSS Disable, prescaler /64), GPIO (PB6=CS output, PB1=RESET output), EXTI0 (PB0 falling, no pull), USART2 (Async, 921600, NVIC priority 5), NVIC (EXTI0 priority 0 highest)
  - **비활성**: USB_OTG_FS, USB_DEVICE 미들웨어, CDC class 미들웨어 모두 깔지 않음
- **펌웨어 송신 코드**: `HAL_UART_Transmit(&huart2, frame, 198, HAL_MAX_DELAY)` 한 줄로 끝, ST-Link 측은 펌웨어 0줄 추가 작업.
- **Jetson 측에서 보이는 모습**: `/dev/ttyACM0` (ST-Link VCP). USB OTG CDC 와 enumerate 결과 동일 — `stm32_source.py`는 pyserial로 read만 하면 됨, 어느 경로인지 모름.

#### 학습 자산에 미치는 영향 (D2' 결정 결과)
- 다운스트림 (StreamingSmoother / SlidingDetector / cwt_fast / TRT) 코드 변경 0. `MatFileSource` → `STM32Source` 한 줄 교체.
- 폴리페이즈 8/15: input Nyquist 7500 Hz, output Nyquist 4000 Hz → 진정한 decimation (anti-alias 필터). 신호 대역 40-160 Hz는 4000 Hz의 4% 위치이므로 passband ripple 무시 가능, alias rejection ~75 dB.
- `resample_for_ads1256.py`에 (p=8, q=15) 모드 추가 검증 필요 — 기존 16/15 검증 코드 패턴 그대로.

#### 작업 순서 (다음 세션 ~ 2026-06)
1. **ADS1256 모듈 식별** (Waveshare HAT vs generic breakout) → 핀헤더 위치 확정.
2. **STM32 펌웨어 작성** (CubeIDE, HAL): SPI1 + DMA, EXTI0 (DRDY), USART2 송신 (921600), TIM watchdog, 링버퍼. CubeMX 설정은 위 "전송 경로 / 펌웨어 스타일 결정" 섹션 그대로.
3. **bring-up 단계별 검증** (각 단계 통과 후 다음으로):
   (a) USART2로 1초마다 "tick" 송신 → 노트북 CubeIDE/Jetson에서 `/dev/ttyACM0` 확인
   (b) PB1 (RESET) GPIO 토글 → 멀티미터 확인
   (c) ADS1256 RREG(STATUS) → 0x01 응답 (SPI 통신 OK)
   (d) Self-cal + RDATAC + 1 sample → 0V 입력 시 ~수 µV 노이즈 확인
   (e) 15000 SPS sustained 30초 → frame 카운트 = 7031 ± 1
4. **`stm32_source.py` 작성** (Jetson side): pyserial로 `/dev/ttyACM0` open, frame parser (SYNC/SEQ/CRC8 검증), int24 BE decoder, 폴리페이즈 8/15.
5. **`stm32_validate.py`**: STM32 캡처 30초 → numpy 처리 vs scipy `resample_poly(x, 8, 15)` reference, max|d| < 1e-15 확인.
6. **calibration**: 30초 캡처 RMS vs `P14_1.mat` 첫 30초 RMS → scale 결정.
7. **`jetson_realtime.py` 통합**: MatFileSource → STM32Source 교체.
8. **본인 발걸음 raw → top-1 분포 spot-check** (100-class 샤프함은 기대 X, 일관성만 확인).
9. **(선택) 4명 데이터 수집 + transfer learning** (1280→4 분류기 교체).

#### 폴백 (Jetson-direct, Stage 6a 자산)
- `python/ads1256_source.py`, `ads1256_bench.py`, `resample_for_ads1256.py` (16/15) 그대로 보존.
- STM32 루트 막힘 시 (예: USB CDC throughput 부족, 펌웨어 디버깅 timeout) 즉시 활성화.

### 보류 항목
- 벡터화 GMM (현 10.6 ms/window는 budget 안, 우선순위 낮음)
- LUT + cv2.resize 추가 최적화 (현 141 ms는 budget 안)
- Stage 1b 카운트 정밀화 (GMM 시드 그리드, 5 vs 4 파일) — 정확도 영향 미미

---

## 🚧 센서 도착 전 병행 가능 작업 (2026-05-01 ~ ADS1256 도착)

ADS1256 모듈 식별 + STM32 펌웨어 작성이 메인 블로커. 그 사이 다음 4 트랙 병행:

### 트랙 A — VIBeID A2/A3 통합 backbone 재학습 (Colab A100, 가장 임팩트 큼)
**목표**: A1 단독 학습된 v2 의 floor/거리 일반화 한계를 극복. 본인 집 floor type 이 A1 의 floor 와 다를 가능성 큼 → A3 (multi-floor: wood/carpet/cement) 가 핵심.

| 단계 | 작업 | 환경 | 예상 시간 |
|---|---|---|---|
| A.1 | VIBeID 프로젝트 페이지 (`vibeidiclr.github.io`) 에서 A2 + A3 raw 다운로드. 용량 미상 — 53시간 8 kHz 데이터, 20-30 GB 추정 | Windows | 다운로드 시간만 |
| A.2 | A2/A3 raw → frozen GMM 통과 → 1500-sample footsteps. `extract.py` 그대로 reuse, 클래스 ID 충돌 없게 P101~ 부여 | Windows | 1-2일 |
| A.3 | CWT → LUT → PNG 일괄 렌더 (`cwt_fast` + `render_lut`). A1 PNG 와 같은 disk 레이아웃 | Windows | 1일 (멀티프로세스) |
| A.4 | 통합 split: A1 100 + A2 30 + A3 40 = 최대 170 클래스 (subject overlap 미고려). Per-class stratified 80/20, seed=42 동일 | Colab | < 1시간 |
| A.5 | v3 학습: MobileNetV3-Large, 동일 augmentation (ColorJitter↑/RandomErasing/MixUp/label smoothing 0.15), 15-25 epochs, AMP, cosine | **A100** | 4-6시간 |
| A.6 | 검증: A1 holdout val acc 비교 (v2 86.76% baseline), 더 중요한 cross-domain — A3.1 학습 → A3.2 평가 (논문 protocol) | Colab | < 1시간 |
| A.7 | ONNX → Jetson TRT FP16 plan 재빌드 (`export_onnx.py` → `trtexec`) | Jetson | 30분 |
| A.8 | `mnv3_v3_fp16.plan` 으로 P14_1.mat 재추론 — Stage 5c 와 분포 비교 | Jetson | < 1시간 |

**기대 효과**: 본인 집 시연 시 floor 다른 환경에서도 일관된 embedding → 가족 transfer learning 적은 데이터로 high accuracy.

**리스크**:
- A2/A3 는 A1 과 다른 피험자 집합 → 통합 시 단순 라벨 부여 시 170-way classification 으로 task 어려워짐 (각 클래스당 데이터 양은 줄어들지만 backbone embedding 다양성 ↑).
- Domain-aware 학습 (Domain Adversarial / GRL) 까지 가면 작업량 큼. **1차는 단순 통합 학습** 으로 baseline 잡고, 효과 미흡 시 도메인 적대 학습 검토.

### 트랙 B — 다중 인원 voter 합성 검증 (Windows, 빠름) ✅ DONE 2026-05-02
**목표**: Stage 5e voter 가 진짜 multi-presence 케이스에서 작동 확인.

**결과** (`python/validate_multi_presence.py`, 90s/case, K=10/M=3):
- Solo P14/P50/P100 controls: **3/3 PASS** (각자만 confirm, 오인 0)
- Temporal interleave 45s P14+P50: **PASS** (P14 #4, P50 #52)
- Temporal interleave 30s P14+P100: **PASS** (P14 #4 → P100 #34 → P14 #53 re-emit)
- Temporal interleave 15s P14+P100: **PASS** (빠른 alternation, forget/re-emit 동작 실증)
- OOD sum 1:1 (sample-level 합산): **FAIL → 예상된 한계**. 분류기가 P41 (학습된 제3자) 출력 → 학습 분포 밖 입력은 단일 채널로 풀 수 없음. 종설 narrative 에 한계 명시.

P50_1.mat 의 footstep 밀도 0.11 Hz (가정 실측 1.5 Hz 의 1/14) → 짧은 블록은 K=10 안에 P50 ≥M 누적 안 됨. 가정 보행 정상 밀도 (P14, P100) 는 15s 블록도 통과.

### 트랙 C — STM32 사전 준비 (CubeIDE 프로젝트 골격, 모듈 도착 무관)
**목표**: ADS1256 모듈 도착 즉시 펌웨어 디버깅 시작 가능하도록 사전 작업.

| 단계 | 작업 | 비고 |
|---|---|---|
| C.1 | NUCLEO-F411RE 보드 단독으로 CubeIDE 프로젝트 생성 + CubeMX 설정 (SPI1/EXTI0/USART2/GPIO — 본 문서의 "전송 경로 / 펌웨어 스타일 결정" 섹션 그대로) | ADS1256 0V 입력으로 진행 가능 |
| C.2 | bring-up (a): USART2 1초 tick — `/dev/ttyACM0` 에서 노트북/Jetson 로 tick 수신 확인 | ADS1256 무관 |
| C.3 | bring-up (b): PB1 (RESET) GPIO 토글 멀티미터 확인 | ADS1256 무관 |
| C.4 | binary frame 파서 (`stm32_source.py` 일부) 미리 작성. 더미 frame 보내서 SYNC/SEQ/CRC8 디코딩 테스트 | Jetson side |
| C.5 | 폴리페이즈 8/15 검증 (`resample_for_ads1256.py` 에 8/15 모드 추가, scipy `resample_poly` reference 와 max\|d\| < 1e-15 확인) | Windows side |

**예상 시간**: 2-3일.

### 트랙 D — 가족 데이터 수집 protocol 설계 (수집은 sensor 도착 후)
**목표**: 센서 도착 즉시 가족 데이터 수집 들어갈 수 있도록 사전 준비.

| 단계 | 작업 | 산출물 |
|---|---|---|
| D.1 | 수집 protocol 문서화: 가족 N명, 각자 5분 × 3 세션 (신발/시간대 다르게), 같은 위치/거치, 라벨링 룰 | docs/COLLECTION_PROTOCOL.md |
| D.2 | `record_session.py` 작성: STM32Source 받아서 30초/5분 세션 단위 파일 저장 (timestamp + label). 라이브 RMS 표시로 신호 정상 여부 확인 | Python script |
| D.3 | Transfer learning 아키텍처 결정: A. classification head 교체 (1280→N+1, +1 = unknown class) vs B. embedding + nearest-centroid. **두 개 다 구현해서 가족 데이터로 비교** | docs/TRANSFER_PLAN.md |
| D.4 | Unknown class 핸들링 디자인: max-prob threshold (0.7?) 또는 centroid 거리 임계. 친척 방문 시 false-confirm 방지 | spec |

**예상 시간**: 1일 (문서 + script 골격, 실제 수집은 sensor 후).

---

## 🔌 센서 도착 후 순차 작업

| # | 작업 | 의존성 | 비고 |
|---|---|---|---|
| S.1 | ADS1256 모듈 식별 (Waveshare HAT vs generic breakout) → 핀헤더 위치 확정 | ADS1256 도착 | |
| S.2 | STM32 펌웨어 ADS1256 통신 부분 (RREG STATUS / Self-cal / RDATAC) 추가 | C.1~C.4 + S.1 | |
| S.3 | 15000 SPS sustained 30초 → frame 카운트 = 7031 ± 1 | S.2 | |
| S.4 | `stm32_source.py` Jetson side 완성 + `stm32_validate.py` (scipy 8/15 reference 비교) | C.4 + S.3 | |
| S.5 | calibration: 30초 캡처 RMS vs `P14_1.mat` 첫 30초 RMS → scale 결정 | S.4 | |
| S.6 | `web_server.py` 의 source 를 `MatFileSource` → `STM32Source` 교체 | S.5 | 한 줄 수정 |
| S.7 | 본인 발걸음 raw → top-1 분포 spot-check (v3 모델로) | A.7 + S.6 | |
| S.8 | 가족 데이터 수집 (D.1, D.2 protocol 따라) | D.* + S.6 | 1주 |
| S.9 | Transfer learning (head 교체 vs embedding) → 가정 deployment 정확도 평가 | D.3 + S.8 | Colab 하루 |
| S.10 | 종설 발표 자료 준비 (최종 시연 영상 + accuracy table + 시스템 다이어그램) | 위 모두 | 마감 2026-06 |

---

## 사용자 정보 / 협업 스타일

- 한국어 소통
- 코드 작성 전 설명/타당성 분석 선호 (큰 결정은 미리 합의)
- Claude Code CLI (로컬 파일 직접 접근)
- 백본/Colab/Jetson은 사용자 직접 다룸. Claude는 코드 작성 + 로컬 검증 담당.
- Windows 11 + bash + Python 3.13.5 (h5py, scipy, numpy, pywavelets, matplotlib, pillow 설치됨)
- ⚠️ Windows 콘솔 cp949 → 한글/이모지 출력 시 `PYTHONIOENCODING=utf-8` 필요
