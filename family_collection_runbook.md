# 가족 수집 Runbook — Phase 0~2

> 오늘 밤 (Phase 0) + 내일 수집 (Phase 1~2) 만 다룸. transfer 학습 (D.4) / 라이브 시연 (Phase 4) 은 수집 끝난 다음에 별도로.
>
> 마지막 수정: 2026-05-19 / 수집일: 2026-05-20

---

## 시간 예산

| Phase | 시간 | 주체 |
|---|---|---|
| Phase 0 — 사전 sanity (오늘 밤) | ~15분 | 본인만 |
| Phase 1 — 셋업 (내일) | ~10분 | 본인 |
| Phase 2 — 가족 4명 수집 | ~80분 (1명당 ~20분 × 4) | 가족 |
| Phase 2 — noise 세션 | ~10분 | 본인 |
| 마무리 백업 | ~5분 | 본인 |
| **합계 (내일)** | **~2시간** | |

---

# Phase 0 — 오늘 밤 사전 점검 (~15분)

## 0-A. 환경 / 하드웨어 체크 (3분)

- [ ] Jetson 부팅 OK (배럴잭 5V/4A)
- [ ] STM32 USB 가 SSD 와 **다른** 물리 포트 (운영 룰 — 같은 hub leg = brownout reset)
- [ ] SSD 여유공간 확인: `df -h /` → 250 MB 이상이면 OK (4명 × 3세션 ≈ 220 MB)
- [ ] 지오폰 배치: 현관-거실 경계 wood floor, walking path 1 m 이내
- [ ] 코일 + 990Ω 션트 정상 (코일 양단 차분 단자)

## 0-B. 본인 발걸음 sanity (5분)

```bash
ssh snup2@snup2-desktop
cd ~/terra
python3 record_session.py --pid presanity --session 0 --duration 30 --outdir data/recordings
```

**PASS 기준** — 출력이 `[PASS]` + 아래 모두 만족:

| 항목 | 기준 |
|---|---|
| frames_ok | ≥ 7000 |
| seq_drops | 0 |
| resets | ≤ 1 |
| footsteps detected | ≥ 5 |

**NG 대응 표** (반드시 가족 오기 전에 해결):

| 증상 | 원인 후보 | 조치 |
|---|---|---|
| `PermissionError: /dev/ttyACM0` | dialout 그룹 누락 | `sudo usermod -aG dialout $USER` 후 재로그인 |
| `ST-Link VCP not found` | USB 미연결 / 케이블 불량 | 다른 포트 / 케이블 |
| seq_drops > 0 | USB 전원 brownout | STM32 를 SSD 와 다른 포트로 |
| resets > 1 | 같은 위 + STM32 재열거 | 같은 위 + USB 케이블 짧은 것 |
| footsteps = 0 | 지오폰 단자 / 신호 부족 | 코일 차분 단자 / 위치 확인 |
| frames_bad > 10 | baud 불일치 / 전원 노이즈 | `--baud 921600` 명시 |

## 0-C. 양말 진폭 검증 (5분)

가족 중 양말 세션 (S2) 가능 여부 결정용.

```bash
python3 record_session.py --pid presanity --session sock --duration 30 --outdir data/recordings
```

- **PASS**: footsteps detected ≥ 3 → 내일 S2 (양말) 진행 OK
- **NG (footsteps 0~2)**:
  - **1차 폴백**: `python/extract.py` 의 envelope detector 호출부에 `k_mad=6.0` 명시 (현재 8.0 default). Jetson 의 `~/terra/extract.py` 도 동시 갱신.
  - **2차 폴백**: S2 (양말) skip, S1/S3 신발 두 세션만 진행. 가족당 ~15분으로 단축됨.

## 0-D. (선택) 신호 시각 확인 (5분)

record_session 의 숫자만으론 불안하면 live_monitor 로 파형 직접 보기:

```bash
python3 live_monitor.py --pid presanity_visual --duration 60
# 브라우저: http://snup2-desktop:8050/
```

체크 포인트:
- 발걸음 peak 가 ±20 mV 이상으로 명확히 보이는지
- idle 구간 noise floor 가 ±2 mV 안인지
- 50 Hz 진동이 두드러지지 않는지 (두드러지면 SNR 이슈 — CLAUDE.md `알려진 이슈` 섹션)

---

# Phase 1 — 내일 셋업 (~10분, 가족 모이기 전)

## 1-A. 부팅 / 연결

- [ ] Jetson 부팅 + 배럴잭 전원
- [ ] STM32 USB 연결 (SSD 와 다른 포트 — 운영 룰 재확인)
- [ ] 지오폰 위치 = Phase 0-A 와 **동일 위치** (옮겼다면 sanity 재실행)

## 1-B. 셋업 직후 30초 sanity (셋업 이동 후 재확인)

```bash
python3 record_session.py --pid presanity --session 1 --duration 30 --outdir data/recordings
```

PASS 안 나오면 가족 입장 전에 해결.

## 1-C. 가족 안내 스크립트 (1분)

> "센서 옆 1 m 안에서 신발장 ↔ 거실 자연스럽게 왕복.
> 의식하지 말고 평소 페이스로.
> 한 명당 5분 × 3세션. 1세션 끝나면 신발 갈아신음.
> 4명 다 끝내고 마지막에 잡음 녹음 5분 더."

---

# Phase 2 — 수집 (~90분)

## 2-A. pids + 조건 매핑

**pids**: A, B, C, D (실제 이름 ↔ A/B/C/D 매핑은 로컬 메모로만 보관, 이 문서는 PUBLIC repo 에 push 됨)

**1명당 3세션** (한 명 끝낸 후 다음 사람):

| 세션 | 조건 |
|---|---|
| S1 | 평소 신는 신발, 평상 페이스 |
| S2 | 양말, 평상 페이스 (Phase 0-C 가 PASS 한 경우만) |
| S3 | 같은 종류 신발, 약간 빠르거나 느린 페이스 |

## 2-B. 세션 실행 (예: A)

```bash
python3 record_session.py --pid A --session 1 --duration 300 --outdir data/recordings
# (신발 갈아신기 ~2분)
python3 record_session.py --pid A --session 2 --duration 300 --outdir data/recordings
python3 record_session.py --pid A --session 3 --duration 300 --outdir data/recordings
```

다른 가족도 동일 (pid 만 바꿔서).

## 2-C. 세션마다 PASS 체크

| 항목 | 기준 |
|---|---|
| `[PASS]` 출력 | yes |
| frames_ok | ≥ 70000 (5분에 7만+) |
| seq_drops | 0 |
| resets | ≤ 2 |
| footsteps detected | **50~250** |

**footsteps 가 < 30** → 그 세션만 재수집. 가족이 너무 가만히 있었거나 센서에서 멀리 걸음.

**seq_drops > 0** → 즉시 중단, STM32 케이블 / 포트 점검 후 재수집.

**다음 세션 진행은 직전 세션 PASS 후에만**. NG 그대로 두면 다음 모든 세션 영향.

## 2-D. Noise 세션 (가족 다 끝낸 후, 본인 진행, ~10분)

```bash
python3 record_session.py --pid noise --session 1 --duration 300 --outdir data/recordings
```

5분 동안 의도적 비-발걸음 임펄스 (분당 10~20회 분산):

- 문 닫기 ~10회
- 책/펜 떨어뜨리기 ~10회
- 의자 끌기 ~5회
- 책상·벽 두드리기 ~5회
- 진공청소기 30초 (가능하면)

→ D.3 에서 `--noise-pid noise` 가 unknown 클래스로 자동 병합. **빠뜨리면 라이브 시연 시 문 닫기 → "A confidence 0.9" 같은 OOD overconfidence 발생.**

---

# 마무리 (가족 헤어진 후 즉시, ~5분)

## 결과 확인

```bash
ls -la ~/terra/data/recordings/
# 13개 폴더 / .npz 가 보여야 함 (4명 × 3세션 + noise × 1 + presanity 들)
du -sh ~/terra/data/recordings/
# ~220 MB 예상
```

## 백업 (필수)

```bash
# Jetson 의 외장 SSD 또는 클라우드로
rsync -av ~/terra/data/recordings/ /mnt/external/terra_backup_2026-05-20/
# 또는 Windows 로 scp
# (Windows 에서)
scp -r snup2@snup2-desktop:~/terra/data/recordings E:/Terra/data/
```

transfer 학습 중 데이터 손상 / 실수로 덮어쓰기 방지.

## 다음 단계 (별도 시간, 권장은 그 다음 날)

1. **D.3** (Windows): `python python/prep_transfer_dataset.py --pids A B C D --out data/transfer/family_v1.npz`
2. **D.4**: `notebooks/transfer_family_v1.ipynb` (작성 예정)
3. **Phase 4**: TRT 빌드 + 라이브 시연

---

# 명령어 cheat sheet (한 줄 정리)

```bash
ssh snup2@snup2-desktop                                                       # Jetson 접속
cd ~/terra                                                                     # 작업 dir
python3 record_session.py --pid <name> --session <N> --duration 300            # 5분 수집
python3 live_monitor.py --pid <name> --duration 60                             # 시각 + 저장
ls data/recordings/                                                            # 결과 확인
du -sh data/recordings/                                                        # 용량 확인
```

라이브 모니터 URL: `http://snup2-desktop:8050/`

---

# 주의사항 (잊지 말 것)

1. **STM32 USB = SSD 와 다른 물리 포트** (운영 룰. 같은 hub leg = brownout reset 으로 seq_drop 폭증)
2. **pid 는 실명** → `data/recordings/<실명>/...` 폴더로 저장. `data/` 는 .gitignore 라 GitHub 푸시 안 됨. 그래도 외부 백업 시 익명화 권장.
3. **지오폰 위치 = Phase 0 와 동일**. 옮기면 footstep 분포 달라짐 (학습-추론 분포 미스매치).
4. **세션마다 PASS 체크**. NG 면 그 세션만 재수집, 다음 세션 진행 OK.
5. **가족 자연스럽게 걷기**. 의식하면 페이스 일정 → 학습 분산 줄어듦 → 일반화 약해짐. 평소처럼 왕복.
6. **Noise 세션 깜빡 금지**. 없으면 OOD overconfidence 로 라이브 시연 무너짐.
7. **Ctrl-C 신중**. record_session 도중 Ctrl-C 하면 `STOPPED EARLY` 로 마킹되고 PASS 안 됨. 끝까지 두기 (5분 = 300초).
8. **record_session.py 와 live_monitor.py 동시 실행 불가** (둘 다 STM32 포트 점유). 살라가락 / 자르고 다시 띄우기.

---

# Q&A (예상 질문)

**Q. 가족이 양말로만 걷고 싶다고 하면?**
A. Phase 0-C 가 PASS 했으면 S2 (양말) 도 받음. 단 S1/S3 = 신발도 받아야 within-person 분산 확보됨. 양말만이면 신발 시연 시 OOD.

**Q. 도중에 친척이 추가로 와서 4명 → 5명 되면?**
A. 5번째 pid 추가해서 받기는 OK. D.3 의 `--pids` 인자에 추가하면 5-class + unknown = 6-class. classifier head 만 다시 짜면 됨 (1280 → 6). 큰 변경 아님.

**Q. 신발 종류 / 발 크기를 라벨로 안 빼도 되나?**
A. 도메인 다양성 포기 결정 (2026-05-19). "집 현관에 설치된 시스템" 컨텍스트 고정. 신발/양말은 within-person 분산일 뿐, 라벨은 사람 1개.

**Q. footsteps detected 가 250 초과면?**
A. 너무 빠른 페이스거나 short-step. 그 세션은 정상 분포에서 벗어남. 재수집은 안 해도 학습엔 들어감 (다양성 +). 단 너무 빈번 (>400) 이면 보고 — 진동 노이즈일 가능성.
