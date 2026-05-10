"""S.2 검증 — STM32 가 921600 baud 로 송신하는 실 ADC binary frame 의

  - SYNC 위치 (0xA5 0x5A, 198 byte 간격)
  - SEQ 카운트
  - CRC8 (poly 0x07)
  - frame rate (목표: 15000 / 64 = 234.375 fps)
  - 샘플 통계 (signed int24, idle ~수만 LSB, stomp 시 수십만)

ramp 검증은 안 함 (Track C 가짜 데이터와 다름). 실 ADC noise/signal 분포만 확인.

사용법:
    1) 다른 시리얼 모니터 (TeraTerm 등) 모두 닫기
    2) python stm32_s2_check.py            # 기본 3초 캡처
       python stm32_s2_check.py 30         # 30초 (S.3 sustained 테스트)
"""
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial 미설치.  pip install pyserial")


PORT = None
BAUD = 921600
FRAME_SIZE = 198
FRAME_N = 64
SAMPLE_RATE = 15000   # SPS


def find_stlink_port():
    for p in list_ports.comports():
        desc = (p.description or "") + " " + (p.manufacturer or "")
        if "STLink" in desc or "ST-Link" in desc or "STMicroelectronics" in desc:
            return p.device
    return None


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc


def be24_to_int(b3: bytes) -> int:
    """signed int24 BE → int."""
    v = (b3[0] << 16) | (b3[1] << 8) | b3[2]
    if v & 0x800000:
        v -= 0x1000000
    return v


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    port = PORT or find_stlink_port()
    if port is None:
        sys.exit("ST-Link VCP 자동 탐지 실패. 장치 관리자에서 COM 번호 확인.")
    print(f"[연결] {port} @ {BAUD} baud, {dur}s 캡처")

    try:
        ser = serial.Serial(port, BAUD, timeout=dur + 0.5)
    except serial.SerialException as e:
        sys.exit(f"포트 열기 실패: {e}\n→ TeraTerm 등 닫혀있는지 확인.")

    # 캡처 시작
    expected_bytes = int(BAUD / 10 * dur)   # 8N1 = 10 bits/byte
    t0 = time.monotonic()
    buf = ser.read(expected_bytes)
    elapsed = time.monotonic() - t0
    ser.close()
    print(f"[수신] {len(buf)} byte (예상 ~{expected_bytes}, elapsed {elapsed:.2f}s)")

    if len(buf) < FRAME_SIZE * 10:
        sys.exit("바이트 너무 적음. 보드 플래시됐는지 / baud 921600 / 케이블 확인.")

    # Anchor + stride 파싱: 첫 valid SYNC (다음 SYNC 가 +198 에 또 있는 위치) 부터
    # 198 byte stride 로 frame 추출. payload 안 가짜 0xA5 0x5A 무시.
    anchor = -1
    for i in range(len(buf) - FRAME_SIZE - 1):
        if (buf[i] == 0xA5 and buf[i + 1] == 0x5A
            and buf[i + FRAME_SIZE] == 0xA5 and buf[i + FRAME_SIZE + 1] == 0x5A):
            anchor = i
            break
    if anchor < 0:
        sys.exit("Anchor SYNC 미발견. baud / 펌웨어 / 케이블 확인.")
    print(f"[Anchor] 첫 valid SYNC = byte {anchor}")

    # 각 frame 검증 + 샘플 수집
    frames_ok = 0
    crc_breaks = 0
    sync_misalign = 0
    seq_seq = []
    all_samples = []

    pos = anchor
    while pos + FRAME_SIZE <= len(buf):
        f = buf[pos:pos + FRAME_SIZE]
        if f[0] != 0xA5 or f[1] != 0x5A:
            sync_misalign += 1
            # resync: 다음 valid anchor 찾기
            new_anchor = -1
            for j in range(pos + 1, len(buf) - FRAME_SIZE - 1):
                if (buf[j] == 0xA5 and buf[j + 1] == 0x5A
                    and buf[j + FRAME_SIZE] == 0xA5 and buf[j + FRAME_SIZE + 1] == 0x5A):
                    new_anchor = j
                    break
            if new_anchor < 0:
                break
            pos = new_anchor
            continue

        if crc8(f[2:FRAME_SIZE - 1]) != f[FRAME_SIZE - 1]:
            crc_breaks += 1
            pos += FRAME_SIZE
            continue

        seq = f[2] | (f[3] << 8)
        seq_seq.append(seq)
        for i in range(FRAME_N):
            all_samples.append(be24_to_int(f[5 + i*3 : 5 + i*3 + 3]))
        frames_ok += 1
        pos += FRAME_SIZE

    print(f"[SYNC] misalign {sync_misalign} (= UART 손실 / drop event 횟수)")

    # SEQ 누락
    seq_missing = 0
    for i in range(len(seq_seq) - 1):
        d = (seq_seq[i + 1] - seq_seq[i]) & 0xFFFF
        if d != 1:
            seq_missing += d - 1

    # 통계
    n_samples = len(all_samples)
    fs_full = 8388608  # signed int24 max
    if n_samples:
        vmin = min(all_samples)
        vmax = max(all_samples)
        vmean = sum(all_samples) / n_samples
        # RMS (AC, mean 제거)
        ac = [x - vmean for x in all_samples]
        rms = (sum(x*x for x in ac) / n_samples) ** 0.5
        peak = max(vmax - vmean, vmean - vmin)
    else:
        vmin = vmax = vmean = rms = peak = 0

    fps = frames_ok / elapsed if elapsed > 0 else 0
    eff_sps = fps * FRAME_N
    expected_fps = SAMPLE_RATE / FRAME_N

    print(f"[Frame] OK {frames_ok} / CRC fail {crc_breaks} / SEQ 누락 {seq_missing}")
    if seq_seq:
        print(f"        SEQ 범위: {seq_seq[0]} ~ {seq_seq[-1]}")
    print(f"[Rate]  {fps:.1f} fps ({eff_sps:.0f} SPS), 목표 {expected_fps:.1f} fps ({SAMPLE_RATE} SPS)")
    print(f"[Stats] N={n_samples}  min={vmin}  max={vmax}  mean={vmean:.1f}")
    print(f"        peak={peak}  ({100.0*peak/fs_full:.2f}% FS)")
    print(f"        RMS={rms:.0f}    ({100.0*rms/fs_full:.3f}% FS)")

    # 합격 판정
    print()
    sps_tol = 0.05  # ±5% 허용
    sps_ok = abs(eff_sps - SAMPLE_RATE) / SAMPLE_RATE < sps_tol
    if frames_ok > 0 and crc_breaks == 0 and seq_missing == 0 and sps_ok:
        print("[PASS] S.2 통과 — frame format / CRC / SEQ / 샘플레이트 정상")
    else:
        print("[FAIL] 위 카운트 확인. 자주 나는 원인:")
        print("   - CRC fail: STM32 / Python crc8 폴리노미얼 불일치")
        print("   - SEQ 누락: ISR 처리 못 따라가서 ring buffer overflow → ads_drops 누적")
        print(f"   - SPS 부족 ({eff_sps:.0f} vs {SAMPLE_RATE}): UART 송신이 frame interval 보다 김")


if __name__ == "__main__":
    main()
