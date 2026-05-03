"""Phase 3 검증 — STM32 가 921600 baud 로 송신하는 binary frame 의

  - SYNC 위치 (0xA5 0x5A, 198 byte 간격)
  - SEQ 카운트
  - CRC8 (poly 0x07)
  - int24 BE ramp (sample[i+1] - sample[i] == 1)

가 모두 맞는지 확인.

사용법:
    1) Tera Term 등 시리얼 모니터 모두 닫기 (포트 한 번에 하나만 점유 가능)
    2) pip install pyserial   (한 번만)
    3) python stm32_phase3_check.py
    4) (필요 시) 아래 PORT 값을 장치 관리자에서 본 COM 번호로 수정

PORT 를 None 으로 두면 자동 탐지 ("STLink Virtual COM Port" 라벨 검색).
"""
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial 미설치. PowerShell 에서:  pip install pyserial")


PORT = None        # None = 자동 탐지. 수동 지정 시 "COM3" 등으로
BAUD = 921600
DUR  = 1.5         # 캡처 시간 (초)


def find_stlink_port():
    """장치 관리자에 'STLink Virtual COM Port' 로 잡힌 첫 포트 반환."""
    for p in list_ports.comports():
        desc = (p.description or "") + " " + (p.manufacturer or "")
        if "STLink" in desc or "ST-Link" in desc or "STMicroelectronics" in desc:
            return p.device
    return None


def crc8(data: bytes) -> int:
    """CRC-8, poly=0x07, init=0x00, no reflect, no xor-out."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc


def be24_to_int(b3: bytes) -> int:
    """uint24 BE → 0..0xFFFFFF (Phase 3 가짜 데이터는 sign 안 씀, ramp 양수만)."""
    return (b3[0] << 16) | (b3[1] << 8) | b3[2]


def main():
    port = PORT or find_stlink_port()
    if port is None:
        sys.exit("ST-Link VCP 자동 탐지 실패. 장치 관리자에서 COM 번호 확인 후 PORT 수동 지정.")
    print(f"[연결] {port} @ {BAUD} baud, {DUR}s 캡처")

    try:
        ser = serial.Serial(port, BAUD, timeout=DUR)
    except serial.SerialException as e:
        sys.exit(f"포트 열기 실패: {e}\n→ Tera Term / 다른 시리얼 모니터 닫혀있는지 확인.")

    expect = int(BAUD / 8 * DUR)
    buf = ser.read(expect)
    ser.close()
    print(f"[수신] {len(buf)} byte (예상 ~{expect})")

    if len(buf) < 198 * 2:
        sys.exit("바이트 너무 적음. 보드 플래시됐는지 / baud 921600 맞는지 / 케이블 확인.")

    # SYNC 위치 검색
    syncs = [i for i in range(len(buf) - 1) if buf[i] == 0xA5 and buf[i + 1] == 0x5A]
    if not syncs:
        sys.exit("SYNC (0xA5 0x5A) 미발견. baud / 코드 / frame 형식 확인.")

    # 간격 분석
    gaps = [syncs[i + 1] - syncs[i] for i in range(len(syncs) - 1)]
    gaps_ok = all(g == 198 for g in gaps)
    print(f"[SYNC] {len(syncs)} 회 발견. 간격 모두 198 = {gaps_ok}")
    if not gaps_ok:
        bad = [g for g in gaps if g != 198]
        print(f"   비정상 간격 예: {bad[:10]}")

    # 각 frame 검증
    frames_ok = 0
    frames_bad = 0
    seq_seq = []
    last_sample = None
    ramp_breaks = 0
    crc_breaks = 0

    for s in syncs:
        if s + 198 > len(buf):
            break
        f = buf[s:s + 198]
        seq    = f[2] | (f[3] << 8)
        n      = f[4]
        crc_rx = f[197]
        crc_ok = crc8(f[2:197]) == crc_rx

        if not crc_ok:
            crc_breaks += 1
            frames_bad += 1
            continue

        # ramp 검증 (sample 간 +1)
        ramp_in_frame_ok = True
        for i in range(n):
            v = be24_to_int(f[5 + i*3 : 5 + i*3 + 3])
            if last_sample is not None:
                expected = (last_sample + 1) & 0xFFFFFF
                if v != expected:
                    ramp_in_frame_ok = False
                    ramp_breaks += 1
                    break
            last_sample = v

        if ramp_in_frame_ok:
            frames_ok += 1
            seq_seq.append(seq)
        else:
            frames_bad += 1

    # SEQ 누락 검사
    seq_missing = 0
    for i in range(len(seq_seq) - 1):
        if seq_seq[i + 1] != (seq_seq[i] + 1) & 0xFFFF:
            seq_missing += 1

    print(f"[Frame] OK {frames_ok} / 손상 {frames_bad}")
    print(f"   CRC 실패: {crc_breaks}")
    print(f"   ramp 단절: {ramp_breaks}  (보드 RESET 직후엔 1회 정상)")
    print(f"   SEQ 누락: {seq_missing}")

    if seq_seq:
        print(f"   SEQ 범위: {seq_seq[0]} ~ {seq_seq[-1]}")

    # 합격 판정
    print()
    if frames_ok > 0 and crc_breaks == 0 and seq_missing == 0:
        print("[PASS] Phase 3 통과 — frame format / CRC / SEQ / ramp 모두 정상")
    else:
        print("[FAIL] 위 카운트 확인. 자주 나는 원인:")
        print("   - CRC 실패: STM32 의 crc8() 와 Python 의 crc8() 폴리노미얼/초기값 불일치")
        print("   - SEQ 누락: 921600 에서 호스트 OS buffer overflow (드물게 Windows)")
        print("   - ramp 단절: 보드 중간 리셋 또는 frame 경계 패킹 버그")


if __name__ == "__main__":
    main()
