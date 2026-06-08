#!/usr/bin/env bash
# run_family_demo.sh — Terra 가족 발걸음 인식 라이브 시연 런처 (Jetson)
#
# 시연용 튜닝: voter K=4/M=2/conf=0.45 (2~3걸음 ~2초 확정, 빠릿한 체감).
# production 안정 모드로 돌리려면: ./run_family_demo.sh --stable
#
# 사용:
#   cd ~/terra
#   tmux new -s demo            # 끊김 방지 (권장)
#   ./run_family_demo.sh        # 시연 모드 (snappy)
#   ./run_family_demo.sh --stable   # 안정 모드 (K=5/M=3, 깜빡임 최소)
#   브라우저: http://snup2-desktop:8000
set -e

cd "$(dirname "$0")"

PLAN="${TERRA_TRT_PLAN:-mnv3_family_v1_fp16.plan}"
PORT="${TERRA_STM32_PORT:-}"          # 비우면 자동 탐지 (ST-Link VCP)

# ---- voter 프리셋 ----
if [ "$1" = "--stable" ]; then
  VK=5; VM=3; VC=0.50; MODE="stable (K=5/M=3)"
else
  VK=4; VM=2; VC=0.45; MODE="demo/snappy (K=4/M=2/conf=0.45)"
fi

# ---- 사전 점검 ----
if [ ! -f "$PLAN" ]; then
  echo "[!] TRT plan 없음: $PLAN"
  echo "    먼저 빌드: /usr/src/tensorrt/bin/trtexec --onnx=mobilenet_v3_family_v1.onnx --fp16 --saveEngine=$PLAN --workspace=512"
  exit 1
fi
if [ ! -f web/people.json ]; then
  echo "[!] web/people.json 없음 — 라벨->이름 매핑 필요"
  exit 1
fi

echo "=================================================="
echo " Terra Family Demo"
echo "   plan   : $PLAN"
echo "   voter  : $MODE"
echo "   labels : $(python3 -c "import json;m=json.load(open('web/people.json'));print(', '.join('%s=%s'%(k,v['name']) for k,v in sorted(m.items())))" 2>/dev/null)"
echo "   open   : http://snup2-desktop:8000"
echo "=================================================="

ARGS="--stm32 --plan $PLAN --vote-k $VK --vote-m $VM --vote-conf $VC"
[ -n "$PORT" ] && ARGS="$ARGS --stm32-port $PORT"

exec python3 web_server.py $ARGS
