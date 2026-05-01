# %%
# === CELL 13b — RESUME (세션 복구, 재렌더 없음) ===
# Drive에 보존된 history.json + .pth만으로 cell 14를 돌릴 수 있게 최소 상태 복구.
# val 재실행 안 하고 ckpt['val_acc']로 대체.

import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "device:", DEVICE)

from google.colab import drive
drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive/vibeid_capstone")
WEIGHTS_DIR = DRIVE / "weights"


def load_hist(name):
    p = WEIGHTS_DIR / f"{name}_v2_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


mnv3_hist = load_hist("mobilenet_v3_large")
effb0_hist = load_hist("efficientnet_b0")
print(f"mnv3_hist : {len(mnv3_hist) if mnv3_hist else 0} epochs  "
      f"best={max((x['val_acc'] for x in mnv3_hist), default=0)*100:.2f}%")
print(f"effb0_hist: {len(effb0_hist) if effb0_hist else 0} epochs  "
      f"best={max((x['val_acc'] for x in effb0_hist), default=0)*100:.2f}%"
      if effb0_hist else "effb0_hist: 없음 (cell 13 미실행)")


def build_model(arch, num_classes, pretrained=False):
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(arch)


print("\n저장된 체크포인트 무결성:")
NUM_CLASSES = None
for path in sorted(WEIGHTS_DIR.glob("*_v2_best.pth")):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    NUM_CLASSES = int(ck["num_classes"])
    m = build_model(ck["arch"], NUM_CLASSES, pretrained=False)
    miss, unexp = m.load_state_dict(ck["model_state_dict"], strict=False)
    status = "OK" if not miss and not unexp else f"miss={len(miss)} unexp={len(unexp)}"
    n_param = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  {path.name}: arch={ck['arch']}, ep={ck['epoch']+1}, "
          f"val={ck['val_acc']*100:.2f}%, {n_param:.1f}M  [{status}]")
    del m, ck

print(f"\nNUM_CLASSES = {NUM_CLASSES}")
print("복구 완료. cell 14 실행 가능.")
