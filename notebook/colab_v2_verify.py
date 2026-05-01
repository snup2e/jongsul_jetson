# %% [markdown]
# # v2 brittleness 해결 검증 (Colab, A100)
#
# v1 .pth는 Colab matplotlib 출력에만 동작 → P50=7%, P100=8% 처참.
# v2 .pth는 LUT 파이프라인으로 학습됨 (val 86.76%) → 모든 사람 LUT 입력에서 정상 분류돼야 함.
#
# 이 노트북은:
# 1. v2 .pth 로드
# 2. a1.mat에서 P1/P14/P50/P100의 row 200개씩 → LUT → 모델 추론 → per-person top-1
# 3. 추가로 val_idx subset 재현 (선택, 86.76% 재확인)
#
# raw .mat 추출은 안 함 — Stage 1b에서 이미 byte-equal 검증됨.
# 여기서 통과하면 Jetson 배포 OK.

# %%
# === CELL 1 — 환경 + 모델 로드 ===
import subprocess
subprocess.run(["pip", "install", "-q",
                "pywavelets==1.8.0",
                "opencv-python-headless==4.13.0.92"], check=True)

import time, hashlib
from pathlib import Path
import numpy as np
import scipy.io
import pywt
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from joblib import Parallel, delayed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive/vibeid_capstone")
WEIGHTS_PATH = DRIVE / "weights/mobilenet_v3_large_v2_best.pth"
A1_PATH = DRIVE / "data/a1.mat"


def build_model(arch, num_classes):
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(arch)


print("\nloading model...")
ck = torch.load(str(WEIGHTS_PATH), map_location=DEVICE, weights_only=False)
NUM_CLASSES = int(ck["num_classes"])
model = build_model(ck["arch"], NUM_CLASSES).to(DEVICE)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print(f"  arch={ck['arch']}, num_classes={NUM_CLASSES}, "
      f"saved val_acc={ck['val_acc']*100:.2f}%, ep={ck['epoch']+1}")


# %%
# === CELL 2 — LUT 렌더 함수 + a1.mat 4명 로드 ===
JET_LUT_RGB_U8 = (cm.get_cmap("jet")(np.arange(256) / 255.0)[:, :3] * 255.0
                  ).round().astype(np.uint8)


def render_one(sig_1500, size=(224, 224)):
    coeff, _ = pywt.cwt(sig_1500, np.arange(1, 257), "morl")
    cmin, cmax = float(coeff.min()), float(coeff.max())
    if cmax - cmin < 1e-12:
        idx = np.zeros(coeff.shape, dtype=np.uint8)
    else:
        idx = np.clip(((coeff - cmin) / (cmax - cmin) * 255.0).round(),
                      0, 255).astype(np.uint8)
    return cv2.resize(JET_LUT_RGB_U8[idx], (size[1], size[0]),
                      interpolation=cv2.INTER_AREA)


# byte-equal 검증 (학습 때와 동일 hash 나와야 정상)
EXPECTED = {
    0:   "1e77eccbbe5dd3f78e7cbf3f7c0016af1a641de8391ea46c74b419cd3d526e13",
    100: "6a389b24c6994970bd2142fe5022872abeb970c0fb1e401aadadd71595cf3c7d",
}
print("loading a1.mat...")
A1 = scipy.io.loadmat(str(A1_PATH))["footstep_feat"]
LABELS = A1[:, -1].astype(np.int16)
print(f"  shape={A1.shape}")

print("byte-equal hash 검증:")
for i, exp in EXPECTED.items():
    got = hashlib.sha256(render_one(A1[i, :-1]).tobytes()).hexdigest()
    print(f"  row {i:>3}: {got}  [{'OK' if got == exp else 'MISMATCH'}]")


# %%
# === CELL 3 — Path B: a1.mat 4명 × 200개씩 LUT → 추론 ===
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


@torch.no_grad()
def predict_batch(imgs_uint8, batch=128):
    N = len(imgs_uint8)
    preds = np.empty(N, dtype=np.int64)
    for i in range(0, N, batch):
        chunk = imgs_uint8[i:i + batch]
        t = torch.from_numpy(chunk).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
        t = NORMALIZE(t)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(t)
        preds[i:i + batch] = logits.argmax(dim=1).cpu().numpy()
    return preds


def render_many(rows):
    return np.stack(Parallel(n_jobs=-1, backend="loky")(
        delayed(render_one)(r) for r in rows
    ))


N_PER_PERSON = 200
PERSONS = [1, 14, 50, 100]

print(f"\n{'person':<8} {'pid':>4} {'tgt':>4}  {'rendered':>10}  "
      f"{'top1':>8}  {'render time':>12}")
print("-" * 65)

for pid in PERSONS:
    rows = A1[LABELS == pid][:N_PER_PERSON, :-1]
    t0 = time.time()
    inputs = render_many(rows)
    tr = time.time() - t0

    preds = predict_batch(inputs)
    target = pid - 1
    acc = (preds == target).mean() * 100

    print(f"P{pid:<7} {pid:>4} {target:>4}  {len(rows):>10}  "
          f"{acc:>7.2f}%  {tr:>10.1f}s")

    if acc < 90:
        wrong = preds[preds != target]
        if len(wrong):
            v, c = np.unique(wrong, return_counts=True)
            top = np.argsort(-c)[:3]
            miscl = ", ".join(f"P{v[i] + 1}({c[i]})" for i in top)
            print(f"        miscls: {miscl}")

print("\nv1 비교용 (참고):")
print("  v1: P1=99%, P14=92%, P50= 7%, P100= 8%   ← brittle")
print("  v2 기대: 모두 85%+ (val 86.76% 평균)")


# %%
# === CELL 4 (선택) — val_idx 전체 subset으로 86.76% 재현 확인 ===
# 시간: ~3-5분 (28K samples 렌더 + 추론).
# split seed=42로 재생성하면 학습 때와 동일 val_idx 나옴.
# 굳이 필요 없으면 skip.

RUN_FULL_VAL = False

if RUN_FULL_VAL:
    SEED = 42
    rng = np.random.default_rng(SEED)
    val_idx = []
    for c in range(NUM_CLASSES):
        cls_i = np.flatnonzero((LABELS - 1) == c)
        rng.shuffle(cls_i)
        n_val = max(1, int(0.2 * len(cls_i)))
        val_idx.extend(cls_i[:n_val].tolist())
    rng.shuffle(val_idx)
    val_idx = np.array(val_idx)
    print(f"val subset: {len(val_idx)} samples")

    # 청크로 렌더 (메모리 절약)
    BATCH = 4096
    correct = total = 0
    t0 = time.time()
    for s in range(0, len(val_idx), BATCH):
        e = min(s + BATCH, len(val_idx))
        rows = A1[val_idx[s:e], :-1]
        targets = (LABELS[val_idx[s:e]] - 1)
        inputs = render_many(rows)
        preds = predict_batch(inputs)
        correct += int((preds == targets).sum())
        total += len(targets)
        if (s // BATCH) % 4 == 0 or e == len(val_idx):
            elapsed = time.time() - t0
            print(f"  {e}/{len(val_idx)}  acc so far={correct/total*100:.2f}%  ({elapsed:.0f}s)")

    print(f"\n전체 val acc: {correct/total*100:.2f}%  (저장된 ckpt: {ck['val_acc']*100:.2f}%)")
