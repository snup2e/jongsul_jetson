# %% [markdown]
# # Terra v2 — LUT 재렌더 + augmentation 강화 학습
#
# 기존 v1 모델 (`mobilenetv3_vibeid_a1_best.pth`, val 84.69%)은 Colab matplotlib
# 출력에만 동작. 로컬 재렌더 RMSE 1%만 발생해도 P50/P100이 P10으로 오분류 →
# 환경 drift에 brittle. v2 목표:
#
# 1. a1.mat 144,371 footsteps을 **결정론적 LUT 파이프라인**으로 직접 224x224 RGB로 렌더
#    (Jetson에서 돌릴 코드와 byte-equal 출력)
# 2. **augmentation 강화** (ColorJitter↑ + RandomErasing + MixUp)로 렌더 미세 차이에 robust
# 3. MobileNetV3-Large를 메인으로, EfficientNet-B0을 곁다리 비교
#
# 입력: `/content/drive/MyDrive/vibeid_capstone/data/a1.mat`  (1.5 GB, 사용자 업로드)
# 출력: `/content/drive/MyDrive/vibeid_capstone/weights/{mnv3,effb0}_v2_best.pth`
#
# 권장 런타임: **A100** (학습 빠름). T4도 가능 (1.5x 시간).

# %%
# === CELL 1 — 환경 + Drive mount ===
import subprocess
subprocess.run(["pip", "install", "-q",
                "pywavelets==1.8.0",
                "opencv-python-headless==4.13.0.92"], check=True)

import os, time, json, gc, math, random
from pathlib import Path
import numpy as np
import scipy.io
import pywt
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
from joblib import Parallel, delayed

print("torch     :", torch.__version__, "cuda:", torch.cuda.is_available())
print("matplotlib:", matplotlib.__version__)
print("pywt      :", pywt.__version__)
print("cv2       :", cv2.__version__)
print("numpy     :", np.__version__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU       :", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive/vibeid_capstone")
DATA_DIR = DRIVE / "data"           # a1.mat 여기에 업로드 ([CHECK] 경로 맞는지 확인)
WEIGHTS_DIR = DRIVE / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

LOCAL = Path("/content/terra_v2")   # 로컬 SSD에 npy 저장 (Drive보다 빠름)
LOCAL.mkdir(exist_ok=True)

A1_PATH = DATA_DIR / "a1.mat"
NPY_DATA = LOCAL / "lut_v2_data.npy"          # (N, 224, 224, 3) uint8
NPY_LABELS = LOCAL / "lut_v2_labels.npy"       # (N,) int16
SPLIT_PATH = LOCAL / "split_v2.npz"            # train/val indices

assert A1_PATH.exists(), f"a1.mat 없음: {A1_PATH}\n→ 로컬에서 업로드: drive에 data/ 만들고 a1.mat 넣으세요"
print(f"\na1.mat OK: {A1_PATH.stat().st_size/1e9:.2f} GB")

# 결정론
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# %%
# === CELL 2 — a1.mat 로드 ===
print("loading a1.mat (~30s for 1.5 GB)...")
t0 = time.time()
A1 = scipy.io.loadmat(str(A1_PATH))["footstep_feat"]
print(f"  shape={A1.shape}, dtype={A1.dtype}, took {time.time()-t0:.1f}s")

LABELS = A1[:, -1].astype(np.int16)
SIGNALS = A1[:, :-1]   # (N, 1500) float64
N = len(A1)
print(f"  N={N}, label range={LABELS.min()}..{LABELS.max()}")

# per-class count
unique, counts = np.unique(LABELS, return_counts=True)
print(f"  classes={len(unique)}, min/max per-class = {counts.min()}/{counts.max()}, "
      f"median = {int(np.median(counts))}")


# %%
# === CELL 3 — LUT 렌더 함수 (E:/Terra/python/render_lut.py와 동일 코드) ===
# matplotlib jet LUT, 256 entries, RGB uint8.
# 이 LUT은 matplotlib 3.x 전체에서 동일하게 정의됨. cm.get_cmap('jet')는
# pywt+cv2와 달리 시각화 라이브러리라 버전간 LUT 안정. 그래도 안전하게
# 한 번 만들고 상수로 박아둠.
JET_LUT_RGB_U8 = (cm.get_cmap("jet")(np.arange(256) / 255.0)[:, :3] * 255.0
                  ).round().astype(np.uint8)
print("JET LUT shape:", JET_LUT_RGB_U8.shape, "dtype:", JET_LUT_RGB_U8.dtype)
print("JET[0]   =", JET_LUT_RGB_U8[0])     # deep blue
print("JET[128] =", JET_LUT_RGB_U8[128])   # green
print("JET[255] =", JET_LUT_RGB_U8[255])   # deep red


def coeffs_to_indices(coefficients):
    cmin = float(coefficients.min()); cmax = float(coefficients.max())
    if cmax - cmin < 1e-12:
        return np.zeros(coefficients.shape, dtype=np.uint8)
    scaled = (coefficients - cmin) / (cmax - cmin) * 255.0
    return np.clip(scaled.round(), 0, 255).astype(np.uint8)


def render_one(sig_1500, size=(224, 224)):
    """1500-sample footstep -> (224, 224, 3) uint8 RGB.
    Jetson과 byte-equal 동일 코드 (E:/Terra/python/render_lut.py).
    """
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig_1500, scales, "morl")
    idx = coeffs_to_indices(coefficients)
    rgb = JET_LUT_RGB_U8[idx]                          # (256, 1500, 3)
    H, W = size
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


# 워밍업 + 한 footstep 렌더 시간 측정
t0 = time.time()
sample = render_one(SIGNALS[0])
dt = (time.time() - t0) * 1000
print(f"\nsingle render: {dt:.1f} ms, output {sample.shape} {sample.dtype}")
plt.figure(figsize=(4, 4)); plt.imshow(sample); plt.title(f"P{LABELS[0]} idx 0"); plt.axis('off'); plt.show()


# %%
# === CELL 4 — 144K footsteps 병렬 렌더 (~10-15분) ===
# joblib n_jobs=-1로 모든 vCPU 사용. mmap에 직접 쓰기 (RAM 폭증 방지).

if NPY_DATA.exists() and NPY_LABELS.exists():
    print(f"이미 렌더 완료, skip. ({NPY_DATA.stat().st_size/1e9:.1f} GB)")
    print("재렌더하려면 NPY_DATA 파일을 지우세요:")
    print(f"  !rm {NPY_DATA} {NPY_LABELS}")
else:
    out = np.lib.format.open_memmap(str(NPY_DATA), mode="w+", dtype=np.uint8,
                                     shape=(N, 224, 224, 3))
    np.save(str(NPY_LABELS), LABELS - 1)   # 0-indexed class

    CHUNK = 2048
    t0 = time.time()
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(render_one)(SIGNALS[i]) for i in range(start, end)
        )
        out[start:end] = np.stack(results)
        if (start // CHUNK) % 4 == 0 or end == N:
            elapsed = time.time() - t0
            eta = elapsed * (N - end) / max(end, 1)
            print(f"  {end}/{N} ({100*end/N:5.1f}%)   "
                  f"elapsed {elapsed:5.0f}s   eta {eta:5.0f}s")
    out.flush()
    print(f"\n완료: {NPY_DATA} ({NPY_DATA.stat().st_size/1e9:.2f} GB)")

# 재로드 (mmap)
DATA = np.load(str(NPY_DATA), mmap_mode="r")
LBL = np.load(str(NPY_LABELS))
print(f"\nDATA  shape={DATA.shape}  dtype={DATA.dtype}")
print(f"LBL   shape={LBL.shape}    range={LBL.min()}..{LBL.max()}")


# %%
# === CELL 4b — byte-equal 검증 (로컬 LUT 출력과 일치 확인) ===
# 로컬 (Windows, pywt 1.8.0, cv2 4.13.0.92, matplotlib 3.10.9)에서
# render_one(SIGNALS[0]) → sha256(img.tobytes())을 미리 계산해둔 값.
# Colab에서 같은 hash가 나오면 환경 drift 없음 = 학습/추론 byte-equal 보장.
import hashlib

EXPECTED = {
    0:   "1e77eccbbe5dd3f78e7cbf3f7c0016af1a641de8391ea46c74b419cd3d526e13",
    100: "6a389b24c6994970bd2142fe5022872abeb970c0fb1e401aadadd71595cf3c7d",
}
for i, expected in EXPECTED.items():
    img = render_one(SIGNALS[i])
    got = hashlib.sha256(img.tobytes()).hexdigest()
    ok = "OK" if got == expected else "MISMATCH"
    print(f"  row {i:>4}: {got}  [{ok}]")
    if got != expected:
        print(f"    expected: {expected}")
# MISMATCH가 나면: pywt/opencv/matplotlib 버전 차이.
# CELL 1의 pip install 버전 명시가 적용 안 됐거나, numpy 차이일 수 있음.
# 이 경우에도 학습 자체는 가능 (Colab과 Jetson이 같으면 됨) — 하지만
# 로컬에서의 byte-level 검증은 못 함. 그땐 로컬에 Colab 환경 동일 docker 띄우거나,
# Jetson에서 Colab과 동일 버전 pin.


# %%
# === CELL 5 — 렌더 시각 sanity check ===
# 클래스 4명 (P1, P14, P50, P100) × 첫 footstep 보여주기. 로컬 무지개 spectrogram
# 모양과 같으면 OK.
fig, axs = plt.subplots(1, 4, figsize=(16, 3))
for ax, pid in zip(axs, [1, 14, 50, 100]):
    idx = np.flatnonzero(LBL == pid - 1)[0]
    ax.imshow(DATA[idx]); ax.set_title(f"P{pid} (idx {idx})"); ax.axis('off')
plt.tight_layout(); plt.show()


# %%
# === CELL 6 — train/val split (per-class stratified 80/20, seed=42) ===
if SPLIT_PATH.exists():
    sp = np.load(str(SPLIT_PATH))
    train_idx, val_idx = sp["train"], sp["val"]
    print(f"split 재사용: train={len(train_idx)}, val={len(val_idx)}")
else:
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in range(LBL.max() + 1):
        cls_i = np.flatnonzero(LBL == c)
        rng.shuffle(cls_i)
        n_val = max(1, int(0.2 * len(cls_i)))
        val_idx.extend(cls_i[:n_val].tolist())
        train_idx.extend(cls_i[n_val:].tolist())
    train_idx = np.array(train_idx); val_idx = np.array(val_idx)
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    np.savez(str(SPLIT_PATH), train=train_idx, val=val_idx, seed=SEED)
    print(f"split 생성: train={len(train_idx)}, val={len(val_idx)}")

print(f"  train per-class min/max = "
      f"{np.bincount(LBL[train_idx]).min()}/{np.bincount(LBL[train_idx]).max()}")
print(f"  val   per-class min/max = "
      f"{np.bincount(LBL[val_idx]).min()}/{np.bincount(LBL[val_idx]).max()}")


# %%
# === CELL 7 — Dataset (mmap에서 로드, transform 적용) ===
class LUTDataset(Dataset):
    def __init__(self, data, labels, indices, transform=None):
        self.data = data; self.labels = labels
        self.indices = indices; self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        arr = np.asarray(self.data[idx])     # (224, 224, 3) uint8
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


# %%
# === CELL 8 — augmentation (강화) ===
IMG_SIZE = 224
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

# v1 augmentation: ColorJitter(0.1, 0.1)만. v2는 다음을 추가:
#   - ColorJitter 강화 (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05)
#     → LUT/matplotlib 미세 차이를 흡수 (brittleness 직접 처방)
#   - RandomErasing (p=0.25) → CWT 패턴의 일부 가려도 사람 식별 가능하게
#   - MixUp(alpha=0.2) → 학습 loop에서 적용
# Flip 안 함 (CWT의 시간/주파수 축은 비대칭, flip이 의미 손상)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    NORMALIZE,
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])

train_ds = LUTDataset(DATA, LBL, train_idx, train_tf)
val_ds   = LUTDataset(DATA, LBL, val_idx,   val_tf)

print(f"train: {len(train_ds)} / val: {len(val_ds)}")
print(f"sample shape: {train_ds[0][0].shape}")


# %%
# === CELL 9 — 모델 factory (MobileNetV3-Large / EfficientNet-B0) ===
def build_model(arch: str, num_classes: int, pretrained=True):
    if arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        in_feat = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_feat, num_classes)
        return m
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
        return m
    raise ValueError(arch)


NUM_CLASSES = int(LBL.max() + 1)
print("NUM_CLASSES:", NUM_CLASSES)


# %%
# === CELL 10 — MixUp helper ===
def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# %%
# === CELL 11 — 학습 루프 (재사용 가능 함수) ===
def train_one_arch(arch: str, epochs=30, batch=128, lr=3e-4, wd=1e-4,
                   mixup_alpha=0.2, label_smoothing=0.15,
                   save_name=None):
    if save_name is None:
        save_name = f"{arch}_v2_best.pth"
    save_path = WEIGHTS_DIR / save_name

    model = build_model(arch, NUM_CLASSES, pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch * 2, shuffle=False,
                        num_workers=4, pin_memory=True,
                        persistent_workers=True)

    best_acc = 0.0; history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time(); tloss = 0.0; nseen = 0
        for x, y in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x_m, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(x_m)
                loss = mixup_loss(crit, logits, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tloss += float(loss) * x.size(0); nseen += x.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        epoch_dt = time.time() - t0
        history.append({"ep": ep, "tloss": tloss / nseen, "val_acc": val_acc, "lr": sched.get_last_lr()[0]})
        msg = f"[{arch}] ep {ep+1:2d}/{epochs}  loss={tloss/nseen:.4f}  " \
              f"val={val_acc*100:.2f}%  lr={sched.get_last_lr()[0]:.2e}  ({epoch_dt:.0f}s)"
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "arch": arch, "num_classes": NUM_CLASSES,
                "model_state_dict": model.state_dict(),
                "val_acc": best_acc, "epoch": ep,
                "config": dict(epochs=epochs, batch=batch, lr=lr, wd=wd,
                               mixup_alpha=mixup_alpha,
                               label_smoothing=label_smoothing,
                               seed=SEED),
            }, str(save_path))
            msg += "  ✓ saved"
        print(msg)

    print(f"\n[{arch}] BEST val_acc = {best_acc*100:.2f}%  →  {save_path}")
    with open(WEIGHTS_DIR / f"{arch}_v2_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return best_acc, history


# %%
# === CELL 12 — MobileNetV3-Large 학습 (메인) ===
# A100 기준 ~25-35분 (30 epoch, ~115K train images, batch 128).
mnv3_acc, mnv3_hist = train_one_arch("mobilenet_v3_large", epochs=30, batch=128,
                                       lr=3e-4, wd=1e-4,
                                       mixup_alpha=0.2, label_smoothing=0.15)


# %%
# === CELL 13 (선택) — EfficientNet-B0 비교 ===
# A100 기준 ~30-40분. MobileNetV3 acc보다 +1~3pp 정도 기대.
effb0_acc, effb0_hist = train_one_arch("efficientnet_b0", epochs=30, batch=128,
                                        lr=3e-4, wd=1e-4,
                                        mixup_alpha=0.2, label_smoothing=0.15)


# %%
# === CELL 14 — 학습 곡선 시각화 + 비교 ===
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, hist in [("mobilenet_v3_large", mnv3_hist),
                    ("efficientnet_b0", effb0_hist if 'effb0_hist' in dir() else None)]:
    if hist is None: continue
    eps = [h["ep"] for h in hist]
    ax[0].plot(eps, [h["tloss"] for h in hist], label=name)
    ax[1].plot(eps, [h["val_acc"]*100 for h in hist], label=name)
ax[0].set_title("train loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid()
ax[1].set_title("val acc (%)"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid()
plt.tight_layout(); plt.show()


# %%
# === CELL 15 — 최종 체크포인트 검증 ===
# 저장된 .pth가 정상 로드되고 val 정확도 재현되는지 확인 (Jetson 이식 전 sanity).
def quick_verify(path, arch):
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    m = build_model(arch, NUM_CLASSES, pretrained=False).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            correct += (m(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    print(f"{path.name}: ckpt val={ckpt['val_acc']*100:.2f}%  → reload val={correct/total*100:.2f}%")

quick_verify(WEIGHTS_DIR / "mobilenet_v3_large_v2_best.pth", "mobilenet_v3_large")
if (WEIGHTS_DIR / "efficientnet_b0_v2_best.pth").exists():
    quick_verify(WEIGHTS_DIR / "efficientnet_b0_v2_best.pth", "efficientnet_b0")


# %%
# === CELL 16 — 끝. 다음 단계 ===
# 1. 가장 좋은 모델 .pth 다운로드 → 로컬 E:/Terra/weights/ 에 복사
# 2. 로컬 infer.py로 Path A vs Path C 재검증 — 이번엔 두 path가 LUT로 동일하므로
#    정확도 차 0pp 나와야 정상
# 3. ONNX export (notebook의 cell-23 코드 참고)
# 4. Jetson TensorRT FP16 변환 + 실시간 stream
print("done. 다음:")
print("  1. weights/*v2_best.pth 로컬로 복사")
print("  2. 로컬 infer.py 재실행으로 Path A == Path C 확인")
print("  3. ONNX export → Jetson")
