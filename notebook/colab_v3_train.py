# %% [markdown]
# # Terra v3 — VIBeID A1+A2+A3 통합 backbone 학습 (170-class)
#
# **목표**: v2 (A1 100명 단독, val 86.76%) 의 floor/거리 일반화 한계 극복.
# A2 (cement, 거리 1.5/2.5/4.0m) + A3 (wood/carpet/cement) 추가 → 총 170 클래스.
# 다양한 floor/거리에서 학습된 backbone → 가정 deployment 시 transfer learning에 강함.
#
# **원칙**: OSF interim/{a1,a2,a3}.zip (footstep_feat) 만 사용. **OSF processed PNG 절대 X**
# (그건 paper 저자 환경 matplotlib 으로 렌더된 거라 v1 brittleness 함정 그대로 재현됨).
# 우리 LUT (`weights/jet_lut_v2.npy` 와 동일한 matplotlib 3.x cm.get_cmap('jet')) 로
# 학습 단계에서 직접 렌더 → Jetson 추론과 byte-equal 분포 보장.
#
# **글로벌 라벨 매핑 (170-class)**:
# - A1 footstep_feat pid 1~100  → global label 0~99
# - A2 footstep_feat pid 1~30   → global label 100~129  (A2_1/A2_2/A2_3 모두 동일 매핑, 같은 30명을 거리만 바꿔 측정)
# - A3 footstep_feat pid 1~40   → global label 130~169  (A3_1/A3_2/A3_3 모두 동일 매핑, 같은 40명을 floor만 바꿔 측정)
#
# **A1/A2/A3 피험자 overlap 가능성**: paper supplementary 미확인. 1차는 별 클래스 가정.
# overlap 있어도 backbone embedding 다양성에는 손해 없음 (같은 사람이 두 라벨 받으면
# 분류기가 둘 다 학습할 뿐).
#
# **권장 런타임**: A100 (학습 ~1.5h, 25 epoch).

# %%
# === CELL 1 — 환경 + Drive mount + OSF 직링크 다운로드 ===
import subprocess
subprocess.run(["pip", "install", "-q",
                "pywavelets==1.8.0",
                "opencv-python-headless==4.13.0.92"], check=True)

import os, time, json, gc, math, random, zipfile
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
from torch.utils.data import Dataset, DataLoader
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
WEIGHTS_DIR = DRIVE / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Colab local SSD (Drive 보다 빠름, 56 GB pre-render 용)
LOCAL = Path("/content/terra_v3")
LOCAL.mkdir(exist_ok=True)
INTERIM = LOCAL / "interim"
INTERIM.mkdir(exist_ok=True)

NPY_DATA = LOCAL / "lut_v3_data.npy"          # (N, 224, 224, 3) uint8
NPY_LABELS = LOCAL / "lut_v3_labels.npy"       # (N,) int16 (0~169)
NPY_DOMAINS = LOCAL / "lut_v3_domains.npy"     # (N,) int8 (0=A1, 1=A2_1, 2=A2_2, 3=A2_3, 4=A3_1, 5=A3_2, 6=A3_3)
SPLIT_PATH = LOCAL / "split_v3.npz"

# 결정론
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# %%
# === CELL 2 — OSF interim/{a1,a2,a3}.zip wget + 압축해제 (~3.73 GB, 5-10분) ===
# A1: 100명 (1.41 GB), A2: 30명×3 cond (0.78 GB), A3: 40명×3 cond (1.54 GB)
OSF_URLS = {
    "a1.zip": "https://osf.io/download/2pmyk/",
    "a2.zip": "https://osf.io/download/mu8sx/",
    "a3.zip": "https://osf.io/download/jkrg5/",
}

for name, url in OSF_URLS.items():
    target = INTERIM / name
    if target.exists() and target.stat().st_size > 1e8:
        print(f"  {name}: 이미 있음 ({target.stat().st_size/1e9:.2f} GB), skip")
        continue
    print(f"  {name}: wget {url} ...")
    t0 = time.time()
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(target), url], check=True)
    print(f"    완료 {target.stat().st_size/1e9:.2f} GB ({time.time()-t0:.0f}s)")

# 압축해제 (각 zip 안에 .mat 1개 또는 3개)
for name in OSF_URLS:
    zpath = INTERIM / name
    extract_dir = INTERIM / name.replace(".zip", "")
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  {name}: 이미 풀려있음, skip")
        continue
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(extract_dir)
    print(f"  {name}: 풀기 완료 → {extract_dir}")

# 풀린 .mat 파일 목록 출력 (구조 확인용)
for sub in ["a1", "a2", "a3"]:
    d = INTERIM / sub
    mats = sorted(d.rglob("*.mat"))
    print(f"\n{sub}/ ({len(mats)} .mat):")
    for m in mats:
        print(f"  {m.relative_to(INTERIM)}  ({m.stat().st_size/1e6:.1f} MB)")


# %%
# === CELL 3 — 7개 .mat 로드 + 글로벌 라벨 매핑 ===
# 예상 구조 (CLAUDE.md OSF 섹션 참고, 실제 zip 풀어보고 fallback 처리):
#   a1/  → a1.mat (또는 A1.mat) — 100명 통합
#   a2/  → A2_1.mat / A2_2.mat / A2_3.mat (30명 × 1.5/2.5/4.0m cement)
#   a3/  → A3_1.mat / A3_2.mat / A3_3.mat (40명 × wood/carpet/cement)
# 변수명 모두 footstep_feat (N, 1501) float64. col[:1500]=waveform, col[1500]=1-indexed pid.

def load_footstep_feat(mat_path):
    """scipy.io.loadmat 로 footstep_feat 추출 + (signals, pids_1idx) 반환."""
    d = scipy.io.loadmat(str(mat_path))
    # 변수명 보수적으로 — 'footstep_feat' 가 정석이지만 혹시 다르면 첫 2D ndarray 키 자동 선택
    if "footstep_feat" in d:
        arr = d["footstep_feat"]
    else:
        candidates = [(k, v) for k, v in d.items()
                      if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2]
        if not candidates:
            raise RuntimeError(f"{mat_path}: footstep_feat 없음, 후보 {list(d.keys())}")
        # 가장 큰 ndarray 선택 (footstep_feat 가 보통 가장 큼)
        k, arr = max(candidates, key=lambda kv: kv[1].size)
        print(f"  ⚠ {mat_path.name}: 'footstep_feat' 없음, '{k}' 사용 (shape={arr.shape})")
    assert arr.shape[1] == 1501, f"{mat_path}: 예상 cols=1501, 실제 {arr.shape[1]}"
    signals = arr[:, :1500].astype(np.float64)
    pids = arr[:, 1500].astype(np.int32)  # 1-indexed
    return signals, pids


# 각 subset의 .mat 경로 결정 (zip 안 풀린 모양 따라 자동 탐색)
def find_mats(subdir, expected_count):
    mats = sorted((INTERIM / subdir).rglob("*.mat"))
    assert len(mats) == expected_count, \
        f"{subdir}: 예상 {expected_count} .mat, 실제 {len(mats)}: {mats}"
    return mats

a1_mats = find_mats("a1", 1)              # a1.mat 1개
a2_mats = find_mats("a2", 3)              # A2_1/A2_2/A2_3
a3_mats = find_mats("a3", 3)              # A3_1/A3_2/A3_3
print("a1:", [m.name for m in a1_mats])
print("a2:", [m.name for m in a2_mats])
print("a3:", [m.name for m in a3_mats])

# 로드 + 글로벌 라벨 매핑
SIGNAL_LIST = []
LABEL_LIST = []
DOMAIN_LIST = []   # 0=A1, 1=A2_1, 2=A2_2, 3=A2_3, 4=A3_1, 5=A3_2, 6=A3_3
DOMAIN_NAMES = ["A1", "A2_1", "A2_2", "A2_3", "A3_1", "A3_2", "A3_3"]

# A1: pid 1~100 → global 0~99
print("\nloading A1...")
sig, pid = load_footstep_feat(a1_mats[0])
print(f"  shape={sig.shape}, pid range={pid.min()}..{pid.max()}")
assert pid.min() >= 1 and pid.max() <= 100, f"A1 pid 범위 비정상: {pid.min()}..{pid.max()}"
SIGNAL_LIST.append(sig)
LABEL_LIST.append(pid - 1)            # 0~99
DOMAIN_LIST.append(np.zeros(len(sig), dtype=np.int8))

# A2: pid 1~30 → global 100~129 (3 .mat 모두 동일 매핑)
print("\nloading A2...")
for di, m in enumerate(a2_mats):
    sig, pid = load_footstep_feat(m)
    print(f"  {m.name}: shape={sig.shape}, pid range={pid.min()}..{pid.max()}")
    assert pid.min() >= 1 and pid.max() <= 30, f"{m.name} pid 범위 비정상"
    SIGNAL_LIST.append(sig)
    LABEL_LIST.append(pid - 1 + 100)   # 100~129
    DOMAIN_LIST.append(np.full(len(sig), 1 + di, dtype=np.int8))   # 1, 2, 3

# A3: pid 1~40 → global 130~169 (3 .mat 모두 동일 매핑)
print("\nloading A3...")
for di, m in enumerate(a3_mats):
    sig, pid = load_footstep_feat(m)
    print(f"  {m.name}: shape={sig.shape}, pid range={pid.min()}..{pid.max()}")
    assert pid.min() >= 1 and pid.max() <= 40, f"{m.name} pid 범위 비정상"
    SIGNAL_LIST.append(sig)
    LABEL_LIST.append(pid - 1 + 130)   # 130~169
    DOMAIN_LIST.append(np.full(len(sig), 4 + di, dtype=np.int8))   # 4, 5, 6

SIGNALS = np.concatenate(SIGNAL_LIST, axis=0)         # (N_total, 1500)
LABELS = np.concatenate(LABEL_LIST, axis=0).astype(np.int16)  # (N_total,) 0~169
DOMAINS = np.concatenate(DOMAIN_LIST, axis=0)         # (N_total,) 0~6
del SIGNAL_LIST, LABEL_LIST, DOMAIN_LIST; gc.collect()

N = len(SIGNALS)
print(f"\n총 {N} footsteps, {SIGNALS.nbytes/1e9:.1f} GB raw float64")
print(f"label range: {LABELS.min()}..{LABELS.max()}  (170 classes 기대)")

# per-class / per-domain 분포 출력
unique, counts = np.unique(LABELS, return_counts=True)
print(f"classes: {len(unique)}, min/max per-class = {counts.min()}/{counts.max()}, "
      f"median = {int(np.median(counts))}")
for di, dn in enumerate(DOMAIN_NAMES):
    n_d = int((DOMAINS == di).sum())
    n_pid = len(np.unique(LABELS[DOMAINS == di]))
    print(f"  domain {dn}: {n_d} footsteps, {n_pid} unique pids")


# %%
# === CELL 4 — LUT 렌더 함수 (Jetson python/render_lut.py와 byte-equal) ===
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
    """1500-sample footstep -> (224, 224, 3) uint8 RGB."""
    scales = np.arange(1, 257)
    coefficients, _ = pywt.cwt(sig_1500, scales, "morl")
    idx = coeffs_to_indices(coefficients)
    rgb = JET_LUT_RGB_U8[idx]                          # (256, 1500, 3)
    H, W = size
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


# 워밍업
t0 = time.time()
sample = render_one(SIGNALS[0])
dt = (time.time() - t0) * 1000
print(f"\nsingle render: {dt:.1f} ms, output {sample.shape} {sample.dtype}")

# 도메인별 첫 footstep 시각화 sanity
fig, axs = plt.subplots(1, 7, figsize=(21, 3))
for ax, di in zip(axs, range(7)):
    idx = int(np.flatnonzero(DOMAINS == di)[0])
    ax.imshow(render_one(SIGNALS[idx]))
    ax.set_title(f"{DOMAIN_NAMES[di]}\nlabel={LABELS[idx]}")
    ax.axis("off")
plt.tight_layout(); plt.show()


# %%
# === CELL 5 — 전체 pre-render (~40-80분, joblib n_jobs=-1) ===
# A100 Colab vCPU 12개 기준 ~40분. 로컬 SSD에 mmap, RAM 폭증 방지.
# 크기 예상: 약 220K~250K footsteps × 224×224×3 uint8 = 33~38 GB. Colab disk 100 GB 안에 들어감.

if NPY_DATA.exists() and NPY_LABELS.exists() and NPY_DOMAINS.exists():
    print(f"이미 렌더 완료, skip. ({NPY_DATA.stat().st_size/1e9:.1f} GB)")
    print("재렌더하려면:")
    print(f"  !rm {NPY_DATA} {NPY_LABELS} {NPY_DOMAINS}")
else:
    out = np.lib.format.open_memmap(str(NPY_DATA), mode="w+", dtype=np.uint8,
                                     shape=(N, 224, 224, 3))
    np.save(str(NPY_LABELS), LABELS)
    np.save(str(NPY_DOMAINS), DOMAINS)

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
DOM = np.load(str(NPY_DOMAINS))
print(f"\nDATA  shape={DATA.shape}  dtype={DATA.dtype}")
print(f"LBL   shape={LBL.shape}    range={LBL.min()}..{LBL.max()}")
print(f"DOM   shape={DOM.shape}    domains={np.unique(DOM)}")

# RAM 확보를 위해 SIGNALS 해제 (이후 학습에선 DATA mmap 만 사용)
del SIGNALS; gc.collect()


# %%
# === CELL 6 — train/val split (per-class stratified 80/20, seed=42) ===
# 도메인 정보도 보존 → val 단계에서 도메인별 acc 분리 가능
if SPLIT_PATH.exists():
    sp = np.load(str(SPLIT_PATH))
    train_idx, val_idx = sp["train"], sp["val"]
    print(f"split 재사용: train={len(train_idx)}, val={len(val_idx)}")
else:
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in range(int(LBL.max()) + 1):
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

# val set 도메인 분포
print("\nval set 도메인 분포:")
for di, dn in enumerate(DOMAIN_NAMES):
    n = int((DOM[val_idx] == di).sum())
    print(f"  {dn}: {n}")


# %%
# === CELL 7 — Dataset + augmentation (v2와 동일) ===
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


IMG_SIZE = 224
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

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
# === CELL 8 — 모델 (MobileNetV3-Large, 170-class) + MixUp ===
def build_model(num_classes: int, pretrained=True):
    w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v3_large(weights=w)
    in_feat = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_feat, num_classes)
    return m


NUM_CLASSES = int(LBL.max() + 1)
print("NUM_CLASSES:", NUM_CLASSES)
assert NUM_CLASSES == 170, f"170 클래스 기대, 실제 {NUM_CLASSES}"


def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# %%
# === CELL 9 — 학습 루프 (도메인별 val acc 기록 추가) ===
def train_v3(epochs=25, batch=128, lr=3e-4, wd=1e-4,
             mixup_alpha=0.2, label_smoothing=0.15,
             save_name="mobilenet_v3_large_v3_best.pth"):
    save_path = WEIGHTS_DIR / save_name

    model = build_model(NUM_CLASSES, pretrained=True).to(DEVICE)
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

    val_dom = DOM[val_idx]   # (len(val_ds),) 도메인 레이블

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

        # validation + 도메인별 acc
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(x)
                all_pred.append(logits.argmax(dim=1).cpu().numpy())
                all_true.append(y.cpu().numpy())
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        val_acc = float((all_pred == all_true).mean())

        per_dom = {}
        for di, dn in enumerate(DOMAIN_NAMES):
            mask = (val_dom == di)
            if mask.sum() > 0:
                per_dom[dn] = float((all_pred[mask] == all_true[mask]).mean())

        epoch_dt = time.time() - t0
        history.append({
            "ep": ep, "tloss": tloss / nseen, "val_acc": val_acc,
            "lr": sched.get_last_lr()[0], "per_domain": per_dom,
        })
        msg = f"ep {ep+1:2d}/{epochs}  loss={tloss/nseen:.4f}  " \
              f"val={val_acc*100:.2f}%  lr={sched.get_last_lr()[0]:.2e}  ({epoch_dt:.0f}s)"
        msg += "  | " + " ".join(f"{k}={v*100:.1f}" for k, v in per_dom.items())

        # === disconnect-safe checkpointing — 매 epoch Drive sync ===
        # (1) history.json: 학습 끊겨도 곡선 살아남도록
        with open(WEIGHTS_DIR / "mobilenet_v3_large_v3_history.json", "w") as f:
            json.dump(history, f, indent=2)
        # (2) last.pth: 가장 최근 epoch state (optimizer + scheduler 포함 → 다음 세션 resumption 가능)
        torch.save({
            "arch": "mobilenet_v3_large", "version": "v3",
            "num_classes": NUM_CLASSES,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": ep, "val_acc": val_acc, "best_acc": best_acc,
            "per_domain": per_dom, "domain_names": DOMAIN_NAMES,
            "config": dict(epochs=epochs, batch=batch, lr=lr, wd=wd,
                           mixup_alpha=mixup_alpha,
                           label_smoothing=label_smoothing,
                           seed=SEED),
        }, str(WEIGHTS_DIR / "mobilenet_v3_large_v3_last.pth"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "arch": "mobilenet_v3_large",
                "version": "v3",
                "num_classes": NUM_CLASSES,
                "model_state_dict": model.state_dict(),
                "val_acc": best_acc, "epoch": ep,
                "per_domain_at_best": per_dom,
                "domain_names": DOMAIN_NAMES,
                "config": dict(epochs=epochs, batch=batch, lr=lr, wd=wd,
                               mixup_alpha=mixup_alpha,
                               label_smoothing=label_smoothing,
                               seed=SEED),
            }, str(save_path))
            # (3) best 갱신 시마다 ONNX export — 학습 끊겨도 가장 최근 best ONNX 가 Drive 에 있음
            try:
                onnx_path = WEIGHTS_DIR / "mobilenet_v3_large_v3.onnx"
                model.eval()
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
                    torch.onnx.export(
                        model, dummy, str(onnx_path),
                        export_params=True, opset_version=13,
                        do_constant_folding=True,
                        input_names=["input"], output_names=["logits"],
                        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                    )
                model.train()
                msg += "  ✓ saved (.pth + .onnx)"
            except Exception as e:
                msg += f"  ✓ saved (.pth)  [onnx skip: {e}]"
        print(msg)

    print(f"\nBEST val_acc = {best_acc*100:.2f}%  →  {save_path}")
    return best_acc, history


# %%
# === CELL 10 — 학습 실행 (A100 ~1.5h) ===
best_acc, history = train_v3(epochs=25, batch=128, lr=3e-4, wd=1e-4,
                              mixup_alpha=0.2, label_smoothing=0.15)


# %%
# === CELL 11 — 학습 곡선 + 도메인별 acc 시각화 ===
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
eps = [h["ep"] for h in history]

ax[0].plot(eps, [h["tloss"] for h in history])
ax[0].set_title("train loss"); ax[0].set_xlabel("epoch"); ax[0].grid()

ax[1].plot(eps, [h["val_acc"]*100 for h in history], lw=2, label="overall")
ax[1].set_title("val acc (%)"); ax[1].set_xlabel("epoch"); ax[1].grid(); ax[1].legend()

for dn in DOMAIN_NAMES:
    vals = [h["per_domain"].get(dn, 0)*100 for h in history]
    ax[2].plot(eps, vals, label=dn)
ax[2].set_title("per-domain val acc (%)"); ax[2].set_xlabel("epoch")
ax[2].grid(); ax[2].legend(fontsize=8)

plt.tight_layout(); plt.show()


# %%
# === CELL 12 — A1-only val acc (v2 baseline 86.76%와 비교) ===
# v3 에서도 A1 만 떼서 보면 v2 와 비슷하거나 더 나아야 함 (training data 늘어났으니).
ckpt_path = WEIGHTS_DIR / "mobilenet_v3_large_v3_best.pth"
ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
model = build_model(NUM_CLASSES, pretrained=False).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

all_pred, all_true = [], []
with torch.no_grad():
    for x, y in val_dl:
        x = x.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(x)
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(y.numpy())
all_pred = np.concatenate(all_pred); all_true = np.concatenate(all_true)
val_dom = DOM[val_idx]

print(f"reload overall val: {(all_pred == all_true).mean()*100:.2f}%  "
      f"(ckpt={ckpt['val_acc']*100:.2f}%)")
print()
for di, dn in enumerate(DOMAIN_NAMES):
    mask = val_dom == di
    if mask.sum() > 0:
        acc = (all_pred[mask] == all_true[mask]).mean()
        print(f"  {dn}: {acc*100:.2f}%  (n={mask.sum()})")

# A1 만 따로 (v2 baseline 비교 용도)
a1_mask = val_dom == 0
a1_acc = (all_pred[a1_mask] == all_true[a1_mask]).mean()
print(f"\n[v2 비교] A1-only val acc = {a1_acc*100:.2f}%  (v2 baseline: 86.76%)")


# %%
# === CELL 13 — Top-3 worst class confusion (각 도메인별) ===
# 가장 많이 틀리는 pid 와 어떤 pid 로 잘못 가는지 확인
from collections import Counter

print("[Top-5 worst classes overall]")
err_counter = Counter()
for t, p in zip(all_true, all_pred):
    if t != p:
        err_counter[int(t)] += 1
for c, n in err_counter.most_common(5):
    cls_total = int((all_true == c).sum())
    # most common wrong prediction
    wrong_preds = [int(p) for t, p in zip(all_true, all_pred) if t == c and p != c]
    top_conf = Counter(wrong_preds).most_common(3)
    dn = DOMAIN_NAMES[int(DOM[val_idx][np.flatnonzero(all_true == c)[0]])]
    print(f"  global pid {c} ({dn}): {n}/{cls_total} 오분류, "
          f"가장 자주 → {top_conf}")


# %%
# === CELL 14 — ONNX verify (학습 루프 안에서 매 best 갱신 시 export 했으므로 검증만) ===
# 만약 어떤 이유로 학습 중 ONNX export 가 skip 됐으면 (메시지 "[onnx skip: ...]" 찍혔으면)
# 여기서 best.pth 로 재 export.
ONNX_PATH = WEIGHTS_DIR / "mobilenet_v3_large_v3.onnx"

if not ONNX_PATH.exists():
    print(f"⚠ {ONNX_PATH} 없음 — best.pth 로 재 export")
    exp_model = build_model(NUM_CLASSES, pretrained=False)
    exp_model.load_state_dict(ckpt["model_state_dict"])
    exp_model.eval()
    dummy_cpu = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        exp_model, dummy_cpu, str(ONNX_PATH),
        export_params=True, opset_version=13,
        do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

sz = ONNX_PATH.stat().st_size / 1e6
print(f"ONNX: {ONNX_PATH}  ({sz:.1f} MB)")
assert sz > 10, "ONNX < 10 MB → external data 분리됐을 가능성. 단일 파일 export 실패!"

# 재로드 sanity (onnxruntime 가 깔려있으면)
try:
    import onnxruntime as ort
    exp_model = build_model(NUM_CLASSES, pretrained=False)
    exp_model.load_state_dict(ckpt["model_state_dict"])
    exp_model.eval()
    dummy_cpu = torch.randn(1, 3, 224, 224)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy_cpu.numpy()})[0]
    with torch.no_grad():
        torch_out = exp_model(dummy_cpu).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    print(f"ONNX vs PyTorch max|diff| = {diff:.2e}  (1e-4 미만이면 OK)")
except ImportError:
    print("(onnxruntime 없음, runtime 검증 skip)")


# %%
# === CELL 15 — Drive 백업 + 다음 단계 안내 ===
print("학습 완료 — Drive 저장본:")
for p in [WEIGHTS_DIR / "mobilenet_v3_large_v3_best.pth",
          WEIGHTS_DIR / "mobilenet_v3_large_v3_last.pth",
          WEIGHTS_DIR / "mobilenet_v3_large_v3.onnx",
          WEIGHTS_DIR / "mobilenet_v3_large_v3_history.json"]:
    if p.exists():
        print(f"  ✓ {p}  ({p.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  ✗ {p}  (없음!)")

print("""
다음 단계:
1. weights/mobilenet_v3_large_v3.onnx 를 Jetson 으로 scp
   scp -P <port> ./mobilenet_v3_large_v3.onnx jetson@<ip>:~/terra/weights/

2. Jetson 에서 trtexec FP16 변환:
   trtexec --onnx=weights/mobilenet_v3_large_v3.onnx \\
           --fp16 --saveEngine=weights/mnv3_v3_fp16.plan \\
           --workspace=512

3. web_server.py / jetson_realtime.py 의 --plan 인자를 mnv3_v3_fp16.plan 으로 변경
   → P14_1/P50_1/P100_1/P1_1.mat 재추론. v2 분포 (P14 93.4%, P50 95.1%, P100 94.2%, P1=P2 97.4%)와 비교.

4. v2 vs v3 분포 차이 분석 → CLAUDE.md Stage 5c 결과 옆에 v3 컬럼 추가.

---

[disconnect 복구 가이드]
세션이 중간에 끊겼을 때 무엇을 다시 해야 하는지:

A. 학습 도중 끊김 (best.pth + last.pth + history.json + onnx 가 Drive에 있음)
   → 새 Colab 세션에서:
     1. CELL 1~7 재실행 (OSF 재다운로드 5-10분 + pre-render 40-80분 — 이건 어쩔 수 없음, local SSD 임시 파일이라)
     2. CELL 8 (모델 정의) 재실행
     3. CELL 9 의 train_v3 함수 정의 재실행
     4. last.pth 로드 후 남은 epoch 만 학습 (아래 셀 추가하면 됨):

       lp = torch.load(str(WEIGHTS_DIR / "mobilenet_v3_large_v3_last.pth"),
                        map_location=DEVICE, weights_only=False)
       resume_ep = lp["epoch"] + 1
       remaining = 25 - resume_ep
       print(f"resume from epoch {resume_ep}, {remaining} epochs 남음")
       # train_v3 를 epochs=remaining 로 호출하되, 모델/optimizer 사전 로드:
       # (간단히는 그냥 best.pth 로 ImageNet 대신 시작해도 OK, val acc 거의 유지)

B. Pre-render 도중 끊김 (best/last 없음)
   → 처음부터 다시. 1.5h 손실. 어쩔 수 없음.

C. ONNX 만 missing, best.pth 는 있음
   → CELL 1, 8, 14 만 재실행하면 ONNX 복구 (5초). val_ds 없이도 export 가능.
""")
