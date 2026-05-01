# %% [markdown]
# # v2 → ONNX export (Jetson TensorRT 8.2 호환)
#
# `weights/mobilenet_v3_large_v2_best.pth` → `.onnx`.
# 입력 이름 = `input`, shape = (N, 3, 224, 224) float32 (dynamic batch).
# Opset 11 (TensorRT 8.2.1 안정 동작 범위).
#
# 검증:
# 1. PyTorch 출력 vs onnxruntime CPU 출력 max|Δ| < 1e-4
# 2. P1/P14/P50/P100 a1.mat 100개씩 → 두 backend top-1 동일 비율 100% 기대

# %%
# === CELL 1a — 설치 (실패 시 stderr 직접 보임) ===
# onnx/onnxruntime은 핀 안 함 (Colab Python 버전에 맞는 latest 자동 선택).
# pywt/cv2는 LUT/렌더 byte-equal 위해 학습/검증 노트북과 동일 버전 핀.
get_ipython().system("pip install --quiet onnx onnxruntime onnxscript")
get_ipython().system("pip install --quiet pywavelets==1.8.0 opencv-python-headless==4.13.0.92")

# %%
# === CELL 1b — 모델 로드 ===
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import onnx, onnxruntime as ort

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "device:", DEVICE)
print("onnx:", onnx.__version__, "  ort:", ort.__version__)

from google.colab import drive
drive.mount("/content/drive")

WEIGHTS_DIR = Path("/content/drive/MyDrive/vibeid_capstone/weights")
PTH_PATH = WEIGHTS_DIR / "mobilenet_v3_large_v2_best.pth"
ONNX_PATH = WEIGHTS_DIR / "mobilenet_v3_large_v2.onnx"

ck = torch.load(str(PTH_PATH), map_location="cpu", weights_only=False)
NUM_CLASSES = int(ck["num_classes"])
ARCH = ck["arch"]
print(f"\nckpt: arch={ARCH}, num_classes={NUM_CLASSES}, "
      f"val_acc={ck['val_acc']*100:.2f}%, ep={ck['epoch']+1}")

assert ARCH == "mobilenet_v3_large", f"unexpected arch: {ARCH}"


def build_mnv3(num_classes):
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


model = build_mnv3(NUM_CLASSES)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print("model loaded.")


# %%
# === CELL 2 — ONNX export ===
# TensorRT 8.2.1은 opset 11~14 안정. 13으로 export (경험상 가장 호환성 좋음).
# Dynamic batch axis만 — H/W는 고정 224 (TRT engine plan 단순화).
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input":  {0: "batch"},
                  "logits": {0: "batch"}},
)

# onnx 모델 무결성 검증
m_onnx = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(m_onnx)
print(f"saved : {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1e6:.2f} MB)")
print(f"opset : {m_onnx.opset_import[0].version}")
print(f"inputs : {[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m_onnx.graph.input]}")
print(f"outputs: {[(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m_onnx.graph.output]}")


# %%
# === CELL 3 — PyTorch vs onnxruntime 출력 일치 검증 ===
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

# 임의 입력 8개로 일치 검증
np.random.seed(0)
x_np = np.random.randn(8, 3, 224, 224).astype(np.float32)

with torch.no_grad():
    y_torch = model(torch.from_numpy(x_np)).numpy()
y_ort = sess.run(["logits"], {"input": x_np})[0]

diff = np.abs(y_torch - y_ort)
print(f"max|Δ| logits: {diff.max():.3e}  (acceptance < 1e-4)")
print(f"max|Δ| softmax: {np.abs(np.exp(y_torch - y_torch.max(1, keepdims=True)) / np.exp(y_torch - y_torch.max(1, keepdims=True)).sum(1, keepdims=True) - np.exp(y_ort - y_ort.max(1, keepdims=True)) / np.exp(y_ort - y_ort.max(1, keepdims=True)).sum(1, keepdims=True)).max():.3e}")

t1 = y_torch.argmax(1)
t2 = y_ort.argmax(1)
print(f"top-1 일치: {(t1 == t2).all()}  (torch={t1.tolist()} vs ort={t2.tolist()})")

assert diff.max() < 1e-4, "logit drift too large"
assert (t1 == t2).all(), "top-1 mismatch"
print("\nONNX export OK.")


# %%
# === CELL 4 — 실데이터로 end-to-end 일치 검증 (선택) ===
# a1.mat에서 P1/P14/P50/P100 100개씩 → LUT → 두 backend top-1 비교.
# 둘 다 같은 prediction이면 ONNX 변환 무손실 확정.

import scipy.io
import pywt
import cv2

A1_PATH = Path("/content/drive/MyDrive/vibeid_capstone/data/a1.mat")
LUT_PATH = WEIGHTS_DIR / "jet_lut_v2.npy"

A1 = scipy.io.loadmat(str(A1_PATH))["footstep_feat"]
LABELS = A1[:, -1].astype(np.int16)
JET_LUT = np.load(str(LUT_PATH))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def render_one(sig):
    coeff, _ = pywt.cwt(sig, np.arange(1, 257), "morl")
    cmin, cmax = float(coeff.min()), float(coeff.max())
    if cmax - cmin < 1e-12:
        idx = np.zeros(coeff.shape, dtype=np.uint8)
    else:
        idx = np.clip(((coeff - cmin) / (cmax - cmin) * 255.0).round(),
                      0, 255).astype(np.uint8)
    return cv2.resize(JET_LUT[idx], (224, 224), interpolation=cv2.INTER_AREA)


def to_input(imgs_uint8):
    # NHWC uint8 -> NCHW float [0,1] -> ImageNet normalize
    x = imgs_uint8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


print(f"\n{'person':<7}  {'n':>4}  {'torch acc':>11}  {'ort acc':>9}  {'top-1 일치':>11}")
print("-" * 60)
for pid in [1, 14, 50, 100]:
    rows = A1[LABELS == pid][:100, :-1]
    imgs = np.stack([render_one(r) for r in rows])
    x = to_input(imgs)

    with torch.no_grad():
        y_t = model(torch.from_numpy(x)).numpy()
    y_o = sess.run(["logits"], {"input": x})[0]

    p_t = y_t.argmax(1)
    p_o = y_o.argmax(1)
    target = pid - 1
    print(f"P{pid:<6}  {len(rows):>4}  "
          f"{(p_t == target).mean()*100:>10.2f}%  "
          f"{(p_o == target).mean()*100:>8.2f}%  "
          f"{(p_t == p_o).mean()*100:>10.2f}%")

print(f"\nONNX 위치: {ONNX_PATH}")
print("로컬로 다운로드 → E:/Terra/weights/mobilenet_v3_large_v2.onnx")


# %%
# === CELL 5 — TensorRT 변환용 메모 ===
print("""
다음 단계 (Jetson에서):

1. ONNX + LUT npy + GMM npz + python 스크립트들을 Jetson으로 옮김
   - mobilenet_v3_large_v2.onnx  (~17 MB)
   - jet_lut_v2.npy              (896 B)
   - gmm_params.npz              (3.5 KB)
   - extract.py, render_lut.py, stream.py

2. TensorRT FP16 engine 생성:
   $ trtexec --onnx=mobilenet_v3_large_v2.onnx \\
             --saveEngine=mnv3_v2_fp16.plan \\
             --fp16 \\
             --workspace=256 \\
             --minShapes=input:1x3x224x224 \\
             --optShapes=input:1x3x224x224 \\
             --maxShapes=input:4x3x224x224

3. Python에서 TRT engine 로드 → 실시간 stream 코드 작성
   - tensorrt + pycuda 사용 (JetPack 4.6.6에 기본 포함)
   - ADS1256 SPI → ring buffer → frozen GMM → footstep → LUT → TRT → top-1
""")
