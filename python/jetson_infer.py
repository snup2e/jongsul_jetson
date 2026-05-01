"""TensorRT inference wrapper for Jetson Nano (JetPack 4.6.6, TRT 8.2.1).

오프라인 검증:
    python3 jetson_infer.py --verify \
        --onnx mobilenet_v3_large_v2.onnx \
        --plan mnv3_v2_fp16.plan

엔드투엔드 (raw .mat 1개로):
    python3 jetson_infer.py --raw P14_1.mat --plan mnv3_v2_fp16.plan

런타임 의존성: tensorrt + pycuda (JetPack 사전설치) + numpy + pywt + cv2.
PyTorch/onnxruntime 불필요 (검증 시만 onnxruntime 사용).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (CUDA context 초기화)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    """TRT engine wrapper. Single-batch (1×3×224×224) 고정으로 빌드된 plan 가정."""

    def __init__(self, plan_path: str):
        with open(plan_path, "rb") as f:
            engine_bytes = f.read()
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        # I/O 바인딩 정보 (TRT 8.x API)
        self.input_name = "input"
        self.output_name = "logits"
        self.input_idx = self.engine.get_binding_index(self.input_name)
        self.output_idx = self.engine.get_binding_index(self.output_name)

        in_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        out_shape = tuple(self.engine.get_binding_shape(self.output_idx))
        # dynamic이면 -1로 나옴 → 고정 batch 1로 set
        if in_shape[0] == -1:
            in_shape = (1,) + in_shape[1:]
            self.context.set_binding_shape(self.input_idx, in_shape)
            out_shape = tuple(self.context.get_binding_shape(self.output_idx))
        self.in_shape = in_shape
        self.out_shape = out_shape

        # GPU/host 버퍼 사전 할당
        self.h_input = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

        print(f"[TRT] in {in_shape} -> out {out_shape}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (1, 3, 224, 224) float32. returns (1, num_classes) float32."""
        assert x.shape == self.in_shape and x.dtype == np.float32
        np.copyto(self.h_input, x.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape(self.out_shape).copy()


# ---- 전처리 ----

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def to_input(img_uint8_hwc: np.ndarray) -> np.ndarray:
    """(224, 224, 3) uint8 RGB -> (1, 3, 224, 224) float32 ImageNet-normalized."""
    x = img_uint8_hwc.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


# ---- 검증 모드 ----

def cmd_verify(args):
    """ONNX (onnxruntime) vs TRT plan 출력 비교."""
    import onnxruntime as ort

    print(f"loading ONNX  : {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print(f"loading plan  : {args.plan}")
    engine = TRTEngine(args.plan)

    rng = np.random.default_rng(0)
    N = 16
    x = rng.standard_normal(size=(N, 3, 224, 224), dtype=np.float32)

    y_onnx = sess.run(["logits"], {"input": x})[0]
    y_trt = np.empty_like(y_onnx)
    for i in range(N):
        y_trt[i] = engine.infer(x[i:i + 1])[0]

    diff = np.abs(y_onnx - y_trt)
    p_onnx = y_onnx.argmax(1)
    p_trt = y_trt.argmax(1)
    print(f"\nrandom inputs (N={N}):")
    print(f"  max|Δ|  logits: {diff.max():.3e}")
    print(f"  mean|Δ| logits: {diff.mean():.3e}")
    print(f"  top-1 일치    : {(p_onnx == p_trt).sum()}/{N}")
    if diff.max() > 5e-2:
        print("  ⚠ FP16 양자화 오차 큼 — workspace 늘리거나 FP32 쪽으로 fallback 검토")


# ---- 엔드투엔드 모드 (raw .mat 1개 → footsteps → 추론) ----

def cmd_raw(args):
    import scipy.io
    from extract import load_gmm_npz, apply_frozen_gmm
    from stream import footsteps_to_inputs

    gmm_path = os.environ.get("TERRA_GMM", "gmm_params.npz")
    print(f"loading GMM   : {gmm_path}")
    gmm = load_gmm_npz(gmm_path)

    print(f"loading raw   : {args.raw}")
    geo = scipy.io.loadmat(args.raw)["geo_data"].ravel().astype(np.float64)
    print(f"  signal: {len(geo)} samples = {len(geo)/8000:.1f}s")

    print("extracting footsteps...")
    t0 = time.time()
    footsteps = apply_frozen_gmm(geo, gmm)
    print(f"  {len(footsteps)} footsteps ({time.time()-t0:.1f}s)")

    print("rendering 224x224 inputs (CWT + LUT)...")
    t0 = time.time()
    imgs = footsteps_to_inputs(footsteps, size=(224, 224))
    print(f"  shape {imgs.shape} ({time.time()-t0:.1f}s)")

    print(f"loading plan  : {args.plan}")
    engine = TRTEngine(args.plan)

    print("inference...")
    t0 = time.time()
    preds = np.empty(len(imgs), dtype=np.int64)
    confs = np.empty(len(imgs), dtype=np.float32)
    for i, img in enumerate(imgs):
        x = to_input(img)
        logits = engine.infer(x)[0]
        # softmax max
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        preds[i] = int(p.argmax())
        confs[i] = float(p.max())
    dt = time.time() - t0
    print(f"  {len(imgs)} inferences in {dt:.1f}s = {1000*dt/len(imgs):.1f} ms/sample")

    # 분포 출력
    vals, counts = np.unique(preds, return_counts=True)
    order = np.argsort(-counts)
    print("\n분류 분포 (top 5):")
    for j in order[:5]:
        cls = vals[j]
        idx = preds == cls
        print(f"  P{cls + 1:<3d}: {counts[j]:>4d} ({100*counts[j]/len(preds):5.1f}%)  "
              f"평균 conf {confs[idx].mean()*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.environ.get("TERRA_TRT_PLAN", "mnv3_v2_fp16.plan"))
    ap.add_argument("--verify", action="store_true",
                    help="ONNX vs TRT 출력 비교 (--onnx 필요)")
    ap.add_argument("--onnx", default="mobilenet_v3_large_v2.onnx")
    ap.add_argument("--raw", default=None,
                    help="end-to-end 테스트할 .mat 파일")
    args = ap.parse_args()

    if args.verify:
        cmd_verify(args)
    elif args.raw:
        cmd_raw(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
