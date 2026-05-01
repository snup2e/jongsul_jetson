# Jetson Nano P3450 셋업 (JetPack 4.6.6 / Python 3.6 또는 3.8)

## 0. 기본 가정
- JetPack 4.6.6: CUDA 10.2, cuDNN 8.2.1, **TensorRT 8.2.1.9**, OpenCV 4.1.1 (system)
- Python 3.6 (시스템 기본) — TRT/pycuda는 시스템 파이썬에 사전설치됨
- swap 4 GB 이상 권장 (4 GB RAM 한계 보완)

## 1. 의존성 설치

```bash
# 시스템 패키지
sudo apt update
sudo apt install -y python3-pip python3-dev libatlas-base-dev libopenblas-dev \
                    libhdf5-serial-dev hdf5-tools

# Python 패키지 (시스템 pip)
python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy==1.19.5 scipy==1.5.4 \
                              pywavelets==1.1.1 opencv-python-headless==4.6.0.66

# TensorRT/pycuda는 JetPack에 사전설치 — 확인만
python3 -c "import tensorrt as trt; print('TRT', trt.__version__)"
python3 -c "import pycuda.driver; print('pycuda OK')"
```

⚠️ pywavelets 1.8.0 (학습 환경)와 1.1.1 (Jetson)은 **버전 다름**. 추출 byte-equal 보장 안 됨.
실측 차이 측정해서 robustness margin 안에 있는지 확인 필요 (CWT 수치 차이 → LUT 픽셀 → 모델).
처음에는 numpy/scipy/pywt 모두 가능한 한 새 버전으로 try (Jetson용 wheel 있는 한도 내).

## 2. ONNX → TensorRT FP16 plan

```bash
cd ~/terra
trtexec --onnx=mobilenet_v3_large_v2.onnx \
        --saveEngine=mnv3_v2_fp16.plan \
        --fp16 \
        --workspace=512 \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:1x3x224x224 \
        --maxShapes=input:1x3x224x224 \
        --verbose 2>&1 | tee trt_build.log
```

- `--workspace=512` (MB): Jetson Nano 4GB에서 안전. 부족 시 256으로 줄임.
- batch 고정 1로 빌드 (실시간 추론은 한 번에 1 footstep).
- 빌드 시간 5-15분 예상. 메모리 압박 시 swap 활용.

## 3. TRT plan vs ONNX 출력 일치 검증

빌드 직후:
```bash
python3 jetson_infer.py --verify --onnx mobilenet_v3_large_v2.onnx \
                                  --plan mnv3_v2_fp16.plan
```

기대: top-1 일치 100%, 로짓 max|Δ| < 5e-2 (FP16 양자화 오차 범위).

## 4. 환경 변수

```bash
# ~/.bashrc 또는 실행 직전
export TERRA_JET_LUT=/home/USER/terra/jet_lut_v2.npy
export TERRA_GMM=/home/USER/terra/gmm_params.npz
export TERRA_TRT_PLAN=/home/USER/terra/mnv3_v2_fp16.plan
```

## 5. 디렉토리 권장 배치

```
~/terra/
├── mobilenet_v3_large_v2.onnx     # ONNX (TRT 빌드용 + 검증용)
├── mnv3_v2_fp16.plan              # TRT engine
├── jet_lut_v2.npy                 # LUT
├── gmm_params.npz                 # frozen GMM
├── extract.py                      # 포팅 코드
├── render_lut.py                   # LUT 렌더
├── stream.py                       # raw → inputs
├── jetson_infer.py                 # TRT inference + verify
└── (선택) sample_data/             # 오프라인 테스트 .mat
```

## 6. 다음 단계 (코드 작성 필요)

- `jetson_infer.py`: TRT engine 로드, infer_batch(np uint8 N×224×224×3) → logits N×100
- `jetson_realtime.py`: ADS1256 SPI ring buffer + sliding GMM + footstep detect → infer
