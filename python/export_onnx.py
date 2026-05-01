"""Re-export MobileNetV3-Large v2 .pth → .onnx (단일 파일, external data 없음).

기존 weights/mobilenet_v3_large_v2.onnx 는 external data 형식인데
.onnx.data 가 없어서 TRT 빌드 실패. 이 스크립트로 self-contained 단일 파일 재생성.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


PTH = Path("E:/Terra/weights/mobilenet_v3_large_v2_best.pth")
ONNX = Path("E:/Terra/weights/mobilenet_v3_large_v2.onnx")
OPSET = 13
NUM_CLASSES = 100


def main():
    print(f"[load] {PTH}")
    ckpt = torch.load(PTH, map_location="cpu", weights_only=False)
    print(f"  arch={ckpt.get('arch')}, num_classes={ckpt.get('num_classes')}, val_acc={ckpt.get('val_acc')}")
    sd = ckpt["model_state_dict"]

    print(f"[build] torchvision mobilenet_v3_large with classifier[3]=Linear(1280, {NUM_CLASSES})")
    model = mobilenet_v3_large(weights=None)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, NUM_CLASSES)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"  missing={missing}, unexpected={unexpected}")
        sys.exit(1)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"  forward OK: out shape {tuple(out.shape)}")

    print(f"[export] {ONNX} (opset={OPSET}, dynamic batch)")
    torch.onnx.export(
        model,
        dummy,
        str(ONNX),
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    size_mb = ONNX.stat().st_size / 1024 / 1024
    print(f"[done] {ONNX} ({size_mb:.2f} MB)")
    if size_mb < 10:
        print(f"  ⚠️ size too small — weights likely not embedded")
        sys.exit(1)


if __name__ == "__main__":
    main()
