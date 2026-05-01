# %% [markdown]
# # JET LUT export (v2)
#
# 학습 시 사용한 jet LUT (matplotlib 3.7+, Colab 환경)을 `.npy`로 박아서
# 로컬/Jetson에서도 byte-equal 재현.
#
# 학습 노트북 끝에서 한 번만 실행. 결과: `weights/jet_lut_v2.npy` (256×3 uint8, 768 bytes).

# %%
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.cm as cm

from google.colab import drive
drive.mount("/content/drive")
WEIGHTS_DIR = Path("/content/drive/MyDrive/vibeid_capstone/weights")

JET_LUT_RGB_U8 = (cm.get_cmap("jet")(np.arange(256) / 255.0)[:, :3] * 255.0
                  ).round().astype(np.uint8)

assert JET_LUT_RGB_U8.shape == (256, 3) and JET_LUT_RGB_U8.dtype == np.uint8

out = WEIGHTS_DIR / "jet_lut_v2.npy"
np.save(str(out), JET_LUT_RGB_U8)

h = hashlib.sha256(JET_LUT_RGB_U8.tobytes()).hexdigest()
print(f"saved : {out}  ({out.stat().st_size} bytes)")
print(f"shape : {JET_LUT_RGB_U8.shape}  dtype: {JET_LUT_RGB_U8.dtype}")
print(f"sha256: {h}")
print(f"\nJET[0]   = {JET_LUT_RGB_U8[0].tolist()}  (deep blue)")
print(f"JET[128] = {JET_LUT_RGB_U8[128].tolist()}  (green)")
print(f"JET[255] = {JET_LUT_RGB_U8[255].tolist()}  (deep red)")
print(f"\n로컬에서 검증할 때 사용할 hash: {h}")
print(f"이 .npy를 다운로드해서 E:/Terra/weights/jet_lut_v2.npy 에 저장하세요.")
