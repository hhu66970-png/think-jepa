"""Keep only the encoder and predictor of the official checkpoint (smaller, loads faster).
  python tools/slim_ckpt.py ckpt/vjepa2-ac-vitg.pt ckpt/vjepa2-ac-vitg-slim.pt
  export VJEPA2_AC_CKPT=ckpt/vjepa2-ac-vitg-slim.pt"""
import sys

import torch

sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(list(sd.keys()))
torch.save({k: sd[k] for k in ("encoder", "predictor")}, sys.argv[2])
