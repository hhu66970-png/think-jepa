"""Per-token saliency targets for the learned relevance scorer (TRAIN clips only).

A vision-only probe trained on dense tokens (train_probe.py --save_model) is differentiated
with respect to its input tokens; saliency of token i = |sum_d grad_id * x_id| (gradient x
input), summed over the 32 future frames' loss. Output <FEAT_ROOT>/saliency_train.npz:
rows [n], sal [n,4096] float32 (per-clip mean normalized to 1). Test clips are never touched.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import FEAT_ROOT
from train_probe import FeatureBank, ForecastProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.model, map_location="cpu")
    model = ForecastProbe(True, False, d=ck["dim"], depth=ck["depth"], D=ck["D"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
    P = np.load(FEAT_ROOT / "poses.npz")
    fut = torch.from_numpy(P["future"]).to(dev)
    past = torch.from_numpy(P["past"]).to(dev)
    rows = np.flatnonzero(~P["is_test"])
    bank = FeatureBank("dense")
    out = np.zeros((len(rows), 4096), np.float32)   # raw values are ~1e-9: never store as fp16
    for i in range(0, len(rows), 16):
        r = rows[i:i + 16]
        x = bank.batch(r, dev).float().requires_grad_(True)
        pred = mu.unsqueeze(0) + model(x, past[r] * 0).float() * sd
        loss = F.mse_loss(pred, fut[r], reduction="sum")
        g, = torch.autograd.grad(loss, x)
        sal = (g * x).sum(-1).abs().detach().double()
        sal = sal / sal.mean(dim=1, keepdim=True).clamp_min(1e-30)    # per-clip mean = 1
        out[i:i + 16] = sal.float().cpu().numpy()
    np.savez(FEAT_ROOT / "saliency_train.npz", rows=rows, sal=out)
    print("saliency", out.shape, "zeros", float((out == 0).mean()), "per-clip max/mean median",
          float(np.median(out.max(1))))


if __name__ == "__main__":
    main()
