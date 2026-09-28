"""Learned relevance scorer: layer-L hidden token -> probe saliency (TRAIN split only).

Per-token MLP with factorized space-time position embeddings; target is the per-clip
standardized log saliency from saliency.py. 10 % of train clips (fixed seed) are held out to
report the per-clip Spearman correlation. Saves <FEAT_ROOT>/scorer_L<L>.pt.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from common import FEAT_ROOT


class Scorer(nn.Module):
    def __init__(self, D=1024, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d), nn.GELU())
        self.pt = nn.Parameter(torch.zeros(1, 16, 1, 1, d))
        self.ph = nn.Parameter(torch.zeros(1, 1, 16, 1, d))
        self.pw = nn.Parameter(torch.zeros(1, 1, 1, 16, d))
        self.head = nn.Sequential(nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):                      # x [B, 4096, D] -> [B, 4096]
        B = x.shape[0]
        h = self.net(x.float()).view(B, 16, 16, 16, -1) + self.pt + self.ph + self.pw
        return self.head(h).view(B, -1)


def spearman(a, b):
    ra = a.argsort(-1).argsort(-1).float()
    rb = b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    H = np.load(FEAT_ROOT / f"hidden_L{a.layer}_train.npy", mmap_mode="r")
    hrows = np.load(FEAT_ROOT / f"hidden_L{a.layer}_train_rows.npy")
    S = np.load(FEAT_ROOT / "saliency_train.npz")
    assert np.array_equal(hrows, S["rows"])
    y = torch.from_numpy(np.log(S["sal"].astype(np.float32) + 1e-6))
    y = (y - y.mean(1, keepdim=True)) / y.std(1, keepdim=True).clamp_min(1e-6)
    n = len(hrows)
    perm = np.random.RandomState(12345).permutation(n)
    va, tr = np.sort(perm[: n // 10]), np.sort(perm[n // 10:])
    model = Scorer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    for ep in range(a.epochs):
        model.train()
        order = np.random.RandomState(ep).permutation(tr)
        for i in range(0, len(order), 16):
            b = np.sort(order[i:i + 16])
            x = torch.from_numpy(np.asarray(H[b])).to(dev)
            loss = ((model(x) - y[b].to(dev)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        rs = []
        with torch.no_grad():
            for i in range(0, len(va), 16):
                b = va[i:i + 16]
                rs.append(spearman(model(torch.from_numpy(np.asarray(H[b])).to(dev)), y[b].to(dev)).cpu())
        print(f"epoch {ep+1}: train mse {float(loss):.3f}  val per-clip Spearman {float(torch.cat(rs).mean()):.3f}", flush=True)
    torch.save(dict(state=model.state_dict(), layer=a.layer), FEAT_ROOT / f"scorer_L{a.layer}.pt")


if __name__ == "__main__":
    main()
