"""B1: learned (decoupled) matching embedding, trained by distillation (TRAIN split only).

g: LN + Linear(1024 -> 64) on the dense hidden state after block 12 (the input to the first
merge). Target for a token pair (a, b) of the same clip: cosine of their DENSE final-layer
features, i.e. whether the frozen encoder ends up treating them as interchangeable. Pairs are
sampled the way bipartite matching sees them: for each sampled token, its 8 nearest neighbours
by post-merge-input feature cosine plus 8 random tokens of the same clip.
The learned cosine replaces the post-RoPE Key cosine as the merge metric at every merge layer
(relevance_source untouched, similarity order kept). 10 % of train clips (seed 12345) are held
out to report the pair-level correlation. Saves <FEAT_ROOT>/matcher_L12.pt.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FEAT_ROOT
from train_probe import FeatureBank


class Matcher(nn.Module):
    def __init__(self, D=1024, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))

    def forward(self, x):                                   # [B, N, D] -> [B, N, d]
        return self.net(x.float())


def sample_pairs(h, n_anchor, gen):
    """h [N, D] one clip -> anchor idx [n_anchor], partner idx [n_anchor, 16]."""
    N = h.shape[0]
    a = torch.randint(0, N, (n_anchor,), generator=gen, device="cpu").to(h.device)
    hn = F.normalize(h.float(), dim=-1)
    sim = hn[a] @ hn.T                                        # [n_anchor, N]
    sim[torch.arange(n_anchor), a] = -2
    nn_idx = sim.topk(8, dim=1).indices
    rnd = torch.randint(0, N, (n_anchor, 8), generator=gen, device="cpu").to(h.device)
    return a, torch.cat([nn_idx, rnd], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--anchors", type=int, default=512)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    H = np.load(FEAT_ROOT / "hidden_L12_train.npy", mmap_mode="r")
    hrows = np.load(FEAT_ROOT / "hidden_L12_train_rows.npy")
    dense = FeatureBank("dense")
    n = len(hrows)
    perm = np.random.RandomState(12345).permutation(n)
    va, tr = np.sort(perm[: n // 10]), np.sort(perm[n // 10:])
    model = Matcher().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)

    def batch_loss(i, train):
        h = torch.from_numpy(np.asarray(H[i])).to(dev)                          # [4096,1024]
        s, j = dense.where[int(hrows[i])]
        fdense = F.normalize(torch.from_numpy(np.asarray(dense.tok[s][j])).to(dev).float(), dim=-1)
        anc, par = sample_pairs(h, a.anchors, gen)
        g = F.normalize(model(h[None])[0], dim=-1)
        pred = (g[anc][:, None, :] * g[par]).sum(-1)                          # [A,16]
        tgt = (fdense[anc][:, None, :] * fdense[par]).sum(-1)
        key_proxy = (F.normalize(h.float(), dim=-1)[anc][:, None, :] * F.normalize(h.float(), dim=-1)[par]).sum(-1)
        return F.mse_loss(pred, tgt), pred.detach(), tgt, key_proxy

    for ep in range(a.epochs):
        model.train()
        for i in np.random.RandomState(ep).permutation(tr):
            loss, *_ = batch_loss(i, True)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        P, T, K = [], [], []
        with torch.no_grad():
            for i in va:
                _, p, t, k = batch_loss(i, False)
                P.append(p.flatten()); T.append(t.flatten()); K.append(k.flatten())
        P, T, K = torch.cat(P), torch.cat(T), torch.cat(K)
        r = torch.corrcoef(torch.stack([P, T]))[0, 1].item()
        rk = torch.corrcoef(torch.stack([K, T]))[0, 1].item()
        print(f"epoch {ep+1}: val pair corr(learned, final-dense cos) {r:.3f} | "
              f"corr(layer-12 feature cos, final-dense cos) {rk:.3f}", flush=True)
    torch.save(dict(state=model.state_dict()), FEAT_ROOT / "matcher_L12.pt")


if __name__ == "__main__":
    main()
