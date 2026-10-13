"""Locate the first non-finite value in B2 training (debug only)."""
import sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as cp
import extract as E
from common import FEAT_ROOT
from dtem_feas import SoftMerge
from fit_matcher import Matcher
from train_probe import ForecastProbe

torch.manual_seed(0)
dev = torch.device("cuda")
enc = E.load_encoder()
for q in enc.parameters(): q.requires_grad_(False)
H = np.load(FEAT_ROOT / "hidden_L12_train.npy", mmap_mode="r")
P = np.load(FEAT_ROOT / "poses.npz"); fut = torch.from_numpy(P["future"]).to(dev)
rows = np.flatnonzero(~P["is_test"])
mu = fut[rows].mean(dim=(0, 1), keepdim=True)[0]; sd = fut[rows].std()
merges = nn.ModuleList(SoftMerge() for _ in range(9)).to(dev)
b1 = Matcher(); b1.load_state_dict(torch.load(FEAT_ROOT / "matcher_L12.pt", map_location="cpu")["state"])
for m in merges: m.g.load_state_dict(b1.net.state_dict())
probe = ForecastProbe(True, False).to(dev)
opt = torch.optim.AdamW(list(merges.parameters()) + list(probe.parameters()), lr=1e-4)
for step in range(30):
    idx = np.arange(step * 4, step * 4 + 4)
    x = torch.from_numpy(np.asarray(H[idx])).to(dev)
    print(f"step {step}: input finite {bool(torch.isfinite(x).all())} max|x| {float(x.float().abs().max()):.1f}")
    with torch.autocast("cuda", dtype=torch.float16):
        p = torch.ones(4, 4096, device=dev); s = torch.ones(4, 4096, device=dev); h = x
        for li in range(12, 24):
            if li <= 20:
                h, p, s = merges[li - 12](h, p, s, parity=(li - 12) % 2)
                bad = [n for n, t in (("h", h), ("p", p), ("s", s)) if not torch.isfinite(t).all()]
                if bad: print(f"  non-finite after merge {li}: {bad}, max|h| {float(h.float().abs().max())}"); sys.exit()
            if li < 23:
                bias = p.clamp_min(1e-6).log()
                h = enc.blocks[li + 1](h, mask=None, attn_mask=bias[:, None, None, :].to(torch.float16), T=16, H_patches=16, W_patches=16)
                if not torch.isfinite(h).all():
                    print(f"  non-finite after block {li+1}, max|h| before? dtype {h.dtype}"); sys.exit()
        feats = enc.norm(h)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = probe(feats, torch.zeros(4, 32, 52, 3, device=dev), key_bias=(p * s).clamp_min(1e-6).log())
    loss = F.mse_loss(mu.unsqueeze(0) + y.float() * sd, fut[rows[idx]])
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(list(merges.parameters()) + list(probe.parameters()), 1.0)
    print(f"  loss {float(loss):.4f} gradnorm {float(gn):.3e} kept {float(p.sum(1).mean()):.0f} max|h| {float(h.float().abs().max()):.0f}")
    if not torch.isfinite(gn): print("  non-finite gradient"); sys.exit()
    opt.step()
