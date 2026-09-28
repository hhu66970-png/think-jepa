"""B2 feasibility only (no results): memory/time of one forward+backward of end-to-end
DTEM-style training from the cached post-block-12 hidden states.

Soft merge at each merge layer (after blocks 12..20): tokens are never dropped; each token
has a presence p (1 -> 0 when absorbed) and a size s. A learned 64-d matching embedding g_l
gives source->receiver probabilities P (softmax over receivers, temperature tau); a soft
top-r gate w removes about 25 % of the present mass per layer; receivers absorb
size-weighted features. Later blocks see log(p) as an attention bias (and log(s) with PA).
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import extract as E
from common import FEAT_ROOT


class SoftMerge(nn.Module):
    def __init__(self, D=1024, d=64, tau=0.05, tau_s=0.02, ratio=0.25):
        super().__init__()
        self.g = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.tau, self.tau_s, self.ratio = tau, tau_s, ratio

    def forward(self, x, p, s, parity=0):
        # all merge arithmetic in fp32: size-weighted sums overflow fp16 once groups grow
        with torch.autocast("cuda", enabled=False):
            return self._forward(x.float(), p.float(), s.float(), parity)

    def _forward(self, x, p, s, parity=0):
        B, N, D = x.shape
        m = F.normalize(self.g(x.float()), dim=-1)
        # alternate the source set between layers so absorbed receivers can later be sources
        ia = torch.arange(parity, N, 2, device=x.device)
        ib = torch.arange(1 - parity, N, 2, device=x.device)
        S = m[:, ia] @ m[:, ib].transpose(1, 2)                           # [B,Na,Nb]
        # absorbed tokens (presence < 0.05) cannot receive; partially present ones are down-weighted
        pb = p[:, ib]
        S = S + torch.where(pb < 0.05, torch.full_like(pb, -1e4), pb.clamp_min(1e-6).log())[:, None, :]
        P = (S / self.tau).softmax(-1)
        score = (P * S).sum(-1)                                            # [B,Na]
        pa = p[:, ia]
        # threshold = weighted quantile: remove ~ratio of the present mass (capped by source mass)
        target = torch.minimum(self.ratio * p.sum(1), 0.95 * pa.sum(1)).detach()
        sd, order = score.detach().sort(1, descending=True)
        cum = pa.detach().gather(1, order).cumsum(1)
        pos = (cum < target[:, None]).sum(1).clamp(max=score.shape[1] - 1)
        thr = sd.gather(1, pos[:, None])[:, 0]
        w = pa * torch.sigmoid((score - thr[:, None]) / self.tau_s)        # [B,Na]
        flow = P * (w * s[:, ia])[..., None]                               # mass a -> b
        xa, xb = x[:, ia].float(), x[:, ib].float()
        mb = s[:, ib] * p[:, ib]
        den = mb + flow.sum(1)
        new_xb = (mb[..., None] * xb + flow.transpose(1, 2) @ xa) / den.clamp_min(1e-3)[..., None]
        new_xb = torch.where((den > 1e-3)[..., None], new_xb, xb)   # numerically empty receivers keep x
        x = x.clone().float()
        x[:, ib] = new_xb
        p = p.clone(); s = s.clone()
        p[:, ia] = (pa - w).clamp_min(0)
        s[:, ib] = s[:, ib] + flow.sum(1)
        return x, p, s          # keep the residual stream in fp32 (deep ViT activations overflow fp16)


def main(B=2, pa=True):
    torch.manual_seed(0)
    enc = E.load_encoder()
    for q in enc.parameters():
        q.requires_grad_(False)
    H = np.load(FEAT_ROOT / "hidden_L12_train.npy", mmap_mode="r")
    x = torch.from_numpy(np.asarray(H[:B])).cuda()
    merges = nn.ModuleList(SoftMerge() for _ in range(9)).cuda()
    head = nn.Linear(1024, 52 * 3 * 32).cuda()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        p = torch.ones(B, 4096, device="cuda"); s = torch.ones(B, 4096, device="cuda")
        h = x
        for li in range(12, 24):
            if 12 <= li <= 20:
                h, p, s = merges[li - 12](h, p, s, parity=(li - 12) % 2)
            if li < 23:
                bias = p.clamp_min(1e-6).log() + (s.clamp_min(1e-6).log() if pa else 0)
                blk = enc.blocks[li + 1]
                h = cp.checkpoint(lambda hh, bb: blk(hh, mask=None, attn_mask=bb[:, None, None, :].to(torch.float16),
                                                     T=16, H_patches=16, W_patches=16), h, bias, use_reentrant=False)
        out = enc.norm(h)
        wts = (p * s)[..., None]
        y = head((out * wts).sum(1) / wts.sum(1))
        loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"B={B}: fwd+bwd {time.time()-t0:.2f} s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB, "
          f"final present mass {float(p.sum(1).mean()):.0f} tokens, grad norm g0 "
          f"{float(merges[0].g[1].weight.grad.norm()):.3e}")


if __name__ == "__main__":
    for b in (1, 2, 4):
        try:
            main(b)
        except torch.OutOfMemoryError as e:
            print(f"B={b}: OOM"); break
        torch.cuda.empty_cache()
