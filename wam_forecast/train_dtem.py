"""B2: end-to-end learned matching embeddings (DTEM-style) for the frozen V-JEPA ViT-L.

Training starts from the cached dense hidden state after block 12 (train split only) and
runs blocks 13..23 with a differentiable soft merge after blocks 12..20 (the L9 schedule):
each token keeps a presence p and size s, a per-layer 64-d matching embedding g_l gives
source->receiver probabilities, a soft top-r gate removes ~25 % of the present mass per layer,
and later blocks see log(p) (+ log(s) with --pa) as an attention bias. A temporary attentive
probe on the final tokens (key bias log(p*s)) is trained jointly with the forecasting loss.
Only the g_l are kept; downstream evaluation re-extracts features with HARD merging using
these embeddings and trains fresh probes as for every other arm.

Hyper-parameters are chosen on the fixed 10 % validation split of train (same split as
train_probe --val). Saves <out>/matchers.pt (list of 9 state dicts) and <out>/log.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import extract as E
from common import FEAT_ROOT, write_json
from dtem_feas import SoftMerge
from fit_matcher import Matcher
from train_probe import ForecastProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--tau_s", type=float, default=0.02)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_p", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--pa", action="store_true")
    ap.add_argument("--init", choices=["b1", "random"], default="b1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")

    enc = E.load_encoder()
    for q in enc.parameters():
        q.requires_grad_(False)
    H = np.load(FEAT_ROOT / "hidden_L12_train.npy", mmap_mode="r")
    hrows = np.load(FEAT_ROOT / "hidden_L12_train_rows.npy")
    P = np.load(FEAT_ROOT / "poses.npz")
    fut = torch.from_numpy(P["future"]).to(dev)
    train_rows = np.flatnonzero(~P["is_test"])
    assert np.array_equal(np.sort(hrows), train_rows)
    pos_of = {int(r): i for i, r in enumerate(hrows)}
    perm = np.random.RandomState(12345).permutation(train_rows)       # same split as probe --val
    n_val = len(perm) // 10
    val_rows, tr_rows = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    mu = fut[tr_rows].mean(dim=(0, 1), keepdim=True)[0]
    sd = fut[tr_rows].std()

    merges = nn.ModuleList(SoftMerge(tau=a.tau, tau_s=a.tau_s) for _ in range(9)).to(dev)
    if a.init == "b1":
        b1 = Matcher(); b1.load_state_dict(torch.load(FEAT_ROOT / "matcher_L12.pt", map_location="cpu")["state"])
        for m in merges:
            m.g.load_state_dict(b1.net.state_dict())
    probe = ForecastProbe(True, False).to(dev)
    opt = torch.optim.AdamW([{"params": merges.parameters(), "lr": a.lr_g},
                             {"params": probe.parameters(), "lr": a.lr_p}], weight_decay=0.05)
    steps = a.epochs * math.ceil(len(tr_rows) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    def forward(rows):
        x = torch.from_numpy(np.stack([np.asarray(H[pos_of[int(r)]]) for r in rows])).to(dev)
        B = x.shape[0]
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.ones(B, 4096, device=dev); s = torch.ones(B, 4096, device=dev)
            h = x
            for li in range(12, 24):
                if li <= 20:
                    h, p, s = merges[li - 12](h, p, s, parity=(li - 12) % 2)
                if li < 23:
                    bias = p.clamp_min(1e-6).log() + (s.clamp_min(1e-6).log() if a.pa else 0)
                    blk = enc.blocks[li + 1]
                    h = cp.checkpoint(lambda hh, bb, blk=blk: blk(hh, mask=None,
                                      attn_mask=bb[:, None, None, :].to(torch.float16), T=16, H_patches=16,
                                      W_patches=16), h, bias, use_reentrant=False)
            feats = enc.norm(h)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = probe(feats, torch.zeros(B, 32, 52, 3, device=dev),
                      key_bias=(p * s).clamp_min(1e-6).log())
        return mu.unsqueeze(0) + y.float() * sd, float(p.sum(1).mean())

    log, best = [], (float("inf"), None)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        merges.train(); probe.train()
        order = tr_rows[torch.randperm(len(tr_rows), generator=g).numpy()]
        tot = 0.0
        for i in range(0, len(order), a.bs):
            rows = order[i:i + a.bs]
            pred, kept = forward(rows)
            loss = F.mse_loss(pred, fut[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(merges.parameters()) + list(probe.parameters()), 1.0)
            opt.step(); sched.step()
            tot += float(loss) * len(rows)
        merges.eval(); probe.eval()
        errs, kepts = [], []
        with torch.no_grad():
            for i in range(0, len(val_rows), 8):
                rows = val_rows[i:i + 8]
                pred, kept = forward(rows)
                errs.append((torch.linalg.norm(pred - fut[rows], dim=-1) * 1000).mean(dim=(1, 2)).cpu())
                kepts.append(kept)
        vade = float(torch.cat(errs).mean())
        log.append(dict(epoch=ep + 1, train_mse=tot / len(order), val_ade_soft=vade,
                        kept=float(np.mean(kepts)), minutes=(time.time() - t0) / 60))
        print(json.dumps(log[-1]), flush=True)
        if vade < best[0]:
            best = (vade, [m.g.state_dict() for m in merges])
            torch.save(dict(matchers=best[1], args=vars(a), val_ade_soft=vade), out / "matchers.pt")
    write_json(dict(log=log, best_val_ade_soft=best[0], args=vars(a)), out / "log.json")
    print("B2 TRAIN DONE", best[0], flush=True)


if __name__ == "__main__":
    main()
