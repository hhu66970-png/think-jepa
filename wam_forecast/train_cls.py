"""Second task (seventh review): classify the Diving48 dive from the frozen (merged) tokens of its 32 frames. Attentive
probe as the forecasting probe (cross attention blocks over the 4096 restored tokens, with the same position
embeddings), 4 learned queries, mean pooled, linear head; video only. Five fold CV over every clip with features
(RandomState(777) split), seeds as given.
  WAM_FEAT_ROOT=<feats> WAM_SLIM=... WAM_SPLITS=... python train_cls.py --cfg kbsmpa_L9 --seed 0 --cv_fold 0 --out <dir>
Writes metrics.json, perclip_acc.npy (0/1), perclip_nll.npy (nats), eval_rows.npy."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FEAT_ROOT, SLIM, TOKENS_PAST, all_clips, read_split, write_json
from train_probe import FeatureBank, XBlock, out_pool, set_determinism


class ClsProbe(nn.Module):
    def __init__(self, n_cls, d=256, heads=8, depth=3, D=1024, nq=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, nq, d) * 0.02)
        self.vin = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pt = nn.Parameter(torch.zeros(1, 16, 1, 1, d))
        self.ph = nn.Parameter(torch.zeros(1, 1, 16, 1, d))
        self.pw = nn.Parameter(torch.zeros(1, 1, 1, 16, d))
        for p in (self.pt, self.ph, self.pw):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(XBlock(d, heads) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))

    def forward(self, vis, idx=None):
        B = vis.shape[0]
        if idx is None:                           # restore readout (and dense): published arithmetic
            x = self.vin(vis.float()).view(B, 16, 16, 16, -1)
            x = (x + self.pt + self.ph + self.pw).reshape(B, TOKENS_PAST, -1)
        else:                                     # tenth review, compact readout as the forecasting probe: each token's
            grid = (self.pt + self.ph + self.pw).reshape(1, TOKENS_PAST, -1)       # position is the mean over its cells
            xc = self.vin(vis.float())
            M = F.one_hot(idx, xc.shape[1]).float()
            pos = torch.einsum("bik,id->bkd", M, grid[0].float()) / M.sum(1).clamp_min(1)[..., None]
            x = xc + pos.to(xc.dtype)
        q = self.queries.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, x)
        return self.out(q.mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv_fold", type=int, default=-1, help="-1: official split (train_cache -> test_cache)")
    ap.add_argument("--rows_from", default="dense")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--readout", choices=["restore", "compact"], default="restore")
    ap.add_argument("--outpool", default="", help="tenth review: kt,kh,kw pooling of the restored output (compact)")
    ap.add_argument("--save_model", action="store_true", help="also save the trained probe (model.pt; figures only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    opool = tuple(int(v) for v in a.outpool.split(",")) if a.outpool else None
    assert opool is None or a.readout == "compact"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    set_determinism(a.seed)
    dev = torch.device("cuda")
    lab = json.loads((SLIM.parent / "labels.json").read_text())
    keys = all_clips()
    y = np.array([lab[k[1]] for k in keys])
    bank = FeatureBank(a.cfg)
    allrows = np.array(sorted(set(FeatureBank(a.rows_from).where) & set(bank.where)))
    if a.cv_fold >= 0:
        folds = np.array_split(np.random.RandomState(777).permutation(allrows), 5)
        eval_rows = np.sort(folds[a.cv_fold]); train_rows = np.setdiff1d(allrows, eval_rows)
    else:                                         # eighth review: official train -> official test
        n_train = len(read_split("train"))
        train_rows = allrows[allrows < n_train]; eval_rows = allrows[allrows >= n_train]
    if bank.nbytes() < 7e9:                       # small banks live on the GPU; large ones are read per batch
        bank.to_device(dev)
    yt = torch.from_numpy(y).to(dev)

    def visual(rows):
        if opool is not None:
            return out_pool(bank.batch(rows, dev), opool)
        if a.readout == "compact":
            return bank.batch_compact(rows, dev)
        return bank.batch(rows, dev), None
    model = ClsProbe(int(y.max()) + 1, D=int(bank.D)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    spe = math.ceil(len(train_rows) / a.bs); total = a.epochs * spe; warm = 2 * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows), generator=g).numpy()]
        for i in range(0, len(order), a.bs):
            rows = order[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*visual(rows))
            loss = F.cross_entropy(logits.float(), yt[rows])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    model.eval(); nll, acc = [], []
    with torch.no_grad():
        for i in range(0, len(eval_rows), 50):
            rows = eval_rows[i:i + 50]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(*visual(rows)).float()
            nll.append(F.cross_entropy(lg, yt[rows], reduction="none").cpu()); acc.append((lg.argmax(1) == yt[rows]).float().cpu())
    nll, acc = torch.cat(nll).numpy(), torch.cat(acc).numpy()
    write_json(dict(cfg=a.cfg, seed=a.seed, split=f"cv{a.cv_fold}" if a.cv_fold >= 0 else "official", n_eval=len(eval_rows), n_train=len(train_rows),
                    nll=float(nll.mean()), acc=float(acc.mean()), seconds=time.time() - t0, readout=a.readout, outpool=a.outpool,
                    K=int(bank.K), args=vars(a)), out / "metrics.json")
    if a.save_model:
        torch.save(dict(state=model.state_dict(), n_cls=int(y.max()) + 1, D=int(bank.D), readout=a.readout), out / "model.pt")
    np.save(out / "perclip_nll.npy", nll); np.save(out / "perclip_acc.npy", acc); np.save(out / "eval_rows.npy", eval_rows.astype(np.int32))
    print(f"{a.cfg} s{a.seed} cv{a.cv_fold}: acc {acc.mean():.3f} nll {nll.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
