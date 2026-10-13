"""Checks for the order-preserving variants: kbsm0 == kbsm bitwise; wam unchanged vs the cached
full-run features; every new arm yields the scheduled K, finite output, and protects hands."""
import numpy as np
import torch

import extract as E
from common import FEAT_ROOT, METHODS, all_clips, clip_path, merge_config
from train_probe import FeatureBank

keys = all_clips()
rows = list(range(4))
u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[r]))["imgs"][:32]) for r in rows]).cuda()
x = E.preprocess(u8)
H = np.load(FEAT_ROOT / "hand_masks.npz")
hm = H["mask"].reshape(len(keys), -1)
E.TMD.set_external_relevance(torch.from_numpy(H["density"].reshape(len(keys), -1)[rows].astype(np.float32)).cuda())
enc = E.load_encoder()


def run(m, s="L9"):
    E.set_merge(enc, merge_config(m, s))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = enc(x, restore_dense=False)
    tid, _, rep = enc.last_merge_state
    return out.float().cpu(), E.rep_index(tid, rep).cpu()


ref, _ = run("kbsm")
k0, _ = run("kbsm0")
print("kbsm0 == kbsm:", torch.equal(ref, k0))
for m in ("kbsm", "wam"):
    bank = FeatureBank(f"{m}_L9")
    w, _ = run(m)
    cached = torch.stack([torch.from_numpy(np.asarray(bank.tok[bank.where[r][0]][bank.where[r][1]]).astype(np.float32)) for r in rows])
    d = (w.half().float() - cached).abs().amax(dim=(1, 2))
    print(f"{m} rerun (batch 4) vs cached (batch 8) per-clip max|diff|:", d.tolist())
for m in ("hrecv15", "hrecv30", "hexcl15", "hexcl30", "hanc1", "hanc4", "hrecvanc", "mrecv15", "manc4"):
    out, idx = run(m)
    h = torch.from_numpy(hm[rows]).bool()
    size = torch.stack([torch.bincount(idx[b], minlength=out.shape[1])[idx[b]] for b in range(len(rows))])
    single = (size[h] == 1).float().mean().item()
    print(f"{m:>9s}: K={out.shape[1]} finite={bool(torch.isfinite(out).all())} "
          f"hand-unmerged={single:.3f}")
