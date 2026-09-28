"""Smoke checks for the forecasting pipeline (run once after any merge-code change).

1. restore map: tokens[idx] from restore_dense=False equals the encoder's restore_dense=True output;
2. every WAM-v2 method runs and yields the expected token count;
3. patched merge code with default knobs reproduces the published K-BSM/WAM bit-exactly
   (compares against the saved *.orig modules).
"""
import importlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import extract as E
from common import METHODS, REPO, SCHEDULES, W, all_clips, clip_path, merge_config

keys = [k for k in all_clips() if clip_path(k).exists()][:4]
u8 = torch.stack([torch.from_numpy(np.load(clip_path(k))["imgs"][:32]) for k in keys]).cuda()
x = E.preprocess(u8)
enc = E.load_encoder()


def run(cfg, restore):
    E.set_merge(enc, cfg)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return enc(x, restore_dense=restore)


# 1. restore map
cfg = merge_config("wam", "L9")
comp = run(cfg, False)
tid, _, rep = enc.last_merge_state
idx = E.rep_index(tid, rep)
dense_like = comp.gather(1, idx.unsqueeze(-1).expand(-1, -1, comp.shape[-1]))
ref = run(cfg, True)
print("1) restore map max|diff| =", float((dense_like.float() - ref.float()).abs().max()))

# 2. all methods
for m in METHODS:
    if METHODS[m] is None:
        continue
    for s in ("s25", "L9", "L12"):
        out = run(merge_config(m, s), False)
        print(f"2) {m:>9s}_{s:<4s} K={out.shape[1]}  finite={bool(torch.isfinite(out).all())}")

# 3. bit-exactness vs published code
new = {m: run(merge_config(m, s), False).float().cpu() for m in ("kbsm", "wam") for s in ("L9",)}
ud = REPO / "vjepa2/src/models/utils"
import src.models.utils.token_merge as tm, src.models.utils.token_merge_diagnostics as tmd
bak = {}
for f in ("token_merge.py", "token_merge_diagnostics.py"):
    bak[f] = (ud / f).read_text()
    shutil.copy(W / "tmp" / (f + ".orig"), ud / f)
try:
    importlib.reload(tm); importlib.reload(tmd)
    import src.models.vision_transformer as vt
    importlib.reload(vt)
    E.normalize_merge_config, E._build_token_merger = tm.normalize_merge_config, vt._build_token_merger
    for m in ("kbsm", "wam"):
        cfg = {k: v for k, v in merge_config(m, "L9").items()}
        old = run(cfg, False).float().cpu()
        print(f"3) {m} published vs patched: identical={torch.equal(old, new[m])} "
              f"max|diff|={float((old - new[m]).abs().max()):.3e}")
finally:
    for f, txt in bak.items():
        (ud / f).write_text(txt)
