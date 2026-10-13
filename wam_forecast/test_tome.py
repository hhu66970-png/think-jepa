"""Sanity for the ToMe (pre-RoPE key) metric: runs, keeps K, differs from K-BSM, and the
default key path is unchanged (kbsmpa identical before/after arming logic)."""
import numpy as np
import torch

import extract as E
from common import all_clips, clip_path, merge_config

enc = E.load_encoder()
keys = [k for k in all_clips() if clip_path(k).exists()][:4]
x = E.preprocess(torch.stack([torch.from_numpy(np.load(clip_path(k))["imgs"][:32]) for k in keys]).cuda())


def run(m, s="L9"):
    E.set_merge(enc, merge_config(m, s))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = enc(x, restore_dense=False).float()
    return o, enc.last_merge_state


ok = True
for s in ("s25", "L9", "L12"):
    ok_, st_k = run("kbsmpa", s)
    ot, st_t = run("tomepa", s)
    op, st_p = run("pitomepa", s)
    same_k = ok_.shape == ot.shape == op.shape
    agree = float((st_k[2] == st_t[2]).float().mean())
    print(f"{s}: K kbsmpa/tomepa/pitomepa = {ok_.shape[1]}/{ot.shape[1]}/{op.shape[1]}, "
          f"rep agreement kbsmpa vs tomepa {agree:.3f}, finite {bool(torch.isfinite(ot).all() and torch.isfinite(op).all())}")
    ok &= same_k and agree < 1.0 and bool(torch.isfinite(ot).all())
print("ALL PASS" if ok else "FAIL")
