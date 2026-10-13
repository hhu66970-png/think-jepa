"""Checks for rounds CF and SS (CPU only, no encoder):
  1. hand_masks.npz: mask_cur equals the last tubelet of mask repeated; mask_f31 lies inside mask_cur, is constant in
     time, and is empty for at most 1% of the clips whose mask_cur is not.
  2. shuffle_tokens permutes values within each clip (same multiset), is deterministic, and differs between clips.
Exits nonzero on any failure."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("VJEPA2_ROOT", str(Path(__file__).resolve().parents[1] / "vjepa2")))
from src.models.utils.token_merge_diagnostics import shuffle_tokens  # noqa: E402
from common import FEAT_ROOT  # noqa: E402

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

H = np.load(FEAT_ROOT / "hand_masks.npz")
m, cur, f31 = H["mask"], H["mask_cur"], H["mask_f31"]
check("mask_cur == last tubelet repeated", np.array_equal(cur, np.repeat(m[:, -1:], 16, axis=1)))
check("mask_f31 inside mask_cur", (f31 <= cur).all())
check("mask_f31 constant in time", (f31 == f31[:, :1]).all())
lost = int((cur.reshape(len(cur), -1).any(1) & ~f31.reshape(len(f31), -1).any(1)).sum())
print("clips whose hand is visible in frame 30 but not in frame 31:", lost)
check("mask_f31 empty only where frame 31 shows no hand (at most 1% of clips)", lost <= 0.01 * len(f31))
print("protected fraction: mask %.3f mask_cur %.3f mask_f31 %.3f" % (m.mean(), cur.mean(), f31.mean()))

torch.manual_seed(0)
v = torch.rand(3, 500); x = torch.randn(3, 500, 8)
a, b = shuffle_tokens(v, x), shuffle_tokens(v, x)
check("shuffle keeps values", torch.equal(a.sort(1).values, v.sort(1).values))
check("shuffle deterministic", torch.equal(a, b))
check("shuffle moves values", (a != v).float().mean() > 0.9)
print("ALL OK:", ok)
sys.exit(0 if ok else 1)
