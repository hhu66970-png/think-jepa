"""Input invariant of the current pose setting (input vispose, pose_frames 1), as train_probe.py builds it.
Asserts that (1) with pose_frames 1 the pose branch receives exactly zero (the last pose repeated, minus itself),
(2) with 32 past poses it does not, and (3) the last pose enters only as the residual anchor of the output
(base_last for inputs pose and vispose). The current pose setting is therefore video plus a forecast of the
displacement from the current pose. Exits nonzero on any failure.  python test_current_pose.py"""
import re
import sys
from pathlib import Path

import numpy as np

from common import FEAT_ROOT

ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


P = np.load(FEAT_ROOT / "poses.npz")
past = P["past"].astype(np.float64)                     # [N, 32, 52, 3]


def pose_input(past, k):                                # the transform of train_probe.py (pose_frames k)
    T0 = past.shape[1]
    if k < T0:
        past = np.concatenate([np.repeat(past[:, T0 - k:T0 - k + 1], T0 - k, axis=1), past[:, T0 - k:]], 1)
    return past - past[:, -1:]


z1, z32 = pose_input(past, 1), pose_input(past, 32)
check("(1) pose_frames 1: pose branch input is exactly zero", np.abs(z1).max() == 0.0, f"max |input| = {np.abs(z1).max()}")
check("(2) pose_frames 32: pose branch input is not zero", np.abs(z32).max() > 0.0, f"max |input| = {np.abs(z32).max():.3f} m")
src = Path(__file__).with_name("train_probe.py").read_text()
check("(3) train_probe.py repeats the last pose and subtracts it, and anchors vispose at the last pose",
      "past[:, T0 - k:T0 - k + 1].expand(-1, T0 - k, -1, -1)" in src and "(pst - pst[:, -1:]) / sd" in src
      and re.search(r'base_last = a\.input in \("pose", "vispose"\)', src) is not None)
sys.exit(0 if ok else 1)
