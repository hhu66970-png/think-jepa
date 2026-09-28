"""Per-clip budget policies for the D (adaptive budget) exclusion test.

Motion score (fixed before any run): mean absolute grayscale difference between consecutive
observed frames 0..31 (uint8 scale), computable before encoding. Fractions are fixed so that
the expected token count on the TRAIN split is 309: 10 % of clips get 972 tokens, 37.25 % get
131, the rest 309 (663 * 0.10 = 178 * 0.3725). Thresholds are train-split quantiles and are
applied unchanged to the test split.
  motion : highest-motion 10 % -> 972, lowest-motion 37.25 % -> 131
  inverse: lowest-motion 10 % -> 972, highest-motion 37.25 % -> 131
  random : same fractions, assignment by a fixed permutation (seed 0) within each split
Writes <FEAT_ROOT>/policy_{motion,inverse,random}.npz and motion_score.npy.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import FEAT_ROOT, T_PAST, all_clips, clip_path

CFGS = np.array(["kbsm_s25", "kbsm_L9", "kbsm_L12"])
P_HI, P_LO = 0.10, 0.3725


def score(key):
    with np.load(clip_path(key)) as z:
        v = z["imgs"][:T_PAST].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.abs(np.diff(v, axis=0)).mean())


def main():
    keys = all_clips()
    with ThreadPoolExecutor(16) as ex:
        m = np.array(list(ex.map(score, keys)))
    np.save(FEAT_ROOT / "motion_score.npy", m)
    is_test = np.load(FEAT_ROOT / "poses.npz")["is_test"]
    tr = ~is_test
    hi, lo = np.quantile(m[tr], 1 - P_HI), np.quantile(m[tr], P_LO)
    ihi, ilo = np.quantile(m[tr], P_HI), np.quantile(m[tr], 1 - P_LO)
    pol = {}
    a = np.ones(len(m), np.int64)
    a[m >= hi] = 0
    a[m <= lo] = 2
    pol["motion"] = a
    a = np.ones(len(m), np.int64)
    a[m <= ihi] = 0
    a[m >= ilo] = 2
    pol["inverse"] = a
    a = np.ones(len(m), np.int64)
    rng = np.random.RandomState(0)
    for split in (tr, is_test):
        idx = rng.permutation(np.flatnonzero(split))
        n = len(idx)
        a[idx[: int(round(P_HI * n))]] = 0
        a[idx[int(round(P_HI * n)): int(round(P_HI * n)) + int(round(P_LO * n))]] = 2
    pol["random"] = a
    K = np.array([972, 309, 131])
    for name, a in pol.items():
        np.savez(FEAT_ROOT / f"policy_{name}.npz", cfgs=CFGS, assign=a)
        print(f"{name:>8s}: mean tokens train {K[a[tr]].mean():.1f}  test {K[a[is_test]].mean():.1f}  "
              f"test mix 972/309/131 = {np.bincount(a[is_test], minlength=3).tolist()}")
    print(f"motion score: train median {np.median(m[tr]):.2f}, thresholds hi {hi:.2f} lo {lo:.2f}")


if __name__ == "__main__":
    main()
