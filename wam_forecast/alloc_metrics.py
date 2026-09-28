"""Allocation-level diagnostics from the feature caches (no training).

For every merged config and every clip, against the dense cache of the same clip:
  fid_hand / fid_bg   mean cosine(restored token, dense token) over hand / background tokens
  gsize_hand/_bg      mean group size (#original tokens sharing the representative)
  singleton_hand      fraction of hand tokens that were never merged
  gini                Gini coefficient of group sizes (capacity concentration, R15)
  gs_median/_max/_var median / max / variance of the K group sizes (Phase A)
  gs_tok              token-weighted mean group size (size of the group an original token is in)
  gs_p99              99th percentile of the group sizes
Clip-level means are reported with a clip-bootstrap 95% CI, plus the paired difference to a
reference config (default: kbsm of the same schedule).

  python alloc_metrics.py --cfgs kbsm_L9,wam_L9,wamcomp_L9 [--limit 400]
"""
import argparse
import json

import numpy as np
import torch

from common import FEAT_ROOT, write_json
from train_probe import FeatureBank


def gini(x):
    x = np.sort(x.astype(np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def per_clip(cfg, dense, hand, rows, dev):
    bank = FeatureBank(cfg)
    out = []
    for r in rows:
        s, j = bank.where[int(r)]
        tok = torch.from_numpy(np.asarray(bank.tok[s][j])).to(dev).float()
        idx = torch.from_numpy(np.asarray(bank.idx[s][j]).astype(np.int64)).to(dev)
        ds, dj = dense.where[int(r)]
        den = torch.from_numpy(np.asarray(dense.tok[ds][dj])).to(dev).float()
        cos = torch.nn.functional.cosine_similarity(tok[idx], den, dim=-1).cpu().numpy()
        size = np.bincount(idx.cpu().numpy(), minlength=tok.shape[0])
        g = size[idx.cpu().numpy()]
        h = hand[r].reshape(-1)
        out.append(dict(fid_hand=cos[h].mean() if h.any() else np.nan, fid_bg=cos[~h].mean(),
                        gsize_hand=g[h].mean() if h.any() else np.nan, gsize_bg=g[~h].mean(),
                        singleton_hand=(g[h] == 1).mean() if h.any() else np.nan,
                        gini=gini(size), gs_median=float(np.median(size)), gs_max=float(size.max()),
                        gs_var=float(size.var()), gs_tok=float(g.mean()),
                        gs_p99=float(np.percentile(size, 99))))
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def boot_ci(v, rng, n=2000):
    v = v[np.isfinite(v)]
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--ref", default="kbsm", help="reference method for paired differences")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="method selection must use train clips only")
    ap.add_argument("--out", default=str(FEAT_ROOT / "alloc_metrics.json"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    hand = np.load(FEAT_ROOT / "hand_masks.npz")["mask"]
    dense = FeatureBank("dense")
    cfgs = a.cfgs.split(",")
    common_rows = set(dense.where)
    for c in cfgs:
        common_rows &= set(FeatureBank(c).where)
    if a.split != "all":
        is_test = np.load(FEAT_ROOT / "poses.npz")["is_test"]
        common_rows = {r for r in common_rows if bool(is_test[r]) == (a.split == "test")}
    rows = sorted(common_rows)[: a.limit or None]
    rng = np.random.default_rng(0)
    res = {c: per_clip(c, dense, hand, rows, dev) for c in cfgs}
    report = {"clips": len(rows), "configs": {}}
    print(f"clips={len(rows)}")
    print(f"{'cfg':>16s} {'fid_hand':>22s} {'fid_bg':>8s} {'gsize_hand':>10s} {'gsize_bg':>9s} "
          f"{'single_h':>8s} {'gini':>6s} {'gs_med':>6s} {'gs_max':>7s} {'gs_var':>8s} {'gs_tok':>7s}")
    for c in cfgs:
        m = res[c]
        fh = boot_ci(m["fid_hand"], rng)
        row = {k: float(np.nanmean(v)) for k, v in m.items()}
        row["fid_hand_ci"] = fh
        method, sched = c.split("_", 1)
        ref = f"{a.ref}_{sched}"
        if ref in res and ref != c:
            d = m["fid_hand"] - res[ref]["fid_hand"]
            row["d_fid_hand_vs_ref"] = boot_ci(d, rng)
            row["d_fid_bg_vs_ref"] = boot_ci(m["fid_bg"] - res[ref]["fid_bg"], rng)
        report["configs"][c] = row
        extra = ""
        if "d_fid_hand_vs_ref" in row:
            dh, db = row["d_fid_hand_vs_ref"], row["d_fid_bg_vs_ref"]
            extra = f"  Δhand vs {ref} {dh[0]:+.4f} [{dh[1]:+.4f},{dh[2]:+.4f}]  Δbg {db[0]:+.4f}"
        print(f"{c:>16s} {fh[0]:.4f} [{fh[1]:.4f},{fh[2]:.4f}] {row['fid_bg']:.4f} "
              f"{row['gsize_hand']:10.2f} {row['gsize_bg']:9.2f} {row['singleton_hand']:8.3f} "
              f"{row['gini']:6.3f} {row['gs_median']:6.1f} {row['gs_max']:7.1f} {row['gs_var']:8.1f} "
              f"{row['gs_tok']:7.1f}{extra}")
    write_json(report, a.out)


if __name__ == "__main__":
    main()
