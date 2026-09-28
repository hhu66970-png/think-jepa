"""Encoder efficiency on an otherwise idle GPU: latency (median, IQR), throughput, peak memory.

  CUDA_VISIBLE_DEVICES=0 python bench.py --configs dense,kbsm_L9,wam_L9,handmult_L9 --batch 1,8
8 warm-up and 30 timed forwards per (config, batch) on the same real clips; fp16 autocast,
as in extraction. The hand-prior arms include setting the external relevance tensor.
"""
import argparse
import json
import time

import numpy as np
import torch

import extract as E
from common import FEAT_ROOT, all_clips, clip_path, merge_config, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=str(FEAT_ROOT / "bench.json"))
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    keys = all_clips()[:8]
    u8 = torch.stack([torch.from_numpy(np.load(clip_path(k))["imgs"][:32]) for k in keys]).cuda()
    x8 = E.preprocess(u8)
    hm = torch.from_numpy(np.load(FEAT_ROOT / "hand_masks.npz")["mask"][:8].reshape(8, -1)
                          .astype(np.float32)).cuda()
    enc = E.load_encoder()
    res = {}
    for name in a.configs.split(","):
        cfg = None if name == "dense" else merge_config(*name.split("_", 1))
        E.set_merge(enc, cfg)
        for b in [int(v) for v in a.batch.split(",")]:
            x = x8[:b]
            ts = []
            torch.cuda.reset_peak_memory_stats()
            for i in range(a.warmup + a.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if cfg is not None and cfg.get("relevance_source") == "external":
                    E.TMD.set_external_relevance(hm[:b])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    enc(x, restore_dense=False)
                torch.cuda.synchronize()
                if i >= a.warmup:
                    ts.append((time.perf_counter() - t0) * 1000)
            ts = np.array(ts)
            q1, med, q3 = np.percentile(ts, [25, 50, 75])
            res[f"{name}@b{b}"] = dict(median_ms=med, iqr_ms=[q1, q3], clips_per_s=1000 * b / med,
                                       peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            print(f"{name:>14s} b{b}: {med:7.1f} ms  IQR [{q1:.1f},{q3:.1f}]  "
                  f"{1000*b/med:6.1f} clip/s  peak {res[f'{name}@b{b}']['peak_mem_gb']:.2f} GB", flush=True)
    res["_device"] = torch.cuda.get_device_name(0)
    res["_torch"] = torch.__version__
    write_json(res, a.out)


if __name__ == "__main__":
    main()
