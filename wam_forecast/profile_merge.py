"""Round PROF (ninth review): where merging spends time. CUDA time per forward of all attention modules, all MLPs, and
all merge steps of a frozen V-JEPA 2 ViT-L, for dense tokens and for K-BSM+PA to 972 and 309 tokens, batch 1 and 8
(fp16, 5 warm up and 20 timed forwards, forward hooks with CUDA events); and, for comparison, the same merge steps
timed with a minimal reimplementation of ToMe's bipartite soft matching (alternating split, cosine scores, top r,
size weighted scatter) on random tokens of the same sizes.
  CUDA_VISIBLE_DEVICES=0 python profile_merge.py -> runs/phaseA/profile_merge.json"""
import json
import time

import numpy as np
import torch

import extract as E
from common import W, all_clips, clip_path, merge_config, write_json


def tome_merge(x, metric, size, r):
    """ToMe bipartite soft matching with size weighted averaging (as tome/merge.py, without the protected class token)."""
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = edge_idx[..., r:, :], edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(t):
        src, dst = t[..., ::2, :], t[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        s = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), s, reduce="sum")
        return torch.cat([unm, dst], dim=1)
    x = merge(x * size); size = merge(size)
    return x / size, size


def timed(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    torch.backends.cudnn.benchmark = False
    keys = all_clips()[:8]
    u8 = torch.stack([torch.from_numpy(np.load(clip_path(k))["imgs"][:32]) for k in keys]).cuda()
    x8 = E.preprocess(u8)
    enc = E.load_encoder()
    events = []

    def pre(name):
        def f(m, inp):
            e = torch.cuda.Event(enable_timing=True); e.record(); events.append([name, e, None])
        return f

    def post(name):
        def f(m, inp, out):
            for ev in reversed(events):
                if ev[0] == name and ev[2] is None:
                    e = torch.cuda.Event(enable_timing=True); e.record(); ev[2] = e
                    break
        return f
    for blk in enc.blocks:
        blk.attn.register_forward_pre_hook(pre("attention")); blk.attn.register_forward_hook(post("attention"))
        blk.mlp.register_forward_pre_hook(pre("mlp")); blk.mlp.register_forward_hook(post("mlp"))
    res = {}
    for name in ("dense", "kbsmpa_s25", "kbsmpa_L9"):
        cfg = None if name == "dense" else merge_config(*name.split("_", 1))
        E.set_merge(enc, cfg)
        if enc.token_merger is not None and not getattr(enc.token_merger, "_hooked", False):
            enc.token_merger.register_forward_pre_hook(pre("merge")); enc.token_merger.register_forward_hook(post("merge"))
            enc.token_merger._hooked = True
        for b in (1, 8):
            x = x8[:b]
            tot = {"attention": [], "mlp": [], "merge": [], "total": []}
            for i in range(25):
                events.clear()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    enc(x, restore_dense=False)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                if i < 5:
                    continue
                acc = {"attention": 0.0, "mlp": 0.0, "merge": 0.0}
                for n, s, e in events:
                    if e is not None:
                        acc[n] += s.elapsed_time(e)
                for k in acc:
                    tot[k].append(acc[k])
                tot["total"].append((t1 - t0) * 1000)
            res[f"{name}@b{b}"] = {k: float(np.median(v)) for k, v in tot.items()}
            print(name, b, {k: round(v, 1) for k, v in res[f"{name}@b{b}"].items()}, flush=True)
        # ToMe reference on the same schedule
        if cfg is not None:
            layers = [int(v) for v in str(cfg["merge_layers"]).split(",")]
            for b in (1, 8):
                n, ms = 4096, 0.0
                for _ in layers:
                    r = n - int(round(n * 0.75))
                    xx = torch.randn(b, n, 1024, device="cuda", dtype=torch.float16)
                    mm = torch.randn(b, n, 64, device="cuda", dtype=torch.float16)
                    sz = torch.ones(b, n, 1, device="cuda", dtype=torch.float16)
                    ms += timed(lambda: tome_merge(xx, mm, sz, r))
                    n -= r
                res[f"{name}@b{b}"]["tome_reference_merge"] = ms
                print(name, b, "ToMe reference merge total ms", round(ms, 2), flush=True)
    res["_device"] = torch.cuda.get_device_name(0)
    write_json(res, W / "runs" / "phaseA" / "profile_merge.json")


if __name__ == "__main__":
    main()
