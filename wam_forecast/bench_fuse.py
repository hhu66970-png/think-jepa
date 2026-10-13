"""Round FUSE (tenth review): wall clock of the frozen V-JEPA 2 ViT-L without host overhead. For every arm of the latency
table, the unchanged forward (WAM_LEAN=1: the merge step skips its host synchronising diagnostics and gathers survivors
instead of boolean indexing; same values) is (a) run eagerly and (b) captured once as a CUDA graph and replayed, so kernel
launch and Python overhead are removed. Before timing, the lean output is checked against the reference forward
(WAM_LEAN=0 semantics, loaded in a subprocess) on 8 real clips: tokens and representative indices must agree.
Attention with the proportional attention bias already runs a fused kernel (memory efficient SDPA; flash without bias).
Batch 1 and 8, fp16, 10 warm up and 50 timed replays, median; otherwise idle RTX 3090.
  WAM_LEAN=1 CUDA_VISIBLE_DEVICES=0 python bench_fuse.py --out runs/phaseA/bench_fuse.json"""
import argparse, json, os, sys, time
import numpy as np
import torch
import extract as E
from common import W, all_clips, past_frames, merge_config, write_json

ARMS = ["dense", "in24f", "in16f", "in12f", "in8f", "in4f", "kbsmpa_s25", "wampa_s25", "kbsmpa_L9", "wampa_L9",
        "kbsmpa_L12", "in24f+kbsmpa_s25", "in24f+wampa_s25", "in16f+kbsmpa_s25", "in16f+wampa_s25", "in16f+kbsmpa_L9"]


def cfg_of(name):
    if name == "dense" or name in E.INPUT_DS:
        return None
    base, rest = (name.split("+", 1) + [None])[:2] if "+" in name else (None, name)
    cfg = merge_config(*rest.split("_", 1))
    if base:
        layers = [int(v) for v in str(cfg["merge_layers"]).split(",")]
        cfg["merge_counts"] = E.composite_counts(E.INPUT_TOKENS[base], layers, E.composite_target(base, rest.split("_", 1)[1]))
    return cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--check", default="")
    a = ap.parse_args()
    assert os.environ.get("WAM_LEAN") == "1"
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    keys = all_clips()[:8]
    u8 = torch.stack([torch.from_numpy(np.ascontiguousarray(past_frames(k))) for k in keys]).cuda()
    x8 = E.preprocess(u8)
    enc = E.load_encoder()
    ref = torch.load(a.check) if a.check else {}
    res = {"_device": torch.cuda.get_device_name(0)}

    def fwd(xin):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = enc(xin, restore_dense=False)
        st = getattr(enc, "last_merge_state", None)
        return out, st

    def tm(fn, it=50, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(it):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))

    for name in ARMS:
        cfg = cfg_of(name)
        E.set_merge(enc, cfg)
        base = name.split("+", 1)[0]
        if name in ref:   # equality with the reference forward (8 clips)
            xin = E.INPUT_DS[base](x8) if base in E.INPUT_DS else x8
            out, st = fwd(xin)
            r = ref[name]
            res[f"{name}@check"] = dict(max_abs_diff=float((out.float().cpu() - r["out"].float()).abs().max()),
                                        ids_equal=bool(cfg is None or torch.equal(st[0].cpu(), r["ids"])))
            print(name, "check", res[f"{name}@check"], flush=True)
        for b in (1, 8):
            xs = x8[:b].clone()
            static = (E.INPUT_DS[base](xs) if base in E.INPUT_DS else xs).clone()
            eager = tm(lambda: fwd(static))
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fwd(static)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout, _ = fwd(static)
            ref_out, _ = fwd(static)
            g.replay(); torch.cuda.synchronize()
            same = float((gout.float() - ref_out.float()).abs().max())
            graph = tm(g.replay)
            res[f"{name}@b{b}"] = dict(eager_ms=eager, graph_ms=graph, graph_vs_eager_max_abs_diff=same,
                                       tokens=int(gout.shape[1]))
            print(name, b, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f"{name}@b{b}"].items()}, flush=True)
            del g
            torch.cuda.empty_cache()
    write_json(res, Path_out(a.out))


def Path_out(p):
    from pathlib import Path
    return Path(p)


if __name__ == "__main__":
    main()
