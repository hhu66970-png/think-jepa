#!/usr/bin/env python
"""Clean encoder-forward latency: dense vs K-BSM vs WAM at working points.
restore_dense=False (pure compute on the reduced token set). Free GPU, warmup +
median over N runs. Reports median ms, tokens, speedup vs dense."""
import argparse, statistics, sys, time
import torch
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E


@torch.no_grad()
def run_cfg(model, video, *, enabled, strategy, layers, ratio, relevance=None, n=30, warm=8):
    E.apply_merge_config(model, enabled=enabled, strategy=strategy, merge_layers=layers,
                         merge_ratio=ratio, restore_dense=False, bsm_match_metric="key")
    if not enabled:
        model.merge_config.enabled = False
    if relevance is not None:
        model.merge_config.relevance_source = relevance[0]
        model.merge_config.relevance_lambda = float(relevance[1])
    tok = None
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize(); ts = []
        torch.cuda.synchronize(); t0 = time.perf_counter()
        outs, infos = model(video, return_merge_info=True, restore_dense=False)
        torch.cuda.synchronize()
        if i >= warm:
            ts.append((time.perf_counter() - t0) * 1000.0)
        if tok is None:
            tok = int(infos[-1]["num_tokens_after"]) if infos else int(outs[0].shape[1])
    return statistics.median(ts), min(ts), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="outputs/wam_frontier/enc_speed.tsv")
    args = ap.parse_args()
    GR5 = [12, 14, 16, 18, 20]; L9 = list(range(12, 21))
    cfgs = [
        ("dense",       dict(enabled=False, strategy="bsm_ksim_gradual_vec", layers=[12], ratio=0.0)),
        ("kbsm_r15",    dict(enabled=True,  strategy="bsm_ksim_gradual_vec", layers=GR5, ratio=0.15)),
        ("wam_r15",     dict(enabled=True,  strategy="bsm_taware_gradual_vec", layers=GR5, ratio=0.15, relevance=("motion", 1.0))),
        ("kbsm_L9_r25", dict(enabled=True,  strategy="bsm_ksim_gradual_vec", layers=L9, ratio=0.25)),
        ("wam_L9_r25",  dict(enabled=True,  strategy="bsm_taware_gradual_vec", layers=L9, ratio=0.25, relevance=("motion", 1.0))),
    ]
    model = E.build_model(args.checkpoint, args.num_frames, args.img_size,
                          args.patch_size, "bsm_ksim_gradual_vec", "cuda")
    video, _ = E.load_video(args.clip, args.num_frames, args.img_size, "cuda")
    rows = []
    dense_ms = None
    for name, kw in cfgs:
        med, mn, tok = run_cfg(model, video, n=args.n, **kw)
        if name == "dense":
            dense_ms = med
        sp = (dense_ms / med) if dense_ms else 1.0
        rows.append((name, tok, med, mn, sp))
        print(f"{name:12s} tokens={tok:5d} median={med:7.2f}ms min={mn:7.2f}ms speedup={sp:.3f}x", flush=True)
    with open(args.out, "w") as f:
        f.write("name\ttokens\tmedian_ms\tmin_ms\tspeedup_vs_dense\n")
        for n_, tk, md, mn, sp in rows:
            f.write(f"{n_}\t{tk}\t{md:.2f}\t{mn:.2f}\t{sp:.3f}\n")
    print(f"[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
