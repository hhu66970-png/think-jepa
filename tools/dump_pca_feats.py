#!/usr/bin/env python
"""Dump features + REAL merge partitions for the unified four-method PCA figure.

For each clip we run ONE forward per method (dense / scheme-A / K-BSM / PiToMe)
with restore_dense=True, and save to a small .npz:

  - dense_L5      : [N, D] float16  -- clean shallow-layer features (color source,
                    identical across methods since merging starts at L>=8).
  - <method>_gids : [N]   int32     -- group id per ORIGINAL token = which surviving
                    token each patch was merged into. Recovered from the restored
                    deep layer: merged members are bit-identical copies, so
                    torch.unique(dim=0) gives the exact partition.
  - <method>_ng   : int             -- #groups (== #surviving tokens; asserted).
  - input_frame   : [S, S, 3] uint8 -- representative frame for the thumbnail.
  - meta          : t, h, w, token_t, clip id/task, finals.

Rendering (PCA, foreground recipe, block painting, layout) is done separately by
render_unified.py on CPU so it can be iterated without re-running the GPU.

Model / merge plumbing reused verbatim from run_token_merge_pca_experiment.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E  # noqa: E402


@torch.no_grad()
def forward_capture(model, video, *, enabled, strategy, layers, ratio, metric, capture,
                    relevance=None):
    """Run one forward (restore_dense=True); return {layer: [N,D] cpu float}, final."""
    E.apply_merge_config(model, enabled=enabled, strategy=strategy, merge_layers=layers,
                         merge_ratio=ratio, restore_dense=True, bsm_match_metric=metric)
    if relevance is not None:
        model.merge_config.relevance_source = relevance[0]
        model.merge_config.relevance_lambda = float(relevance[1])
    if not enabled:
        model.merge_config.enabled = False  # force true dense
    model.out_layers = sorted(capture)
    outs, infos = model(video, return_merge_info=True, restore_dense=True)
    final = int(infos[-1]["num_tokens_after"]) if infos else int(outs[0].shape[1])
    feats = {int(l): outs[i][0].float().cpu() for i, l in enumerate(sorted(capture))}
    return feats, final


def partition_from_restored(feat_restored):
    """feat_restored [N,D] (restore_dense): merged members are identical copies.
    Exact unique over rows -> group id per token. Returns (gids[N] int32, ngroups)."""
    uniq, inv = torch.unique(feat_restored, dim=0, return_inverse=True)
    return inv.to(torch.int32).cpu().numpy(), int(uniq.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--clips", required=True,
                    help="comma list of absolute npz paths")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--ref_layer", type=int, default=5)
    ap.add_argument("--deep_layer", type=int, default=20)
    ap.add_argument("--scheme_a_layer", type=int, default=8)
    ap.add_argument("--scheme_a_ratio", type=float, default=0.25)
    ap.add_argument("--bsm_layers", default="12,13,14,15,16,17,18,19,20")
    ap.add_argument("--bsm_ratio", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t = args.num_frames // 2
    h = w = args.img_size // args.patch_size
    token_t = t // 2
    bsm_layers = [int(v) for v in args.bsm_layers.split(",")]
    clip_paths = [c.strip() for c in args.clips.split(",") if c.strip()]
    print(f"[INFO] grid={h}x{w} t={t} token_t={token_t} clips={len(clip_paths)}")

    model = E.build_model(args.checkpoint, args.num_frames, args.img_size,
                          args.patch_size, "bsm_ksim_gradual_vec", args.device)

    # (label, enabled, strategy, layers, ratio, metric, relevance)
    methods = [
        ("dense",    False, "local_2x2_same_time_vec", [], 0.0, "key", None),
        ("scheme_a", True,  "local_2x2_same_time_vec", [args.scheme_a_layer],
         args.scheme_a_ratio, "key", None),
        ("kbsm",     True,  "bsm_ksim_gradual_vec", bsm_layers, args.bsm_ratio, "key", None),
        ("pitome",   True,  "bsm_pitome_gradual_vec", bsm_layers, args.bsm_ratio, "key", None),
        ("wam",      True,  "bsm_taware_gradual_vec", bsm_layers, args.bsm_ratio, "key",
         ("motion", 1.0)),
    ]

    for path in clip_paths:
        cid = os.path.basename(path).split("_")[0]
        task = os.path.basename(os.path.dirname(path))
        video, _ = E.load_video(path, args.num_frames, args.img_size, args.device)
        imgs = np.load(path, allow_pickle=True)["imgs"]
        mid = imgs.shape[0] // 2
        from PIL import Image
        frame = np.asarray(Image.fromarray(imgs[mid]).resize(
            (args.img_size, args.img_size), Image.Resampling.BICUBIC), dtype=np.uint8)

        save = dict(t=t, h=h, w=w, token_t=token_t, cid=cid, task=task,
                    ref_layer=args.ref_layer, deep_layer=args.deep_layer,
                    input_frame=frame)
        finals = {}
        for label, en, strat, lays, r, metric, rel in methods:
            cap = [args.ref_layer, args.deep_layer] if label == "dense" else [args.deep_layer]
            feats, final = forward_capture(model, video, enabled=en, strategy=strat,
                                           layers=lays, ratio=r, metric=metric, capture=cap,
                                           relevance=rel)
            finals[label] = final
            if label == "dense":
                save["dense_L5"] = feats[args.ref_layer].half().numpy()
            gids, ng = partition_from_restored(feats[args.deep_layer])
            save[f"{label}_gids"] = gids
            save[f"{label}_ng"] = ng
            flag = "OK" if ng == final else f"MISMATCH(final={final})"
            print(f"  [{cid}] {label:9s} final={final:5d} ngroups={ng:5d} {flag}", flush=True)
        save["finals"] = finals
        out = os.path.join(args.out_dir, f"feats_{cid}.npz")
        np.savez_compressed(out, **save)
        sz = os.path.getsize(out) / 1e6
        print(f"  [{cid}] saved {out} ({sz:.1f} MB)", flush=True)
    print("[DONE]")


if __name__ == "__main__":
    main()
