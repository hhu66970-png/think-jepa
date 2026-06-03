#!/usr/bin/env python
"""Encoder-side cos-vs-dense frontier for dense/方案A/K-BSM/PiToMe/WAM at an r grid.
Reuses run_token_merge_pca_experiment (build_model/apply_merge_config/load_video).
256px/64f geometry (=downstream: gradual L12-20 r0.15 -> 3637 tokens). Reports
cos_vs_dense (mean over tokens, restore_dense=True) + survivor token count."""
import argparse, glob, os, sys
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E


@torch.no_grad()
def final_feat(model, video, *, enabled, strategy, layers, ratio, metric="key",
               relevance=None, axis="free", pmargin=0.0):
    E.apply_merge_config(model, enabled=enabled, strategy=strategy, merge_layers=layers,
                         merge_ratio=ratio, restore_dense=True, bsm_match_metric=metric,
                         merge_axis=axis, pitome_margin=pmargin)
    if not enabled:
        model.merge_config.enabled = False
    if relevance is not None:
        model.merge_config.relevance_source = relevance[0]
        model.merge_config.relevance_lambda = float(relevance[1])
    outs, infos = model(video, return_merge_info=True, restore_dense=True)
    final = int(infos[-1]["num_tokens_after"]) if infos else int(outs[0].shape[1])
    return outs[0][0].float(), final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--final_layer", type=int, default=23)
    ap.add_argument("--clips", required=True, help="comma list of npz paths")
    ap.add_argument("--ratios", default="0.08,0.15,0.20,0.25")
    ap.add_argument("--out", default="outputs/wam_frontier/enc_cos.tsv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    GR5 = [12, 14, 16, 18, 20]
    LSETS = {"L9": list(range(12, 21)), "L10": list(range(12, 22)), "L12": list(range(10, 22))}
    METHODS = [("kbsm", "bsm_ksim_gradual_vec", None),
               ("pitome", "bsm_pitome_gradual_vec", None),
               ("wam", "bsm_taware_gradual_vec", ("motion", 1.0))]

    # config NAMES match the frozen-frontier drivers so ADE<->tokens merge by name.
    cfgs = [("dense", dict(enabled=False, strategy="bsm_ksim_gradual_vec", layers=[12], ratio=0.0)),
            ("A_r25", dict(enabled=True, strategy="local_2x2_same_time_vec", layers=[8], ratio=0.25))]
    for r in [0.08, 0.15, 0.20, 0.25]:                       # 5-layer grid
        rr = f"r{int(round(r*100)):02d}"
        for nm, strat, rel in METHODS:
            cfgs.append((f"{nm}_{rr}", dict(enabled=True, strategy=strat, layers=GR5,
                                            ratio=r, relevance=rel)))
    for ln, layers in LSETS.items():                          # aggressive layer-sets
        for r in [0.20, 0.25]:
            rr = f"r{int(round(r*100)):02d}"
            for nm, strat, rel in METHODS:
                cfgs.append((f"{nm}_{ln}_{rr}", dict(enabled=True, strategy=strat,
                                                     layers=layers, ratio=r, relevance=rel)))

    model = E.build_model(args.checkpoint, args.num_frames, args.img_size,
                          args.patch_size, "bsm_ksim_gradual_vec", args.device)
    model.out_layers = [args.final_layer]

    acc = {name: {"cos": [], "tok": []} for name, _ in cfgs}
    for ci, npz in enumerate(clips):
        video, _ = E.load_video(npz, args.num_frames, args.img_size, args.device)
        dense_feat, _ = final_feat(model, video, **cfgs[0][1])
        for name, kw in cfgs:
            feat, tok = final_feat(model, video, **kw)
            cos = F.cosine_similarity(dense_feat, feat, dim=-1).mean().item()
            acc[name]["cos"].append(cos); acc[name]["tok"].append(tok)
            print(f"  clip{ci} {name:11s} tokens={tok:5d} cos={cos:.4f}", flush=True)

    with open(args.out, "w") as f:
        f.write("name\ttokens\tcos_vs_dense\n")
        for name, _ in cfgs:
            tok = int(np.mean(acc[name]["tok"])); cos = float(np.mean(acc[name]["cos"]))
            f.write(f"{name}\t{tok}\t{cos:.4f}\n")
            print(f"[AVG] {name:11s} tokens={tok:5d} cos={cos:.4f}")
    print(f"[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
