# -*- coding: utf-8 -*-
"""深层多层特征抽取(供 F9 多层 U 形 / F10 深层热力图 / P3 RankMe-存活 / P4 检索)。
复用 dump_pca_feats 同款 E.build_model / forward_capture / partition_from_restored(API 逐行一致)。
每 clip:① dense forward 捕获多层 [2,5,8,11,14,17,20,23] → dense_L{n};② 各方法 gids(@bsm_ratio).
P3 存活特征 = 用 gids 对 dense 深层做组均值(合并≈组内平均,无需另跑 merge);P4 池化在本地算。
零训练,纯前向。运行(λ-n=12 跑完释放 GPU 后):
  PYTHONPATH=vjepa2 python tools/redundancy_extract.py --clips "<paths>" --out_dir outputs/redundancy_dump"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E  # noqa: E402


@torch.no_grad()
def forward_capture(model, video, *, enabled, strategy, layers, ratio, metric, capture, relevance=None):
    E.apply_merge_config(model, enabled=enabled, strategy=strategy, merge_layers=layers,
                         merge_ratio=ratio, restore_dense=True, bsm_match_metric=metric)
    if relevance is not None:
        model.merge_config.relevance_source = relevance[0]
        model.merge_config.relevance_lambda = float(relevance[1])
    if not enabled:
        model.merge_config.enabled = False
    model.out_layers = sorted(capture)
    outs, infos = model(video, return_merge_info=True, restore_dense=True)
    final = int(infos[-1]["num_tokens_after"]) if infos else int(outs[0].shape[1])
    feats = {int(l): outs[i][0].float().cpu() for i, l in enumerate(sorted(capture))}
    return feats, final


def partition_from_restored(feat_restored):
    uniq, inv = torch.unique(feat_restored, dim=0, return_inverse=True)
    return inv.to(torch.int32).cpu().numpy(), int(uniq.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--cap_layers", default="2,5,8,11,14,17,20,23")
    ap.add_argument("--deep_layer", type=int, default=20)
    ap.add_argument("--bsm_layers", default="12,13,14,15,16,17,18,19,20")
    ap.add_argument("--bsm_ratio", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t = args.num_frames // 2
    h = w = args.img_size // args.patch_size
    token_t = t // 2
    cap = [int(v) for v in args.cap_layers.split(",")]
    bsm_layers = [int(v) for v in args.bsm_layers.split(",")]
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    print(f"[INFO] grid={h}x{w} t={t} cap_layers={cap} bsm_ratio={args.bsm_ratio} clips={len(clips)}")

    model = E.build_model(args.checkpoint, args.num_frames, args.img_size,
                          args.patch_size, "bsm_ksim_gradual_vec", args.device)
    methods = [
        ("kbsm",   True, "bsm_ksim_gradual_vec",   bsm_layers, args.bsm_ratio, "key", None),
        ("pitome", True, "bsm_pitome_gradual_vec", bsm_layers, args.bsm_ratio, "key", None),
        ("wam",    True, "bsm_taware_gradual_vec",  bsm_layers, args.bsm_ratio, "key", ("motion", 1.0)),
    ]
    for path in clips:
        cid = os.path.basename(path).split("_")[0]
        task = os.path.basename(os.path.dirname(path))
        video, _ = E.load_video(path, args.num_frames, args.img_size, args.device)
        save = dict(t=t, h=h, w=w, token_t=token_t, cid=cid, task=task, cap_layers=np.array(cap))
        # ① dense multi-layer
        feats, final = forward_capture(model, video, enabled=False, strategy="local_2x2_same_time_vec",
                                       layers=[], ratio=0.0, metric="key", capture=cap)
        for L in cap:
            save[f"dense_L{L}"] = feats[L].half().numpy()
        print(f"  [{cid}] dense final={final} layers={cap}", flush=True)
        # ② per-method gids @ deep_layer
        for label, en, strat, lays, r, metric, rel in methods:
            f2, fin = forward_capture(model, video, enabled=en, strategy=strat, layers=lays,
                                      ratio=r, metric=metric, capture=[args.deep_layer], relevance=rel)
            gids, ng = partition_from_restored(f2[args.deep_layer])
            save[f"{label}_gids"] = gids
            save[f"{label}_ng"] = ng
            print(f"  [{cid}] {label:7s} final={fin:5d} ng={ng:5d}", flush=True)
        out = os.path.join(args.out_dir, f"feats_{cid}.npz")
        np.savez_compressed(out, **save)
        print(f"  [{cid}] saved {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    print("[DONE]")


if __name__ == "__main__":
    main()
