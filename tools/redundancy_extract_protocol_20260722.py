# -*- coding: utf-8 -*-
"""Protocol-locked feature and allocation dump for the AAAI figure audit.

The original dump used 16 frames at 512 px.  That geometry also produces 8192
tokens, but it does not match the paper's 64-frame, 256-px setup.  This script
creates a new, non-overwriting dump with explicit geometry metadata.

For each clip it stores:
  * dense features at the eight paper layers;
  * exact final merge partitions for K-BSM and PiToMe;
  * exact WAM partitions for a configurable lambda sweep;
  * a representative input frame and provenance fields.

This is a forward-only diagnostic.  It does not train or select checkpoints.
"""
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


def same_partition(left, right):
    """Compare partitions up to arbitrary integer group labels."""
    if left.shape != right.shape:
        return False
    n_left = len(np.unique(left))
    n_right = len(np.unique(right))
    n_pairs = len(np.unique(np.stack([left, right], axis=1), axis=0))
    return n_pairs == n_left == n_right


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
    ap.add_argument("--wam_lambdas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t = args.num_frames // 2
    h = w = args.img_size // args.patch_size
    token_t = t // 2
    cap = [int(v) for v in args.cap_layers.split(",")]
    bsm_layers = [int(v) for v in args.bsm_layers.split(",")]
    wam_lambdas = [float(v) for v in args.wam_lambdas.split(",")]
    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    if len(set(wam_lambdas)) != len(wam_lambdas):
        raise ValueError(f"duplicate WAM lambdas: {wam_lambdas}")
    if any(v < 0.0 or v > 1.0 for v in wam_lambdas):
        raise ValueError(f"WAM lambdas must lie in [0,1]: {wam_lambdas}")
    if t * h * w != 8192:
        raise ValueError(f"paper protocol requires 8192 tokens, got {t}x{h}x{w}")
    print(
        f"[INFO] grid={t}x{h}x{w} cap_layers={cap} bsm_ratio={args.bsm_ratio} "
        f"wam_lambdas={wam_lambdas} clips={len(clips)}"
    )

    model = E.build_model(args.checkpoint, args.num_frames, args.img_size,
                          args.patch_size, "bsm_ksim_gradual_vec", args.device)
    methods = [
        ("kbsm",   True, "bsm_ksim_gradual_vec",   bsm_layers, args.bsm_ratio, "key", None),
        ("pitome", True, "bsm_pitome_gradual_vec", bsm_layers, args.bsm_ratio, "key", None),
    ]
    for lam in wam_lambdas:
        label = f"wam_lam{int(round(lam * 100)):03d}"
        methods.append(
            (label, True, "bsm_taware_gradual_vec", bsm_layers,
             args.bsm_ratio, "key", ("motion", lam))
        )
    for path in clips:
        cid = os.path.basename(path).split("_")[0]
        task = os.path.basename(os.path.dirname(path))
        video, _ = E.load_video(path, args.num_frames, args.img_size, args.device)
        imgs = np.load(path, allow_pickle=True)["imgs"]
        from PIL import Image
        input_frame = np.asarray(
            Image.fromarray(imgs[len(imgs) // 2]).resize(
                (args.img_size, args.img_size), Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
        save = dict(
            t=t,
            h=h,
            w=w,
            token_t=token_t,
            num_frames=args.num_frames,
            img_size=args.img_size,
            patch_size=args.patch_size,
            token_count=t * h * w,
            cid=cid,
            task=task,
            source_path=path,
            cap_layers=np.array(cap),
            bsm_layers=np.array(bsm_layers),
            relevance_layer=min(bsm_layers),
            bsm_ratio=args.bsm_ratio,
            wam_lambdas=np.array(wam_lambdas, dtype=np.float32),
            input_frame=input_frame,
        )
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
            if ng != fin:
                raise RuntimeError(f"{cid} {label}: groups={ng}, tokens_after={fin}")
            save[f"{label}_gids"] = gids
            save[f"{label}_ng"] = ng
            print(f"  [{cid}] {label:7s} final={fin:5d} ng={ng:5d}", flush=True)
        if "wam_lam100_gids" in save:
            save["wam_gids"] = save["wam_lam100_gids"]
            save["wam_ng"] = save["wam_lam100_ng"]
        if "wam_lam000_gids" in save:
            save["lambda0_same_partition_kbsm"] = bool(
                same_partition(save["wam_lam000_gids"], save["kbsm_gids"])
            )
            print(
                f"  [{cid}] lambda0_same_partition_kbsm="
                f"{bool(save['lambda0_same_partition_kbsm'])}",
                flush=True,
            )
        out = os.path.join(args.out_dir, f"feats_{cid}.npz")
        np.savez_compressed(out, **save)
        print(f"  [{cid}] saved {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    print("[DONE]")


if __name__ == "__main__":
    main()
