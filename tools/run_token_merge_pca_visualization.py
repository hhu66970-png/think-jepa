"""3-group PCA RGB visualization: dense vs scheme-A vs Gradual K-BSM.

Supports several rendering "paths" so you can compare what looks best:
  - default        : two-stage PCA + foreground mask on FINAL-layer features
  - --feature_layer L : extract a MID encoder layer L (via out_layers) instead of the
                        final layer (mid-layer ViT features are often spatially cleaner)
  - --per_frame_pca  : fit PCA INDEPENDENTLY per frame (sharper per-frame contrast)
                       instead of one shared two-stage basis for the whole clip

All modes share the same basis/mask within a given path so colors stay comparable;
merged tokens show as same-color blocks. Reuses model/merge plumbing from
run_token_merge_pca_experiment.py.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E


def normalize_l2_center(tokens):
    tokens = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    return tokens - tokens.mean(0, keepdim=True)


def fit_pca(tokens, k=3, drop_pc1=False):
    center = tokens.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(tokens - center, full_matrices=False)
    comp = vh[1:1 + k] if drop_pc1 else vh[:k]
    return center, comp.T  # [D, k]


def rank01(v):
    v = v.float().reshape(-1)
    lo, hi = torch.quantile(v, 0.01), torch.quantile(v, 0.99)
    return ((v - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def foreground_mask(score_grid, q):
    work = F.avg_pool2d(score_grid.unsqueeze(1), 3, 1, 1).squeeze(1)
    thr = torch.quantile(work.reshape(-1), float(q))
    m = (work >= thr).float().unsqueeze(1)
    m = F.max_pool2d(m, 3, 1, 1)
    m = F.avg_pool2d(m, 3, 1, 1)
    return (m.squeeze(1) >= 0.35)


def robust_rgb(proj, norm_mask, pct=0.01):
    vals = proj[norm_mask] if bool(norm_mask.any()) else proj
    chans = []
    for c in range(3):
        lo, hi = torch.quantile(vals[:, c], pct), torch.quantile(vals[:, c], 1 - pct)
        chans.append(((proj[:, c] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1))
    return torch.stack(chans, dim=-1)


def postprocess(rgb, smooth=0.3, saturation=1.3, gamma=0.9):
    rgb = rgb.clamp(0, 1)
    if smooth > 0:
        v = rgb.permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, 1)
        rgb = ((1 - smooth) * v + smooth * pooled).permute(0, 2, 3, 1)
    g = rgb.mean(-1, keepdim=True)
    rgb = g + (rgb - g) * saturation
    return rgb.clamp(0, 1).pow(gamma)


def upsample_smooth(img_hw3, size):
    t = torch.from_numpy(np.ascontiguousarray(img_hw3)).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def build_model(checkpoint, num_frames, img_size, patch_size, device, out_layers=None):
    from src.models import vision_transformer
    mc = {"enabled": True, "merge_layers": (8,), "merge_ratio": 0.0,
          "strategy": "local_2x2_same_time_vec", "receiver": "max_norm",
          "restore_dense": False, "importance_source": "none"}
    model = vision_transformer.vit_large_rope(
        img_size=(img_size, img_size), num_frames=num_frames, patch_size=patch_size,
        tubelet_size=2, out_layers=out_layers, use_sdpa=True, use_silu=False,
        wide_silu=True, uniform_power=False, merge_config=mc)
    try:
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(str(checkpoint), map_location="cpu")
    state = blob.get("encoder", blob.get("model", blob)) if isinstance(blob, dict) else blob
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(clean, strict=False)
    print(f"[INFO] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} out_layers={out_layers}")
    return model.to(device).eval()


@torch.no_grad()
def run_mode(model, video, multi_out, *, enabled, strategy, layers, ratio, bsm_metric="key"):
    E.apply_merge_config(model, enabled=enabled, strategy=strategy, merge_layers=layers,
                         merge_ratio=ratio, restore_dense=True, bsm_match_metric=bsm_metric)
    if not enabled:
        model.merge_config.enabled = False
    res, infos = model(video, return_merge_info=True, restore_dense=True)
    out = res[0] if multi_out else res            # out_layers -> list; else tensor
    final = int(infos[-1]["num_tokens_after"]) if infos else int(out.shape[1])
    return out[0].float(), final


def render_two_stage(modes, t, h, w, fg_q, drop_pc1):
    dense = list(modes.values())[0]
    nd = normalize_l2_center(dense)
    c1, b1 = fit_pca(nd, 3)
    proj1 = (nd - c1) @ b1
    score = 0.65 * rank01(proj1[:, 0].abs()) + 0.35 * rank01((dense - dense.mean(0, keepdim=True)).norm(dim=1))
    mask = foreground_mask(score.reshape(t, h, w), fg_q).reshape(-1)
    cov = float(mask.float().mean())
    if cov < 0.15 or cov > 0.85:
        mask = torch.ones_like(mask)
    c2, b2 = fit_pca(nd[mask], 3, drop_pc1=drop_pc1)
    gray, mix = 60 / 255.0, 0.8
    out = {}
    for name, feat in modes.items():
        proj = (normalize_l2_center(feat) - c2) @ b2
        rgb = robust_rgb(proj, mask)
        rgb[~mask] = (1 - mix) * rgb[~mask] + mix * gray
        out[name] = postprocess(rgb.reshape(t, h, w, 3)).cpu().numpy()
    return out, cov


def render_per_frame(modes, t, h, w, frames, drop_pc1):
    """Independent PCA basis per displayed frame (fit on dense frame, applied to all modes)."""
    out = {name: np.zeros((t, h, w, 3), dtype=np.float32) for name in modes}
    for ft in frames:
        dframe = normalize_l2_center(list(modes.values())[0].reshape(t, h * w, -1)[ft])
        c, b = fit_pca(dframe, 3, drop_pc1=drop_pc1)
        allmask = torch.ones(h * w, dtype=torch.bool)
        for name, feat in modes.items():
            f = normalize_l2_center(feat.reshape(t, h * w, -1)[ft])
            rgb = robust_rgb((f - c) @ b, allmask)
            out[name][ft] = postprocess(rgb.reshape(1, h, w, 3)).cpu().numpy()[0]
    return out, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--npz_glob", default="/root/autodl-tmp/thinkjepa-work/tiny-cache/part2/*/*.npz")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--frames", default="4,12,20,28")
    ap.add_argument("--scheme_a_layer", type=int, default=8)
    ap.add_argument("--scheme_a_ratio", type=float, default=0.25)
    ap.add_argument("--bsm_layers", default="4,6,8,10,12")
    ap.add_argument("--bsm_ratio", type=float, default=0.056)
    ap.add_argument("--fg_quantile", type=float, default=0.7)
    ap.add_argument("--feature_layer", type=int, default=-1, help=">=0 to read a mid encoder layer via out_layers")
    ap.add_argument("--per_frame_pca", action="store_true", help="fit PCA independently per frame")
    ap.add_argument("--drop_pc1", action="store_true", help="use PC2..4 (skip dominant axis)")
    ap.add_argument("--display_size", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t = args.num_frames // 2
    h = w = args.img_size // args.patch_size
    frames = [int(v) for v in args.frames.split(",")]
    bsm_layers = [int(v) for v in args.bsm_layers.split(",")]
    multi_out = args.feature_layer >= 0
    out_layers = [args.feature_layer] if multi_out else None
    tag = ("layer%d" % args.feature_layer if multi_out else "final") + ("_perframe" if args.per_frame_pca else "") + ("_droppc1" if args.drop_pc1 else "")

    model = build_model(args.checkpoint, args.num_frames, args.img_size, args.patch_size, args.device, out_layers)

    for npz in sorted(glob.glob(args.npz_glob)):
        vid = os.path.basename(npz).split("_")[0]
        task = os.path.basename(os.path.dirname(npz))
        video, _ = E.load_video(npz, args.num_frames, args.img_size, args.device)
        imgs = np.load(npz, allow_pickle=True)["imgs"]

        modes = {}
        f, n = run_mode(model, video, multi_out, enabled=False, strategy="local_2x2_same_time_vec", layers=[], ratio=0.0)
        modes[f"(1) dense N={n}"] = f
        f, n = run_mode(model, video, multi_out, enabled=True, strategy="local_2x2_same_time_vec",
                        layers=[args.scheme_a_layer], ratio=args.scheme_a_ratio)
        modes[f"(2) scheme-A N={n}"] = f
        f, n = run_mode(model, video, multi_out, enabled=True, strategy="bsm_ksim_gradual_vec",
                        layers=bsm_layers, ratio=args.bsm_ratio, bsm_metric="key")
        modes[f"(3) Gradual K-BSM N={n}"] = f

        if args.per_frame_pca:
            rgb_maps, cov = render_per_frame(modes, t, h, w, frames, args.drop_pc1)
        else:
            rgb_maps, cov = render_two_stage(modes, t, h, w, args.fg_quantile, args.drop_pc1)

        row_names = ["input frame"] + list(rgb_maps.keys())
        nrows, ncols = len(row_names), len(frames)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows))
        if ncols == 1:
            axes = axes.reshape(-1, 1)
        for c, ft in enumerate(frames):
            axes[0, c].imshow(imgs[min(ft * 2, imgs.shape[0] - 1)])
            axes[0, c].set_title(f"frame t={ft}", fontsize=10)
            for r, name in enumerate(rgb_maps.keys(), start=1):
                axes[r, c].imshow(upsample_smooth(rgb_maps[name][ft], args.display_size))
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, 0].set_ylabel(row_names[r], fontsize=11)
        feat_desc = (f"mid-layer {args.feature_layer}" if multi_out else "final layer")
        pca_desc = ("per-frame PCA" if args.per_frame_pca else "two-stage PCA + fg-mask")
        if args.drop_pc1:
            pca_desc += " (drop PC1)"
        fig.suptitle(f"3-group PCA RGB | {feat_desc} | {pca_desc} | {h}x{w} grid bicubic | video {vid} {task}",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(args.out_dir, f"pca_compare_{vid}_{tag}.png")
        fig.savefig(out, dpi=95)
        plt.close(fig)
        print(f"saved {out} | path={tag} coverage={cov:.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
