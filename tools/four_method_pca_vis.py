#!/usr/bin/env python
"""Four-method dense-feature PCA-RGB comparison for ThinkJEPA token merging.

Produces, per tiny-cache clip:
  (A) A clean "spatial-redundancy" reference: dense features @ a pre-merge layer
      (default L5), rendered with the historical-best two-stage foreground PCA
      recipe (foreground coloured, background desaturated). Adjacent patches with
      similar features -> smooth PCA-RGB colour blobs => visual evidence of spatial
      redundancy that token-merging exploits.

  (B) A horizontal 4-panel (dense | scheme-A | K-BSM | PiToMe) at a POST-merge
      layer (default encoder final L23; also renders L12 / L20 for selection),
      rendered with ALL tokens coloured (no desaturate) under a SINGLE shared PCA
      basis fit on the dense features. Because restore_dense=True copies the merged
      representative back onto every absorbed patch, merged regions appear as
      perfectly uniform colour blocks. The shared basis means colour is directly
      comparable across panels, so the only thing that differs is the merge.

Plus a 3-clip contact sheet stacking the per-clip 4-panels.

All features come from REAL encoder forward passes. Merge plumbing
(apply_merge_config / build_model / load_video) is reused verbatim from
tools/run_token_merge_pca_experiment.py so the configs are validated exactly as
in production.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "tools")
import run_token_merge_pca_experiment as E  # noqa: E402


# --------------------------------------------------------------------------
# PCA / rendering primitives (historical-best two-stage recipe)
# --------------------------------------------------------------------------
def normalize_l2_center(tokens):
    t = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    return t - t.mean(0, keepdim=True)


def fit_pca(tokens, k=3):
    center = tokens.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(tokens - center, full_matrices=False)
    return center, vh[:k].T  # [D, k]


def apply_pca_signed(tokens, center, basis):
    """Project and stabilise sign per channel (largest-|value| token -> positive)."""
    proj = (tokens - center) @ basis
    for c in range(proj.size(1)):
        idx = torch.argmax(proj[:, c].abs())
        if proj[idx, c] < 0:
            proj[:, c] *= -1
    return proj


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
    return m.squeeze(1) >= 0.35


def robust_rgb(proj, norm_mask, pct=0.01):
    vals = proj[norm_mask] if bool(norm_mask.any()) else proj
    chans = []
    for c in range(3):
        lo, hi = torch.quantile(vals[:, c], pct), torch.quantile(vals[:, c], 1 - pct)
        chans.append(((proj[:, c] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1))
    return torch.stack(chans, dim=-1)


def postprocess(rgb_grid, smooth=0.3, saturation=1.35, gamma=0.9):
    rgb = rgb_grid.clamp(0, 1)
    if smooth > 0:
        v = rgb.permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, 1)
        rgb = ((1 - smooth) * v + smooth * pooled).permute(0, 2, 3, 1)
    g = rgb.mean(-1, keepdim=True)
    rgb = g + (rgb - g) * saturation
    return rgb.clamp(0, 1).pow(gamma)


# --------------------------------------------------------------------------
# Feature extraction: one forward per method, capturing several layers
# --------------------------------------------------------------------------
@torch.no_grad()
def extract_layers(model, video, *, enabled, strategy, layers, ratio, bsm_metric="key"):
    """Forward with restore_dense=True; return {layer_idx: [N, D] tensor}, final_tokens.

    model.out_layers must be set; outs is a list aligned to sorted(out_layers).
    With restore_dense=True each captured layer is already dense-restored, so
    merged tokens carry their representative's feature (== uniform blocks).
    """
    E.apply_merge_config(model, enabled=enabled, strategy=strategy, merge_layers=layers,
                         merge_ratio=ratio, restore_dense=True, bsm_match_metric=bsm_metric)
    if not enabled:
        model.merge_config.enabled = False  # force true dense
    outs, infos = model(video, return_merge_info=True, restore_dense=True)
    final = int(infos[-1]["num_tokens_after"]) if infos else int(outs[0].shape[1])
    # Move to CPU: rendering is cheap there and keeps all PCA tensors co-located,
    # and it releases GPU memory between methods.
    layer_feats = {int(l): outs[i][0].float().cpu()
                   for i, l in enumerate(sorted(model.out_layers))}
    return layer_feats, final


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def render_clean_reference(feat, t, h, w, token_t, fg_q, bg_desat=0.35, bg_gray=145):
    """Two-stage foreground PCA on a single layer's features; bg desaturated.

    Returns an [h, w, 3] float array for the chosen token_t.
    """
    nd = normalize_l2_center(feat)
    c1, b1 = fit_pca(nd, 3)
    proj1 = apply_pca_signed(nd, c1, b1)
    raw_norm = (feat - feat.mean(0, keepdim=True)).norm(dim=1)
    score = 0.65 * rank01(proj1[:, 0].abs()) + 0.35 * rank01(raw_norm)
    mask = foreground_mask(score.reshape(t, h, w), fg_q).reshape(-1)
    cov = float(mask.float().mean())
    if cov < 0.15 or cov > 0.85:
        mask = torch.ones_like(mask)
    c2, b2 = fit_pca(nd[mask], 3)
    proj = apply_pca_signed(nd, c2, b2)
    rgb = robust_rgb(proj, mask)
    gray = float(bg_gray) / 255.0
    mix = float(bg_desat)
    rgb[~mask] = (1 - mix) * rgb[~mask] + mix * gray
    rgb = postprocess(rgb.reshape(t, h, w, 3))
    return rgb[token_t].cpu().numpy(), cov


def render_shared_basis_panels(method_feats, t, h, w, token_t):
    """Fit ONE PCA basis on dense, apply to all methods, colour ALL tokens.

    method_feats: ordered dict {label: [N, D] tensor}; first entry must be dense.
    Returns {label: [h, w, 3] float array at token_t}.
    """
    labels = list(method_feats.keys())
    dense = method_feats[labels[0]]
    nd = normalize_l2_center(dense)
    center, basis = fit_pca(nd, 3)
    # Lock sign on the dense projection, reuse the SAME basis (no re-sign) for all.
    proj_dense = (nd - center) @ basis
    signs = torch.ones(3)
    for c in range(3):
        idx = torch.argmax(proj_dense[:, c].abs())
        if proj_dense[idx, c] < 0:
            signs[c] = -1
    allmask = torch.ones(dense.shape[0], dtype=torch.bool)
    # Robust per-channel range fit on dense, reused for every method.
    proj_dense_signed = proj_dense * signs
    vals = proj_dense_signed
    los = torch.stack([torch.quantile(vals[:, c], 0.01) for c in range(3)])
    his = torch.stack([torch.quantile(vals[:, c], 0.99) for c in range(3)])
    out = {}
    for label, feat in method_feats.items():
        proj = ((normalize_l2_center(feat) - center) @ basis) * signs
        rgb = ((proj - los) / (his - los).clamp_min(1e-6)).clamp(0, 1)
        rgb = postprocess(rgb.reshape(t, h, w, 3))
        out[label] = rgb[token_t].cpu().numpy()
    return out


# --------------------------------------------------------------------------
# Image assembly
# --------------------------------------------------------------------------
def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# Module-level render interpolation (set in main from --render_interp).
# "nearest" keeps the raw mosaic look; "bicubic" smooths the PCA-RGB token grid
# up to display resolution (recommended for high-quality figures @512).
RENDER_INTERP = "bicubic"


def _resample(name):
    return {"nearest": Image.Resampling.NEAREST,
            "bicubic": Image.Resampling.BICUBIC}[name]


def up_nearest(img_hw3, size, interp=None):
    """Upsample an [h,w,3] float array to (size,size). Despite the legacy name,
    the interpolation is controlled by `interp` (defaults to module RENDER_INTERP)."""
    arr = np.clip(img_hw3 * 255.0, 0, 255).astype(np.uint8)
    mode = _resample(interp or RENDER_INTERP)
    return Image.fromarray(arr).resize((size, size), mode)


def panel_row(title_imgs, cell, header_h=26, gutter=6, pad=6):
    """title_imgs: list of (title, PIL.Image). Lay out horizontally with titles."""
    n = len(title_imgs)
    W = pad * 2 + n * cell + (n - 1) * gutter
    H = pad * 2 + header_h + cell
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for i, (title, img) in enumerate(title_imgs):
        x0 = pad + i * (cell + gutter)
        bbox = draw.textbbox((0, 0), title, font=font)
        tx = x0 + (cell - (bbox[2] - bbox[0])) // 2
        draw.text((max(x0, tx), pad + 4), title, fill=(0, 0, 0), font=font)
        sheet.paste(img, (x0, pad + header_h))
    return sheet


def stack_rows(rows, row_labels, label_w=120, gutter=8):
    """rows: list of PIL row images (same width). Add a left vertical label."""
    w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + gutter * (len(rows) - 1)
    sheet = Image.new("RGB", (label_w + w, total_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(20)
    y = 0
    for row, lab in zip(rows, row_labels):
        label = Image.new("RGB", (row.height, label_w), "white")
        ld = ImageDraw.Draw(label)
        bbox = ld.textbbox((0, 0), lab, font=font)
        ld.text(((row.height - (bbox[2] - bbox[0])) // 2,
                 (label_w - (bbox[3] - bbox[1])) // 2), lab, fill=(0, 0, 0), font=font)
        sheet.paste(label.rotate(90, expand=True), (0, y))
        sheet.paste(row, (label_w, y))
        y += row.height + gutter
    return sheet


def background_frame_index(token_t, tubelet, nframes):
    pos = token_t * max(1, tubelet) + max(0, tubelet // 2)
    return int(np.clip(pos, 0, nframes - 1))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--npz_glob",
                    default="/root/autodl-tmp/thinkjepa-work/tiny-cache/part2/*/*.npz")
    ap.add_argument("--clip_ids", default="542,1873,5522",
                    help="comma list of clip-id prefixes to keep from the glob")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--token_t", type=int, default=-1, help="time-token index; <0 -> middle")
    # method configs
    ap.add_argument("--scheme_a_layer", type=int, default=8)
    ap.add_argument("--scheme_a_ratio", type=float, default=0.25)
    ap.add_argument("--bsm_layers", default="12,13,14,15,16,17,18,19,20")
    ap.add_argument("--bsm_ratio", type=float, default=0.15)
    # which layers to capture / render
    ap.add_argument("--ref_layer", type=int, default=5,
                    help="pre-merge layer for the clean spatial-redundancy reference")
    ap.add_argument("--panel_layers", default="12,20,23",
                    help="post-merge layers to render the 4-panel for (pick best visually)")
    ap.add_argument("--fg_quantile", type=float, default=0.75)
    ap.add_argument("--display_size", type=int, default=256)
    ap.add_argument("--render_interp", choices=("nearest", "bicubic"),
                    default="bicubic",
                    help="how to upsample the PCA-RGB token grid to display_size.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    global RENDER_INTERP
    RENDER_INTERP = args.render_interp
    os.makedirs(args.out_dir, exist_ok=True)
    t = args.num_frames // 2  # tubelet 2
    h = w = args.img_size // args.patch_size
    token_t = t // 2 if args.token_t < 0 else int(np.clip(args.token_t, 0, t - 1))
    bsm_layers = [int(v) for v in args.bsm_layers.split(",")]
    panel_layers = [int(v) for v in args.panel_layers.split(",")]
    keep_ids = [s.strip() for s in args.clip_ids.split(",") if s.strip()]
    capture_layers = sorted(set([args.ref_layer] + panel_layers))

    # select clips
    all_npz = sorted(glob.glob(args.npz_glob))
    npz_files = []
    for cid in keep_ids:
        hit = [p for p in all_npz if os.path.basename(p).split("_")[0] == cid]
        if not hit:
            raise FileNotFoundError(f"no npz with clip-id {cid} under {args.npz_glob}")
        npz_files.append(hit[0])
    print(f"[INFO] clips={[os.path.basename(p) for p in npz_files]}")
    print(f"[INFO] grid={h}x{w} t_tokens={t} token_t={token_t} "
          f"capture_layers={capture_layers} panel_layers={panel_layers}")

    model = E.build_model(args.checkpoint, args.num_frames, args.img_size,
                          args.patch_size, "bsm_ksim_gradual_vec", args.device)
    model.out_layers = capture_layers  # capture all layers in one forward

    # method spec: (label, enabled, strategy, layers, ratio, metric)
    method_specs = [
        ("dense", False, "local_2x2_same_time_vec", [], 0.0, "key"),
        ("scheme-A", True, "local_2x2_same_time_vec", [args.scheme_a_layer],
         args.scheme_a_ratio, "key"),
        ("K-BSM", True, "bsm_ksim_gradual_vec", bsm_layers, args.bsm_ratio, "key"),
        ("PiToMe", True, "bsm_pitome_gradual_vec", bsm_layers, args.bsm_ratio, "key"),
    ]

    # accumulate per-clip per-layer panel rows for the contact sheet
    contact_rows = {pl: [] for pl in panel_layers}
    contact_labels = {pl: [] for pl in panel_layers}
    ref_rows, ref_labels = [], []

    for npz in npz_files:
        cid = os.path.basename(npz).split("_")[0]
        task = os.path.basename(os.path.dirname(npz))
        video, _ = E.load_video(npz, args.num_frames, args.img_size, args.device)
        imgs = np.load(npz, allow_pickle=True)["imgs"]
        bg_idx = background_frame_index(token_t, 2, imgs.shape[0])
        input_img = Image.fromarray(imgs[bg_idx]).resize(
            (args.display_size, args.display_size), Image.Resampling.BICUBIC)

        # run all 4 methods, capturing all layers each
        per_method = {}   # label -> {layer: [N,D]}
        finals = {}
        for label, en, strat, lays, r, metric in method_specs:
            feats, final = extract_layers(model, video, enabled=en, strategy=strat,
                                          layers=lays, ratio=r, bsm_metric=metric)
            per_method[label] = feats
            finals[label] = final
            print(f"  [{cid}] {label:9s} final_tokens={final} "
                  f"(layers captured: {sorted(feats)})")

        # ---- (A) clean spatial-redundancy reference: dense @ ref_layer ----
        ref_rgb, cov = render_clean_reference(
            per_method["dense"][args.ref_layer], t, h, w, token_t, args.fg_quantile)
        ref_img = up_nearest(ref_rgb, args.display_size)
        ref_row = panel_row([(f"input  t={token_t}  ({cid} {task[:18]})", input_img),
                             (f"dense L{args.ref_layer}  fg-PCA  cov={cov:.2f}", ref_img)],
                            args.display_size)
        ref_row.save(os.path.join(args.out_dir, f"ref_L{args.ref_layer}_{cid}.png"))
        ref_rows.append(ref_row)
        ref_labels.append(cid)

        # ---- (B) 4-panel comparison at each panel layer (shared basis) ----
        for pl in panel_layers:
            method_feats = {lab: per_method[lab][pl] for lab, *_ in method_specs}
            rgb_maps = render_shared_basis_panels(method_feats, t, h, w, token_t)
            titled = [("input", input_img)]
            for lab in rgb_maps:
                ntok = finals[lab]
                titled.append((f"{lab}  N={ntok}",
                               up_nearest(rgb_maps[lab], args.display_size)))
            row = panel_row(titled, args.display_size)
            row.save(os.path.join(args.out_dir, f"panel_L{pl}_{cid}.png"))
            contact_rows[pl].append(row)
            contact_labels[pl].append(cid)
            print(f"  [{cid}] panel L{pl} saved "
                  f"(N: {[finals[l] for l,*_ in method_specs]})")

    # ---- contact sheets (3 clips stacked) ----
    ref_sheet = stack_rows(ref_rows, ref_labels)
    ref_sheet.save(os.path.join(args.out_dir,
                                f"contact_sheet_ref_L{args.ref_layer}.png"))
    for pl in panel_layers:
        sheet = stack_rows(contact_rows[pl], contact_labels[pl])
        sheet.save(os.path.join(args.out_dir, f"contact_sheet_panel_L{pl}.png"))
    print(f"[DONE] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
