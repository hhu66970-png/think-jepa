#!/usr/bin/env python
"""Unified, clean four-method "tome -> PCA" figure from dumped features.

Reads feats_<cid>.npz produced by dump_pca_feats.py and renders, per clip, a row:

    [ input | dense | scheme-A | K-BSM | PiToMe ]

All five panels share ONE colour basis: the foreground-PCA of the *dense* shallow
layer (L5), where spatial structure is crisp -> smooth, DINO-style colour blobs.
Each method then paints its REAL merge partition: every surviving token's region
is filled with the mean colour of the patches it absorbed. So merging shows up as
redundant background collapsing into large uniform blocks while the foreground
(hands / manipulated object) stays detailed. Identical recipe across methods ->
the only thing that changes panel-to-panel is the merge. Runs on CPU.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# ---------------- PCA / rendering primitives (historical-best recipe) ----------
def normalize_l2_center(tokens):
    t = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    return t - t.mean(0, keepdim=True)


def fit_pca(tokens, k=3):
    center = tokens.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(tokens - center, full_matrices=False)
    return center, vh[:k].T


def apply_pca_signed(tokens, center, basis):
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


def postprocess(rgb_grid, smooth, saturation, gamma):
    rgb = rgb_grid.clamp(0, 1)
    if smooth > 0:
        v = rgb.permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, 1)
        rgb = ((1 - smooth) * v + smooth * pooled).permute(0, 2, 3, 1)
    g = rgb.mean(-1, keepdim=True)
    rgb = g + (rgb - g) * saturation
    return rgb.clamp(0, 1).pow(gamma)


def dense_rgb_and_mask(dense_L5, t, h, w, fg_q, bg_desat, bg_gray):
    """Foreground two-stage PCA on dense L5 -> rgb [t,h,w,3] (bg desaturated) + mask [t*h*w]."""
    nd = normalize_l2_center(dense_L5)
    c1, b1 = fit_pca(nd, 3)
    proj1 = apply_pca_signed(nd, c1, b1)
    raw_norm = (dense_L5 - dense_L5.mean(0, keepdim=True)).norm(dim=1)
    score = 0.65 * rank01(proj1[:, 0].abs()) + 0.35 * rank01(raw_norm)
    mask = foreground_mask(score.reshape(t, h, w), fg_q).reshape(-1)
    cov = float(mask.float().mean())
    if cov < 0.15 or cov > 0.85:
        mask = torch.ones_like(mask)
    c2, b2 = fit_pca(nd[mask], 3)
    proj = apply_pca_signed(nd, c2, b2)
    rgb = robust_rgb(proj, mask)
    gray = float(bg_gray) / 255.0
    rgb[~mask] = (1 - bg_desat) * rgb[~mask] + bg_desat * gray
    return rgb.reshape(t, h, w, 3), mask, cov


def median3(x_hw3):
    """3x3 spatial median filter on [h,w,3] -> kills isolated speckles, keeps blocks."""
    h, w, _ = x_hw3.shape
    v = x_hw3.permute(2, 0, 1).unsqueeze(0)
    p = F.pad(v, (1, 1, 1, 1), mode="replicate")
    patches = p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(1, 3, h, w, 9)
    return patches.median(-1).values[0].permute(1, 2, 0)


def group_sizes(gids_flat):
    """Return per-token group size [N]."""
    _, inv, counts = np.unique(gids_flat, return_inverse=True, return_counts=True)
    return counts[inv]


def group_canonical_color(rgb_flat, gids_flat):
    """Give every token its GROUP's mean colour, averaged over ALL members across
    the full t*h*w token set (not just one frame). rgb_flat [N,3], gids_flat [N]
    -> [N,3]. Merged-background groups (many, all in the smooth L5 region) get
    background-ish colours -> the background reads smooth; preserved foreground
    tokens (singleton groups) keep their own colour -> detail."""
    flat = rgb_flat.numpy().astype(np.float64)
    g = gids_flat.astype(np.int64)
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(sums, inv, flat)
    counts = np.bincount(inv).astype(np.float64)
    means = sums / counts[:, None]
    return torch.from_numpy(means[inv]).float()


# ---------------- image assembly ----------------------------------------------
def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def add_merge_boundaries(pil_img, gids_hw, sizes_hw, display, alpha, color):
    """Overlay thin outlines around MERGED groups (size>1). Dense (all size-1)
    gets no lines -> stays a pure smooth PCA; merged panels show their superpixels."""
    g = np.array(Image.fromarray(gids_hw.astype(np.int32), mode="I").resize(
        (display, display), Image.Resampling.NEAREST))
    s = np.array(Image.fromarray(sizes_hw.astype(np.int32), mode="I").resize(
        (display, display), Image.Resampling.NEAREST))
    bnd = np.zeros((display, display), bool)
    d = g[:, 1:] != g[:, :-1]; m = (s[:, 1:] > 1) | (s[:, :-1] > 1); bnd[:, 1:] |= d & m
    d = g[1:, :] != g[:-1, :]; m = (s[1:, :] > 1) | (s[:-1, :] > 1); bnd[1:, :] |= d & m
    arr = np.array(pil_img).astype(np.float32)
    col = np.array(color, dtype=np.float32)
    arr[bnd] = (1 - alpha) * arr[bnd] + alpha * col
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def up(img_hw3, size, interp):
    arr = np.clip(img_hw3.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    mode = {"nearest": Image.Resampling.NEAREST,
            "bicubic": Image.Resampling.BICUBIC}[interp]
    return Image.fromarray(arr).resize((size, size), mode)


def panel_row(title_imgs, cell, header_h=30, gutter=6, pad=6):
    n = len(title_imgs)
    W = pad * 2 + n * cell + (n - 1) * gutter
    H = pad * 2 + header_h + cell
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(17)
    for i, (title, img) in enumerate(title_imgs):
        x0 = pad + i * (cell + gutter)
        bbox = draw.textbbox((0, 0), title, font=font)
        tx = x0 + (cell - (bbox[2] - bbox[0])) // 2
        draw.text((max(x0, tx), pad + 5), title, fill=(0, 0, 0), font=font)
        sheet.paste(img, (x0, pad + header_h))
    return sheet


def stack_rows(rows, row_labels, label_w=128, gutter=8):
    w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + gutter * (len(rows) - 1)
    sheet = Image.new("RGB", (label_w + w, total_h), "white")
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


import os as _os
_WAM = _os.environ.get("INCLUDE_WAM", "0") == "1"
METHOD_ORDER = [("dense", "dense"), ("scheme_a", "scheme-A"),
                ("kbsm", "K-BSM"), ("pitome", "PiToMe")] + (
                [("wam", "WAM")] if _WAM else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clip_order", default="",
                    help="comma list of cids to fix row order (others appended)")
    ap.add_argument("--fg_quantile", type=float, default=0.72)
    ap.add_argument("--bg_desat", type=float, default=0.35)
    ap.add_argument("--bg_gray", type=int, default=150)
    ap.add_argument("--smooth", type=float, default=0.25)
    ap.add_argument("--saturation", type=float, default=1.35)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--interp", choices=("nearest", "bicubic"), default="bicubic")
    ap.add_argument("--mode", choices=("canon", "dim"), default="canon",
                    help="canon=recolour merged tokens by group-mean L5 PCA; "
                         "dim=keep clean dense PCA, fade merged-away regions by merge intensity")
    ap.add_argument("--median", type=int, default=0, help="apply 3x3 median denoise (canon mode)")
    ap.add_argument("--max_blend", type=float, default=0.82, help="dim-mode max fade-to-gray")
    ap.add_argument("--boundary", type=int, default=1, help="overlay merged-group outlines")
    ap.add_argument("--boundary_alpha", type=float, default=0.4)
    ap.add_argument("--display", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(args.feat_glob))
    order = [c.strip() for c in args.clip_order.split(",") if c.strip()]

    def keyfn(p):
        cid = os.path.basename(p)[len("feats_"):-len(".npz")]
        return (order.index(cid) if cid in order else len(order), cid)
    files = sorted(files, key=keyfn)

    rows, labels = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        t, h, w, tt = int(d["t"]), int(d["h"]), int(d["w"]), int(d["token_t"])
        cid, task = str(d["cid"]), str(d["task"])
        finals = d["finals"].item()
        dense_L5 = torch.from_numpy(d["dense_L5"].astype(np.float32))
        rgb, mask, cov = dense_rgb_and_mask(dense_L5, t, h, w, args.fg_quantile,
                                            args.bg_desat, args.bg_gray)
        frame = Image.fromarray(d["input_frame"]).resize(
            (args.display, args.display), Image.Resampling.BICUBIC)

        rgb_flat = rgb.reshape(-1, 3)
        gray = float(args.bg_gray) / 255.0
        titled = [(f"input ({cid} {task[:16]})", frame)]
        for key, disp in METHOD_ORDER:
            gids_flat = d[f"{key}_gids"].reshape(-1)
            if args.mode == "canon":
                canon = group_canonical_color(rgb_flat, gids_flat).reshape(t, h, w, 3)
                sl = canon[tt]
                if args.median:
                    sl = median3(sl)
            else:  # dim: clean dense PCA, fade merged-away regions by merge intensity
                size = torch.from_numpy(group_sizes(gids_flat).astype(np.float32))
                blend = (args.max_blend * (1.0 - 1.0 / size)).reshape(t, h, w, 1)
                faded = (1 - blend) * rgb + blend * gray
                sl = faded[tt]
            painted = postprocess(sl.unsqueeze(0), args.smooth, args.saturation, args.gamma)[0]
            panel = up(painted, args.display, args.interp)
            if args.boundary:
                gids_hw = gids_flat.reshape(t, h, w)[tt]
                sizes_hw = group_sizes(gids_flat).reshape(t, h, w)[tt]
                panel = add_merge_boundaries(panel, gids_hw, sizes_hw, args.display,
                                             args.boundary_alpha, (25, 25, 25))
            titled.append((f"{disp}  N={finals[key]}", panel))
        row = panel_row(titled, args.display)
        row.save(os.path.join(args.out_dir, f"row_{cid}.png"))
        rows.append(row)
        labels.append(f"{cid}")
        print(f"  [{cid}] cov={cov:.2f} finals={[finals[k] for k,_ in METHOD_ORDER]}", flush=True)

    sheet = stack_rows(rows, labels)
    sheet.save(os.path.join(args.out_dir, "F8b_unified_4method_PCA.png"))
    print(f"[DONE] {len(rows)} rows -> {args.out_dir}/F8b_unified_4method_PCA.png")


if __name__ == "__main__":
    main()
