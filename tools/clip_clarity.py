#!/usr/bin/env python
"""Score EgoDex clips for PCA "cleanliness / easy-redundancy".

For each clip (npz with imgs [T,H,W,3]) we sample a few frames and compute
metrics that predict a clean V-JEPA dense-PCA (strong, simple foreground vs.
uniform background):

  - bg_uniformity : mean cosine similarity between spatially-adjacent 16x16
                    patches (RGB).  High => large smooth background regions =>
                    lots of redundant tokens => clean PCA.
  - bg_fraction   : fraction of patches whose local gradient energy is below a
                    low threshold (i.e. flat background area).
  - fg_bg_contrast: ratio of the (high-gradient foreground) patch-energy to the
                    (low-gradient background) patch-energy.  High => crisp
                    subject popping out of a calm background.
  - edge_density  : mean Sobel gradient magnitude over the frame (LOWER is
                    cleaner -> fewer competing textures).  Reported but the
                    composite penalizes it.

Composite score (higher = cleaner / better for a pretty PCA):
    score = z(bg_uniformity) + z(bg_fraction) + 0.5*z(fg_bg_contrast) - z(edge_density)
where z() is a robust z-score across all clips.
"""
import glob, os, sys
import numpy as np

ROOT = "/root/autodl-tmp/thinkjepa-work/cache_ext30/part2"
PATCH = 16
N_FRAMES = 8  # uniformly sampled frames per clip


def sobel_mag(gray):
    # gray: [H,W] float
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(gray, (3, 3))  # [H-2,W-2,3,3]
    gx = (win * kx).sum(axis=(-1, -2))
    gy = (win * ky).sum(axis=(-1, -2))
    mag = np.sqrt(gx * gx + gy * gy)
    out = np.zeros_like(gray)
    out[1:-1, 1:-1] = mag
    return out


def patchify(img, p=PATCH):
    H, W, C = img.shape
    Hc, Wc = (H // p) * p, (W // p) * p
    img = img[:Hc, :Wc]
    gh, gw = Hc // p, Wc // p
    patches = img.reshape(gh, p, gw, p, C).transpose(0, 2, 1, 3, 4).reshape(gh, gw, p * p * C)
    return patches, gh, gw


def adjacent_cosine(patches):
    # patches: [gh,gw,D]; mean cosine of horizontal + vertical neighbors
    v = patches.astype(np.float32)
    nrm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6
    u = v / nrm
    ch = (u[:, :-1] * u[:, 1:]).sum(-1)
    cv = (u[:-1, :] * u[1:, :]).sum(-1)
    return float(np.concatenate([ch.ravel(), cv.ravel()]).mean())


def score_frame(img):
    img = img.astype(np.float32)
    gray = img.mean(-1)
    mag = sobel_mag(gray)
    # patch-level gradient energy
    pe, gh, gw = patchify(mag[..., None], p=PATCH)
    pe = pe.mean(-1).reshape(gh, gw)  # mean grad per patch
    patches, _, _ = patchify(img, p=PATCH)
    bg_unif = adjacent_cosine(patches)
    thr = np.quantile(pe, 0.45)  # low-energy => background
    bg_mask = pe <= thr
    bg_frac = float(bg_mask.mean())
    fg_e = float(pe[~bg_mask].mean()) if (~bg_mask).any() else 0.0
    bg_e = float(pe[bg_mask].mean()) + 1e-6
    contrast = fg_e / bg_e
    edge = float(mag.mean())
    return bg_unif, bg_frac, contrast, edge


def main():
    clips = sorted(glob.glob(os.path.join(ROOT, "*", "*.npz")))
    rows = []
    for path in clips:
        task = os.path.basename(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            data = np.load(path, allow_pickle=False)
            imgs = data["imgs"]  # [T,H,W,3]
        except Exception as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        T = imgs.shape[0]
        idx = np.linspace(0, T - 1, N_FRAMES).round().astype(int)
        bu, bf, ct, ed = [], [], [], []
        for i in idx:
            a, b, c, d = score_frame(imgs[i])
            bu.append(a); bf.append(b); ct.append(c); ed.append(d)
        rows.append(dict(task=task, name=name, path=path,
                         bg_unif=np.mean(bu), bg_frac=np.mean(bf),
                         contrast=np.mean(ct), edge=np.mean(ed)))
        print(f"  scored {task}/{name}", flush=True)

    def z(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        med = np.median(vals); mad = np.median(np.abs(vals - med)) + 1e-6
        return {id(r): (r[key] - med) / (1.4826 * mad) for r in rows}

    zbu, zbf, zct, zed = z("bg_unif"), z("bg_frac"), z("contrast"), z("edge")
    for r in rows:
        r["score"] = (zbu[id(r)] + zbf[id(r)] + 0.5 * zct[id(r)] - zed[id(r)])

    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== CLIP CLARITY RANKING (higher = cleaner) ===")
    print(f"{'rank':>4} {'score':>7} {'bg_unif':>8} {'bg_frac':>8} {'contrast':>9} {'edge':>7}  task/clip")
    for i, r in enumerate(rows):
        print(f"{i+1:>4} {r['score']:>7.3f} {r['bg_unif']:>8.3f} {r['bg_frac']:>8.3f} "
              f"{r['contrast']:>9.2f} {r['edge']:>7.2f}  {r['task']}/{r['name']}")

    # Pick top clips that are also diverse across tasks (one per task, greedy by score)
    picked, seen_tasks = [], set()
    for r in rows:
        if r["task"] in seen_tasks:
            continue
        picked.append(r); seen_tasks.add(r["task"])
        if len(picked) >= 4:
            break
    print("\n=== PICKED (top, diverse across tasks) ===")
    for r in picked:
        print(f"  {r['task']}/{r['name']}  score={r['score']:.3f}  -> {r['path']}")


if __name__ == "__main__":
    main()
