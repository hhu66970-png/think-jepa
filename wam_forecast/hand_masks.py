"""Project the observed (past) hand joints onto the encoder's 16x16x16 token grid.

Output <FEAT_ROOT>/hand_masks.npz : mask [N,16,16,16] bool (t = tubelet of frames 2t,2t+1),
in all_clips() order. A token is "hand" if any of the 52 joints of either frame of its
tubelet falls in its 16x16 patch after the encoder's resize/crop, dilated by one patch.
Only past frames are used, so the mask is also a legal prior for forecasting (E2).

--check <n> writes an overlay PNG for the first n clips to verify the camera convention.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import FEAT_ROOT, T_PAST, all_clips, clip_path

SHORT, CROP = 292, 256


def pixels(z):
    """[T,52,2] pixel coords in the 256x256 encoder crop, plus visibility [T,52]."""
    xc = z["xyz_cam"][:T_PAST].astype(np.float64)
    K = z["cam_int"].astype(np.float64).reshape(3, 3)
    H, W = z["imgs"].shape[1:3]
    uvw = xc @ K.T
    z_ok = uvw[..., 2] > 1e-6
    u = uvw[..., 0] / np.where(z_ok, uvw[..., 2], 1)
    v = uvw[..., 1] / np.where(z_ok, uvw[..., 2], 1)
    # intrinsics are for the native video; frames were resized to HxW without cropping
    sx, sy = W / (2 * K[0, 2]), H / (2 * K[1, 2])
    u, v = u * sx, v * sy
    s = SHORT / min(H, W)
    u = u * s - round((W * s - CROP) / 2.0)
    v = v * s - round((H * s - CROP) / 2.0)
    vis = z_ok & (u >= 0) & (u < CROP) & (v >= 0) & (v < CROP)
    return np.stack([u, v], -1), vis


def mask_one(key):
    """(binary mask dilated by one patch, graded density) on the 16x16x16 grid."""
    shape = (T_PAST // 2, 16, 16)
    if not clip_path(key).exists():                   # partial download: empty mask
        return np.zeros(shape, bool), np.zeros(shape, np.float32)
    with np.load(clip_path(key)) as z:
        uv, vis = pixels(z)
    cnt = np.zeros(shape, np.float32)
    for f in range(T_PAST):
        p = (uv[f][vis[f]] // 16).astype(int)
        np.add.at(cnt, (f // 2, p[:, 1], p[:, 0]), 1.0)
    m = cnt > 0
    d = m.copy()                                        # 1-patch dilation in space
    d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    d2 = d.copy()
    d2[:, :, 1:] |= d[:, :, :-1]; d2[:, :, :-1] |= d[:, :, 1:]
    # graded relevance: joint hits in the cell (log-scaled) for core cells, a small
    # constant for the dilation ring, so a quota picks the densest hand cells first.
    dens = np.where(m, 1.0 + np.log1p(cnt), np.where(d2, 0.5, 0.0)).astype(np.float32)
    return d2, dens / max(float(dens.max()), 1e-6)


def check(keys, n):
    from PIL import Image, ImageDraw
    tiles = []
    for key in keys[:n]:
        with np.load(clip_path(key)) as z:
            uv, vis = pixels(z)
            img = z["imgs"][T_PAST - 1]
        H, W = img.shape[:2]
        s = SHORT / min(H, W)
        im = Image.fromarray(img).resize((round(W * s), round(H * s)), Image.BILINEAR)
        l, t = round((im.width - CROP) / 2.0), round((im.height - CROP) / 2.0)
        im = im.crop((l, t, l + CROP, t + CROP))
        dr = ImageDraw.Draw(im)
        for (u, v), ok in zip(uv[T_PAST - 1], vis[T_PAST - 1]):
            if ok:
                dr.ellipse([u - 2, v - 2, u + 2, v + 2], outline=(255, 0, 0))
        tiles.append(im)
    canvas = Image.new("RGB", (CROP * len(tiles), CROP))
    for i, im in enumerate(tiles):
        canvas.paste(im, (i * CROP, 0))
    canvas.save(FEAT_ROOT / "hand_projection_check.png")
    print("wrote", FEAT_ROOT / "hand_projection_check.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    keys = all_clips()
    if a.check:
        return check([k for k in keys if clip_path(k).exists()][::7], a.check)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(mask_one, keys))
    masks = np.stack([r[0] for r in res])
    dens = np.stack([r[1] for r in res]).astype(np.float16)
    np.savez_compressed(FEAT_ROOT / "hand_masks.npz", mask=masks, density=dens)
    print(f"hand token fraction: mean {masks.mean():.3f}, clips with no hand token: "
          f"{int((masks.reshape(len(keys), -1).sum(1) == 0).sum())}")


if __name__ == "__main__":
    main()
