"""Hand crop boxes in native 1920x1080 pixels from the OBSERVED past joints only.

Per clip: project the 52 joints of frames 0..31 with cam_int (native resolution), take the
union of visible joints, pad 15 % per side, make square, minimum side 256 px, clamp to the
frame. Also per-hand boxes (right = joints 0..25, left = 26..51) for the two-crop variant.
Writes <FEAT_ROOT>/hand_boxes.npz: box [N,3] (x0, y0, side), rbox/lbox [N,3], valid flags.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import FEAT_ROOT, T_PAST, all_clips, clip_path

FW, FH, PAD, MIN_SIDE = 1920, 1080, 0.15, 256


def square_box(u, v):
    if len(u) == 0:
        return np.array([FW / 2 - 270, FH / 2 - 270, 540.0]), False
    x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max()
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    side = min(max(side, MIN_SIDE), FH)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = float(np.clip(cx - side / 2, 0, FW - side))
    by = float(np.clip(cy - side / 2, 0, FH - side))
    return np.array([bx, by, side]), True


def one(key):
    with np.load(clip_path(key)) as z:
        xc = z["xyz_cam"][:T_PAST].astype(np.float64)
        K = z["cam_int"].astype(np.float64).reshape(3, 3)
        conf = z["confs"][:T_PAST] if "confs" in z.files else np.ones(xc.shape[:2])
    uvw = xc @ K.T
    ok = (uvw[..., 2] > 1e-6) & (conf > 0.5)
    u = uvw[..., 0] / np.where(ok, uvw[..., 2], 1)
    v = uvw[..., 1] / np.where(ok, uvw[..., 2], 1)
    ok &= (u >= 0) & (u < FW) & (v >= 0) & (v < FH)
    out = [square_box(u[ok], v[ok])]
    for sl in (slice(0, 26), slice(26, 52)):
        m = ok[:, sl]
        out.append(square_box(u[:, sl][m], v[:, sl][m]))
    return out


def main():
    keys = all_clips()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, keys))
    box = np.stack([r[0][0] for r in res]); valid = np.array([r[0][1] for r in res])
    rbox = np.stack([r[1][0] for r in res]); rv = np.array([r[1][1] for r in res])
    lbox = np.stack([r[2][0] for r in res]); lv = np.array([r[2][1] for r in res])
    np.savez(FEAT_ROOT / "hand_boxes.npz", box=box, valid=valid, rbox=rbox, rvalid=rv, lbox=lbox, lvalid=lv)
    s = box[:, 2]
    print(f"union box side px: p10 {np.percentile(s,10):.0f} median {np.median(s):.0f} p90 {np.percentile(s,90):.0f} "
          f"max {s.max():.0f}; valid {valid.mean():.3f}")
    print(f"share with side > 720 px: {(s > 720).mean():.3f}; side == 1080 (whole height): {(s >= 1079).mean():.3f}")
    print(f"per-hand valid: right {rv.mean():.3f} left {lv.mean():.3f}; per-hand median side "
          f"{np.median(rbox[rv,2]):.0f} / {np.median(lbox[lv,2]):.0f}")
    # linear magnification of a 128-px crop relative to the 256x256 frame (which squashes 1920 -> 256)
    mag = (128 / s) / (256 / 1920)
    print(f"horizontal pixel density of a 128 crop vs the 256 frame: median {np.median(mag):.2f}x "
          f"(>1 means genuinely more pixels per hand than the global view)")


if __name__ == "__main__":
    main()
