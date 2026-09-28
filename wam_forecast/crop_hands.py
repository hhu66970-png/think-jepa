"""Tracked per-hand crops for direction A (observed past only).

For each clip and each hand (right = joints 0..25, left = 26..51):
  * per tubelet (frames 2t, 2t+1 of the observed 0..31) the box centre is the mean of the hand's
    visible joints (confidence > 0.5, inside the 1920x1080 frame); tubelets without a visible
    joint copy the nearest tubelet that has one; centres are smoothed with a 3-tubelet moving
    average; the side is fixed per clip and hand: 1.3 x median per-tubelet joint extent,
    clamped to [192, 720] px, and the box is clamped to the frame;
  * RAW crop: the two raw 1080p frames of the tubelet (npz frame_indices) cropped with that box
    and resized to S x S (bilinear, antialias) -> genuinely more pixels on the hand;
  * DENSITY control: the same box mapped onto the existing 256x256 frame (which squashes
    1920x1080 non-uniformly) and resized to S x S -> same pixels as the global view.
Outputs under <FEAT_ROOT>/crops_S<S>/: raw.npy and d256.npy [N, 2, 32, S, S, 3] uint8,
boxes.npy [N, 2, 16, 3] (x0, y0, side in 1080p px), hand_ok.npy [N, 2] bool,
and align.npy [N] (correlation between downsampled raw frame 0 and the 256 frame 0; a check
that frame_indices map the clip to the raw video correctly).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import FEAT_ROOT, T_PAST, W, all_clips, clip_path

FW, FH = 1920, 1080
RAW = W / "data" / "raw_videos"
HANDS = (slice(0, 26), slice(26, 52))


def boxes_for(z):
    xc = z["xyz_cam"][:T_PAST].astype(np.float64)
    K = z["cam_int"].astype(np.float64).reshape(3, 3)
    conf = z["confs"][:T_PAST] if "confs" in z.files else np.ones(xc.shape[:2])
    uvw = xc @ K.T
    ok = (uvw[..., 2] > 1e-6) & (conf > 0.5)
    u = uvw[..., 0] / np.where(ok, uvw[..., 2], 1)
    v = uvw[..., 1] / np.where(ok, uvw[..., 2], 1)
    ok &= (u >= 0) & (u < FW) & (v >= 0) & (v < FH)
    out, valid = np.zeros((2, 16, 3)), np.zeros(2, bool)
    for h, js in enumerate(HANDS):
        cen, ext = np.full((16, 2), np.nan), []
        for t in range(16):
            m = ok[2 * t:2 * t + 2, js]
            if m.any():
                uu, vv = u[2 * t:2 * t + 2, js][m], v[2 * t:2 * t + 2, js][m]
                cen[t] = (uu.mean(), vv.mean())
                ext.append(max(np.ptp(uu), np.ptp(vv)))
        have = np.flatnonzero(np.isfinite(cen[:, 0]))
        if len(have) == 0:
            out[h] = (FW / 2 - 200, FH / 2 - 200, 400)
            continue
        valid[h] = True
        for t in range(16):                                   # nearest tubelet with a joint
            if not np.isfinite(cen[t, 0]):
                cen[t] = cen[have[np.argmin(np.abs(have - t))]]
        sm = np.stack([cen[max(0, t - 1):t + 2].mean(0) for t in range(16)])
        side = float(np.clip(1.3 * np.median(ext), 192, 720))
        x0 = np.clip(sm[:, 0] - side / 2, 0, FW - side)
        y0 = np.clip(sm[:, 1] - side / 2, 0, FH - side)
        out[h] = np.stack([x0, y0, np.full(16, side)], 1)
    return out, valid


def crop_resize(frames, boxes, sx, sy, S):
    """frames [T,H,W,3] uint8 (T=32); boxes [16,3] in 1080p px; sx, sy map 1080p px -> frame px."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    out = []
    for f in range(x.shape[0]):
        x0, y0, s = boxes[f // 2]
        c0, r0 = int(round(x0 * sx)), int(round(y0 * sy))
        c1, r1 = max(c0 + 1, int(round((x0 + s) * sx))), max(r0 + 1, int(round((y0 + s) * sy)))
        patch = x[f:f + 1, :, r0:r1, c0:c1]
        out.append(F.interpolate(patch, size=(S, S), mode="bilinear", align_corners=False, antialias=True))
    return torch.cat(out).clamp(0, 255).round().byte().permute(0, 2, 3, 1).numpy()


def one(args):
    i, key, S = args
    import decord
    with np.load(clip_path(key)) as z:
        b, ok = boxes_for(z)
        fi = z["frame_indices"][:T_PAST].astype(np.int64)
        f256 = z["imgs"][:T_PAST]
    vr = decord.VideoReader(str(RAW / key[0] / f"{key[1].split('_', 1)[0]}.mp4"))
    raw = vr.get_batch(list(np.clip(fi, 0, len(vr) - 1))).asnumpy()      # [32,1080,1920,3]
    small = F.interpolate(torch.from_numpy(raw[:1]).permute(0, 3, 1, 2).float(), size=(256, 256),
                          mode="bilinear", antialias=True)[0].permute(1, 2, 0).numpy()
    align = float(np.corrcoef(small.ravel(), f256[0].astype(np.float32).ravel())[0, 1])
    craw = np.stack([crop_resize(raw, b[h], 1.0, 1.0, S) for h in range(2)])
    c256 = np.stack([crop_resize(f256, b[h], 256 / FW, 256 / FH, S) for h in range(2)])
    return i, craw, c256, b, ok, align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(1)
    keys = all_clips()[: a.limit or None]
    d = FEAT_ROOT / f"crops_S{a.S}"
    d.mkdir(parents=True, exist_ok=True)
    shp = (len(keys), 2, T_PAST, a.S, a.S, 3)
    raw = np.lib.format.open_memmap(d / "raw.npy", "w+", np.uint8, shp)
    d256 = np.lib.format.open_memmap(d / "d256.npy", "w+", np.uint8, shp)
    boxes, hok, align = np.zeros((len(keys), 2, 16, 3)), np.zeros((len(keys), 2), bool), np.zeros(len(keys))
    with ThreadPoolExecutor(a.workers) as ex:
        for n, (i, cr, c2, b, ok, al) in enumerate(ex.map(one, [(i, k, a.S) for i, k in enumerate(keys)])):
            raw[i], d256[i], boxes[i], hok[i], align[i] = cr, c2, b, ok, al
            if (n + 1) % 200 == 0:
                print(f"{n+1}/{len(keys)} align median {np.median(align[:n+1]):.3f} min {align[:n+1].min():.3f}", flush=True)
    raw.flush(); d256.flush()
    np.save(d / "boxes.npy", boxes); np.save(d / "hand_ok.npy", hok); np.save(d / "align.npy", align)
    print(f"CROPS DONE: align median {np.median(align):.3f}, <0.9: {(align < 0.9).sum()}, "
          f"hands ok right {hok[:,0].mean():.3f} left {hok[:,1].mean():.3f}")


if __name__ == "__main__":
    main()
