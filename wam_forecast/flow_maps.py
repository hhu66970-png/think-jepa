"""Optical flow relevance (seventh review): Farneback flow between consecutive observed frames, on the frames the
encoder sees (short side 292, centre crop 256), averaged over each 16x16 cell and over the two flows of each tubelet.
  flow   raw flow magnitude
  flowc  magnitude after subtracting the median flow vector of the frame (camera compensated)
Each map is [16,16,16] per clip, divided by its clip maximum; both are added to <FEAT_ROOT>/hand_masks.npz.
  python flow_maps.py [--workers 32]"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import FEAT_ROOT, T_PAST, all_clips, clip_path, past_frames

SHORT, CROP = 292, 256


def frames(key):
    imgs = past_frames(key)
    H, W = imgs.shape[1:3]
    s = SHORT / min(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    top, left = int(round((nh - CROP) / 2.0)), int(round((nw - CROP) / 2.0))
    out = []
    for im in imgs:
        g = cv2.cvtColor(cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)
        out.append(g[top:top + CROP, left:left + CROP])
    return out


def one(key):
    shape = (T_PAST // 2, 16, 16)
    if not clip_path(key).exists():
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    fr = frames(key)
    mag, magc = [], []
    for a, b in zip(fr[:-1], fr[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        m = np.linalg.norm(f, axis=-1)
        fc = f - np.median(f.reshape(-1, 2), axis=0)
        mc = np.linalg.norm(fc, axis=-1)
        mag.append(m.reshape(16, 16, 16, 16).mean(axis=(1, 3)))
        magc.append(mc.reshape(16, 16, 16, 16).mean(axis=(1, 3)))
    mag.append(mag[-1]); magc.append(magc[-1])                 # pad: flow of the last frame = previous one
    mag, magc = np.stack(mag), np.stack(magc)                   # [32,16,16]
    t = mag.reshape(16, 2, 16, 16).mean(1); tc = magc.reshape(16, 2, 16, 16).mean(1)
    return (t / max(float(t.max()), 1e-6)).astype(np.float32), (tc / max(float(tc.max()), 1e-6)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    keys = all_clips()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, keys, chunksize=8))
    d = dict(np.load(FEAT_ROOT / "hand_masks.npz"))
    d["flow"] = np.stack([r[0] for r in res]).astype(np.float16)
    d["flowc"] = np.stack([r[1] for r in res]).astype(np.float16)
    np.savez_compressed(FEAT_ROOT / "hand_masks.npz", **d)
    print("flow maps", d["flow"].shape, "mean", float(d["flow"].mean()), "compensated mean", float(d["flowc"].mean()))


if __name__ == "__main__":
    main()
