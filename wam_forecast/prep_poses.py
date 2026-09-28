"""Build the pose tensors of the forecasting protocol for all 2000 official clips.

Output  <FEAT_ROOT>/poses.npz
  keys          [N]            "task/file" in all_clips() order (train first, then test)
  is_test       [N] bool
  past          [N,32,52,3]    joints of frames 0..31 in the camera frame of frame 31 (m)
  future        [N,32,52,3]    joints of frames 32..63 in the same frame (m)
  conf_future   [N,32,52]      tracking confidence of the targets
  task          [N]            task name (for per-task breakdowns)
  avail         [N] bool       clip file present (pilot runs use a partial download)
Also checks xyz_world == cam_ext @ xyz_cam on every clip, so the frame convention is verified.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import FEAT_ROOT, T_PAST, all_clips, clip_path, key_str, read_split


def one(key):
    if not clip_path(key).exists():
        z0 = np.zeros((T_PAST, 52, 3), np.float32)
        return z0, z0, np.zeros((T_PAST, 52), np.float32), 0.0, False
    with np.load(clip_path(key)) as z:
        xw = z["xyz_world"].astype(np.float64)           # [64,52,3]
        xc = z["xyz_cam"].astype(np.float64)
        E = z["cam_ext"].astype(np.float64)               # [64,4,4] cam -> world
        conf = z["confs"].astype(np.float32) if "confs" in z.files else np.ones(xw.shape[:2], np.float32)
    recon = np.einsum("tij,tkj->tki", E[:, :3, :3], xc) + E[:, None, :3, 3]
    err = float(np.abs(recon - xw).max())
    Einv = np.linalg.inv(E[T_PAST - 1])                   # world -> cam(t=31)
    xr = np.einsum("ij,tkj->tki", Einv[:3, :3], xw) + Einv[None, None, :3, 3]
    return xr[:T_PAST].astype(np.float32), xr[T_PAST:].astype(np.float32), conf[T_PAST:], err, True


def main():
    keys = all_clips()
    n_test = len(read_split("test"))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, keys))
    past = np.stack([r[0] for r in res])
    fut = np.stack([r[1] for r in res])
    conf = np.stack([r[2] for r in res])
    errs = np.array([r[3] for r in res])
    avail = np.array([r[4] for r in res])
    print(f"available clips: {avail.sum()} (train {avail[:len(keys)-n_test].sum()}, test {avail[len(keys)-n_test:].sum()})")
    print(f"clips={len(keys)}  cam/world consistency max|err|={errs.max():.2e} m  "
          f"(median {np.median(errs):.2e})")
    if errs.max() > 1e-3:
        bad = [key_str(keys[i]) for i in np.argsort(-errs)[:5]]
        print("WARNING: frame convention mismatch on", bad)
    disp = np.linalg.norm(fut - past[:, -1:], axis=-1)[avail].mean()
    print(f"mean |future - last observed| = {disp*1000:.1f} mm (copy-last ADE)")
    FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT_ROOT / "poses.npz",
             keys=np.array([key_str(k) for k in keys]),
             is_test=np.arange(len(keys)) >= len(keys) - n_test, avail=avail,
             past=past, future=fut, conf_future=conf,
             task=np.array([k[0] for k in keys]))
    print("wrote", FEAT_ROOT / "poses.npz")


if __name__ == "__main__":
    sys.exit(main())
