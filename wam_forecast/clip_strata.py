"""Phase B: model-free per-clip strata, fixed before any Phase B result.

  motion     copy-last ADE of the clip (mm): how far the hands actually move in the future window
  horizon_s  real duration of the forecast window, (frame_indices[63] - frame_indices[31]) / 30 fps
  obs_s      real duration of the observed window
  task       EgoDex task name
Writes <FEAT_ROOT>/clip_strata.npz (row order = poses.npz / all_clips()).
"""
import numpy as np

from common import FEAT_ROOT, all_clips, clip_path

P = np.load(FEAT_ROOT / "poses.npz")
motion = np.linalg.norm(P["future"] - P["past"][:, -1:], axis=-1).mean(axis=(1, 2)) * 1000
keys = all_clips()
assert len(keys) == len(P["keys"]) and all(str(a).endswith(b[1]) for a, b in zip(P["keys"], keys))
hor = np.full(len(keys), np.nan, np.float32)
obs = np.full(len(keys), np.nan, np.float32)
for i, k in enumerate(keys):
    p = clip_path(k)
    if p.exists():
        with np.load(p) as z:
            fi = z["frame_indices"]
        hor[i] = (fi[63] - fi[31]) / 30.0
        obs[i] = (fi[31] - fi[0]) / 30.0
np.savez(FEAT_ROOT / "clip_strata.npz", motion=motion.astype(np.float32), horizon_s=hor, obs_s=obs,
         task=P["task"], is_test=P["is_test"])
t = P["is_test"]
for name, v in (("motion (mm)", motion), ("horizon (s)", hor)):
    q = np.nanpercentile(v[~t], [25, 50, 75])
    print(f"{name}: train quartiles {q.round(2)}, test mean {np.nanmean(v[t]):.2f}")
