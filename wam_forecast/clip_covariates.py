"""Per clip covariates known at forecast time, used only for the descriptive strata (Tab. S28).

  past_motion_m  observed path length of the hand centroid (mean of the 52 joints) over frames 0-31, in metres
  horizon_s      seconds from the last observed frame (31) to the last forecast frame (63), at 30 fps
  stride_frames  frames of the 30 fps video between consecutive sampled observed frames, (f[31] - f[0]) / 31
Rows follow all_clips() (= evaluation rows). Writes <out>/clip_covariates.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import FEAT_ROOT, W, all_clips, clip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(W / "runs" / "paper_data"))
    a = ap.parse_args()
    P = np.load(FEAT_ROOT / "poses.npz")
    past = P["past"].astype(np.float64)                          # [N, 32, 52, 3], camera frame of frame 31 (m)
    c = past.mean(2)
    motion = np.linalg.norm(np.diff(c, axis=1), axis=-1).sum(1)
    keys = all_clips()
    hor = np.zeros(len(keys)); stride = np.zeros(len(keys))
    for i, k in enumerate(keys):
        fi = np.load(clip_path(k))["frame_indices"]
        hor[i] = (fi[63] - fi[31]) / 30.0
        stride[i] = (fi[31] - fi[0]) / 31.0
    out = dict(note="per clip covariates known at forecast time, in all_clips() order (= evaluation rows): observed path "
                    "length of the hand centroid over frames 0-31 (m), forecast horizon (s, 30 fps), and sampling stride (video frames between consecutive sampled frames)",
               past_motion_m=[round(float(v), 5) for v in motion], horizon_s=[round(float(v), 4) for v in hor],
               stride_frames=[round(float(v), 4) for v in stride])
    Path(a.out, "clip_covariates.json").write_text(json.dumps(out))
    print("median past motion %.3f m, median horizon %.2f s" % (np.median(motion), np.median(hor)))


if __name__ == "__main__":
    main()
