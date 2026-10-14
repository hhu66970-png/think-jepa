"""S3 data (PROTOCOL.md): random play of the simulated Franka for fine tuning the AC predictor (CPU only).
Seeds 1000 + k: random start offset (as T1), then 16 actions of smoothed random displacement (each axis
a_t = 0.6 a_{t-1} + 0.4 u, u uniform in [-0.05, 0.05], reflected at the safe box), gripper open.
  python s3_collect.py --seeds 1000-1099 --out data/sim/play_1000.npz"""
import argparse

import numpy as np

from sim_env import FrankaReach

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", required=True)
ap.add_argument("--steps", type=int, default=16)
ap.add_argument("--out", required=True)
args = ap.parse_args()
lo, hi = map(int, args.seeds.split("-"))
BOX = np.array([[0.35, 0.75], [-0.25, 0.25], [0.12, 0.45]])
env = FrankaReach()
F, S = [], []
for seed in range(lo, hi + 1):
    rng = np.random.default_rng(seed)
    env.reset(seed)
    p, Rb = env._base()
    f, s = env.move_to(env.eef_world() + Rb @ (rng.uniform(-1, 1, 3) * [0.12, 0.15, 0.10]))
    fr, st = [f], [s]
    a = np.zeros(3)
    for t in range(args.steps):
        a = np.clip(0.6 * a + 0.4 * rng.uniform(-0.05, 0.05, 3), -0.05, 0.05)
        nxt = s[:3] + a
        a = np.where((nxt < BOX[:, 0]) | (nxt > BOX[:, 1]), -a, a)
        f, s = env.apply(np.concatenate([a, np.zeros(4)]))
        fr.append(f)
        st.append(s)
    F.append(np.stack(fr))
    S.append(np.stack(st))
    print(seed, flush=True)
np.savez(args.out, frames=np.stack(F), states=np.stack(S), seeds=np.arange(lo, hi + 1))
