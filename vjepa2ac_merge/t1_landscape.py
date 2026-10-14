"""T1 (PROTOCOL.md): T0's fidelity and energy landscape measures on 100 simulated transitions.
Seeds 100..199 (disjoint from the S2 episodes): from the start pose, a random displacement of up to 15 cm sets the
current state; a random action (each axis uniform in [-0.05, 0.05]) gives the next frame, used as the goal.
  python t1_landscape.py --states 100-199 --out results/t1.jsonl"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import acmerge as M
from sim_env import FrankaReach

ap = argparse.ArgumentParser()
ap.add_argument("--states", default="100-199")
ap.add_argument("--out", required=True)
ap.add_argument("--pred_ckpt", default="")              # T1ft: the S3 fine tuned predictor
args = ap.parse_args()
lo, hi = map(int, args.states.split("-"))
dev = "cuda"
enc, P = M.load_models(M.CKPT, dev)
if args.pred_ckpt:
    P.load_state_dict(torch.load(args.pred_ckpt, map_location=dev))
env = FrankaReach()
g = torch.linspace(-0.05, 0.05, 5)
grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
acts = torch.cat([grid, torch.zeros(len(grid), 4)], -1)[:, None].to(dev)
ARMS = ["sim", "gate", "gate_shuf", "rand", "pool"]
NS = [128, 64, 32, 16]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["seed"] for l in open(args.out)}

for seed in range(lo, hi + 1):
    if seed in done:
        continue
    rng = np.random.default_rng(seed)
    env.reset(seed)
    p, Rb = env._base()
    d = rng.uniform(-1, 1, 3) * [0.12, 0.15, 0.10]
    f0, s0 = env.move_to(env.eef_world() + Rb @ d)
    a = np.concatenate([rng.uniform(-0.05, 0.05, 3), np.zeros(4)])
    rel = torch.from_numpy(env.gripper_relevance())
    f1, s1 = env.apply(a)
    true = s1[:3] - s0[:3]                                    # displacement actually reached
    z = M.encode(enc, np.stack([f0, f1]), dev)
    z0, z1 = z[0], z[1]
    st0 = torch.from_numpy(s0).to(dev)
    at = torch.from_numpy(np.concatenate([true, np.zeros(3), [0.0]])).float().to(dev)
    rec = dict(seed=seed, action=a.tolist(), true=true.tolist(), copy_L1=float((z0 - z1).abs().mean()), arms={})
    full = None
    for arm, n in [("full", 256)] + [(x, n) for n in NS for x in ARMS]:
        m = M.make_merge("identity" if arm == "full" else arm, z0, n, rel=rel, seed=seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = M.predict(P, m.reduce(z0)[None], at[None, None], st0[None, None], m)
        y = F.layer_norm(y.float(), (y.size(-1),))[0]
        e = M.rollout_energy(P, z0, st0, z1, acts, m).cpu().numpy()
        if full is None:
            full = e
        am = grid[e.argmin()].numpy()
        rec["arms"][f"{arm}_{n}"] = dict(
            L1=float((m.restore(y) - z1).abs().mean()),
            rho=float(spearmanr(e, full)[0]),
            argmin_cos=float(np.dot(am, true) / (np.linalg.norm(am) * np.linalg.norm(true) + 1e-9)),
            argmin_eq_full=bool(np.allclose(am, grid[full.argmin()].numpy())))
    with open(args.out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(seed, {k: round(v["rho"], 2) for k, v in rec["arms"].items() if k.endswith("_16")}, flush=True)
