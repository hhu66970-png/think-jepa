"""T0 (PROTOCOL.md): equivalence, fidelity, energy landscape and timing on the official example trajectory.
  python t0_offline.py ckpt/vjepa2-ac-vitg.pt -> results/t0.json
Installation check: "equivalence max abs diff" must be tiny (the merged predictor with identity groups is the
official predictor)."""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import acmerge as M

sys.path.insert(0, os.path.join(M.REPO, "notebooks"))
from utils.mpc_utils import poses_to_diff  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
enc, P = M.load_models(sys.argv[1], dev)
d = np.load(os.path.join(M.REPO, "notebooks", "franka_example_traj.npz"))
frames, states = d["observations"][0], torch.tensor(d["states"][0]).float().to(dev)
act = poses_to_diff(states[0].cpu(), states[1].cpu()).float().to(dev)
z = M.encode(enc, frames, dev)                     # [2, 256, D]
z0, z1 = z[0], z[1]
out = {"gt_action": act.tolist()}

# 1. equivalence (fp32, no autocast)
with torch.no_grad():
    ref = P(z0[None], act[None, None], states[None, :1])
    mine = M.predict(P, z0[None], act[None, None], states[None, :1], M.identity_merge(dev), pa=True)
out["equiv_max_abs"] = float((ref - mine).abs().max())
print("equivalence max abs diff", out["equiv_max_abs"], flush=True)

ARMS = ["sim", "rand", "pool"]
NS = [128, 64, 32, 16]


def merges():
    yield "full", 256, M.identity_merge(dev)
    for n in NS:
        for a in ARMS:
            yield a, n, M.make_merge(a, z0, n, seed=0)


# 2. fidelity: one step prediction of frame 1 from frame 0 with the true action
fid = {"copy": float((z0 - z1).abs().mean())}
with torch.no_grad():
    for a, n, m in merges():
        for pa in (True, False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = M.predict(P, m.reduce(z0)[None], act[None, None], states[None, :1], m, pa=pa)
            y = F.layer_norm(y.float(), (y.size(-1),))[0]
            fid[f"{a}_{n}_pa{int(pa)}"] = float((m.restore(y) - z1).abs().mean())
            # how much of the error is the merging of the target itself (best a group constant prediction can do)
            fid[f"{a}_{n}_floor"] = float((m.restore(m.reduce(z1)) - z1).abs().mean())
out["fidelity_L1"] = fid
print(json.dumps(fid, indent=1), flush=True)

# 3. energy landscape on a 5^3 grid, goal = frame 1
g = torch.linspace(-0.075, 0.075, 5)
grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
acts = torch.cat([grid, torch.zeros(len(grid), 4)], -1)[:, None].to(dev)    # [125, 1, 7]
land = {}
for a, n, m in merges():
    e = M.rollout_energy(P, z0, states[0], z1, acts, m, readout="restore")
    land[f"{a}_{n}"] = e.cpu().numpy()
    if a != "full":
        e2 = M.rollout_energy(P, z0, states[0], z1, acts, m, readout="compact")
        land[f"{a}_{n}_compact"] = e2.cpu().numpy()
full = land["full_256"]
amin_full = grid[full.argmin()].numpy()
ls = {}
for k, e in land.items():
    am = grid[e.argmin()].numpy()
    ls[k] = dict(spearman=float(spearmanr(e, full)[0]), argmin=am.tolist(),
                 argmin_dist_to_full=float(np.linalg.norm(am - amin_full)),
                 argmin_cos_gt=float(np.dot(am, act[:3].cpu().numpy()) /
                                     (np.linalg.norm(am) * np.linalg.norm(act[:3].cpu().numpy()) + 1e-9)))
out["landscape"] = ls
print(json.dumps(ls, indent=1), flush=True)

# 4. timing: encoder and one CEM call per arm
torch.cuda.synchronize()
t = []
for _ in range(5):
    t0 = time.perf_counter(); M.encode(enc, frames[:1], dev); torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
tim = {"encoder_frame_s": float(np.median(t[1:]))}
for a, n, m in [("full", 256, M.identity_merge(dev))] + [("sim", n, M.make_merge("sim", z0, n)) for n in NS]:
    ts = []
    for r in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        M.cem_plan(P, z0, states[0], z1, m, gen=torch.Generator().manual_seed(r))
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    tim[f"cem_{n}_s"] = float(np.median(ts))
for n in NS:   # merge cost itself
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        M.make_merge("sim", z0, n)
    torch.cuda.synchronize(); tim[f"merge_sim_{n}_s"] = (time.perf_counter() - t0) / 5
out["timing"] = tim
print(json.dumps(tim, indent=1), flush=True)
os.makedirs(os.path.join(M.ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(M.ROOT, "results", "t0.json"), "w"), indent=1)
