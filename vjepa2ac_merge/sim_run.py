"""S1 / S2 (PROTOCOL.md): closed loop reach with V-JEPA 2-AC planning on the simulated Franka.
  python sim_run.py --arm full --n 256 --episodes 0-19 --out results/s1/full_256.jsonl [--samples 400 --rollout 2]
Each line of the output is one episode: goal, per step planned action / cosine to the goal direction / timings,
final distance."""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import acmerge as M
from sim_env import FrankaReach

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="full")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--episodes", default="0-19")
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--rollout", type=int, default=2)
ap.add_argument("--cem_steps", type=int, default=10)
ap.add_argument("--readout", default="restore")
ap.add_argument("--pa", type=int, default=1)
ap.add_argument("--ckpt", default=M.CKPT)
ap.add_argument("--pred_ckpt", default="")              # S3: fine tuned predictor state dict
ap.add_argument("--out", required=True)
ap.add_argument("--snap", default="")
ap.add_argument("--goal", default="0.10,0.20")                # C2: farther goals
args = ap.parse_args()
a, b = map(int, args.episodes.split("-"))
dev = "cuda"
enc, P = M.load_models(args.ckpt, dev)
if args.pred_ckpt:
    P.load_state_dict(__import__("torch").load(args.pred_ckpt, map_location=dev))
env = FrankaReach()
mpc = dict(samples=args.samples, rollout=args.rollout, cem_steps=args.cem_steps)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["episode"] for l in open(args.out)}


def goal_for(seed, env):
    """Random displacement of 10 to 20 cm (--goal) from the start, kept inside a safe box in the base frame."""
    rng = np.random.default_rng(1000 + seed)
    s0 = env.state()
    for _ in range(1000):
        v = rng.normal(size=3)
        v[2] *= 0.6
        v /= np.linalg.norm(v)
        d = v * rng.uniform(*map(float, args.goal.split(",")))
        g = s0[:3] + d
        if 0.35 <= g[0] <= 0.75 and -0.25 <= g[1] <= 0.25 and 0.12 <= g[2] <= 0.45:  # state z is above the table top
            return d
    raise RuntimeError("no goal")


for ep in range(a, b + 1):
    if ep in done:
        continue
    f0, s0 = env.reset(ep)
    d = goal_for(ep, env)
    p, Rb = env._base()
    fg, sg = env.move_to(env.eef_world() + Rb @ d)
    goal_world = env.eef_world()
    zg = M.encode(enc, fg[None], dev)[0]
    f, s = env.reset(ep)
    start_dist = float(np.linalg.norm(env.eef_world() - goal_world))
    rec = dict(episode=ep, arm=args.arm, n=args.n, mpc=mpc, readout=args.readout, pa=args.pa, goal_state=sg.tolist(),
               start_state=s.tolist(), start_dist=start_dist, steps=[])
    snaps = [f, fg]
    gen = torch.Generator().manual_seed(ep)
    for t in range(args.steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        z = M.encode(enc, f[None], dev)[0]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        rel = torch.from_numpy(env.gripper_relevance())
        m = M.make_merge(args.arm if args.arm != "full" else "identity", z, args.n, rel=rel, seed=ep * 100 + t)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        act = M.cem_plan(P, z, torch.from_numpy(s).to(dev), zg, m, mpc=mpc, readout=args.readout, pa=bool(args.pa),
                         gen=gen)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        to_goal = np.linalg.inv(Rb) @ (goal_world - env.eef_world())
        cos = float(np.dot(act[:3], to_goal) / (np.linalg.norm(act[:3]) * np.linalg.norm(to_goal) + 1e-9))
        f, s = env.apply(act)
        dist = float(np.linalg.norm(env.eef_world() - goal_world))
        rec["steps"].append(dict(action=act.tolist(), cos_goal=cos, dist=dist, t_enc=t1 - t0, t_merge=t2 - t1,
                                 t_plan=t3 - t2, n=int(m.n)))
        print(f"ep {ep} t {t} dist {dist:.3f} cos {cos:+.2f} plan {t3 - t2:.1f}s", flush=True)
    snaps.append(f)
    rec["final_dist"] = rec["steps"][-1]["dist"]
    rec["min_dist"] = min(x["dist"] for x in rec["steps"])
    rec["success"] = rec["final_dist"] < 0.05
    with open(args.out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    if args.snap and ep < a + 3:
        os.makedirs(args.snap, exist_ok=True)
        Image.fromarray(np.concatenate(snaps, 1)).save(os.path.join(args.snap, f"{args.arm}_{args.n}_ep{ep}.png"))
