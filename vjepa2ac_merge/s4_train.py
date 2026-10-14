"""S4 (PROTOCOL.md): merge aware fine tuning of the S3 predictor on the same simulated play data.
  python s4_train.py rel                                   (gripper relevance of every play frame -> data/sim/rel.npy)
  python -m torch.distributed.run --nproc_per_node 4 s4_train.py train
  python s4_train.py eval                                  (held out: S3 vs S4 predictor on every merged arm)
C1 (PROTOCOL.md): more training seeds, and the equal training control (same recipe on the full tokens):
  python -m torch.distributed.run --nproc_per_node 4 s4_train.py train --seed 1 --out ckpt/c1_s4_k1.pt
  python -m torch.distributed.run --nproc_per_node 4 s4_train.py train --seed 0 --plain --out ckpt/c1_s3b_k0.pt"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as Rot

import acmerge as M

ROOT = M.ROOT
TOK = os.path.join(M.CACHE, "s3_tok.npy")                                            # frozen encoder tokens from s3_train.py
S3_CKPT = os.path.join(ROOT, "ckpt", "ac_pred_sim_ft.pt")
files = sorted(__import__("glob").glob(os.path.join(ROOT, "data", "sim", "play_*.npz")))
seeds = np.concatenate([np.load(f)["seeds"] for f in files])
E, T17 = len(seeds), 17
TRAIN = seeds < 1760


def all_states():
    return np.concatenate([np.load(f)["states"] for f in files]).astype(np.float32)        # [E, 17, 7]


def actions_of(st):
    """Official poses_to_diff, vectorised (as in s3_train.py): [..., T, 7] -> [..., T-1, 7]."""
    a = np.zeros(st.shape[:-2] + (st.shape[-2] - 1, 7), np.float32)
    a[..., :3] = st[..., 1:, :3] - st[..., :-1, :3]
    r0 = Rot.from_euler("xyz", st[..., :-1, 3:6].reshape(-1, 3))
    r1 = Rot.from_euler("xyz", st[..., 1:, 3:6].reshape(-1, 3))
    a[..., 3:6] = (r1 * r0.inv()).as_euler("xyz").reshape(a[..., 3:6].shape)
    a[..., 6] = st[..., 1:, 6] - st[..., :-1, 6]
    return a

mode = sys.argv[1]
ap = __import__("argparse").ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--plain", action="store_true")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--out", default=os.path.join(ROOT, "ckpt", "ac_pred_sim_ft_merge.pt"))
args = ap.parse_args(sys.argv[2:])
OUT = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
REL = os.path.join(ROOT, "data", "sim", "rel.npy")
METHODS, NS, CLIP = ["sim", "gate", "gate_shuf", "rand", "pool"], [64, 16], 3


def load_pred(path, dev):
    _, P = M.load_models(M.CKPT, dev)
    P.load_state_dict(torch.load(path, map_location="cpu"))
    return P


def merged_losses(P, z, a, s, m):
    """z [CLIP, 256, D] layer normed; a [CLIP-1, 7]; s [CLIP, 7]. Official teacher forcing + 2 step rollout L1, with
    the context reduced by groups m and predictions restored to 256 tokens (as at planning time)."""
    zc = m.reduce(z[:-1])                                                   # [2, n, D]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = M.predict(P, zc.flatten(0, 1)[None], a[None], s[None, :-1], m)
    y = F.layer_norm(y.float(), (y.size(-1),)).view(CLIP - 1, m.n, -1)
    jloss = (m.restore(y) - z[1:]).abs().mean()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y2 = M.predict(P, torch.cat([zc[0], y[0]])[None], a[None], s[None, :-1], m)
    y2 = F.layer_norm(y2.float()[0, -m.n:], (y2.size(-1),))
    sloss = (torch.cat([m.restore(y[:1]), m.restore(y2)[None]]) - z[1:]).abs().mean()
    return jloss, sloss


if mode == "rel":
    from sim_env import FrankaReach
    env = FrankaReach()
    env.reset(1000)
    p, Rb = env._base()

    def relevance(st):                                                      # [K, 7] states -> [K, 256]
        xyz = st[:, :3] - [0, 0, env.z_off]                                 # state z is measured from the table
        uv = env.project(p + xyz @ Rb.T) / 16.0
        gh, gw = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        d2 = (gh.flatten()[None] + 0.5 - uv[:, :1]) ** 2 + (gw.flatten()[None] + 0.5 - uv[:, 1:]) ** 2
        return np.exp(-d2 / (2 * 2.0 ** 2)).astype(np.float32)

    for d in ([0.1, 0.1, 0.05], [-0.1, -0.12, -0.08]):                      # check against the simulator's own
        env.move_to(env.eef_world() + Rb @ np.array(d))
        print("max diff to env.gripper_relevance:", float(np.abs(env.gripper_relevance() - relevance(env.state()[None])[0]).max()))
    st = all_states()
    rel = relevance(st.reshape(-1, 7)).reshape(E, T17, 256)
    print("rel", rel.shape)
    np.save(REL, rel)

elif mode == "train":                                                       # S4 (seed 0) and C1, via ftlib
    import torch.distributed as dist
    import ftlib
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    P = load_pred(S3_CKPT, f"cuda:{dist.get_rank()}")
    tok = np.load(TOK, mmap_mode="r")
    st_all, rel_all = all_states(), np.load(REL)
    tr = np.where(TRAIN)[0]

    def sample(rng):
        e, t = int(rng.choice(tr)), int(rng.integers(0, T17 - CLIP + 1))
        return tok[e, t:t + CLIP], st_all[e, t:t + CLIP], rel_all[e, t]

    ftlib.train(P, sample, OUT, os.path.join(ROOT, "results", os.path.basename(OUT).replace(".pt", "_log.json")),
                plain=args.plain, seed=args.seed, steps=args.steps)

elif mode == "eval":
    dev = "cuda"
    Ps = {"s3": load_pred(S3_CKPT, dev).eval(), "s4": load_pred(OUT, dev).eval()}
    tok = np.load(TOK, mmap_mode="r")
    st_all, rel_all = all_states(), np.load(REL)
    g = torch.linspace(-0.05, 0.05, 5)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    acts = torch.cat([grid, torch.zeros(len(grid), 4)], -1)[:, None].to(dev)
    rows = []
    for e in np.where(~TRAIN)[0]:
        for t in range(0, 16, 4):
            z = F.layer_norm(torch.from_numpy(tok[e, t:t + 2].astype(np.float32)).to(dev), (1408,))
            st = st_all[e, t:t + 2]
            a = torch.from_numpy(actions_of(st)).to(dev)
            s0 = torch.from_numpy(st[0]).to(dev)
            true = a[0, :3].cpu().numpy()
            rec = dict(e=int(e), t=t, moving=bool(np.linalg.norm(true) >= 0.02), copy=float((z[0] - z[1]).abs().mean()))
            for meth in METHODS:
                for n in NS:
                    m = M.make_merge(meth, z[0], n, rel=torch.from_numpy(rel_all[e, t]), seed=int(e) * 100 + t)
                    rec[f"floor_{meth}_{n}"] = float((m.restore(m.reduce(z[1])) - z[1]).abs().mean())   # best restore L1
                    for name, P in Ps.items():
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            y = M.predict(P, m.reduce(z[0])[None], a[None], s0[None, None], m)
                        y = F.layer_norm(y.float(), (y.size(-1),))[0]
                        en = M.rollout_energy(P, z[0], s0, z[1], acts, m).cpu().numpy()
                        am = grid[en.argmin()].numpy()
                        rec[f"{name}_{meth}_{n}"] = dict(
                            L1=float((m.restore(y) - z[1]).abs().mean()),
                            cos=float(np.dot(am, true) / (np.linalg.norm(am) * np.linalg.norm(true) + 1e-9)))
            rows.append(rec)
        print(e, flush=True)
    mv = [r for r in rows if r["moving"]]
    summ = {k: dict(L1=float(np.mean([r[k]["L1"] for r in rows])), cos_moving=float(np.mean([r[k]["cos"] for r in mv])))
            for k in rows[0] if k.startswith("s")}
    summ["copy_L1"] = float(np.mean([r["copy"] for r in rows]))
    summ["floor_L1"] = {k[6:]: float(np.mean([r[k] for r in rows])) for k in rows[0] if k.startswith("floor_")}
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(ROOT, "results", "s4_eval.json"), "w"))
