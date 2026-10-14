"""S3 (PROTOCOL.md): fine tune the AC predictor on simulated random play, encoder frozen.
  python s3_train.py encode --part 0/4           (one per GPU; frozen encoder tokens -> $VJEPA2_CACHE memmap, fp16)
  python -m torch.distributed.run --nproc_per_node 4 s3_train.py train
  python s3_train.py eval                         (held out seeds: one step L1 vs copy, landscape argmin cosine)"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

import acmerge as M

mode = sys.argv[1]
ROOT = M.ROOT
TOK = os.path.join(M.CACHE, "s3_tok.npy")
OUT = os.path.join(ROOT, "ckpt", "ac_pred_sim_ft.pt")
files = sorted(glob.glob(os.path.join(ROOT, "data", "sim", "play_*.npz")))
seeds = np.concatenate([np.load(f)["seeds"] for f in files])
E, T17 = len(seeds), 17
TRAIN = seeds < 1760


def all_states():
    return np.concatenate([np.load(f)["states"] for f in files]).astype(np.float32)        # [E, 17, 7]


def actions_of(st):
    """Official poses_to_diff, vectorised: [..., T, 7] -> [..., T-1, 7]."""
    a = np.zeros(st.shape[:-2] + (st.shape[-2] - 1, 7), np.float32)
    a[..., :3] = st[..., 1:, :3] - st[..., :-1, :3]
    r0 = Rot.from_euler("xyz", st[..., :-1, 3:6].reshape(-1, 3))
    r1 = Rot.from_euler("xyz", st[..., 1:, 3:6].reshape(-1, 3))
    a[..., 3:6] = (r1 * r0.inv()).as_euler("xyz").reshape(a[..., 3:6].shape)
    a[..., 6] = st[..., 1:, 6] - st[..., :-1, 6]
    return a


if mode == "encode":
    os.makedirs(M.CACHE, exist_ok=True)
    k, K = map(int, sys.argv[3].split("/"))
    mm = np.lib.format.open_memmap(TOK, mode="r+" if os.path.exists(TOK) else "w+", dtype=np.float16,
                                   shape=(E, T17, 256, 1408)) if k == 0 or os.path.exists(TOK) else None
    while mm is None:                                     # parts > 0 wait for part 0 to create the file
        time.sleep(5)
        if os.path.exists(TOK):
            time.sleep(5)
            mm = np.lib.format.open_memmap(TOK, mode="r+")
    enc, _ = M.load_models(M.CKPT, "cuda")
    e0 = 0
    for f in files:
        fr = np.load(f)["frames"]
        for j in range(len(fr)):
            if (e0 + j) % K == k:
                mm[e0 + j] = M.encode(enc, fr[j], "cuda").half().cpu().numpy()
        e0 += len(fr)
        print(f, flush=True)
    mm.flush()
    open(TOK + f".done{k}", "w").write("ok")

elif mode == "train":
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    torch.manual_seed(239 + rank)
    sd = torch.load(M.CKPT, map_location="cpu", weights_only=False)
    from src.models import ac_predictor as acp
    P = acp.vit_ac_predictor(img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=1408,
                             use_activation_checkpointing=True)
    P.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v for k, v in sd["predictor"].items()})
    del sd
    P = torch.nn.parallel.DistributedDataParallel(P.to(dev), device_ids=[rank], find_unused_parameters=True)
    opt = torch.optim.AdamW(P.parameters(), lr=1e-4, weight_decay=0.04)
    tok = np.load(TOK, mmap_mode="r")
    st_all = all_states()
    tr = np.where(TRAIN)[0]
    STEPS, B, CLIP, WARM = 2000, 32 // world, 8, 100
    rng = np.random.default_rng(rank)
    log = []
    t0 = time.time()
    for it in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = 1e-4 * min(1.0, (it + 1) / WARM)
        ep = rng.choice(tr, B)
        s0 = rng.integers(0, T17 - CLIP + 1, B)
        z = torch.from_numpy(np.stack([tok[e, s:s + CLIP] for e, s in zip(ep, s0)])).to(dev).float()
        z = F.layer_norm(z, (1408,)).flatten(1, 2)                              # [B, CLIP*256, D]
        st = np.stack([st_all[e, s:s + CLIP] for e, s in zip(ep, s0)])
        a = torch.from_numpy(actions_of(st)).to(dev)                             # [B, CLIP-1, 7]
        st = torch.from_numpy(st).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            def step(_z, _a, _s):
                return F.layer_norm(P(_z, _a, _s).float(), (1408,))
            z_tf = step(z[:, :-256], a, st[:, :-1])
            jloss = (z_tf - z[:, 256:]).abs().mean()
            _z = torch.cat([z[:, :256], z_tf[:, :256]], 1)
            _z = torch.cat([_z, step(_z, a[:, :2], st[:, :2])[:, -256:]], 1)
            sloss = (_z[:, 256:] - z[:, 256:768]).abs().mean()
            loss = jloss + sloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
        opt.step()
        if rank == 0 and (it % 20 == 0 or it == STEPS - 1):
            log.append(dict(it=it, jloss=float(jloss), sloss=float(sloss), t=time.time() - t0))
            print(log[-1], flush=True)
        if rank == 0 and (it + 1) % 500 == 0:
            torch.save(P.module.state_dict(), OUT)
            json.dump(log, open(os.path.join(ROOT, "results", "s3_trainlog.json"), "w"))
    dist.destroy_process_group()

elif mode == "eval":
    from scipy.stats import spearmanr  # noqa: F401
    dev = "cuda"
    enc, P0 = M.load_models(M.CKPT, dev)
    import copy
    P1 = copy.deepcopy(P0)
    P1.load_state_dict(torch.load(OUT, map_location="cpu"))
    tok = np.load(TOK, mmap_mode="r")
    st_all = all_states()
    ho = np.where(~TRAIN)[0]
    g = torch.linspace(-0.05, 0.05, 5)
    grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    acts = torch.cat([grid, torch.zeros(len(grid), 4)], -1)[:, None].to(dev)
    res = {"official": [], "finetuned": []}
    idm = M.identity_merge(dev)
    for e in ho:
        for t in range(0, 16, 4):
            z0 = F.layer_norm(torch.from_numpy(tok[e, t].astype(np.float32)).to(dev), (1408,))
            z1 = F.layer_norm(torch.from_numpy(tok[e, t + 1].astype(np.float32)).to(dev), (1408,))
            st = st_all[e, t:t + 2]
            a = torch.from_numpy(actions_of(st)[0]).to(dev)
            s0 = torch.from_numpy(st[0]).to(dev)
            true = a[:3].cpu().numpy()
            for name, P in (("official", P0), ("finetuned", P1)):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    y = M.predict(P, z0[None], a[None, None], s0[None, None], idm)
                y = F.layer_norm(y.float(), (1408,))[0]
                en = M.rollout_energy(P, z0, s0, z1, acts, idm).cpu().numpy()
                am = grid[en.argmin()].numpy()
                res[name].append(dict(L1=float((y - z1).abs().mean()), copy=float((z0 - z1).abs().mean()),
                                      cos=float(np.dot(am, true) / (np.linalg.norm(am) * np.linalg.norm(true) + 1e-9)),
                                      moving=bool(np.linalg.norm(true) >= 0.02)))
    summ = {}
    for name, r in res.items():
        mv = [x for x in r if x["moving"]]
        summ[name] = dict(n=len(r), L1=float(np.mean([x["L1"] for x in r])), copy=float(np.mean([x["copy"] for x in r])),
                          argmin_cos_moving=float(np.mean([x["cos"] for x in mv])), n_moving=len(mv))
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=res), open(os.path.join(ROOT, "results", "s3_eval.json"), "w"))
