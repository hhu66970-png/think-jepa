"""Shared fine tuning pieces for S4, C1 and D1 (PROTOCOL.md): the official pair of L1 losses (teacher forcing and
2 step rollout) on clips of 3 frames, either on merged tokens with the restore readout (merge aware) or on the full
256 tokens (plain, the equal training control). Manual data parallel over torchrun ranks."""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

import acmerge as M

CLIP, D = 3, 1408
METHODS, NS = ["sim", "gate", "gate_shuf", "rand", "pool"], [64, 16]


def actions_of(st):
    """Official poses_to_diff, vectorised (as in s3_train.py): [..., T, 7] -> [..., T-1, 7]."""
    a = np.zeros(st.shape[:-2] + (st.shape[-2] - 1, 7), np.float32)
    a[..., :3] = st[..., 1:, :3] - st[..., :-1, :3]
    r0 = Rot.from_euler("xyz", st[..., :-1, 3:6].reshape(-1, 3))
    r1 = Rot.from_euler("xyz", st[..., 1:, 3:6].reshape(-1, 3))
    a[..., 3:6] = (r1 * r0.inv()).as_euler("xyz").reshape(a[..., 3:6].shape)
    a[..., 6] = st[..., 1:, 6] - st[..., :-1, 6]
    return a


def merged_losses(P, z, a, s, m):
    """z [CLIP, 256, D] layer normed; a [CLIP-1, 7]; s [CLIP, 7]. Context reduced by groups m, predictions restored
    to 256 tokens (as at planning time)."""
    zc = m.reduce(z[:-1])                                                   # [2, n, D]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = M.predict(P, zc.flatten(0, 1)[None], a[None], s[None, :-1], m)
    y = F.layer_norm(y.float(), (D,)).view(CLIP - 1, m.n, -1)
    jloss = (m.restore(y) - z[1:]).abs().mean()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y2 = M.predict(P, torch.cat([zc[0], y[0]])[None], a[None], s[None, :-1], m)
    y2 = F.layer_norm(y2.float()[0, -m.n:], (D,))
    sloss = (torch.cat([m.restore(y[:1]), m.restore(y2)[None]]) - z[1:]).abs().mean()
    return jloss, sloss


def plain_losses(P, z, a, s):
    """Batched, full tokens: z [B, CLIP, 256, D]; a [B, CLIP-1, 7]; s [B, CLIP, 7]. Same pair of losses."""
    B = z.size(0)
    idm = M.identity_merge(z.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = M.predict(P, z[:, :-1].flatten(1, 2), a, s[:, :-1], idm)
    y = F.layer_norm(y.float(), (D,)).view(B, CLIP - 1, M.NTOK, -1)
    jloss = (y - z[:, 1:]).abs().mean()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y2 = M.predict(P, torch.cat([z[:, 0], y[:, 0]], 1), a, s[:, :-1], idm)
    y2 = F.layer_norm(y2.float()[:, -M.NTOK:], (D,))
    sloss = (torch.cat([y[:, :1], y2[:, None]], 1) - z[:, 1:]).abs().mean()
    return jloss, sloss


def to_dev(zt, st, dev):
    z = F.layer_norm(torch.from_numpy(np.asarray(zt, np.float32)).to(dev), (D,))
    return z, torch.from_numpy(actions_of(st)).to(dev), torch.from_numpy(np.asarray(st, np.float32)).to(dev)


def train(P, sample, out, log_path, plain, seed, steps=2000, lr=1e-4, warm=100, save_at=(500, 1000, 1500, 2000),
          keep_all=False, mb=4):
    """sample(rng) -> (tokens [CLIP, 256, D] fp16, states [CLIP, 7], relevance [256]). Batch 32 over all ranks.
    Merge aware: one merger (uniform over METHODS) and n (uniform over NS) per sample, groups from the context frame.
    Saves rank 0's weights at save_at (one file per step if keep_all, else overwriting out); out + '.done' at the end."""
    import torch.distributed as dist
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = f"cuda:{rank}"
    torch.manual_seed(4000 + 100 * seed + rank)
    rng = np.random.default_rng(4000 + 100 * seed + rank)
    P.train()
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=0.04)
    B = 32 // world
    log, t0 = [], time.time()
    for it in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (it + 1) / warm)
        opt.zero_grad(set_to_none=True)
        js, ss = 0.0, 0.0
        if plain:
            for _ in range(0, B, mb):
                bt = [sample(rng) for _ in range(mb)]
                z, a, s = to_dev(np.stack([b[0] for b in bt]), np.stack([b[1] for b in bt]), dev)
                jl, sl = plain_losses(P, z, a, s)
                ((jl + sl) * mb / B).backward()
                js, ss = js + float(jl) * mb / B, ss + float(sl) * mb / B
        else:
            for _ in range(B):
                zt, st, rel = sample(rng)
                z, a, s = to_dev(zt, st, dev)
                with torch.no_grad():
                    m = M.make_merge(METHODS[rng.integers(5)], z[0], NS[rng.integers(2)],
                                     rel=torch.from_numpy(np.asarray(rel)), seed=int(rng.integers(1 << 30)))
                jl, sl = merged_losses(P, z, a, s, m)
                ((jl + sl) / B).backward()
                js, ss = js + float(jl) / B, ss + float(sl) / B
        for prm in P.parameters():
            if prm.grad is not None:
                dist.all_reduce(prm.grad)
                prm.grad /= world
        torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
        opt.step()
        if rank == 0 and (it % 20 == 0 or it == steps - 1):
            log.append(dict(it=it, jloss=js, sloss=ss, t=time.time() - t0))
            print(log[-1], flush=True)
        if rank == 0 and (it + 1) in save_at:
            torch.save(P.state_dict(), out.replace(".pt", f"_{it + 1}.pt") if keep_all else out)
            json.dump(log, open(log_path, "w"))
    if rank == 0:
        open(out + ".done", "w").write("ok")
    dist.barrier()
    dist.destroy_process_group()
