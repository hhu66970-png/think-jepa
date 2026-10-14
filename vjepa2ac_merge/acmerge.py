"""Token merging between the frozen V-JEPA 2 encoder and the frozen V-JEPA 2-AC predictor, and CEM planning on top.

The encoder runs once per control step; the predictor runs samples x cem_steps x rollout times. Every merger here maps
the 256 tokens of the current frame to n groups (assign: token -> group); the predictor then reads n tokens per frame,
each at the size weighted centroid of its group (RoPE accepts fractional positions), with proportional attention
(log group size added to the key logits). Predicted next frames stay merged; the energy is read out either by copying
each group's prediction back to its member positions (restore, compared with the 256 goal tokens) or against the goal
merged with the same groups (compact, size weighted).

Mergers: sim (bipartite matching on encoder output cosine, several sources may share a receiver, sizes add),
gate (sim with edge score s * (1 - r_a)(1 - r_b), r = relevance / max relevance), gate_shuf (same relevance values,
positions permuted), rand (same procedure with random edges), pool (fixed grid blocks).
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("VJEPA2_REPO", os.path.join(ROOT, "vjepa2_official"))     # clone of facebookresearch/vjepa2
sys.path.insert(0, REPO)
from src.models import ac_predictor as acp  # noqa: E402
from src.models import vision_transformer as vit  # noqa: E402
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402

# Official V-JEPA 2-AC checkpoint (or the slim copy from tools/slim_ckpt.py, which loads faster).
CKPT = os.environ.get("VJEPA2_AC_CKPT", os.path.join(ROOT, "ckpt", "vjepa2-ac-vitg.pt"))
# Large token caches (frozen encoder outputs, fp16 memmaps); a fast local disk or a RAM disk is best.
CACHE = os.environ.get("VJEPA2_CACHE", os.path.join(ROOT, "cache"))
GRID = 16
NTOK = GRID * GRID
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GH, GW = torch.meshgrid(torch.arange(GRID).float(), torch.arange(GRID).float(), indexing="ij")
GH, GW = GH.flatten(), GW.flatten()


def load_models(ckpt, device="cuda"):
    enc = vit.vit_giant_xformers(img_size=(256, 256), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
                                 wide_SiLU=True, uniform_power=False, use_rope=True)
    pred = acp.vit_ac_predictor(img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2,
                                embed_dim=enc.embed_dim)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    def clean(d):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}

    msg = enc.load_state_dict(clean(sd["encoder"]), strict=False)
    assert not msg.missing_keys, msg
    pred.load_state_dict(clean(sd["predictor"]), strict=True)
    return enc.to(device).eval(), pred.to(device).eval()


@torch.no_grad()
def encode(enc, frames, device="cuda"):
    """frames: uint8 [T, 256, 256, 3] -> layer normed tokens [T, 256, 1408] (float32)."""
    x = torch.as_tensor(np.ascontiguousarray(frames)).float().div(255).permute(0, 3, 1, 2)
    x = ((x - MEAN) / STD).unsqueeze(2).repeat(1, 1, 2, 1, 1).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(x)
    return F.layer_norm(h.float(), (h.size(-1),))


# ----------------------------------------------------------------------------------------------------------- mergers
class Merge:
    """Groups of the 256 tokens of one frame. assign [256] in 0..n-1; sizes, ph, pw [n] (centroids in grid units)."""

    def __init__(self, assign):
        self.assign = assign.long()
        dev = self.assign.device
        self.n = int(self.assign.max()) + 1
        self.sizes = torch.zeros(self.n, device=dev).index_add_(0, self.assign, torch.ones(NTOK, device=dev))
        self.ph = torch.zeros(self.n, device=dev).index_add_(0, self.assign, GH.to(dev)) / self.sizes
        self.pw = torch.zeros(self.n, device=dev).index_add_(0, self.assign, GW.to(dev)) / self.sizes

    def reduce(self, z):
        """z [..., 256, D] -> group means [..., n, D]."""
        out = torch.zeros(*z.shape[:-2], self.n, z.shape[-1], device=z.device, dtype=z.dtype)
        out.index_add_(-2, self.assign, z)
        return out / self.sizes.to(z.dtype)[:, None]

    def restore(self, y):
        """y [..., n, D] -> [..., 256, D]."""
        return y.index_select(-2, self.assign)


def identity_merge(device="cuda"):
    return Merge(torch.arange(NTOK, device=device))


def pool_merge(n, device="cuda"):
    kh, kw = {256: (1, 1), 128: (1, 2), 64: (2, 2), 32: (2, 4), 16: (4, 4)}[n]
    g = (GH // kh) * (GRID // kw) + (GW // kw)
    return Merge(g.to(device))


def bipartite_merge(z, n, score_fn=None, rand_gen=None):
    """Iterated bipartite matching on the current groups until n remain.

    z [256, D] layer normed tokens. Each round splits the current groups alternately (raster order of their first
    member) into sources A and receivers B; every source picks its most similar receiver (several may share one), the
    r strongest edges are merged (r at most half of A, so each round drops at most a quarter of the groups and the
    ranking, plain or gated, decides which sources merge), sizes add. score_fn(best_sim, a_idx, b_idx) -> ranking score (gate). rand_gen: if
    given, receivers and ranking are random (rand control)."""
    dev = z.device
    x = z.float().clone()
    size = torch.ones(NTOK, device=dev)
    rep = torch.arange(NTOK, device=dev)          # original token -> current group (index into x)
    alive = torch.arange(NTOK, device=dev)        # current groups (indices into x), in raster order
    while len(alive) > n:
        a_idx, b_idx = alive[0::2], alive[1::2]
        r = min(len(alive) - n, len(a_idx) // 2)             # at most half the sources: the ranking decides who merges
        if rand_gen is None:
            xa = F.normalize(x[a_idx], dim=-1)
            xb = F.normalize(x[b_idx], dim=-1)
            best_sim, best_b = (xa @ xb.T).max(dim=-1)
            score = best_sim if score_fn is None else score_fn(best_sim, a_idx, b_idx[best_b])
        else:
            best_b = torch.randint(len(b_idx), (len(a_idx),), generator=rand_gen, device="cpu").to(dev)
            score = torch.rand(len(a_idx), generator=rand_gen, device="cpu").to(dev)
        top = score.topk(r).indices
        src, dst = a_idx[top], b_idx[best_b[top]]
        # size weighted mean into receivers (several sources may share one receiver)
        acc = torch.zeros_like(x).index_add_(0, dst, x[src] * size[src, None])
        wsum = torch.zeros_like(size).index_add_(0, dst, size[src])
        ud = torch.unique(dst)
        x[ud] = (x[ud] * size[ud, None] + acc[ud]) / (size[ud] + wsum[ud])[:, None]
        size[ud] = size[ud] + wsum[ud]
        remap = torch.arange(NTOK, device=dev)
        remap[src] = dst
        rep = remap[rep]
        keep = torch.ones(NTOK, dtype=torch.bool, device=dev)
        keep[src] = False
        alive = alive[keep[alive]]
    lut = torch.full((NTOK,), -1, dtype=torch.long, device=dev)
    lut[alive] = torch.arange(len(alive), device=dev)
    return Merge(lut[rep])


def gate_score(rel, lam=1.0):
    """rel [256] relevance per original token (current groups keep their representative's own score)."""
    r = rel / rel.max().clamp_min(1e-8)
    g = (1.0 - lam * r).clamp(0, 1)

    def fn(best_sim, a_idx, b_idx):
        return best_sim * g[a_idx] * g[b_idx]
    return fn


def make_merge(method, z, n, rel=None, seed=0):
    """method in identity|sim|gate|gate_shuf|rand|pool."""
    dev = z.device
    if method == "identity" or n >= NTOK:
        return identity_merge(dev)
    if method == "pool":
        return pool_merge(n, dev)
    if method == "sim":
        return bipartite_merge(z, n)
    if method == "rand":
        return bipartite_merge(z, n, rand_gen=torch.Generator().manual_seed(seed))
    if method in ("gate", "gate_shuf"):
        rel = rel.to(dev).float()
        if method == "gate_shuf":
            perm = torch.randperm(NTOK, generator=torch.Generator().manual_seed(seed)).to(dev)
            rel = rel[perm]
        return bipartite_merge(z, n, score_fn=gate_score(rel))
    raise ValueError(method)


# --------------------------------------------------------------------------------------------- predictor on groups
def _attn(A, x, pd, ph, pw, mask):
    B, N, C = x.shape
    q, k, v = A.qkv(x).unflatten(-1, (3, A.num_heads, -1)).permute(2, 0, 3, 1, 4)
    d = A.d_dim
    parts_q, parts_k = [], []
    for i, pos in enumerate((pd, ph, pw)):
        parts_q.append(rotate_queries_or_keys(q[..., i * d:(i + 1) * d], pos=pos))
        parts_k.append(rotate_queries_or_keys(k[..., i * d:(i + 1) * d], pos=pos))
    if 3 * d < A.head_dim:
        parts_q.append(q[..., 3 * d:])
        parts_k.append(k[..., 3 * d:])
    q, k = torch.cat(parts_q, -1), torch.cat(parts_k, -1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return A.proj(y.transpose(1, 2).reshape(B, N, C))


def predict(P, z, actions, states, m, pa=True):
    """z [B, T*n, D] merged context frames (all frames share the groups m); actions, states [B, T, 7].
    Returns [B, T*n, D] predictions (frame t -> t+1), before the outer layer norm."""
    x = P.predictor_embed(z)
    B, N, D = x.shape
    T = actions.size(1)
    n = N // T
    s = P.state_encoder(states).unsqueeze(2)
    a = P.action_encoder(actions).unsqueeze(2)
    x = torch.cat([a, s, x.view(B, T, n, D)], dim=2).flatten(1, 2)
    nt = n + 2
    dev = x.device
    fid = torch.arange(T, device=dev).repeat_interleave(nt)
    zero2 = torch.zeros(2, device=dev)
    pd = fid.float()
    ph = torch.cat([zero2, m.ph]).repeat(T)
    pw = torch.cat([zero2, m.pw]).repeat(T)
    allow = fid[:, None] >= fid[None, :]
    bias = torch.cat([zero2, m.sizes.log() if pa else torch.zeros(n, device=dev)]).repeat(T)
    mask = torch.where(allow, bias[None, :], torch.tensor(float("-inf"), device=dev)).to(x.dtype)
    for blk in P.predictor_blocks:
        x = x + _attn(blk.attn, blk.norm1(x), pd, ph, pw, mask)
        x = x + blk.mlp(blk.norm2(x))
    x = x.view(B, T, nt, D)[:, :, 2:].flatten(1, 2)
    return P.predictor_proj(P.predictor_norm(x))


def next_pose(pose, action):
    """Batched torch version of mpc_utils.compute_new_pose for actions with zero rotation (planner never rotates):
    xyz adds, gripper adds and clips, orientation unchanged."""
    out = pose.clone()
    out[..., :3] = pose[..., :3] + action[..., :3]
    out[..., 6:] = (pose[..., 6:] + action[..., 6:]).clamp(0, 1)
    return out


# ---------------------------------------------------------------------------------------------------------- planning
DEFAULT_MPC = dict(rollout=2, samples=400, topk=10, cem_steps=10, momentum_mean=0.15, momentum_std=0.15,
                   momentum_mean_gripper=0.15, momentum_std_gripper=0.15, maxnorm=0.05)


@torch.no_grad()
def rollout_energy(P, z0, pose0, goal, acts, m, readout="restore", pa=True, chunk=200):
    """z0 [256, D] current frame (full), pose0 [7], goal [256, D], acts [S, H, 7] -> energy [S]."""
    S, H, _ = acts.shape
    zc = m.reduce(z0)                                   # [n, D]
    gc = m.reduce(goal) if readout == "compact" else goal
    out = []
    for i in range(0, S, chunk):
        a = acts[i:i + chunk]
        b = a.size(0)
        frames = zc[None, None].expand(b, 1, -1, -1)
        poses = pose0[None, None].expand(b, 1, -1)
        for h in range(H):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = predict(P, frames.flatten(1, 2), a[:, :h + 1], poses, m, pa=pa)
            y = F.layer_norm(y.float()[:, -m.n:], (y.size(-1),))
            frames = torch.cat([frames, y[:, None]], dim=1)
            poses = torch.cat([poses, next_pose(poses[:, -1:], a[:, h:h + 1])], dim=1)
        last = frames[:, -1]
        if readout == "compact":
            e = (torch.abs(last - gc[None]).mean(-1) * m.sizes[None]).sum(-1) / NTOK
        else:
            e = torch.abs(m.restore(last) - gc[None]).mean(dim=(-1, -2))
        out.append(e)
    return torch.cat(out)


@torch.no_grad()
def cem_plan(P, z0, pose0, goal, m, mpc=None, readout="restore", pa=True, gen=None, close_gripper=None):
    """Official CEM (mpc_utils.cem) with the predictor on merged groups. Returns the first action [7] (numpy)."""
    c = dict(DEFAULT_MPC, **(mpc or {}))
    dev = z0.device
    H, S, K, mx = c["rollout"], c["samples"], c["topk"], c["maxnorm"]
    mean = torch.zeros(H, 4, device=dev)
    std = torch.cat([torch.full((H, 3), mx, device=dev), torch.ones(H, 1, device=dev)], -1)
    for _ in range(c["cem_steps"]):
        eps = torch.randn(S, H, 4, generator=gen, device="cpu").to(dev)
        smp = eps * std + mean
        smp[..., :3] = smp[..., :3].clamp(-mx, mx)
        smp[..., 3:] = smp[..., 3:].clamp(-0.75, 0.75)
        acts = torch.cat([smp[..., :3], torch.zeros(S, H, 3, device=dev), smp[..., 3:]], -1)
        if close_gripper is not None:
            acts[:, close_gripper:, -1] = 1.0
        e = rollout_energy(P, z0, pose0, goal, acts, m, readout=readout, pa=pa)
        sel = smp[e.topk(K, largest=False).indices]
        mu, sd = sel.mean(0), sel.std(0)
        mean = torch.cat([mu[:, :3] * (1 - c["momentum_mean"]) + mean[:, :3] * c["momentum_mean"],
                          mu[:, 3:] * (1 - c["momentum_mean_gripper"]) + mean[:, 3:] * c["momentum_mean_gripper"]], -1)
        std = torch.cat([sd[:, :3] * (1 - c["momentum_std"]) + std[:, :3] * c["momentum_std"],
                         sd[:, 3:] * (1 - c["momentum_std_gripper"]) + std[:, 3:] * c["momentum_std_gripper"]], -1)
    g = mean[0, 3:]
    g = torch.where(g.abs() < 0.25, torch.zeros_like(g), g)
    return torch.cat([mean[0, :3], torch.zeros(3, device=dev), g]).cpu().numpy()


def predictor_tokens(n, rollout):
    """Frame tokens the predictor processes per sample for one CEM evaluation (sum over rollout steps)."""
    return sum((h + 1) * (n + 2) for h in range(rollout))
