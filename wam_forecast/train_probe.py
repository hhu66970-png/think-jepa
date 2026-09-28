"""Attentive forecasting probe on frozen (possibly merged) V-JEPA tokens.

Task: from the observed past (frames 0..31) predict the 52 hand joints of frames 32..63 in the
camera frame of frame 31. Inputs (--input):
  vis      restored-dense visual tokens only                         (main WAM read-out)
  pose     past joints only                                          (no vision, gate control)
  vispose  both                                                      (practical setting)
  zero     visual tokens replaced by zeros                           (lower bound)
The probe is cross-attention only (32 future-frame queries over the 4096 restored tokens), so
every op is deterministic under torch.use_deterministic_algorithms(True).

Outputs <out>/metrics.json and <out>/perclip_ade.npy (test clips in poses.npz order).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FEAT_ROOT, TOKENS_PAST, write_json

# ----------------------------------------------------------------------------- data
class FeatureBank:
    """All shards of one config, memory-mapped (page cache is shared across processes)."""

    def __init__(self, cfg):
        d = FEAT_ROOT / cfg
        self.tok, self.idx, self.where = [], [], {}
        for s, tf in enumerate(sorted(d.glob("tokens_*.npy"))):
            tag = tf.stem.split("_", 1)[1]
            tok = np.load(tf, mmap_mode="r")
            idxf = d / f"idx_{tag}.npy"
            idx = np.load(idxf, mmap_mode="r") if idxf.exists() else None
            rows = np.load(d / f"rows_{tag}.npy")
            assert len(rows) == tok.shape[0], (tf, len(rows), tok.shape)
            for j, r in enumerate(rows):
                self.where[int(r)] = (s, j)
            self.tok.append(tok)
            self.idx.append(idx)
        assert self.tok, f"no features under {d}"
        self.K = self.tok[0].shape[1]
        self.D = self.tok[0].shape[2]

    def to_device(self, device):
        """Load every shard once into one device tensor (removes the CPU read bottleneck)."""
        order = sorted(self.where)
        self.lut = torch.full((max(order) + 1,), -1, dtype=torch.long)
        self.lut[torch.tensor(order)] = torch.arange(len(order))
        toks, idxs = [], []
        for r in order:
            s, j = self.where[r]
            toks.append(np.asarray(self.tok[s][j]))
            if self.idx[s] is not None:
                idxs.append(np.asarray(self.idx[s][j]))
        self.gtok = torch.from_numpy(np.stack(toks)).to(device)
        self.gidx = torch.from_numpy(np.stack(idxs).astype(np.int64)).to(device) if idxs else None
        self.lut = self.lut.to(device)
        return self

    def batch_compact(self, rows, device):
        """Compressed tokens [B,K,D] and representative index [B,4096] (None for dense)."""
        p = self.lut[torch.as_tensor(np.asarray(rows), device=device)]
        return self.gtok[p], (self.gidx[p] if self.gidx is not None else None)

    def batch(self, rows, device):
        if getattr(self, "gtok", None) is not None:
            p = self.lut[torch.as_tensor(np.asarray(rows), device=device)]
            tok = self.gtok[p]
            if self.gidx is not None:
                tok = tok.gather(1, self.gidx[p].unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
            return tok
        toks, idxs = [], []
        for r in rows:
            s, j = self.where[int(r)]
            toks.append(torch.from_numpy(np.asarray(self.tok[s][j])))
            if self.idx[s] is not None:
                idxs.append(torch.from_numpy(np.asarray(self.idx[s][j]).astype(np.int64)))
        tok = torch.stack(toks).to(device, non_blocking=True)                  # [B,K,D] fp16
        if idxs:
            idx = torch.stack(idxs).to(device, non_blocking=True)              # [B,4096]
            tok = tok.gather(1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        assert tok.shape[1] == TOKENS_PAST
        return tok


class MixedBank:
    """Per-clip budget: clip r reads its features from cfgs[assign[r]] (restore readout)."""

    def __init__(self, policy_path):
        pz = np.load(policy_path, allow_pickle=True)
        self.cfgs = [str(c) for c in pz["cfgs"]]
        self.assign = pz["assign"].astype(np.int64)
        self.banks = [FeatureBank(c) for c in self.cfgs]
        self.Ks = np.array([b.K for b in self.banks])
        self.D = self.banks[0].D
        common = set(self.banks[0].where)
        for b in self.banks[1:]:
            common &= set(b.where)
        self.where = {r: None for r in common}

    def mean_K(self, rows):
        return self.Ks[self.assign[np.asarray(rows)]].mean()

    def to_device(self, device):
        for b in self.banks:
            b.to_device(device)
        return self

    def batch(self, rows, device):
        rows = np.asarray(rows)
        out = None
        for j, b in enumerate(self.banks):
            sel = np.flatnonzero(self.assign[rows] == j)
            if len(sel) == 0:
                continue
            t = b.batch(rows[sel], device)
            if out is None:
                out = torch.empty(len(rows), *t.shape[1:], dtype=t.dtype, device=device)
            out[torch.as_tensor(sel, device=device)] = t
        return out


# ----------------------------------------------------------------------------- model
class XBlock(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.h = heads
        self.nq, self.nkv, self.ns = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.sqkv, self.so = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.nm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def _attn(self, q, k, v, bias=None):
        B, Nq, d = q.shape
        h, dh = self.h, d // self.h
        q = q.view(B, Nq, h, dh).transpose(1, 2)
        k = k.view(B, -1, h, dh).transpose(1, 2)
        v = v.view(B, -1, h, dh).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)) * dh ** -0.5                     # explicit math path
        if bias is not None:                                           # [B, Nkv] per-key logit bias
            a = a + bias[:, None, None, :].to(a.dtype)
        o = a.softmax(-1) @ v
        return o.transpose(1, 2).reshape(B, Nq, d)

    def forward(self, q, x, bias=None):
        k, v = self.kv(self.nkv(x)).chunk(2, -1)
        q = q + self.o(self._attn(self.q(self.nq(q)), k, v, bias))
        sq, sk, sv = self.sqkv(self.ns(q)).chunk(3, -1)
        q = q + self.so(self._attn(sq, sk, sv))
        return q + self.mlp(self.nm(q))


def fourier(x, n=8):
    """x [..., c] in ~[0,1] -> [..., 2*n*c] sin/cos features."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class CoordEmbed(nn.Module):
    """Continuous version of the published grid embedding.

    Position: the probe's own per-axis tables (pt, ph, pw; 16 entries each) linearly
    interpolated at the continuous (t, y, x) of a token, so global cells at their centres get
    exactly the grid values. Extra cues (log cell width/height, log cluster size, branch) enter
    through ZERO-initialised layers, so at initialisation the read-out equals the grid read-out.
    """

    def __init__(self, d, n_branch=3):
        super().__init__()
        self.extra = nn.Linear(3, d)
        nn.init.zeros_(self.extra.weight)
        nn.init.zeros_(self.extra.bias)
        self.branch = nn.Embedding(n_branch, d)
        nn.init.zeros_(self.branch.weight)
        self.ref = torch.tensor([math.log(1 / 16), math.log(1 / 16), 0.0])   # global cell values

    @staticmethod
    def interp(table, c):
        """table [16, d]; c [...] in [0,1] -> [..., d] (linear between cell centres)."""
        u = (c * 16 - 0.5).clamp(0, 15)
        lo = u.floor().long().clamp(0, 15)
        hi = (lo + 1).clamp(max=15)
        w = (u - lo.float())[..., None]
        return table[lo] * (1 - w) + table[hi] * w

    def forward(self, txy, logs, branch, pt, ph, pw):
        pos = (self.interp(pt.reshape(16, -1), txy[..., 0]) + self.interp(ph.reshape(16, -1), txy[..., 2])
               + self.interp(pw.reshape(16, -1), txy[..., 1]))
        return pos + self.extra(logs - self.ref.to(logs.device)) + self.branch(branch)


class ForecastProbe(nn.Module):
    def __init__(self, use_vis, use_pose, d=256, heads=8, depth=3, J=52, T=32, D=1024, coords=False):
        super().__init__()
        self.use_vis, self.use_pose, self.J, self.T = use_vis, use_pose, J, T
        self.queries = nn.Parameter(torch.randn(1, T, d) * 0.02)
        if use_vis:
            self.vin = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
            self.pt = nn.Parameter(torch.zeros(1, 16, 1, 1, d))       # token order is t, h, w
            self.ph = nn.Parameter(torch.zeros(1, 1, 16, 1, d))
            self.pw = nn.Parameter(torch.zeros(1, 1, 1, 16, d))
            for p in (self.pt, self.ph, self.pw):
                nn.init.trunc_normal_(p, std=0.02)
        if use_pose:
            self.pin = nn.Sequential(nn.Linear(J * 3, d), nn.GELU(), nn.Linear(d, d))
            self.ppos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.blocks = nn.ModuleList(XBlock(d, heads) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, J * 3))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        # created last and only on request, so the published configurations consume the
        # random stream exactly as before (bit-identical restore runs)
        self.coord = CoordEmbed(d) if (use_vis and coords) else None

    def forward_coords(self, groups, past):
        """groups: list of (tokens [B,K,D], coords [B,K,6], branch id, mask [B] or None);
        coords = (t, x, y, log w, log h, log size). Tokens of a masked group get -inf bias."""
        B = past.shape[0]
        kv, bias = [], []
        for tok, co, br, m in groups:
            x = self.vin(tok.float()) + self.coord(co[..., :3], co[..., 3:],
                                                   torch.full(co.shape[:2], br, device=co.device, dtype=torch.long),
                                                   self.pt, self.ph, self.pw)
            kv.append(x)
            b = torch.zeros(B, x.shape[1], device=x.device)
            if m is not None:
                b = b.masked_fill(~m[:, None], float("-inf"))
            bias.append(b)
        if self.use_pose:
            kv.append(self.pin(past.flatten(2)) + self.ppos)
            bias.append(torch.zeros(B, self.T, device=past.device))
        x = torch.cat(kv, 1)
        b = torch.cat(bias, 1)
        b = b if bool((b != 0).any()) else None
        q = self.queries.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, x, b)
        return self.out(q).view(B, self.T, self.J, 3)

    def forward(self, vis, past, idx=None, compact=False, logsize=False, key_bias=None):
        """restore path: vis [B,4096,D] on the dense grid. compact path: vis [B,K,D] merged
        tokens; each token's position embedding is the mean over the cells it represents
        (idx [B,4096] -> token row, None = identity), optional log(cluster size) key bias."""
        B = past.shape[0]
        kv, bias = [], []
        if self.use_vis:
            if not compact or idx is None:
                # published arithmetic, kept bit-for-bit: ((x + pt) + ph) + pw on the grid.
                # (dense in compact mode is the same computation: every token has size 1.)
                x = self.vin(vis.float()).view(B, 16, 16, 16, -1)
                x = x + self.pt + self.ph + self.pw
                kv.append(x.reshape(B, TOKENS_PAST, -1))
                bias.append(key_bias.float() if key_bias is not None
                            else torch.zeros(B, TOKENS_PAST, device=x.device))
            else:
                grid = (self.pt + self.ph + self.pw).reshape(1, TOKENS_PAST, -1)   # [1,4096,d]
                xc = self.vin(vis.float())                                          # [B,K,d]
                K = xc.shape[1]
                M = F.one_hot(idx, K).float()                                       # [B,4096,K]
                size = M.sum(1)                                                     # [B,K]
                pos = torch.einsum("bik,id->bkd", M, grid[0].float()) / size.clamp_min(1)[..., None]
                kv.append(xc + pos.to(xc.dtype))
                bias.append(size.clamp_min(1).log() if logsize else torch.zeros_like(size))
        if self.use_pose:
            kv.append(self.pin(past.flatten(2)) + self.ppos)
            bias.append(torch.zeros(B, self.T, device=past.device))
        x = torch.cat(kv, 1)
        b = torch.cat(bias, 1)
        b = b if bool((b != 0).any()) else None
        q = self.queries.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, x, b)
        return self.out(q).view(B, self.T, self.J, 3)


def cell_coords_global(device):
    """[4096, 5]: t, x, y (frame-normalised centres of the 16x16x16 cells), log w, log h."""
    t, i, j = torch.meshgrid(torch.arange(16), torch.arange(16), torch.arange(16), indexing="ij")
    c = torch.stack([(t + 0.5) / 16, (j + 0.5) / 16, (i + 0.5) / 16,
                     torch.full_like(t, 1 / 16, dtype=torch.float).log(),
                     torch.full_like(t, 1 / 16, dtype=torch.float).log()], -1).float()
    return c.reshape(-1, 5).to(device)


def cell_coords_crop(boxes, n, device):
    """boxes [B,16,3] (x0, y0, side) in 1920x1080 px -> [B, 16*n*n, 5] cell coordinates."""
    b = boxes.to(device).float()
    t, i, j = torch.meshgrid(torch.arange(16, device=device), torch.arange(n, device=device),
                             torch.arange(n, device=device), indexing="ij")
    x0, y0, s = b[:, t, 0], b[:, t, 1], b[:, t, 2]                     # [B,16,n,n]
    x = (x0 + (j + 0.5) / n * s) / 1920.0
    y = (y0 + (i + 0.5) / n * s) / 1080.0
    lw, lh = (s / n / 1920.0).log(), (s / n / 1080.0).log()
    tt = ((t + 0.5) / 16).float().expand_as(x)
    return torch.stack([tt, x, y, lw, lh], -1).reshape(b.shape[0], -1, 5)


def pool_coords(cells, idx, K):
    """cells [B or 1, C, 5], idx [B, C] -> token coords [B, K, 6] (mean over members + log size)."""
    B = idx.shape[0]
    M = F.one_hot(idx, K).float()                                       # [B,C,K]
    size = M.sum(1).clamp_min(1)                                        # [B,K]
    c = torch.einsum("bck,bcf->bkf", M, cells.expand(B, -1, -1).float()) / size[..., None]
    return torch.cat([c, size.log()[..., None]], -1)


# ----------------------------------------------------------------------------- train/eval
def set_determinism(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def errors(pred, fut):
    """Per-clip ADE / FDE / wrist ADE in mm, and per-horizon ADE."""
    e = torch.linalg.norm(pred - fut, dim=-1) * 1000.0                 # [B,T,J]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="feature config dir under FEAT_ROOT (e.g. dense, wam_L9)")
    ap.add_argument("--input", choices=["vis", "pose", "vispose", "zero"], default="vis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--val", action="store_true", help="hold out 10%% of train and report it instead of test")
    ap.add_argument("--rows_from", default="dense",
                    help="feature config whose clip set defines train/eval rows for ALL inputs")
    ap.add_argument("--save_model", action="store_true", help="also save the trained probe (model.pt)")
    ap.add_argument("--readout", choices=["restore", "compact", "compact_logsize"], default="restore",
                    help="restore: tokens[idx] on the dense grid (published protocol); compact: read "
                         "the K merged tokens directly with member-averaged positions; "
                         "compact_logsize: compact + ToMe proportional attention (log size key bias)")
    ap.add_argument("--coords", action="store_true",
                    help="continuous (t,x,y,scale,size,branch) position encoding on merged tokens "
                         "(compact read-out); required for --crops")
    ap.add_argument("--crops", default="", help="add tracked hand-crop tokens: raw | d256")
    ap.add_argument("--crop_sched", default="c11", help="crop feature schedule: dense | c11 | c9")
    ap.add_argument("--crop_S", type=int, default=128)
    ap.add_argument("--mix_policy", default="",
                    help="with --cfg mix: .npz with 'cfgs' (list of feature configs) and 'assign' "
                         "[N rows] index into cfgs; each clip reads its own budget (restore readout)")
    ap.add_argument("--cv_fold", type=int, default=-1,
                    help="A2b replication: evaluate fold f of a fixed 5-fold split of the train clips "
                         "that are NOT in the --val hold-out; train on all other train clips")
    ap.add_argument("--pose_frames", type=int, default=32,
                    help="Phase B: keep only the last k past pose frames (earlier frames are set to "
                         "the oldest kept frame, i.e. no motion information older than k frames)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_determinism(a.seed)
    dev = torch.device("cuda")

    P = np.load(FEAT_ROOT / "poses.npz")
    past = torch.from_numpy(P["past"]).to(dev)
    if a.pose_frames < past.shape[1]:
        k = a.pose_frames
        T0 = past.shape[1]
        past = torch.cat([past[:, T0 - k:T0 - k + 1].expand(-1, T0 - k, -1, -1), past[:, T0 - k:]], 1)
    fut = torch.from_numpy(P["future"]).to(dev)
    is_test = P["is_test"]
    ok = P["avail"].copy() if "avail" in P.files else np.ones(len(is_test), bool)
    use_vis = a.input in ("vis", "vispose", "zero")
    use_pose = a.input in ("pose", "vispose")
    if a.cfg == "mix":
        assert a.readout == "restore" and a.input in ("vis", "vispose")
        bank = MixedBank(a.mix_policy)
    else:
        bank = FeatureBank(a.cfg) if a.input in ("vis", "vispose") else None
    if a.rows_from:                              # restrict every arm to the same clip set
        ok &= np.isin(np.arange(len(ok)), list(FeatureBank(a.rows_from).where))
    if bank is not None:
        ok &= np.isin(np.arange(len(ok)), list(bank.where))
    train_rows = np.flatnonzero(~is_test & ok)
    eval_rows = np.flatnonzero(is_test & ok)
    if a.val:                                   # fixed hold-out, independent of --seed
        perm = np.random.RandomState(12345).permutation(train_rows)
        n_val = len(perm) // 10
        eval_rows, train_rows = np.sort(perm[:n_val]), np.sort(perm[n_val:])

    if a.cv_fold >= 0:                          # never touches test or the selection hold-out
        assert not a.val
        perm = np.random.RandomState(12345).permutation(train_rows)
        pool = np.sort(perm[len(perm) // 10:])
        folds = np.array_split(np.random.RandomState(777).permutation(pool), 5)
        eval_rows = np.sort(folds[a.cv_fold])
        train_rows = np.setdiff1d(train_rows, eval_rows)
    if bank is not None:
        bank.to_device(dev)
    # Normalisation from training targets only.
    mu = fut[train_rows].mean(dim=(0, 1), keepdim=True)[0]               # [1,J,3]
    sd = fut[train_rows].std()                                           # scalar
    base_last = a.input in ("pose", "vispose")                           # predict residual to last pose

    D = int(bank.D) if bank is not None else 1024
    model = ForecastProbe(use_vis, use_pose, d=a.dim, depth=a.depth, D=D, coords=a.coords).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(train_rows) / a.bs)
    total = a.epochs * steps_per_epoch
    warm = 2 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    g = torch.Generator().manual_seed(a.seed)

    compact = a.readout != "restore"
    crop_banks, crop_boxes, crop_ok = [], None, None
    if a.coords:
        assert bank is not None and not isinstance(bank, MixedBank)
        gcells = cell_coords_global(dev)
        if a.crops:
            cd = FEAT_ROOT / f"crops_S{a.crop_S}"
            crop_boxes = torch.from_numpy(np.load(cd / "boxes.npy")).float().to(dev)   # [N,2,16,3]
            crop_ok = torch.from_numpy(np.load(cd / "hand_ok.npy")).to(dev)             # [N,2]
            for h in (0, 1):
                cb = FeatureBank(f"crop_{a.crops}_h{h}_{a.crop_sched}").to_device(dev)
                crop_banks.append(cb)
            ncrop = a.crop_S // 16

    def visual_groups(rows):
        rows_t = torch.as_tensor(np.asarray(rows), device=dev)
        def unmerged(cells, n):                                   # size-1 tokens: cells as-is
            c = cells.expand(n, -1, -1)
            return torch.cat([c, torch.zeros_like(c[..., :1])], -1)
        tok, idx = bank.batch_compact(rows, dev)
        co = unmerged(gcells[None], len(rows)) if idx is None else pool_coords(gcells[None], idx, tok.shape[1])
        groups = [(tok, co, 0, None)]
        for h, cb in enumerate(crop_banks):
            ct, ci = cb.batch_compact(rows, dev)
            cells = cell_coords_crop(crop_boxes[rows_t, h], ncrop, dev)
            cco = unmerged(cells, len(rows)) if ci is None else pool_coords(cells, ci, ct.shape[1])
            groups.append((ct, cco, 1 + h, crop_ok[rows_t, h]))
        return groups

    def visual(rows):
        if bank is not None:
            if compact:
                return bank.batch_compact(rows, dev)
            return bank.batch(rows, dev), None
        if a.input == "zero":
            return torch.zeros(len(rows), TOKENS_PAST, 1024, device=dev, dtype=torch.float16), None
        return None, None

    def predict(rows):
        pst = past[rows]
        base = pst[:, -1:] if base_last else mu.unsqueeze(0)
        if a.coords:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model.forward_coords(visual_groups(rows), (pst - pst[:, -1:]) / sd)
            return base + y.float() * sd
        v, idx = visual(rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(v, (pst - pst[:, -1:]) / sd, idx=idx, compact=compact and bank is not None,
                      logsize=a.readout == "compact_logsize")
        return base + y.float() * sd

    t0 = time.time()
    log = []
    for ep in range(a.epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows), generator=g).numpy()]
        tot = 0.0
        for i in range(0, len(order), a.bs):
            rows = order[i:i + a.bs]
            loss = F.mse_loss(predict(rows), fut[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(rows)
        log.append(dict(epoch=ep + 1, train_mse=tot / len(order)))

    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(eval_rows), 50):
            rows = eval_rows[i:i + 50]
            errs.append(errors(predict(rows), fut[rows]).cpu())
    e = torch.cat(errs)                                                   # [N,T,J] mm
    wr = [24, 50]                                                         # rightHand / leftHand in QUERY_TFS
    perclip = e.mean(dim=(1, 2)).numpy()
    m = dict(cfg=a.cfg if a.cfg != "mix" else f"mix:{Path(a.mix_policy).stem}", input=a.input,
             seed=a.seed, readout=a.readout, coords=a.coords, crops=a.crops, crop_sched=a.crop_sched,
             crop_K=[cb.K for cb in crop_banks],
             K=(float(bank.mean_K(eval_rows)) if isinstance(bank, MixedBank) else (bank.K if bank else 0)),
             split="val" if a.val else (f"cv{a.cv_fold}" if a.cv_fold >= 0 else "test"), n_eval=len(eval_rows), n_train=len(train_rows),
             ade_mm=float(e.mean()), fde_mm=float(e[:, -1].mean()),
             wrist_ade_mm=float(e[:, :, wr].mean()), wrist_fde_mm=float(e[:, -1, wr].mean()),
             ade_per_step=e.mean(dim=(0, 2)).tolist(), train_log=log,
             seconds=time.time() - t0, args=vars(a))
    if a.save_model:
        torch.save(dict(state=model.state_dict(), mu=mu.cpu(), sd=sd.cpu(), D=D, dim=a.dim,
                        depth=a.depth, input=a.input), out / "model.pt")
    write_json(m, out / "metrics.json")
    np.save(out / "perclip_ade.npy", perclip)
    np.save(out / "perclip_fde.npy", e[:, -1].mean(dim=1).numpy())
    np.save(out / "eval_rows.npy", np.asarray(eval_rows, np.int32))
    np.save(out / "perclip_step.npy", e.mean(dim=2).numpy().astype(np.float16))   # [N,32] mm
    print(f"{a.cfg:>12s} {a.input:>7s} s{a.seed}: ADE {m['ade_mm']:.2f}  FDE {m['fde_mm']:.2f}  "
          f"wristADE {m['wrist_ade_mm']:.2f} mm  ({m['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
