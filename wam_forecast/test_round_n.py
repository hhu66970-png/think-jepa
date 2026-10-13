"""Unit test for round N (native vid-TLDR placement). Asserts that
(1) merge_counts removes exactly the requested tokens at each layer;
(2) merge_order=official returns the surviving A tokens first and then all B tokens, each in original order, and is a
    permutation of the keep order (same tokens, sizes, and representative map);
(3) on the real V-JEPA 2 ViT-L encoder, vidtldrnat merges at layers 0-3 between attention and MLP and leaves 309
    finite tokens, and vidtldrmid leaves 309 tokens with merges at layers 12-20.
Exits nonzero on any failure.  VJEPA2_ROOT=<patched checkout> python test_round_n.py"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("VJEPA2_ROOT", str(Path(__file__).resolve().parents[1] / "vjepa2")))
import src.models.utils.token_merge_diagnostics as TMD  # noqa: E402
from src.models.utils.token_merge import normalize_merge_config  # noqa: E402

from common import merge_config  # noqa: E402

ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


torch.manual_seed(0)
B, N, D = 2, 64, 16
x = torch.randn(B, N, D)
ent = torch.randn(B, N)


def run(cfg, block_index):
    m = TMD.DiagnosticTokenMerger(normalize_merge_config(cfg))
    m._attn_entropy = ent
    m._block_index = block_index
    ids = torch.arange(N).unsqueeze(0).expand(B, -1).contiguous()
    return m(x=x, token_ids=ids, token_size=torch.ones(B, N), rep_for_orig=ids.clone(), t_grid=4, h_grid=4, w_grid=4,
             attn_key=None)


keep = dict(merge_config("vidtldrpa", "L9"), merge_layers=[0, 1], merge_counts=[20, 5])
off = dict(keep, merge_order="official")
xk, ik, sk, rk, _ = run(keep, 0)
xo, io, so, ro, _ = run(off, 0)
check("(1) merge_counts removes 20 tokens at layer 0", xk.shape[1] == N - 20, f"{xk.shape[1]} of {N}")
_, i1, _, _, _ = run(keep, 1)
check("(1) merge_counts removes 5 tokens at layer 1", i1.shape[1] == N - 5, f"{i1.shape[1]} of {N}")
same_set = all(set(ik[b].tolist()) == set(io[b].tolist()) for b in range(B))
perm_ok = True
for b in range(B):
    pos = {int(t): j for j, t in enumerate(ik[b])}
    idx = torch.tensor([pos[int(t)] for t in io[b]])
    perm_ok &= torch.equal(xo[b], xk[b][idx]) and torch.equal(so[b], sk[b][idx])
check("(2) official order is a permutation of the keep order", same_set and perm_ok and torch.equal(ro, rk))
a_first = True
for b in range(B):
    ids = io[b].tolist()                     # original ids; A = even positions, B = odd positions of the input
    par = [t % 2 for t in ids]
    a_first &= par == sorted(par) and ids[:par.count(0)] == sorted(ids[:par.count(0)]) and ids[par.count(0):] == sorted(ids[par.count(0):])
check("(2) official order: surviving A tokens, then B tokens, each sorted", a_first)

if torch.cuda.is_available():
    import extract as E
    from common import all_clips, clip_path, T_PAST
    keys = all_clips()
    u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[r]))["imgs"][:T_PAST]) for r in (0, 1)]).cuda()
    enc = E.load_encoder("vitl")
    x = E.preprocess(u8)
    for name, layers, sizes in (("vidtldrnat_L9", [0, 1, 2, 3], [4096, 2049, 1025, 513]),
                                ("vidtldrmid_L9", list(range(12, 21)), None)):
        E.set_merge(enc, merge_config(*name.split("_", 1)))
        calls, mlp_len = [], {}
        f0 = enc.token_merger.forward
        enc.token_merger.forward = lambda *a, **k: (calls.append((int(enc.token_merger._block_index), int(k["x"].shape[1]))), f0(*a, **k))[1]
        hooks = [blk.mlp.register_forward_hook(lambda m, inp, out, j=j: mlp_len.__setitem__(j, int(inp[0].shape[1])))
                 for j, blk in enumerate(enc.blocks) if j in layers]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            tok = enc(x, restore_dense=False)
        for h in hooks:
            h.remove()
        enc.token_merger.forward = f0
        lay = [c[0] for c in calls]
        mid_ok = all(mlp_len[j] == c[1] - (c[1] - mlp_len[j]) and mlp_len[j] < c[1] for j, c in zip(layers, calls))
        check(f"(3) {name}: 309 finite tokens, merges at layers {layers[0]}..{layers[-1]}, MLP sees the merged tokens",
              tok.shape[1] == 309 and bool(torch.isfinite(tok).all()) and lay == layers and mid_ok
              and (sizes is None or [c[1] for c in calls] == sizes),
              f"tokens {tok.shape[1]}, merger inputs {[c[1] for c in calls]}, MLP inputs {[mlp_len[j] for j in layers]}")
else:
    check("(3) GPU available for the encoder check", False)
sys.exit(0 if ok else 1)
