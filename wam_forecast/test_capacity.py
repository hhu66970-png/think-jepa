"""Sanity tests for capacity-constrained K-BSM (bsm_cap_count / bsm_cap_size).

C1 unit: with no binding cap the selected edge set equals K-BSM top-r (continuous scores).
C2 unit: with caps, exactly r distinct sources per sample, per-receiver count <= c and
   receiver size <= cap; every selected edge is a finite score.
C3 end-to-end (4 real clips): capinfpa == kbsmpa; caps2pa final max group <= 2 x mean;
   capc1pa has lower group-size Gini than kbsmpa.

  CUDA_VISIBLE_DEVICES=0 python test_capacity.py
"""
import sys

import numpy as np
import torch

import extract as E
from common import all_clips, clip_path, merge_config
from src.models.utils.token_merge_diagnostics import DiagnosticTokenMerger as TM

ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")


g = torch.Generator(device="cuda").manual_seed(0)
B, Na, Nb, r = 3, 512, 512, 256
scores = torch.rand(B, Na, Nb, device="cuda", generator=g) * 2 - 1
scores[0, :, :5] += 1.5          # make a few receivers very popular -> caps bind
sa = torch.randint(1, 6, (B, Na), device="cuda", generator=g).float()
sb = torch.randint(1, 6, (B, Nb), device="cuda", generator=g).float()

# C1
bs, bb = scores.max(2)
ref_a = bs.topk(r, 1).indices.sort(1).values
ea, eb, info = TM._capacity_select(scores, r, sa, sb, 10 ** 6, 0.0)
check("C1 no-cap == top-r", torch.equal(ea, ref_a) and torch.equal(eb, bb.gather(1, ref_a)),
      f"rounds={info['cap_rounds']} forced={info['cap_forced']}")

# C2
for c, cap in ((1, 0.0), (2, 0.0), (0, 12.0), (2, 12.0)):
    ea, eb, info = TM._capacity_select(scores, r, sa, sb, c, cap)
    uniq = all(len(set(ea[i].tolist())) == r for i in range(B))
    load = torch.zeros(B, Nb, device="cuda").scatter_add_(1, eb, torch.ones_like(eb, dtype=torch.float))
    mass = sb.clone().scatter_add_(1, eb, sa.gather(1, ea))
    touched = load > 0
    fin = bool(torch.isfinite(scores.gather(2, eb.unsqueeze(-1))).all())
    good = uniq and fin and (c == 0 or int(load.max()) <= c) and \
        (cap == 0 or float(mass[touched].max()) <= cap) and info["cap_forced"] == 0
    check(f"C2 caps count={c} size={cap}", good,
          f"max load {int(load.max())}, max recv size {float(mass[touched].max()):.0f}, "
          f"rounds {info['cap_rounds']}, forced {info['cap_forced']}")

# C3
enc = E.load_encoder()
keys = [k for k in all_clips() if clip_path(k).exists()][:4]
u8 = torch.stack([torch.from_numpy(np.load(clip_path(k))["imgs"][:32]) for k in keys]).cuda()
x = E.preprocess(u8)


def run(m, s="L9"):
    E.set_merge(enc, merge_config(m, s))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = enc(x, restore_dense=False).float()
    return out, enc.last_merge_state[1].float()


def gini(v):
    v = np.sort(v.astype(np.float64)); n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


o_k, s_k = run("kbsmpa")
o_i, s_i = run("capinfpa")
check("C3 capinfpa == kbsmpa", torch.equal(o_k, o_i) and torch.equal(s_k, s_i),
      f"max|diff| {float((o_k - o_i).abs().max()):.1e}")
for m in ("capc1pa", "capc2pa", "capc4pa", "caps2pa", "caps4pa", "caps8pa"):
    o, s = run(m)
    K = s.shape[1]
    gk = np.mean([gini(v) for v in s_k.cpu().numpy()]); gm = np.mean([gini(v) for v in s.cpu().numpy()])
    line = (f"K={K} (kbsmpa {s_k.shape[1]}), max group {float(s.max()):.0f} vs {float(s_k.max()):.0f}, "
            f"Gini {gm:.3f} vs {gk:.3f}")
    cond = K == s_k.shape[1] and float(s.sum(1).min()) == 4096.0
    if m.startswith("caps"):
        k = float(m[4:-2])
        cond &= float(s.max()) <= k * 4096 / K + 1e-3
    check(f"C3 {m}", cond, line)
print("ALL PASS" if ok else "SOME TESTS FAILED")
sys.exit(0 if ok else 1)
