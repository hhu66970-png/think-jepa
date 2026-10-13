"""Unit test for round T (vid-TLDR port). Asserts that
(1) vidtldr_objective equals the official get_objective_score (copied verbatim below) on random attention maps;
(2) with all saliencies 1 the vid-TLDR merge equals the plain feature metric merge (same matching, size weighting);
(3) the merge conserves the total size of the kept tokens plus the saliency-weighted sources, and background
    sources (saliency 0) leave their receivers unchanged.
Exits nonzero on any failure.  python test_vidtldr.py"""
import os
import sys
from pathlib import Path

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


def official_get_objective_score(score_attn):      # verbatim from vid-TLDR multi_modality/utils/vid_TLDR.py
    score_attn = score_attn.mean(dim=1)
    scores = (score_attn * torch.log(score_attn)).sum(dim=2).unsqueeze(-1)
    B, T_R, _ = scores.shape
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True)
    score_mask = scores < scores.mean(dim=1, keepdim=True)
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True)
    scores[score_mask] = 0.0
    return scores


torch.manual_seed(0)
B, H, N, D = 2, 4, 64, 32
attn = torch.randn(B, H, N, N).mul(3).softmax(-1).double()
ref = official_get_objective_score(attn.clone())[..., 0]
p_mean = attn.mean(dim=1)
mine = TMD.vidtldr_objective(torch.special.xlogy(p_mean, p_mean).sum(-1))
check("(1) saliency equals official get_objective_score", torch.allclose(ref, mine, atol=1e-10),
      f"max|diff| = {float((ref - mine).abs().max()):.1e}")


def run(cfg, x, ent):
    m = TMD.DiagnosticTokenMerger(normalize_merge_config(cfg))
    m._attn_entropy = ent
    n = x.shape[1]
    ids = torch.arange(n).unsqueeze(0).expand(x.shape[0], -1).contiguous()
    size = torch.ones(x.shape[0], n)
    rep = ids.clone()
    return m(x=x, token_ids=ids, token_size=size, rep_for_orig=rep, t_grid=4, h_grid=4, w_grid=4, attn_key=None)


x = torch.randn(B, N, D)
ent = torch.randn(B, N)
base = dict(merge_config("kbsmpa", "L9"), bsm_match_metric="feature")
tl = merge_config("vidtldrpa", "L9")
orig = TMD.vidtldr_objective
TMD.vidtldr_objective = lambda e: torch.ones_like(e)
xa, ia, sa, ra, _ = run(base, x, ent)
xb, ib, sb, rb, _ = run(tl, x, ent)
TMD.vidtldr_objective = orig
check("(2) saliency 1 reduces to the feature metric merge", torch.allclose(xa, xb) and torch.equal(ia, ib)
      and torch.allclose(sa, sb) and torch.equal(ra, rb), f"max|dx| = {float((xa - xb).abs().max()):.1e}")

xc, ic, sc, rc, _ = run(tl, x, ent)
so = TMD.vidtldr_objective(ent)
check("(3a) same matching and token count as the feature merge", torch.equal(ic, ia) and xc.shape == xa.shape,
      f"tokens {xc.shape[1]} of {N}")
dropped = torch.ones(B, N, dtype=torch.bool)
dropped.scatter_(1, ic, False)                                   # merged away sources
mass = sc.sum(1)
expect = (N - dropped.sum(1)).float() + (so * dropped).sum(1)    # kept tokens (size 1) + saliency of the sources
check("(3b) size mass = kept tokens + saliency weighted sources", torch.allclose(mass, expect, atol=1e-5),
      f"{mass.tolist()} vs {expect.tolist()}")
sys.exit(0 if ok else 1)
