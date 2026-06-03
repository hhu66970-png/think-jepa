#!/usr/bin/env python
"""CPU smoke for WAM (bsm_taware_gradual_vec): correctness + motion protection.

Synthetic video: all tokens share a base (high mutual cosine => mergeable); a few
"mover" spatial positions get a per-frame signal (high motion). K-BSM merges by
similarity only; WAM should PROTECT the movers (drop them last) => more movers
survive under WAM than under K-BSM, at the SAME token budget.
"""
import sys
import torch

sys.path.insert(0, "vjepa2/src")
from src.models.utils.token_merge import MergeConfig, init_token_merge_state  # noqa
from src.models.utils.token_merge_diagnostics import DiagnosticTokenMerger  # noqa

torch.manual_seed(0)
B, t, h, w, D = 2, 4, 6, 6, 48
N = t * h * w  # 144
MOVERS = [0, 7, 14, 21, 28]  # spatial indices (of h*w=36) that move over time


def make_video():
    base = torch.randn(B, 1, 1, D)
    spatial = base + 0.04 * torch.randn(B, 1, h * w, D)        # per-spatial, const over t
    x = spatial.expand(B, t, h * w, D).clone()                 # zero motion, all similar
    for s in MOVERS:
        # small per-frame perturbation: stays HIGH-cosine (mergeable) but nonzero
        # motion => K-BSM will merge these, WAM should protect them.
        x[:, :, s, :] = spatial[:, 0, s, :].unsqueeze(1) + 0.18 * torch.randn(B, t, D)
    return x.reshape(B, N, D)


def mover_orig_ids():
    ids = set()
    for f in range(t):
        for s in MOVERS:
            ids.add(f * (h * w) + s)
    return ids


def run(strategy, video, lam=1.0):
    cfg = MergeConfig(enabled=True, strategy=strategy, merge_layers=tuple(range(12, 21)),
                      merge_ratio=0.20, restore_dense=False, bsm_match_metric="feature",
                      relevance_source="motion", relevance_lambda=lam)
    m = DiagnosticTokenMerger(cfg)
    x = video.clone()
    tid, tsz, rep = init_token_merge_state(B, N, x.device, x.dtype)
    counts = [x.shape[1]]
    info = {}
    for _ in range(9):  # 9 gradual layers
        x, tid, tsz, rep, info = m(x, tid, tsz, rep, t, h, w, attn_key=None)
        counts.append(x.shape[1])
        assert torch.isfinite(x).all(), "NaN/Inf in merged features!"
    survived = set(tid[0].tolist())
    return counts, survived, info


movers = mover_orig_ids()
VIDEO = make_video()                                   # SAME video for all runs
c_k, s_k, _ = run("bsm_ksim_gradual_vec", VIDEO)
c_t, s_t, info_t = run("bsm_taware_gradual_vec", VIDEO, lam=1.0)
c_t0, s_t0, _ = run("bsm_taware_gradual_vec", VIDEO, lam=0.0)  # lam=0 must equal K-BSM

print(f"#movers (orig tokens) = {len(movers)}")
print(f"K-BSM   counts={c_k}  movers_survived={len(movers & s_k)}/{len(movers)}")
print(f"WAM l=1 counts={c_t}  movers_survived={len(movers & s_t)}/{len(movers)}")
print(f"WAM l=0 counts={c_t0} movers_survived={len(movers & s_t0)}/{len(movers)}  (should ~= K-BSM)")
print(f"WAM info: relevance_source={info_t.get('relevance_source')} "
      f"lambda={info_t.get('relevance_lambda')} metric={info_t.get('bsm_match_metric')}")
print(f"final token count equal (K-BSM vs WAM): {c_k[-1] == c_t[-1]}")
assert c_k[-1] == c_t[-1], "token budget must match across methods"
assert len(movers & s_t) >= len(movers & s_k), "WAM must protect movers >= K-BSM"
assert s_t0 == s_k, "WAM(lambda=0) must be identical to K-BSM"
print("\nSMOKE PASS: shapes ok, no NaN, WAM protects movers, lambda=0 == K-BSM.")
