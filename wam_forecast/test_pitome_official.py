"""Check the published-PiToMe partition against the official code (hchautran/PiToMe, algo/pitome/merge.py,
pitome_vision + pitome, copied below): same merge groups and same merged features."""
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, os.environ.get("VJEPA2_ROOT", str(Path(__file__).resolve().parents[1] / "vjepa2")))
from src.models.utils.token_merge import normalize_merge_config  # noqa: E402
from src.models.utils.token_merge_diagnostics import DiagnosticTokenMerger  # noqa: E402


def official(metric, r, margin):          # verbatim logic of pitome_vision(...) -> pitome(...)
    B, T, C = metric.shape
    metric = F.normalize(metric, p=2, dim=-1)
    sim = metric @ metric.transpose(-1, -2)
    energy_score = F.elu((sim - margin), alpha=1.0).mean(dim=-1)
    indices = torch.argsort(energy_score, descending=True, stable=True)
    merge_idx = indices[..., :2 * r]
    protected_idx = indices[..., 2 * r:]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]
    scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r))
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
    _, dst_idx = scores.max(dim=-1)
    return a_idx, b_idx.gather(1, dst_idx), protected_idx


torch.manual_seed(0)
cfg = normalize_merge_config(dict(enabled=True, merge_layers="12", merge_ratio=0.25, strategy="bsm_pitome_gradual_vec",
                                  pitome_official=True, bsm_match_metric="key"))
m = DiagnosticTokenMerger(cfg)
ok = True
for li in (12, 16, 20):
    m._block_index, m._num_blocks = li, 24
    B, N, D, H = 2, 256, 32, 64
    key = torch.randn(B, N, H)
    x = torch.randn(B, N, D)
    r = N // 4
    a_ref, recv_ref, prot_ref = official(key, r, 0.75 - 0.75 * li / 24)
    a, b, _ = m._pitome_official_partition(F.normalize(key, dim=-1), r)
    # full forward through the merger: sources removed = official A, receivers = official choices
    ids = torch.arange(N).unsqueeze(0).repeat(B, 1)
    size = torch.ones(B, N)
    rep = ids.clone()
    xn, idn, sn, repn, info = m._forward_bsm(x, ids, size, rep, 4, 8, 8, key)
    for bb in range(B):
        mapping = {int(s): int(repn[bb, s]) for s in a_ref[bb]}
        ref = {int(s): int(t) for s, t in zip(a_ref[bb], recv_ref[bb])}
        ok &= mapping == ref
        ok &= set(idn[bb].tolist()) == set(range(N)) - set(a_ref[bb].tolist())
        ok &= set(int(t) for t in prot_ref[bb]) <= set(idn[bb].tolist())
    ok &= bool(torch.equal(a, a_ref))
    print(f"block {li}: margin {0.75 - 0.75 * li / 24:.3f}, partition {info.get('bsm_partition')}, removed {N - xn.shape[1]} (r={r})")
print("MATCHES OFFICIAL PITOME:", ok)
