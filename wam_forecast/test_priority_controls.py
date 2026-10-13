"""Priority-signal controls: shuffle keeps the multiset of scores, shuffle_frame keeps each frame's multiset,
invert maps r -> max r - r, none is the identity (WAM unchanged)."""
import os
import sys
from pathlib import Path
import torch
sys.path.insert(0, os.environ.get("VJEPA2_ROOT", str(Path(__file__).resolve().parents[1] / "vjepa2")))
from src.models.utils.token_merge import normalize_merge_config  # noqa: E402
from src.models.utils.token_merge_diagnostics import DiagnosticTokenMerger  # noqa: E402
torch.manual_seed(0)
T, HW = 16, 256
x = torch.randn(2, T * HW, 8)
rel = torch.rand(2, T * HW)
ok = True
for mode in ("none", "shuffle", "shuffle_frame", "invert"):
    m = DiagnosticTokenMerger(normalize_merge_config(dict(enabled=True, merge_layers="12", merge_ratio=0.25,
        strategy="bsm_taware_gradual_vec", relevance_transform=mode)))
    out = m._transform_relevance(rel, x, T, HW)
    if mode == "none":
        ok &= torch.equal(out, rel)
    elif mode == "shuffle":
        ok &= torch.equal(out.sort(1).values, rel.sort(1).values) and not torch.equal(out, rel)
    elif mode == "shuffle_frame":
        a, b = out.reshape(2, T, HW).sort(2).values, rel.reshape(2, T, HW).sort(2).values
        ok &= torch.equal(a, b) and not torch.equal(out, rel)
    else:
        ok &= torch.allclose(out, rel.amax(1, keepdim=True) - rel)
    again = m._transform_relevance(rel, x, T, HW)
    ok &= torch.equal(out, again)                     # deterministic per clip
    print(mode, "ok" if ok else "FAIL")
print("ALL OK:", ok)
sys.exit(0 if ok else 1)
