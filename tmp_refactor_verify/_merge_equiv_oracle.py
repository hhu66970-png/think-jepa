"""Deterministic behavior oracle for token-merge refactor verification.

Runs A/B/C (base LocalTokenMerger) on fixed-seed dense tokens and prints a JSON
fingerprint per strategy. Compare before/after refactor: A/B/C must be
byte-identical. (B2/C2 were removed 2026-05-31 as confirmed No-Go.)
"""
import hashlib
import json
import sys

import torch

from src.models.utils.token_merge import (
    LocalTokenMerger,
    init_token_merge_state,
    normalize_merge_config,
)

try:
    from src.models.utils.token_merge_diagnostics import DiagnosticTokenMerger
    HAVE_DIAG = True
except Exception:
    DiagnosticTokenMerger = None
    HAVE_DIAG = False


def _h(t):
    t = t.detach().float().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def fingerprint(x_new, ids_new, size_new, rep_new, info):
    return {
        "shape": list(x_new.shape),
        "num_after": int(x_new.shape[1]),
        "x_sum": round(float(x_new.float().sum().item()), 5),
        "x_hash": _h(x_new),
        "ids_hash": _h(ids_new.float()),
        "size_sum": round(float(size_new.float().sum().item()), 5),
        "rep_hash": _h(rep_new.float()),
        "method": info.get("method"),
        "num_accepted": info.get("num_accepted"),
        "mean_sel_sim": (
            round(info["mean_selected_similarity"], 6)
            if info.get("mean_selected_similarity") is not None else None
        ),
    }


def run_one(name, cfg_dict, merger_cls, seed=0):
    torch.manual_seed(seed)
    t, h, w, D = 4, 8, 8, 32
    N = t * h * w
    x = torch.randn(2, N, D)
    ids, size, rep = init_token_merge_state(2, N, x.device, x.dtype)
    cfg = normalize_merge_config(cfg_dict)
    merger = merger_cls(cfg)
    out = merger(x, ids, size, rep, t, h, w)
    x_new, ids_new, size_new, rep_new, info = out
    return fingerprint(x_new, ids_new, size_new, rep_new, info)


BASE = dict(enabled=True, merge_layers=(8,), merge_ratio=0.125)
CASES = [
    ("A", {**BASE, "strategy": "local_2x2_same_time_vec", "importance_source": "none"}, "base"),
    ("B", {**BASE, "strategy": "local_2x2_importance_protected_vec",
            "importance_source": "norm", "protect_mode": "local_top1"}, "base"),
    ("C", {**BASE, "strategy": "local_2x2_hybrid_score_vec", "importance_source": "norm"}, "base"),
]

result = {"have_diagnostic_module": HAVE_DIAG}
for name, cfg_dict, kind in CASES:
    if kind == "diag" and HAVE_DIAG:
        cls = DiagnosticTokenMerger
    else:
        cls = LocalTokenMerger
    try:
        result[name] = {"merger": cls.__name__, **run_one(name, cfg_dict, cls)}
    except Exception as e:
        result[name] = {"merger": cls.__name__, "ERROR": f"{type(e).__name__}: {e}"}

print(json.dumps(result, indent=2, ensure_ascii=False))
