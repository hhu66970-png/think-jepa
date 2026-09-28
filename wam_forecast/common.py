"""Shared definitions for the WAM forecasting protocol (CVPR 2027).

Protocol (fixes the AAAI-version leakage):
  * the encoder sees ONLY the observed past: video frames [0, 32) -> 16x16x16 = 4096 tokens;
  * targets are the 52 hand joints of frames [32, 64), expressed in the camera frame of the
    last observed frame (t = 31), so no future camera pose is used anywhere;
  * the encoder is frozen and run once per (method, schedule); downstream heads are trained
    on the cached (restored-dense) tokens with a fixed seed list, so every method sees the
    same clips, the same seeds and the same head initialisation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

W = Path(os.environ.get("WAM_ROOT", "/21231_data1/huhaoming_wam"))
REPO = W / "ThinkJEPA"
SLIM = W / "data" / "egodex_slim" / "part2"
SPLITS = W / "tmp"                        # train_cache.txt / test_cache.txt (official 1800/200)
FEAT_ROOT = Path(os.environ.get("WAM_FEAT_ROOT", W / "feats"))
RUN_ROOT = W / "runs"
VITL = REPO / "vjepa2" / "vitl.pt"
VITG = REPO / "vjepa2" / "vitg.pt"          # official V-JEPA 2 ViT-g/16 (40 blocks, d=1408)

T_ALL, T_PAST = 64, 32
TOKENS_PAST = (T_PAST // 2) * 16 * 16     # 4096

# Merge schedules: (layers, per-layer ratio). Token counts are measured, not assumed.
SCHEDULES = {
    "s15": ("12,14,16,18,20", 0.15),
    "s20": ("12,14,16,18,20", 0.20),
    "s25": ("12,14,16,18,20", 0.25),
    "L9": ("12,13,14,15,16,17,18,19,20", 0.25),
    "L12": ("10,11,12,13,14,15,16,17,18,19,20,21", 0.25),
    # ViT-g (40 blocks): same number of merges at the same relative depth (50-90 %)
    "gs25": ("20,24,28,32,36", 0.25),
    "gL9": ("20,22,24,26,28,30,32,34,36", 0.25),
    # direction A: global path at ~232 tokens, 128-px hand crops (1024 tokens) at ~43 / ~77
    "L10g": ("12,13,14,15,16,17,18,19,20,21", 0.25),
    "c11": ("11,12,13,14,15,16,17,18,19,20,21", 0.25),
    "c9": ("12,13,14,15,16,17,18,19,20", 0.25),
}

# Methods -> merge-config overrides. None = dense (no merging).
METHODS = {
    "dense": None,
    "kbsm": dict(strategy="bsm_ksim_gradual_vec"),
    "pitome": dict(strategy="bsm_pitome_gradual_vec"),
    "wam": dict(strategy="bsm_taware_gradual_vec", relevance_source="motion", relevance_lambda=1.0),
}
_WAM = METHODS["wam"]
# WAM-v2 single-knob ablations (each changes exactly one thing relative to "wam").
METHODS.update({
    "wamrank": dict(_WAM, relevance_norm="rank"),                   # M1
    "wamff": dict(_WAM, relevance_first_frame="ffill"),             # M3
    "wampmax": dict(_WAM, relevance_prop="max"),                    # M4
    "wampmean": dict(_WAM, relevance_prop="mean"),                  # M4
    "wamcomp": dict(_WAM, relevance_comp="median"),                 # E1
    "wamss": dict(_WAM, relevance_gate="signsafe"),                 # M2
    "wamsrc": dict(_WAM, relevance_gate="source"),                  # gate form
    "wamadd": dict(_WAM, relevance_gate="add"),                     # gate form
    "wamq10": dict(_WAM, relevance_gate="quota", relevance_quota=0.10),   # E4
    "randmax": dict(_WAM, relevance_source="random_inplace"),       # control for wam
    "randrank": dict(_WAM, relevance_source="random_inplace", relevance_norm="rank"),
    # E2: observed-past hand prior (legal for forecasting: uses frames 0..31 only)
    "handmult": dict(_WAM, relevance_source="external"),
    "handq10": dict(_WAM, relevance_source="external", relevance_gate="quota", relevance_quota=0.10),
    "handq20": dict(_WAM, relevance_source="external", relevance_gate="quota", relevance_quota=0.20),
    "handsrc": dict(_WAM, relevance_source="external", relevance_gate="source"),
})
# Order-preserving protection (keeps K-BSM's similarity order: relevance_lambda = 0).
_KEEP = dict(_WAM, relevance_lambda=0.0)
METHODS.update({
    # P1: densest hand cells (quota q of current tokens) can absorb but are never removed
    "hrecv15": dict(_KEEP, relevance_source="external", external_map="density",
                    relevance_partition="recv", relevance_quota=0.15),
    "hrecv30": dict(_KEEP, relevance_source="external", external_map="density",
                    relevance_partition="recv", relevance_quota=0.30),
    # P3: the same cells are left out of matching entirely (masked re-matching)
    "hexcl15": dict(_KEEP, relevance_source="external", external_map="density",
                    relevance_partition="excl", relevance_quota=0.15),
    "hexcl30": dict(_KEEP, relevance_source="external", external_map="density",
                    relevance_partition="excl", relevance_quota=0.30),
    # P2: anchored averaging, similarity order untouched
    "hanc1": dict(_KEEP, relevance_source="external", external_map="density", relevance_anchor=1.0),
    "hanc4": dict(_KEEP, relevance_source="external", external_map="density", relevance_anchor=4.0),
    "hrecvanc": dict(_KEEP, relevance_source="external", external_map="density",
                     relevance_partition="recv", relevance_quota=0.15, relevance_anchor=4.0),
    "mrecv15": dict(_KEEP, relevance_source="motion", relevance_partition="recv", relevance_quota=0.15),
    "manc4": dict(_KEEP, relevance_source="motion", relevance_anchor=4.0),
    "mrecv10": dict(_KEEP, relevance_source="motion", relevance_partition="recv", relevance_quota=0.10),
    "mrecv20": dict(_KEEP, relevance_source="motion", relevance_partition="recv", relevance_quota=0.20),
    "rrecv15": dict(_KEEP, relevance_source="random_inplace", relevance_partition="recv", relevance_quota=0.15),
    # content-free anchored partition: fixed lattice tokens are receiver-only
    "lat4": dict(_KEEP, relevance_source="external", external_map="lattice4",
                 relevance_partition="recv", relevance_quota=0.0625),
    "lat2": dict(_KEEP, relevance_source="external", external_map="lattice2",
                 relevance_partition="recv", relevance_quota=0.25),
    "kbsm0": dict(_KEEP),                    # sanity: must equal kbsm bit for bit
    # K-BSM with ToMe proportional attention in the encoder (baseline correction)
    "kbsmpa": dict(strategy="bsm_ksim_gradual_vec", prop_attn=True),
    # B1: learned matching embedding (distilled from dense final-feature similarity)
    "kbsmlm": dict(strategy="bsm_ksim_gradual_vec", bsm_match_metric="learned"),
})
# Every relevance variant again with proportional attention (reference: kbsmpa)
METHODS.update({
    "b2": dict(strategy="bsm_ksim_gradual_vec", bsm_match_metric="learned", matcher_file="b2_best_nopa.pt"),
    "b2pa": dict(strategy="bsm_ksim_gradual_vec", bsm_match_metric="learned", matcher_file="b2_best_pa.pt",
                 prop_attn=True),
})
METHODS.update({f"{m}pa": dict(METHODS[m], prop_attn=True)
                for m in ("wam", "handmult", "mrecv15", "rrecv15", "kbsmlm")})
METHODS.update({
    # P4: learned relevance (scorer on layer-12 hidden tokens, trained on train-split saliency)
    "lrecv15": dict(_KEEP, relevance_source="scorer", relevance_partition="recv", relevance_quota=0.15),
    "lrecv30": dict(_KEEP, relevance_source="scorer", relevance_partition="recv", relevance_quota=0.30),
    "lexcl15": dict(_KEEP, relevance_source="scorer", relevance_partition="excl", relevance_quota=0.15),
    "lmult": dict(_WAM, relevance_source="scorer"),
})

# Phase A (balanced group size): corrected baseline kbsmpa + ONE capacity constraint.
#   capc<c>pa  <= c sources per receiver per merge layer
#   caps<k>pa  group size after each layer <= k x mean group size of that layer
#   capinfpa   sanity: non-binding cap through the capacity code path (must match kbsmpa)
_KPA = METHODS["kbsmpa"]
METHODS.update({f"capc{c}pa": dict(_KPA, bsm_cap_count=c) for c in (1, 2, 4)})
METHODS.update({f"caps{k}pa": dict(_KPA, bsm_cap_size=float(k)) for k in (2, 4, 8)})
METHODS["capinfpa"] = dict(_KPA, bsm_cap_count=10 ** 6)
# A4b mechanism: the same cap WITHOUT proportional attention (reference: kbsm)
METHODS["caps2"] = dict(METHODS["kbsm"], bsm_cap_size=2.0)


def merge_config(method: str, sched: str):
    over = METHODS[method]
    if over is None:
        return None
    layers, ratio = SCHEDULES[sched]
    cfg = dict(enabled=True, merge_layers=layers, merge_ratio=ratio, receiver="max_norm",
               merge_axis="free", restore_dense=False, importance_source="none",
               protect_mode="none", bsm_match_metric="key")
    cfg.update(over)
    return cfg


def cfg_dir(name: str, arch: str) -> str:
    return name if arch == "vitl" else f"{arch}_{name}"


def cfg_name(method: str, sched: str) -> str:
    return "dense" if METHODS[method] is None else f"{method}_{sched}"


def read_split(name: str) -> list[tuple[str, str]]:
    """Official split -> list of (task, file) keys, order preserved."""
    out = []
    for line in open(SPLITS / f"{name}_cache.txt"):
        line = line.strip()
        if line:
            p = Path(line)
            out.append((p.parent.name, p.name))
    return out


def all_clips() -> list[tuple[str, str]]:
    return read_split("train") + read_split("test")


def clip_path(key: tuple[str, str]) -> Path:
    return SLIM / key[0] / key[1]


def key_str(key) -> str:
    return f"{key[0]}/{key[1]}"


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)
