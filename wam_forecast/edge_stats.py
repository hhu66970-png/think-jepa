"""Descriptive: how dissimilar are the pairs a gate merges? For 96 part 2 training clips, the mean cosine similarity
of the merged (source, receiver) pairs and the fraction with negative similarity, per merge layer, as reported by the
merger. python edge_stats.py -> runs/phaseA/edge_stats.json"""
import json

import numpy as np
import torch

import extract as E
import src.models.utils.token_merge_diagnostics as TMD
from common import FEAT_ROOT, T_PAST, W, all_clips, clip_path, merge_config, write_json

CFGS = ["kbsmpa_L9", "wampa_L9", "wamlam05pa_L9", "wamlam025pa_L9", "wamshufpa_L9", "handmultpa_L9",
        "kbsmpa_L12", "handmultpa_L12", "handshuffpa_L12"]
keys = all_clips()
rows = list(range(0, 1800, 1800 // 96))[:96]
H = np.load(FEAT_ROOT / "hand_masks.npz")
ext = {k: torch.from_numpy(H[k].reshape(len(keys), -1).astype(np.float32)) for k in ("mask", "mask_cur")}
enc = E.load_encoder("vitl")
out = {}
for name in CFGS:
    cfg = merge_config(*name.split("_", 1))
    E.set_merge(enc, cfg)
    sims, negs = [], []
    for i in range(0, len(rows), 4):
        rb = rows[i:i + 4]
        u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[r]))["imgs"][:T_PAST]) for r in rb]).cuda()
        if cfg.get("relevance_source") == "external":
            TMD.set_external_relevance(ext[cfg.get("external_map", "mask")][rb].cuda())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _, infos = enc(E.preprocess(u8), return_merge_info=True, restore_dense=False)
        sims.append([i_["mean_selected_similarity"] for i_ in infos])
        negs.append([i_.get("neg_sim_selected_frac", np.nan) for i_ in infos])
    s, n = np.array(sims, float).mean(0), np.array(negs, float).mean(0)
    out[name] = dict(mean_sim_per_layer=s.tolist(), neg_frac_per_layer=n.tolist(), mean_sim=float(s.mean()), neg_frac=float(n.mean()))
    print(f"{name:16s} mean similarity of merged pairs {s.mean():.3f}  negative {100 * n.mean():.1f}%  first layer {s[0]:.3f}")
write_json(out, W / "runs" / "phaseA" / "edge_stats.json")
