"""Descriptive (sixth review): key cosine similarity of the merged (source, receiver) pairs for every merger compared in
the paper's mechanism figure, on 96 part 2 training clips. Content free baselines report the key cosine of the pairs they
merge (WAM_REPORT_KEYS=1; their merging is unchanged).  WAM_REPORT_KEYS=1 python edge_stats3.py -> runs/phaseA/edge_stats3.json (gate arms at 131 tokens, seventh review)"""
import os

import numpy as np
import torch

import extract as E
import src.models.utils.token_merge_diagnostics as TMD
from common import FEAT_ROOT, T_PAST, W, all_clips, clip_path, merge_config, write_json

assert os.environ.get("WAM_REPORT_KEYS") == "1"
CFGS = ["kbsmpa_L12", "handcurpa_L12", "handcurshufpa_L12", "handcurq20pa_L12", "handcurq20shufpa_L12", "handcurq10pa_L12", "handcurq10shufpa_L12", "handcurrecvpa_L12", "handcurrecvshufpa_L12", "handfutpa_L12", "handfutshufpa_L12", "flowpa_L12", "flowshufpa_L12"]
keys = all_clips()
rows = list(range(0, 1800, 1800 // 96))[:96]
H = np.load(FEAT_ROOT / "hand_masks.npz")
ext = {k: torch.from_numpy(H[k].reshape(len(keys), -1).astype(np.float32)) for k in ("mask", "mask_cur", "mask_f31", "mask_fut", "flow")}
enc = E.load_encoder("vitl")
out = {}
for name in CFGS:
    cfg = merge_config(*name.split("_", 1))
    E.set_merge(enc, cfg)
    sims = []
    for i in range(0, len(rows), 4):
        rb = rows[i:i + 4]
        u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[r]))["imgs"][:T_PAST]) for r in rb]).cuda()
        if cfg.get("relevance_source") == "external":
            TMD.set_external_relevance(ext[cfg.get("external_map", "mask")][rb].cuda())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _, infos = enc(E.preprocess(u8), return_merge_info=True, restore_dense=False)
        sims.append([i_["mean_selected_similarity"] for i_ in infos])
    s = np.array(sims, float).mean(0)
    out[name] = dict(mean_sim_per_layer=s.tolist(), mean_sim=float(s.mean()))
    print(f"{name:20s} mean key cosine of merged pairs {s.mean():.3f}", flush=True)
write_json(out, W / "runs" / "phaseA" / "edge_stats3.json")
