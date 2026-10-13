"""Collect every number / map the CVPR paper figures need into runs/paper_data/.

  python paper_data.py
Outputs: paper_data.json (paired statistics, budget curves, group-size distributions, bench)
         qual_maps.npz  (frames, hand masks and per-token group sizes for the qualitative figure)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import FEAT_ROOT, RUN_ROOT, W, all_clips, clip_path, write_json
from pairs import load
from phaseA_analysis import paired
from train_probe import FeatureBank

OUT = W / "runs" / "paper_data"
OUT.mkdir(parents=True, exist_ok=True)
TM, TG = RUN_ROOT / "test_main", RUN_ROOT / "test_vitg"
D = {}


def mean(root, d):
    v = load(root, d)
    if not v:
        return None
    return dict(n=len(v), ade=float(np.mean([x[0] for x in v.values()])), fde=float(np.mean([x[1] for x in v.values()])),
                ade_sd=float(np.std([x[0] for x in v.values()], ddof=1)) if len(v) > 1 else 0.0)


def pair(root, a, b):
    p = paired(load(root, a), load(root, b))
    return p


# 1. reference points and budget curves (original task, vision-only read-out)
D["reference"] = {k: mean(TM, d) for k, d in (("dense_vis", "dense/vis"), ("pose", "none/pose"), ("dense_vispose", "dense/vispose"))}
P = np.load(FEAT_ROOT / "poses.npz")
te = P["is_test"]
copy_err = np.linalg.norm(P["future"] - P["past"][:, -1:], axis=-1) * 1000       # [N,32,52]
D["reference"]["copy_last"] = dict(ade=float(copy_err[te].mean()), fde=float(copy_err[te][:, -1].mean()))
arms_curve = ["kbsm", "kbsmpa", "wam", "handmult", "randrank", "caps2pa", "mrecv15", "rrecv15pa", "wampa", "handmultpa"]
D["curve_vis"] = {a: {b: mean(TM, f"{a}_{b}/vis") for b in ("s25", "L9", "L12")} for a in arms_curve}

# 2. no-PA gates vs K-BSM and PA vs no-PA (paired, test)
D["gates_vs_kbsm"] = {a: {b: pair(TM, f"{a}_{b}/vis", f"kbsm_{b}/vis") for b in ("s25", "L9", "L12")}
                      for a in ("randrank", "wam", "handmult", "handsrc", "mrecv15", "rrecv15", "hrecv15", "kbsmlm", "b2")}
D["pa_vs_nopa"] = {b: pair(TM, f"kbsmpa_{b}/vis", f"kbsm_{b}/vis") for b in ("s25", "L9", "L12")}
D["pa_vs_nopa"]["vitg_gL9"] = pair(TG, "vitg_kbsmpa_gL9/vis", "vitg_kbsm_gL9/vis")
D["pa_arms_vs_kbsmpa"] = {a: {b: pair(TM, f"{a}_{b}/vis", f"kbsmpa_{b}/vis") for b in ("s25", "L9", "L12")}
                          for a in ("wampa", "handmultpa", "mrecv15pa", "rrecv15pa", "kbsmlmpa", "b2pa", "caps2pa",
                                    "capc1pa", "capc2pa", "capc4pa", "caps4pa", "caps8pa")}
D["vitg_vis"] = {a: mean(TG, f"vitg_{a}_gL9/vis") for a in ("kbsm", "kbsmpa", "caps2pa", "wam", "handmult", "randrank", "mrecv15")}

# 3. P1 (video + current pose) curves on both backbones
D["p1"] = {"ViT-L": {a: {b: mean(TM, f"{a}_{b}/vispose_k1") for b in ("s25", "L9", "L12", "L15")}
                     for a in ("kbsmpa", "tomepa", "pitomepa", "caps2pa", "kbsm")},
           "ViT-g": {a: {b: mean(TG, f"vitg_{a}_{b}/vispose_k1") for b in ("gs25", "gL9", "gL12", "gL15")}
                     for a in ("kbsmpa", "tomepa", "pitomepa", "caps2pa", "kbsm")}}
D["p1"]["dense"] = mean(TM, "dense/vispose_k1")
D["p1"]["pose_only"] = mean(TM, "none/pose_k1")
D["p1_pairs"] = {}
for bb, root, pre, buds in (("ViT-L", TM, "", ("s25", "L9", "L12", "L15")), ("ViT-g", TG, "vitg_", ("gs25", "gL9", "gL12", "gL15"))):
    for b in buds:
        for a, r in (("caps2pa", "kbsmpa"), ("caps2pa", "tomepa"), ("caps2pa", "pitomepa"), ("kbsmpa", "tomepa")):
            D["p1_pairs"][f"{bb}/{b}/{a}-{r}"] = pair(root, f"{pre}{a}_{b}/vispose_k1", f"{pre}{r}_{b}/vispose_k1")
    if bb == "ViT-L":
        for b in buds:
            D["p1_pairs"][f"ViT-L/{b}/kbsmpa-dense"] = pair(TM, f"kbsmpa_{b}/vispose_k1", "dense/vispose_k1")

# 4. group-size distributions on the test clips (per original token: size of its group)
edges = np.unique(np.round(np.logspace(0, np.log10(2048), 40)).astype(int))
D["groups"] = {}
test_rows = np.flatnonzero(te)
for cfg in ("kbsm_L9", "kbsmpa_L9", "wam_L9", "handmult_L9", "caps2pa_L9", "capc1pa_L9", "caps4pa_L9",
            "tomepa_L9", "pitomepa_L9", "kbsmpa_L12", "caps2pa_L12"):
    bank = FeatureBank(cfg)
    tok_sizes, grp_sizes, gin, mx = [], [], [], []
    for r in test_rows:
        if int(r) not in bank.where:
            continue
        s, j = bank.where[int(r)]
        idx = np.asarray(bank.idx[s][j]).astype(np.int64)
        size = np.bincount(idx)
        size = size[size > 0]
        tok_sizes.append(np.repeat(size, size))
        grp_sizes.append(size)
        x = np.sort(size.astype(float)); n = len(x)
        gin.append((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
        mx.append(size.max())
    t = np.concatenate(tok_sizes)
    D["groups"][cfg] = dict(ccdf_edges=edges.tolist(), ccdf_tokens=[float((t >= e).mean()) for e in edges],
                            gini_mean=float(np.mean(gin)), max_mean=float(np.mean(mx)),
                            tok_median=float(np.median(t)), frac_tokens_in_groups_ge64=float((t >= 64).mean()))

# 5. copy existing result files
for src in ["runs/phaseA/select_val.json", "runs/phaseA/corr_test.json", "runs/phaseA/corr_val.json",
            "runs/phaseA/corr_nopa_family.json", "runs/phaseA/cv_caps2pa_vs_kbsmpa.json", "runs/phaseA/cv_caps2_vs_kbsm.json",
            "runs/phaseA/cv_caps2pa_vs_kbsmpa_vispose_k1.json", "runs/phaseA/roundM.json", "runs/phaseB/phaseB_val.json",
            "runs_R/roundR.json", "feats/alloc_full.json", "feats/alloc_v3.json", "feats/alloc_p4.json",
            "feats/alloc_parel.json", "feats/alloc_phaseA_L9.json", "feats/alloc_phaseA_L12.json",
            "feats/alloc_phaseA_s25.json", "feats/bench_vitl.json"]:
    p = W / src
    if p.exists():
        shutil.copy(p, OUT / p.name)
write_json(D, OUT / "paper_data.json")
print("paper_data.json written")

# 6. qualitative maps: two fixed example clips + two high-motion test clips of different tasks
keys = all_clips()
names = [str(k) for k in P["keys"]]
pick = [next(i for i, k in enumerate(names) if k.split("/")[1].startswith(s)) for s in ("1082_", "1013_")]
mot = copy_err.mean(axis=(1, 2))
seen = {names[i].split("/")[0] for i in pick}
for i in np.argsort(-mot):
    if te[i] and names[i].split("/")[0] not in seen and mot[i] < np.percentile(mot, 95):
        pick.append(int(i)); seen.add(names[i].split("/")[0])
    if len(pick) == 4:
        break
T_SHOW = 12                                    # tubelet index (frames 24-25)
arms = ["kbsm_L9", "kbsmpa_L9", "wam_L9", "handmult_L9", "tomepa_L9", "caps2pa_L9"]
H = np.load(FEAT_ROOT / "hand_masks.npz")["mask"]
frames, qmaps, hmask, labels = [], [], [], []
for r in pick:
    with np.load(clip_path(keys[r])) as z:
        img = torch.from_numpy(z["imgs"][2 * T_SHOW].astype(np.float32)).permute(2, 0, 1)[None]
    img = F.interpolate(img, size=(292, 292), mode="bilinear", align_corners=False)[0, :, 18:274, 18:274]
    frames.append(img.permute(1, 2, 0).clamp(0, 255).byte().numpy())
    hmask.append(H[r][T_SHOW].astype(np.uint8))
    labels.append(names[r])
    q = []
    for cfg in arms:
        bank = FeatureBank(cfg)
        s, j = bank.where[int(r)]
        idx = np.asarray(bank.idx[s][j]).astype(np.int64)
        size = np.bincount(idx)[idx]                             # group size of every original token
        q.append((1.0 / size).reshape(16, 16, 16)[T_SHOW].astype(np.float32))
    qmaps.append(np.stack(q))
np.savez(OUT / "qual_maps.npz", frames=np.stack(frames), qmaps=np.stack(qmaps), hand=np.stack(hmask),
         labels=np.array(labels), arms=np.array(arms), motion=mot[pick], t_show=T_SHOW)
print("qual_maps.npz written", labels)
