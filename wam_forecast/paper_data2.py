"""Second data export for the revised paper (review response).

1. raw_results.npz + raw_index.json : every probe run (per-seed ADE/FDE and per-clip ADE) = provenance
2. paper_data2.json:
   - paired means (A, B over the common seeds) for every comparison shown in the paper
   - task-clustered two-level bootstrap (seeds, then tasks with all their clips) as a sensitivity analysis
   - per-arm seed-level 95% t intervals of the mean (Figure 7 error bars)
   - ViT-g cap vs K-BSM+PA and the CV factorial means read from the CV result files
  python paper_data2.py
"""
import json
from pathlib import Path

import numpy as np

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, write_json

OUT = W / "runs" / "paper_data"
TM, TG = RUN_ROOT / "test_main", RUN_ROOT / "test_vitg"
P = np.load(FEAT_ROOT / "poses.npz")
te = P["is_test"]
TASK_TEST = P["task"][np.flatnonzero(te)]          # test rows in sorted order = perclip order (200 clips)
BOOT_SEED = 0


def load(root, d):
    out = {}
    for mf in (Path(root) / d).glob("s*/metrics.json"):
        m = json.loads(mf.read_text())
        out[m["seed"]] = (m["ade_mm"], m["fde_mm"], np.load(mf.parent / "perclip_ade.npy"))
    return out


def paired(A, B, tasks=None, nboot=4000):
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    d = np.array([A[k][0] - B[k][0] for k in s])
    half = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    D = np.stack([A[k][2] - B[k][2] for k in s])
    rng = np.random.default_rng(BOOT_SEED)
    clip = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(nboot)]
    res = dict(n=len(s), A=float(np.mean([A[k][0] for k in s])), B=float(np.mean([B[k][0] for k in s])),
               A_fde=float(np.mean([A[k][1] for k in s])), B_fde=float(np.mean([B[k][1] for k in s])),
               d=float(d.mean()), seed_ci=[float(d.mean() - half), float(d.mean() + half)],
               clip_ci=[float(np.percentile(clip, 2.5)), float(np.percentile(clip, 97.5))], wins=int((d < 0).sum()))
    if tasks is not None and D.shape[1] == len(tasks):
        ut = np.unique(tasks)
        idx = [np.flatnonzero(tasks == t) for t in ut]
        rng = np.random.default_rng(BOOT_SEED)
        tb = []
        for _ in range(nboot):
            Ds = D[rng.integers(0, len(s), len(s))]
            pick = np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])
            tb.append(Ds[:, pick].mean())
        res["task_ci"] = [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))]
    return res


def mean_ci(R):
    v = np.array([x[0] for x in R.values()])
    if len(v) < 2:
        return None
    h = tcrit(len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(n=len(v), ade=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h))


D = {}
# --- key test comparisons with paired means and task-clustered intervals
cmp = {}
for b in ("s25", "L9", "L12"):
    for a in ("randrank", "wam", "handmult", "handsrc"):
        cmp[f"{a}-kbsm/{b}"] = paired(load(TM, f"{a}_{b}/vis"), load(TM, f"kbsm_{b}/vis"), TASK_TEST)
    cmp[f"kbsmpa-kbsm/{b}"] = paired(load(TM, f"kbsmpa_{b}/vis"), load(TM, f"kbsm_{b}/vis"), TASK_TEST)
    for a in ("wampa", "handmultpa", "mrecv15pa", "rrecv15pa", "kbsmlmpa", "b2pa", "caps2pa"):
        cmp[f"{a}-kbsmpa/{b}"] = paired(load(TM, f"{a}_{b}/vis"), load(TM, f"kbsmpa_{b}/vis"), TASK_TEST)
cmp["wam-randrank/L9"] = paired(load(TM, "wam_L9/vis"), load(TM, "randrank_L9/vis"), TASK_TEST)
cmp["mrecv15-kbsm/L9"] = paired(load(TM, "mrecv15_L9/vis"), load(TM, "kbsm_L9/vis"), TASK_TEST)
cmp["vitg/kbsmpa-kbsm"] = paired(load(TG, "vitg_kbsmpa_gL9/vis"), load(TG, "vitg_kbsm_gL9/vis"), TASK_TEST)
cmp["vitg/caps2pa-kbsmpa"] = paired(load(TG, "vitg_caps2pa_gL9/vis"), load(TG, "vitg_kbsmpa_gL9/vis"), TASK_TEST)
for b in ("s25", "L9", "L12", "L15"):
    for a, r in (("caps2pa", "kbsmpa"), ("caps2pa", "tomepa"), ("caps2pa", "pitomepa"), ("kbsmpa", "tomepa")):
        cmp[f"P1/{a}-{r}/{b}"] = paired(load(TM, f"{a}_{b}/vispose_k1"), load(TM, f"{r}_{b}/vispose_k1"), TASK_TEST)
    cmp[f"P1/kbsmpa-dense/{b}"] = paired(load(TM, f"kbsmpa_{b}/vispose_k1"), load(TM, "dense/vispose_k1"), TASK_TEST)
cmp["P1/dense-pose"] = paired(load(TM, "dense/vispose_k1"), load(TM, "none/pose_k1"), TASK_TEST)
D["cmp"] = cmp

# --- Figure 7 error bars: per-arm seed-level CI of the mean
D["p1_ci"] = {}
for bb, root, pre, buds in (("ViT-L", TM, "", ("s25", "L9", "L12", "L15")), ("ViT-g", TG, "vitg_", ("gs25", "gL9", "gL12", "gL15"))):
    for a in ("kbsmpa", "tomepa", "pitomepa", "caps2pa"):
        for b in buds:
            D["p1_ci"][f"{bb}/{a}/{b}"] = mean_ci(load(root, f"{pre}{a}_{b}/vispose_k1"))
D["p1_ci"]["ViT-L/dense"] = mean_ci(load(TM, "dense/vispose_k1"))
D["p1_ci"]["ViT-L/pose"] = mean_ci(load(TM, "none/pose_k1"))
RR = W / "runs_R"
D["p1_ci"]["R/dense"] = mean_ci(load(RR / "test_main", "dense/vispose_k1"))
D["p1_ci"]["R/pose"] = mean_ci(load(RR / "test_main", "none/pose_k1"))
for bb, root, pre, buds in (("ViT-L", RR / "test_main", "", ("L9", "L12", "L15")), ("ViT-g", RR / "test_vitg", "vitg_", ("gL9", "gL12", "gL15"))):
    for a in ("kbsmpa", "tomepa", "pitomepa", "caps2pa"):
        for b in buds:
            D["p1_ci"][f"R/{bb}/{a}/{b}"] = mean_ci(load(root, f"{pre}{a}_{b}/vispose_k1"))

# --- CV factorial means straight from the CV result files
cv = {}
for f in ("cv_caps2pa_vs_kbsmpa.json", "cv_caps2_vs_kbsm.json", "cv_caps2pa_vs_kbsmpa_vispose_k1.json"):
    p = W / "runs" / "phaseA" / f
    if p.exists():
        cv[f] = json.loads(p.read_text())
D["cv"] = cv
write_json(D, OUT / "paper_data2.json")
print("paper_data2.json written")

# --- raw export (provenance)
index, arrays = {}, {}
for root in (W / "runs" / "test_main", W / "runs" / "test_vitg", W / "runs" / "select", W / "runs" / "cv",
             W / "runs_R" / "test_main", W / "runs_R" / "test_vitg", W / "runs_X" / "test_main", W / "runs_S", W / "runs_DV", W / "runs_DV2", W / "runs_Q"):
    for mf in root.rglob("metrics.json"):
        rel = str(mf.parent.relative_to(W))
        m = json.loads(mf.read_text())
        index[rel] = dict(seed=m.get("seed"), split=m.get("split"), ade=m.get("ade_mm"), fde=m.get("fde_mm"), acc=m.get("acc"), nll=m.get("nll"),
                          cfg=m.get("cfg"), input=m.get("input"), readout=m.get("readout"), n_eval=m.get("n_eval"),
                          args={k: (str(v).replace(str(W), "$WAM_ROOT") if isinstance(v, str) else v)
                                for k, v in (m.get("args") or {}).items()})
        pc = mf.parent / "perclip_ade.npy"
        if pc.exists():
            arrays[rel.replace("/", "|")] = np.load(pc).astype(np.float32)
        pf = mf.parent / "perclip_fde.npy"
        if pf.exists():
            arrays[rel.replace("/", "|") + "|fde"] = np.load(pf).astype(np.float32)
        for extra in ("acc", "nll"):                  # Diving48 classification runs (round DV)
            pe = mf.parent / f"perclip_{extra}.npy"
            if pe.exists():
                arrays[rel.replace("/", "|") + "|" + extra] = np.load(pe).astype(np.float32)
        er = mf.parent / "eval_rows.npy"
        if er.exists():
            arrays[rel.replace("/", "|") + "|rows"] = np.load(er).astype(np.int16)
np.savez_compressed(OUT / "raw_results.npz", **arrays)
write_json(dict(n_runs=len(index), bootstrap_seed=BOOT_SEED, n_boot=4000,
                note="keys of raw_results.npz are run directories with '/' replaced by '|' (per clip ADE; suffix |fde: per clip FDE; |rows: evaluation rows); perclip order is the "
                     "sorted evaluation rows (test: the 200 official test clips in poses.npz order)",
                runs=index), OUT / "raw_index.json")
print("raw export", len(index), "runs")


# --- clip lists with task labels (anonymized: task and episode id only), for task-clustered statistics
def _keys(path):
    return [(Path(l.strip()).parent.name, Path(l.strip()).name.split("_", 1)[0].replace(".npz", ""))
            for l in open(path) if l.strip()]
from common import SPLITS  # noqa: E402
held = W / "data" / "egodex_testset" / "splits"
write_json(dict(note="(task, episode id) per clip in split-file order; evaluation rows index train+test "
                     "(part 2: 1800 train then 200 test); CV folds: np.array_split(RandomState(777).permutation("
                     "non-selection train rows), 5), see train_probe.py",
                part2=dict(train=_keys(SPLITS / "train_cache.txt"), test=_keys(SPLITS / "test_cache.txt")),
                heldout=dict(train=_keys(held / "train_cache.txt"), test=_keys(held / "test_cache.txt")),
                part2_test_tasks_eval_order=[str(t) for t in TASK_TEST],
                xsplit=dict(note="round X: train = part 2 train; test = held out episodes of the 13 part 2 tasks (ho_ prefix)",
                            test=_keys(W / "data" / "egodex_x" / "splits" / "test_cache.txt"))),
           OUT / "clip_tasks.json")
print("clip_tasks.json written")
