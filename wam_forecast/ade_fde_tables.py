"""ADE and FDE for the paper's Tables 1 and 2, from the released raw archive alone.

  python ade_fde_tables.py --data ../../data   -> <data>/ade_fde_tables.json
Uses the per clip ADE (key) and FDE (key|fde) arrays of raw_results.npz with the bootstraps of rebuild_from_raw.py
(4000 replicates, seed 0): paired seed t interval, clip x seed interval, and on the test split the task x seed interval.
The 30 seed test runs of Table 1 predate per clip FDE; their FDE comparison uses the seed level values of
raw_index.json with a paired t interval over seeds.
Table 1: test split, video only, 309 tokens, gates minus K-BSM (30 seeds). Table 2: pooled 5-fold CV over the 1620
unused clips, arms minus K-BSM+PA, for video only (vis), the compact readout (vis_compact) and the current pose
(vispose_k1, round CP)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
DATA = Path(ap.parse_args().data)
IDX = json.loads((DATA / "raw_index.json").read_text())["runs"]
RAW = np.load(DATA / "raw_results.npz")
TASKS = np.array(json.loads((DATA / "clip_tasks.json").read_text())["part2_test_tasks_eval_order"])
CV_ROWS = {}
CT = json.loads((DATA / "clip_tasks.json").read_text())["part2"]
ROW_TASK = np.array([t for t, _ in CT["train"]] + [t for t, _ in CT["test"]])
SUF = {"ade": "", "fde": "|fde"}
BY_DIR = {}
for k, m in IDX.items():
    BY_DIR.setdefault(k.rsplit("/", 1)[0], []).append((k.replace("/", "|"), m))


def load(run_dir, metric):
    return {m["seed"]: RAW[key + SUF[metric]].astype(np.float64) for key, m in BY_DIR.get(run_dir, [])
            if key + SUF[metric] in RAW.files}


def seed_level(run_dir, metric):
    """{seed: value} from raw_index.json (ade / fde of the whole run), for runs saved before per clip FDE existed"""
    return {m["seed"]: m[metric] for key, m in BY_DIR.get(run_dir, []) if m.get(metric) is not None}


def paired_seed(A, B):
    s = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in s])
    half = stats.t.ppf(0.975, len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    return dict(n=len(s), A=float(np.mean([A[k] for k in s])), B=float(np.mean([B[k] for k in s])), d=float(d.mean()),
                seed_ci=[float(d.mean() - half), float(d.mean() + half)], level="seed")


def load_cv(run_dir, metric):
    per = {}
    for f in range(5):
        for key, m in BY_DIR.get(f"{run_dir}_cv{f}", []):
            if key + SUF[metric] in RAW.files:
                per.setdefault(m["seed"], {})[f] = (RAW[key + "|rows"], RAW[key + SUF[metric]].astype(np.float64))
    out = {}
    for s, v in per.items():
        if len(v) == 5:
            rows = np.concatenate([v[f][0] for f in range(5)])
            out[s] = np.concatenate([v[f][1] for f in range(5)])[np.argsort(rows)]
            CV_ROWS[run_dir] = np.sort(rows)
    return out


def paired(A, B, tasks=None, nboot=4000):
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    D = np.stack([A[k] - B[k] for k in s]); d = D.mean(1)
    half = stats.t.ppf(0.975, len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    rng = np.random.default_rng(0)
    clip = [D[rng.integers(0, len(s), len(s))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(nboot)]
    res = dict(n=len(s), A=float(np.mean([A[k].mean() for k in s])), B=float(np.mean([B[k].mean() for k in s])),
               d=float(d.mean()), seed_ci=[float(d.mean() - half), float(d.mean() + half)],
               clip_ci=[float(np.percentile(clip, 2.5)), float(np.percentile(clip, 97.5))],
               clip_ci_bonf=[float(np.percentile(clip, 1.25)), float(np.percentile(clip, 98.75))])
    if tasks is not None and D.shape[1] == len(tasks):
        ut = np.unique(tasks); idx = [np.flatnonzero(tasks == t) for t in ut]
        rng = np.random.default_rng(0); tb = []
        for _ in range(nboot):
            Ds = D[rng.integers(0, len(s), len(s))]
            tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
        res["task_ci"] = [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))]
    return res


OUT = {"table1": {}, "table2": {}, "means": {}}
for metric in ("ade", "fde"):
    ref = load("runs/test_main/kbsm_L9/vis", metric)
    for a in ("randrank", "wam", "handsrc", "handmult"):
        A = load(f"runs/test_main/{a}_L9/vis", metric)
        OUT["table1"].setdefault(a, {})[metric] = paired(A, ref, TASKS) if A and ref else \
            paired_seed(seed_level(f"runs/test_main/{a}_L9/vis", metric), seed_level("runs/test_main/kbsm_L9/vis", metric))
    for k in ("kbsm_L9", "dense"):
        v = seed_level(f"runs/test_main/{k}/vis", metric)
        OUT["means"][f"test_vis_{k}_{metric}_seedlevel"] = float(np.mean(list(v.values())))
    for inp in ("vis", "vis_compact", "vispose_k1"):
        r = load_cv(f"runs/cv/kbsmpa_L9/{inp}", metric)
        if r:
            OUT["means"][f"cv_{inp}_kbsmpa_L9_{metric}"] = float(np.mean([v.mean() for v in r.values()]))
        for a in ("kbsm", "wampa", "handmultpa", "tomepa", "pitomeopa", "pitomepa", "caps2pa", "wamshufpa", "wamshuffpa",
                  "waminvpa", "wamcomppa", "randrankpa", "vidtldrpa", "pospa", "randmpa", "randdroppa", "in16f", "in128"):
            p = paired(load_cv(f"runs/cv/{a}{'' if a.startswith('in') else '_L9'}/{inp}", metric), r) if r else None
            if p:
                OUT["table2"].setdefault(inp, {}).setdefault(a, {})[metric] = p
# round CP replication on the part 2 test split (current pose, seeds 0-19), not used for the classification
OUT["cp_test"] = {}
for metric in ("ade", "fde"):
    ref = load("runs/test_main/kbsmpa_L9/vispose_k1", metric)
    OUT["means"][f"test_vispose_k1_kbsmpa_L9_{metric}"] = float(np.mean([v.mean() for v in ref.values()])) if ref else None
    for a in ("wampa", "handmultpa", "wamshufpa", "wamshuffpa"):
        A = load(f"runs/test_main/{a}_L9/vispose_k1", metric)
        if A and ref:
            OUT["cp_test"].setdefault(a, {})[metric] = paired(A, ref, TASKS)
# round CP2: replications with the current pose (131 tokens, VideoMAE, compact readout), CV
OUT["cp2"] = {}
CP2 = {"a_L12": ("vispose_k1", {"ref": "kbsmpa_L12", "motion": "wampa_L12", "hand": "handmultpa_L12"}),
       "b_vmae": ("vispose_k1", {"ref": "vmae_kbsmpa_L9", "motion": "vmae_wampa_L9", "hand": "vmae_handmultpa_L9", "dense": "vmae_dense"}),
       "c_compact": ("vispose_k1_compact", {"ref": "kbsmpa_L9", "motion": "wampa_L9", "hand": "handmultpa_L9"})}
for name, (sub, arms) in CP2.items():
    for metric in ("ade", "fde"):
        R = {k: load_cv(f"runs/cv/{c}/{sub}", metric) for k, c in arms.items()}
        if R["ref"]:
            OUT["cp2"].setdefault(name, {})[metric + "_ref"] = float(np.mean([v.mean() for v in R["ref"].values()]))
        for k in arms:
            if k != "ref" and R[k] and R["ref"]:
                OUT["cp2"][name].setdefault(k, {})[metric] = paired(R[k], R["ref"])
# CP2 motion classifications with a Bonferroni correction over the three replications (98.33% clip intervals)
for name in OUT.get("cp2", {}):
    mo = OUT["cp2"][name].get("motion")
    if mo:
        A = load_cv(f"runs/cv/{CP2[name][1]['motion']}/{CP2[name][0]}", "ade")
        B = load_cv(f"runs/cv/{CP2[name][1]['ref']}/{CP2[name][0]}", "ade")
        sd = sorted(set(A) & set(B)); D = np.stack([A[k] - B[k] for k in sd])
        rng = np.random.default_rng(0)
        bo = [D[rng.integers(0, len(sd), len(sd))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
        mo["ade"]["clip_ci_bonf3"] = [float(np.percentile(bo, 100 * 0.05 / 6)), float(np.percentile(bo, 100 - 100 * 0.05 / 6))]
# budget interaction of the hand gate given the current pose: (hand - K-BSM+PA) at 131 minus at 309 tokens
def interaction(a12, r12, a09, r09, metric):
    A12, R12 = load_cv(f"runs/cv/{a12}/vispose_k1", metric), load_cv(f"runs/cv/{r12}/vispose_k1", metric)
    A09, R09 = load_cv(f"runs/cv/{a09}/vispose_k1", metric), load_cv(f"runs/cv/{r09}/vispose_k1", metric)
    sd = sorted(set(A12) & set(R12) & set(A09) & set(R09))
    D = np.stack([(A12[k] - R12[k]) - (A09[k] - R09[k]) for k in sd]); d = D.mean(1)
    rng = np.random.default_rng(0)
    bo = [D[rng.integers(0, len(sd), len(sd))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    tk = ROW_TASK[CV_ROWS[f"runs/cv/{a12}/vispose_k1"]]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, len(sd), len(sd))]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    return dict(n=len(sd), d=float(d.mean()), clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))])
OUT["interactions"] = {}
for metric in ("ade", "fde"):
    OUT["interactions"].setdefault("hand_131_minus_309", {})[metric] = interaction("handmultpa_L12", "kbsmpa_L12", "handmultpa_L9", "kbsmpa_L9", metric)
    OUT["interactions"].setdefault("motion_131_minus_309", {})[metric] = interaction("wampa_L12", "kbsmpa_L12", "wampa_L9", "kbsmpa_L9", metric)
    # round C9: the same budget contrast for the current hand mask; round SM: saliency effect, early minus late schedule
    # (both with mid placement and official order), i.e. (vidtldrnat - fmnat) - (vidtldrmid - fmmid)
    if load_cv("runs/cv/handcurpa_L9/vispose_k1", metric):
        OUT["interactions"].setdefault("curhand_131_minus_309", {})[metric] = interaction("handcurpa_L12", "kbsmpa_L12", "handcurpa_L9", "kbsmpa_L9", metric)
    OUT["interactions"].setdefault("saliency_early_minus_late_post", {})[metric] = interaction("vidtldrnat_L9", "fmnat_L9", "vidtldrpa_L9", "fmpa_L9", metric)
    if load_cv("runs/cv/fmmid_L9/vispose_k1", metric):
        OUT["interactions"].setdefault("saliency_early_minus_late", {})[metric] = interaction("vidtldrnat_L9", "fmnat_L9", "vidtldrmid_L9", "fmmid_L9", metric)
# rounds HS, G, T from the raw archive: paired CV differences with clip, Bonferroni 97.5% and task intervals
def cv_pair(a_dir, b_dir, metric):
    A, B = load_cv(a_dir, metric), load_cv(b_dir, metric)
    sd = sorted(set(A) & set(B))
    if len(sd) < 2:
        return None
    D = np.stack([A[k] - B[k] for k in sd]); d = D.mean(1)
    rng = np.random.default_rng(0)
    bo = [D[rng.integers(0, len(sd), len(sd))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    tk = ROW_TASK[CV_ROWS[a_dir]]; ut = np.unique(tk); idx = [np.flatnonzero(tk == t) for t in ut]
    rng = np.random.default_rng(0); tb = []
    for _ in range(4000):
        Ds = D[rng.integers(0, len(sd), len(sd))]
        tb.append(Ds[:, np.concatenate([idx[i] for i in rng.integers(0, len(ut), len(ut))])].mean())
    return dict(n=len(sd), A=float(np.mean([A[k].mean() for k in sd])), B=float(np.mean([B[k].mean() for k in sd])),
                d=float(d.mean()), clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                clip_ci_bonf=[float(np.percentile(bo, 1.25)), float(np.percentile(bo, 98.75))],
                task_ci=[float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))], wins=int((d < 0).sum()))
PAIRS = {"hs_hand_minus_shuf_frame": ("handmultpa_L12", "handshuffpa_L12", "vispose_k1"),
         "hs_hand_minus_shuf_clip": ("handmultpa_L12", "handshufpa_L12", "vispose_k1"),
         "hs_shuf_frame_minus_ref": ("handshuffpa_L12", "kbsmpa_L12", "vispose_k1"),
         "hs_shuf_clip_minus_ref": ("handshufpa_L12", "kbsmpa_L12", "vispose_k1"),
         "g_motion": ("vitg_wampa_gL9", "vitg_kbsmpa_gL9", "vispose_k1"),
         "g_hand": ("vitg_handmultpa_gL9", "vitg_kbsmpa_gL9", "vispose_k1"),
         "t_current_pose": ("vidtldrpa_L9", "kbsmpa_L9", "vispose_k1"),
         "t_video_only": ("vidtldrpa_L9", "kbsmpa_L9", "vis"),
         "ls_lambda_05": ("wamlam05pa_L9", "kbsmpa_L9", "vispose_k1"),
         "ls_lambda_025": ("wamlam025pa_L9", "kbsmpa_L9", "vispose_k1"),
         "c_cur_minus_ref": ("handcurpa_L12", "kbsmpa_L12", "vispose_k1"),
         "c_cur_minus_cur_shuffled": ("handcurpa_L12", "handcurshuffpa_L12", "vispose_k1"),
         "c_cur_shuffled_minus_ref": ("handcurshuffpa_L12", "kbsmpa_L12", "vispose_k1"),
         "c_cur_minus_full_past": ("handcurpa_L12", "handmultpa_L12", "vispose_k1"),
         "n_native_current_pose": ("vidtldrnat_L9", "kbsmpa_L9", "vispose_k1"),
         "n_native_video_only": ("vidtldrnat_L9", "kbsmpa_L9", "vis"),
         "n_mid_current_pose": ("vidtldrmid_L9", "kbsmpa_L9", "vispose_k1"),
         "n_mid_minus_adapted": ("vidtldrmid_L9", "vidtldrpa_L9", "vispose_k1"),
         "n2_native_minus_nosaliency": ("vidtldrnat_L9", "fmnat_L9", "vispose_k1"),
         "n2_early_nosaliency_minus_ref": ("fmnat_L9", "kbsmpa_L9", "vispose_k1"),
         "n3_adapted_minus_nosaliency": ("vidtldrpa_L9", "fmpa_L9", "vispose_k1"),
         "n3_late_nosaliency_minus_ref": ("fmpa_L9", "kbsmpa_L9", "vispose_k1"),
         "c9_cur_minus_cur_shuffled": ("handcurpa_L9", "handcurshuffpa_L9", "vispose_k1"),
         "c9_cur_minus_ref": ("handcurpa_L9", "kbsmpa_L9", "vispose_k1"),
         "c9_cur_shuffled_minus_ref": ("handcurshuffpa_L9", "kbsmpa_L9", "vispose_k1"),
         "cf_f31_minus_f31_shuffled": ("handf31pa_L12", "handf31shuffpa_L12", "vispose_k1"),
         "cf_f31_minus_ref": ("handf31pa_L12", "kbsmpa_L12", "vispose_k1"),
         "cf_f31_shuffled_minus_ref": ("handf31shuffpa_L12", "kbsmpa_L12", "vispose_k1"),
         "cf_f31_minus_cur": ("handf31pa_L12", "handcurpa_L12", "vispose_k1"),
         "ss_late_real_minus_shuffled": ("vidtldrpa_L9", "vidtldrshufpa_L9", "vispose_k1"),
         "ss_late_shuffled_minus_none": ("vidtldrshufpa_L9", "fmpa_L9", "vispose_k1"),
         "ss_early_real_minus_shuffled": ("vidtldrnat_L9", "vidtldrnatshuf_L9", "vispose_k1"),
         "ss_early_shuffled_minus_none": ("vidtldrnatshuf_L9", "fmnat_L9", "vispose_k1"),
         "sm_late_mid_saliency_minus_none": ("vidtldrmid_L9", "fmmid_L9", "vispose_k1"),
         "sm_late_mid_none_minus_ref": ("fmmid_L9", "kbsmpa_L9", "vispose_k1"),
         "cp_motion_minus_clip_shuffle": ("wampa_L9", "wamshufpa_L9", "vispose_k1"),
         "u_tubelet_minus_clip_shuffle": ("handcurpa_L12", "handcurshufpa_L12", "vispose_k1"),
         "u_frame_minus_clip_shuffle": ("handf31pa_L12", "handf31shufpa_L12", "vispose_k1")}
OUT["rounds"] = {}
for name, (a_, b_, sub) in PAIRS.items():
    for metric in ("ade", "fde"):
        r = cv_pair(f"runs/cv/{a_}/{sub}", f"runs/cv/{b_}/{sub}", metric)
        if r:
            OUT["rounds"].setdefault(name, {})[metric] = r
# strata (descriptive): per task effects, leave one task out, and tertiles of covariates known at forecast time
COV = json.loads((DATA / "clip_covariates.json").read_text()) if (DATA / "clip_covariates.json").exists() else None
STRATA = {"cp_motion_309": ("wampa_L9", "kbsmpa_L9", "vispose_k1"),
          "vis_motion_309": ("wampa_L9", "kbsmpa_L9", "vis"),
          "cp_hand_131": ("handmultpa_L12", "kbsmpa_L12", "vispose_k1"),
          "hs_hand_minus_shuffle_131": ("handmultpa_L12", "handshuffpa_L12", "vispose_k1"),
          "g_motion_vitg": ("vitg_wampa_gL9", "vitg_kbsmpa_gL9", "vispose_k1"),
          "t_vidtldr_adapted": ("vidtldrpa_L9", "kbsmpa_L9", "vispose_k1"),
          "c_current_hand_131": ("handcurpa_L12", "kbsmpa_L12", "vispose_k1"),
          "c_current_hand_minus_shuffle_131": ("handcurpa_L12", "handcurshuffpa_L12", "vispose_k1"),
          "c9_current_hand_309": ("handcurpa_L9", "kbsmpa_L9", "vispose_k1"),
          "cp_motion_minus_shuffle": ("wampa_L9", "wamshufpa_L9", "vispose_k1"),
          "vis_motion_minus_shuffle": ("wampa_L9", "wamshufpa_L9", "vis"),
          "u_frame_hand_minus_shuffle_131": ("handf31pa_L12", "handf31shufpa_L12", "vispose_k1")}
OUT["strata"] = {}
if COV is not None:
    pm, hz = np.array(COV["past_motion_m"]), np.array(COV["horizon_s"])
    st = np.array(COV["stride_frames"]) if "stride_frames" in COV else None
    for name, (a_, b_, sub) in STRATA.items():
        A, B = load_cv(f"runs/cv/{a_}/{sub}", "ade"), load_cv(f"runs/cv/{b_}/{sub}", "ade")
        sd = sorted(set(A) & set(B))
        if len(sd) < 2:
            continue
        D = np.stack([A[k] - B[k] for k in sd]); dclip = D.mean(0)          # per clip difference averaged over seeds
        rows = CV_ROWS[f"runs/cv/{a_}/{sub}"]
        tk = ROW_TASK[rows]; ut = np.unique(tk)
        per_task = {t: float(dclip[tk == t].mean()) for t in ut}
        loto = [float(dclip[tk != t].mean()) for t in ut]
        def bins(cov):
            v = cov[rows]; q = np.quantile(v, [1 / 3, 2 / 3]); lab = np.digitize(v, q)
            res = []
            for g in range(3):
                Dg = D[:, lab == g]; rng = np.random.default_rng(0)
                bo = [Dg[rng.integers(0, len(sd), len(sd))][:, rng.integers(0, Dg.shape[1], Dg.shape[1])].mean() for _ in range(2000)]
                res.append(dict(lo_edge=float(v[lab == g].min()), hi_edge=float(v[lab == g].max()), n=int((lab == g).sum()),
                                d=float(Dg.mean()), ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]))
            # direct contrast of the top and bottom tertiles (exploratory interaction)
            hi, lo = D[:, lab == 2], D[:, lab == 0]; rng = np.random.default_rng(0); bo = []
            for _ in range(2000):
                si = rng.integers(0, len(sd), len(sd))
                bo.append(hi[si][:, rng.integers(0, hi.shape[1], hi.shape[1])].mean() - lo[si][:, rng.integers(0, lo.shape[1], lo.shape[1])].mean())
            res.append(dict(high_minus_low=float(hi.mean() - lo.mean()), ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]))
            return res
        OUT["strata"][name] = dict(d=float(dclip.mean()), tasks_positive=int(sum(v > 0 for v in per_task.values())),
                                   n_tasks=len(ut), per_task=per_task, loto_min=min(loto), loto_max=max(loto),
                                   past_motion_tertiles=bins(pm), horizon_tertiles=bins(hz),
                                   **({"stride_tertiles": bins(st)} if st is not None else {}))
# round X: probes trained on part 2, tested on 552 new episodes of the same 13 tasks (held out archive), 10 seeds
XS = json.loads((DATA / "clip_tasks.json").read_text()).get("xsplit")
OUT["roundX"] = {}
if XS:
    XT = np.array([t for t, _ in XS["test"]])
    XP = {"x1_motion_minus_clip_shuffle": ("wampa_L9", "wamshufpa_L9"), "x1_motion_minus_ref": ("wampa_L9", "kbsmpa_L9"),
          "x2_tubelet_minus_frame_shuffle": ("handcurpa_L12", "handcurshuffpa_L12"),
          "x2_frame_minus_frame_shuffle": ("handf31pa_L12", "handf31shuffpa_L12"),
          "x2_tubelet_minus_clip_shuffle": ("handcurpa_L12", "handcurshufpa_L12"),
          "x2_frame_minus_clip_shuffle": ("handf31pa_L12", "handf31shufpa_L12"),
          "x2_tubelet_minus_ref": ("handcurpa_L12", "kbsmpa_L12"), "x2_frame_minus_ref": ("handf31pa_L12", "kbsmpa_L12"),
          "x3_late_saliency": ("vidtldrpa_L9", "fmpa_L9"), "x3_early_saliency": ("vidtldrnat_L9", "fmnat_L9"),
          "x3_early_schedule_minus_ref": ("fmnat_L9", "kbsmpa_L9"), "cg131_minus_dense": ("kbsmpa_L12", "dense")}
    for name, (a_, b_) in XP.items():
        for metric in ("ade", "fde"):
            A, B = load(f"runs_X/test_main/{a_}/vispose_k1", metric), load(f"runs_X/test_main/{b_}/vispose_k1", metric)
            if len(set(A) & set(B)) >= 2:
                r = paired(A, B, XT)
                sd = sorted(set(A) & set(B)); D = np.stack([A[k] - B[k] for k in sd]); rng = np.random.default_rng(0)
                bo = [D[rng.integers(0, len(sd), len(sd))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
                r["clip_ci_bonf"] = [float(np.percentile(bo, 1.25)), float(np.percentile(bo, 98.75))]
                OUT["roundX"].setdefault(name, {})[metric] = r
    for metric in ("ade", "fde"):
        R = [load(f"runs_X/test_main/{c}/vispose_k1", metric) for c in ("vidtldrnat_L9", "fmnat_L9", "vidtldrpa_L9", "fmpa_L9")]
        sd = sorted(set.intersection(*[set(r) for r in R]))
        if len(sd) >= 2:
            D = np.stack([(R[0][k] - R[1][k]) - (R[2][k] - R[3][k]) for k in sd])
            Z = {k: np.zeros(D.shape[1]) for k in sd}
            OUT["roundX"].setdefault("x3_interaction", {})[metric] = paired({k: D[i] for i, k in enumerate(sd)}, Z, XT)
    OUT["roundX"]["means"] = {c: float(np.mean([v.mean() for v in load(f"runs_X/test_main/{c}/vispose_k1", "ade").values()]))
                             for c in ("dense", "kbsmpa_L9", "kbsmpa_L12") if load(f"runs_X/test_main/{c}/vispose_k1", "ade")}
# all prespecified confirmatory contrasts: power (MDE at 80%), equivalence (TOST, SESOI 1 mm), Holm across contrasts
from scipy.stats import norm  # noqa: E402
CONF = [("L", "motion gate vs K-BSM+PA, video only", "cv", "wampa_L9", "kbsmpa_L9", "vis", "pos"),
        ("L", "hand gate vs K-BSM+PA, video only", "cv", "handmultpa_L9", "kbsmpa_L9", "vis", "pos"),
        ("E", "VideoMAE motion gate, video only", "cv", "vmae_wampa_L9", "vmae_kbsmpa_L9", "vis", "pos"),
        ("E", "VideoMAE hand gate, video only", "cv", "vmae_handmultpa_L9", "vmae_kbsmpa_L9", "vis", "pos"),
        ("E", "VideoMAE PA", "cv", "vmae_kbsmpa_L9", "vmae_kbsm_L9", "vis", "null"),
        ("CP", "motion gate vs K-BSM+PA, current pose", "cv", "wampa_L9", "kbsmpa_L9", "vispose_k1", "pos"),
        ("CP", "motion gate vs clip shuffle, current pose", "cv", "wampa_L9", "wamshufpa_L9", "vispose_k1", "null"),
        ("CP2", "motion gate, 131 tokens", "cv", "wampa_L12", "kbsmpa_L12", "vispose_k1", "pos"),
        ("CP2", "motion gate, compact readout", "cv", "wampa_L9", "kbsmpa_L9", "vispose_k1_compact", "pos"),
        ("CP2", "motion gate, VideoMAE", "cv", "vmae_wampa_L9", "vmae_kbsmpa_L9", "vispose_k1", "null"),
        ("G", "motion gate, ViT-g", "cv", "vitg_wampa_gL9", "vitg_kbsmpa_gL9", "vispose_k1", "pos"),
        ("T", "vid-TLDR, late schedule", "cv", "vidtldrpa_L9", "kbsmpa_L9", "vispose_k1", "pos"),
        ("N", "vid-TLDR, early schedule", "cv", "vidtldrnat_L9", "kbsmpa_L9", "vispose_k1", "pos"),
        ("LS", "motion gate, lambda 0.5", "cv", "wamlam05pa_L9", "kbsmpa_L9", "vispose_k1", "null"),
        ("LS", "motion gate, lambda 0.25", "cv", "wamlam025pa_L9", "kbsmpa_L9", "vispose_k1", "null"),
        ("C", "last tubelet hand vs frame shuffle, 131", "cv", "handcurpa_L12", "handcurshuffpa_L12", "vispose_k1", "neg"),
        ("C9", "last tubelet hand vs frame shuffle, 309", "cv", "handcurpa_L9", "handcurshuffpa_L9", "vispose_k1", "neg"),
        ("CF", "last frame hand vs frame shuffle, 131", "cv", "handf31pa_L12", "handf31shuffpa_L12", "vispose_k1", "neg"),
        ("SS", "saliency vs shuffled saliency, late", "cv", "vidtldrpa_L9", "vidtldrshufpa_L9", "vispose_k1", "null"),
        ("SS", "saliency vs shuffled saliency, early", "cv", "vidtldrnat_L9", "vidtldrnatshuf_L9", "vispose_k1", "null"),
        ("U", "last tubelet hand vs clip shuffle, 131", "cv", "handcurpa_L12", "handcurshufpa_L12", "vispose_k1", "neg"),
        ("U", "last frame hand vs clip shuffle, 131", "cv", "handf31pa_L12", "handf31shufpa_L12", "vispose_k1", "neg"),
        ("X", "new episodes: motion vs clip shuffle", "x", "wampa_L9", "wamshufpa_L9", "", "null"),
        ("X", "new episodes: last tubelet hand vs frame shuffle", "x", "handcurpa_L12", "handcurshuffpa_L12", "", "neg"),
        ("X", "new episodes: last frame hand vs frame shuffle", "x", "handf31pa_L12", "handf31shuffpa_L12", "", "neg")]


def boot_d(A, B):
    sd = sorted(set(A) & set(B))
    if len(sd) < 2:
        return None
    D = np.stack([A[k] - B[k] for k in sd]); rng = np.random.default_rng(0)
    bo = np.array([D[rng.integers(0, len(sd), len(sd))][:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)])
    return float(D.mean()), bo


OUT["confirmatory"] = []
for rnd, name, kind, a_, b_, sub, reading in CONF:
    if kind == "cv":
        r = boot_d(load_cv(f"runs/cv/{a_}/{sub}", "ade"), load_cv(f"runs/cv/{b_}/{sub}", "ade"))
    else:
        r = boot_d(load(f"runs_X/test_main/{a_}/vispose_k1", "ade"), load(f"runs_X/test_main/{b_}/vispose_k1", "ade"))
    if r is None:
        continue
    d, bo = r
    se = float(bo.std(ddof=1)); p = float(2 * norm.sf(abs(d) / se))
    c90 = [float(np.percentile(bo, 5)), float(np.percentile(bo, 95))]
    OUT["confirmatory"].append(dict(round=rnd, name=name, reading=reading, d=d, se=se, p=p, mde80=float((norm.ppf(0.975) + norm.ppf(0.8)) * se),
                                    ci90=c90, tost_1mm=bool(c90[0] > -1 and c90[1] < 1)))
ps = np.array([c["p"] for c in OUT["confirmatory"]]); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
for c, q in zip(OUT["confirmatory"], adj):
    c["p_holm"] = float(q)
    c["reading_kept"] = bool((c["reading"] == "null") or (q < 0.05 and ((c["d"] > 0) == (c["reading"] == "pos"))))
(DATA / "ade_fde_tables.json").write_text(json.dumps(OUT, indent=1))
fmt = lambda c: "[%+.2f, %+.2f]" % tuple(c)
for a, v in OUT["table1"].items():
    print("T1 %-9s ADE %+.2f %s  FDE %+.2f seed %s" % (a, v["ade"]["d"], fmt(v["ade"]["clip_ci"]), v["fde"]["d"], fmt(v["fde"]["seed_ci"])))
for inp, arms in OUT["table2"].items():
    for a, v in arms.items():
        print("T2 %-11s %-10s ADE %+.2f %s  FDE %s" % (inp, a, v["ade"]["d"], fmt(v["ade"]["clip_ci"]),
                                                      ("%+.2f %s" % (v["fde"]["d"], fmt(v["fde"]["clip_ci"]))) if "fde" in v else "-"))
for a, v in OUT["cp_test"].items():
    print("CP test %-10s ADE %+.2f %s task %s  FDE %+.2f %s (seeds %d)" % (a, v["ade"]["d"], fmt(v["ade"]["clip_ci"]), fmt(v["ade"]["task_ci"]),
          v["fde"]["d"], fmt(v["fde"]["clip_ci"]), v["ade"]["n"]))
for name, v in OUT["cp2"].items():
    for k, p in v.items():
        if isinstance(p, dict):
            print("CP2 %-9s %-6s ADE %+.2f 97.5 %s  FDE %+.2f %s" % (name, k, p["ade"]["d"], fmt(p["ade"]["clip_ci_bonf"]), p["fde"]["d"], fmt(p["fde"]["clip_ci"])))
for k, v in OUT["rounds"].items():
    print("round %-26s ADE %+.2f 97.5 %s task %s | FDE %+.2f %s (seeds %d)" % (k, v["ade"]["d"], fmt(v["ade"]["clip_ci_bonf"]), fmt(v["ade"]["task_ci"]),
          v["fde"]["d"], fmt(v["fde"]["clip_ci"]), v["ade"]["n"]))
for k, v in OUT["strata"].items():
    print("strata %-26s %+.2f tasks>0 %d/%d LOTO [%+.2f, %+.2f] motion tertiles %s horizon tertiles %s" % (
        k, v["d"], v["tasks_positive"], v["n_tasks"], v["loto_min"], v["loto_max"],
        " ".join("%+.2f" % b["d"] for b in v["past_motion_tertiles"][:3]), " ".join("%+.2f" % b["d"] for b in v["horizon_tertiles"][:3])))
    print("   high minus low: motion %+.2f %s  horizon %+.2f %s" % (v["past_motion_tertiles"][3]["high_minus_low"], fmt(v["past_motion_tertiles"][3]["ci"]),
          v["horizon_tertiles"][3]["high_minus_low"], fmt(v["horizon_tertiles"][3]["ci"])))
for k, v in OUT["roundX"].items():
    if k != "means":
        print("roundX %-30s ADE %+.2f clip %s task %s | FDE %+.2f clip %s (seeds %d)" % (k, v["ade"]["d"], fmt(v["ade"]["clip_ci"]), fmt(v["ade"]["task_ci"]),
              v["fde"]["d"], fmt(v["fde"]["clip_ci"]), v["ade"]["n"]))
for c in OUT["confirmatory"]:
    print("conf %-4s %-48s d %+.2f se %.2f p %.1e holm %.1e MDE80 %.2f ci90 [%+.2f, %+.2f] TOST1 %s kept %s" % (c["round"], c["name"], c["d"], c["se"], c["p"], c["p_holm"], c["mde80"], c["ci90"][0], c["ci90"][1], c["tost_1mm"], c["reading_kept"]))
for k, v in OUT["interactions"].items():
    print("interaction %-22s ADE %+.2f clip %s task %s | FDE %+.2f clip %s" % (k, v["ade"]["d"], fmt(v["ade"]["clip_ci"]), fmt(v["ade"]["task_ci"]), v["fde"]["d"], fmt(v["fde"]["clip_ci"])))
for name, v in OUT.get("cp2", {}).items():
    if "motion" in v:
        print("CP2 %s motion Bonferroni over three (98.33%%): %s" % (name, fmt(v["motion"]["ade"]["clip_ci_bonf3"])))
print({k: round(v, 2) for k, v in OUT["means"].items() if v is not None})
