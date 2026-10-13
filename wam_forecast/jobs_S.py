"""Round S(b) jobs file (PROTOCOL, seventh review): 13 arms x 2 samplings x 5 folds x 4 seeds = 520 runs."""
import json
import sys

W = "$WAM_ROOT"
cap = sys.argv[1] if len(sys.argv) > 1 else ""
PY = f"{W}/conda/bin/python " + (f"capfrac.py {cap} " if cap else "")


def job(m, cfg, inp, sub, extra, s, f):
    d = "none" if inp == "pose" else cfg
    out = f"{W}/runs_S/{m}/cv/{d}/{sub}_cv{f}/s{s}"
    cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={W}/data/egodex_fps/{m}/slim WAM_SPLITS={W}/data/egodex_fps/{m}/splits "
           f"WAM_FEAT_ROOT={W}/feats_fps/{m} WAM_RUN_ROOT={W}/runs_S/{m} {PY}"
           f"train_probe.py --cfg {cfg} --input {inp} --seed {s} --out {out} --lr 2e-4 --epochs 80 --cv_fold {f} --rows_from kbsmpa_L9{extra}")
    return dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")


K1 = " --pose_frames 1"
cp = ["kbsmpa_L9", "wampa_L9", "wamshufpa_L9", "flowpa_L9", "flowshufpa_L9", "pospa_L9", "kbsmpa_L12", "wampa_L12", "wamshufpa_L12"]
arms = [(c, "vispose", "vispose_k1", K1) for c in cp] + [("kbsmpa_L9", "pose", "pose_k1", K1)]
arms += [(c, "vis", "vis", "") for c in ["kbsmpa_L9", "wampa_L9", "wamshufpa_L9"]]
with open(f"{W}/runs_S/jobs_S.jsonl", "w") as fh:
    for f in range(5):
        for s in range(4):
            for m in ("fixed30", "uniform"):
                for a in arms:
                    fh.write(json.dumps(job(m, *a, s, f)) + "\n")
