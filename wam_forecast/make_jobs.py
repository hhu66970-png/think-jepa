"""Write a gpu_pool jobs file.

  python make_jobs.py <exp> --cfgs dense,kbsm_L9 --inputs vis --seeds 0-4 [--extra "--epochs 40"]
Runs go to <RUN_ROOT>/<exp>/<cfg>/<input>/s<seed>/ ; a repeated seed can be requested with
--rep (writes .../s<seed>_rep<k>/) for the G0 determinism check.
"""
import argparse
import json
import os

from common import FEAT_ROOT, RUN_ROOT


def seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--inputs", default="vis")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--tag", default="", help="suffix for the input directory, e.g. _compact")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    py = os.environ.get("WAM_PY", "/21231_data1/huhaoming_wam/conda/bin/python")
    here = os.path.dirname(os.path.abspath(__file__))
    jobs = []
    for inp in a.inputs.split(","):
        cfgs = a.cfgs.split(",") if inp in ("vis", "vispose") else ["none"]
        for cfg in cfgs:
            for s in seeds(a.seeds):
                for k in range(a.rep + 1):
                    tag = f"s{s}" + (f"_rep{k}" if k else "")
                    if cfg.startswith("mix:"):            # per-clip budget policy
                        pol = cfg.split(":", 1)[1]
                        cdir, carg = f"mix_{pol}", f"mix --mix_policy {FEAT_ROOT}/policy_{pol}.npz"
                    else:
                        cdir, carg = cfg, (cfg if cfg != "none" else "dense")
                    out = RUN_ROOT / a.exp / cdir / f"{inp}{a.tag}" / tag
                    cmd = (f"cd {here} && WAM_FEAT_ROOT={FEAT_ROOT} {py} train_probe.py --cfg "
                           f"{carg} --input {inp} --seed {s} --out {out} {a.extra}")
                    jobs.append(dict(cmd=cmd, done=str(out / "metrics.json"), log=str(out / "log.txt")))
    path = a.out or str(RUN_ROOT / a.exp / "jobs.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs -> {path}")


if __name__ == "__main__":
    main()
