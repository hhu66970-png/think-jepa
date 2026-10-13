#!/bin/bash
# Round Q (eighth review): int8 storage check on the round DV CV (fp16 features quantized on loading).
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
J=$W/runs_Q/jobs_Q.jsonl; mkdir -p $W/runs_Q; rm -f $J
$PY - <<PYEOF
import json
W = "$W"
with open("$J", "w") as fh:
    for cfg in ("kbsmpa_L9", "wampa_L9"):
        for f in range(5):
            for s in range(4):
                out = f"{W}/runs_Q/cv_cls/{cfg}/vis_cv{f}/s{s}"
                cmd = (f"cd {W}/repo/wam_forecast && WAM_SIMQ=1 WAM_SLIM={W}/data/diving48/v2test/slim "
                       f"WAM_SPLITS={W}/data/diving48/v2test/splits WAM_FEAT_ROOT={W}/feats_DV WAM_RUN_ROOT={W}/runs_Q "
                       f"{W}/conda/bin/python train_cls.py --cfg {cfg} --seed {s} --cv_fold {f} --out {out}")
                fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
$PY gpu_pool_claim.py $J --gpus 0,0,4,4,5,5 --per_gpu 1
echo "Q CV DONE"
