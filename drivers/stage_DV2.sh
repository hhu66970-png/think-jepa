#!/bin/bash
# Round DV2 (eighth review): Diving48 official split. Waits for the download, builds the clip set, flow maps,
# int8 features (dense on the local disk /data1), 84 probe runs, analysis.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until grep -q "ALL TRAIN SHARDS DONE" $W/tmp/dl_train.log; do sleep 60; done
$PY - <<PYEOF || { echo "PARQUET CHECK FAILED"; exit 1; }
import glob, pyarrow.parquet as pq
n = sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(glob.glob("$W/data/diving48/mteb/train-*.parquet")))
print("train rows", n); assert n == 15027, n
PYEOF
D=$W/data/diving48/official
$PY build_diving2.py --parquet $W/data/diving48/mteb --out $D || { echo "BUILD FAILED"; exit 1; }
export WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_RUN_ROOT=$W/runs_DV2 WAM_FEAT_INT8=1
mkdir -p $WAM_FEAT_ROOT/logs /data1/anon/wam_feats_DV2/dense
[ -e $WAM_FEAT_ROOT/dense ] || ln -s /data1/anon/wam_feats_DV2/dense $WAM_FEAT_ROOT/dense
$PY -c "
import numpy as np
from common import FEAT_ROOT, all_clips
n = len(all_clips()); print(clips, n)
np.savez_compressed(FEAT_ROOT / hand_masks.npz, mask=np.zeros((n, 16, 16, 16), bool))
"
$PY flow_maps.py --workers 32 > $WAM_FEAT_ROOT/logs/flow.log 2>&1 || { echo "FLOW FAILED"; exit 1; }
tail -1 $WAM_FEAT_ROOT/logs/flow.log
echo "DV2 PREP DONE"
CFGS=dense,in16f,kbsmpa_s25,wampa_s25,wamshufpa_s25,kbsmpa_L9,wampa_L9,wamshufpa_L9,kbsmpa_L12,wampa_L12,wamshufpa_L12,in16f+kbsmpa_L9,in16f+wampa_L9,in16f+wamshufpa_L9,vidtldrpa_L9,vidtldrshufpa_L9,pospa_L9,flowpa_L9,flowshufpa_L9,flowcpa_L9,flowcshufpa_L9
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $WAM_FEAT_ROOT/logs/extract_$s.log 2>&1 &
done
wait
grep -il traceback $WAM_FEAT_ROOT/logs/extract_*.log && { echo "DV2 EXTRACT FAILED"; exit 1; }
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_0.log
echo "DV2 EXTRACT DONE"
J=$W/runs_DV2/jobs_DV2.jsonl; mkdir -p $W/runs_DV2; rm -f $J
$PY - <<PYEOF
import json
W, D = "$W", "$D"
cfgs = "$CFGS".split(",")
big = {"dense", "in16f", "kbsmpa_s25", "wampa_s25", "wamshufpa_s25"}      # read per batch from disk
order = [c for c in cfgs if c in big] + [c for c in cfgs if c not in big]
with open("$J", "w") as fh:
    for c in order:
        for s in range(4):
            out = f"{W}/runs_DV2/off_cls/{c}/vis_official/s{s}"
            cmd = (f"cd {W}/repo/wam_forecast && WAM_SLIM={D}/slim WAM_SPLITS={D}/splits WAM_FEAT_ROOT={W}/feats_DV2 "
                   f"WAM_RUN_ROOT={W}/runs_DV2 {W}/conda/bin/python train_cls.py --cfg {c} --seed {s} --cv_fold -1 --epochs 20 "
                   f"--out {out}")
            fh.write(json.dumps(dict(cmd=cmd, done=out + "/metrics.json", log=out + "/log.txt")) + "\n")
PYEOF
wc -l $J
$PY gpu_pool_claim.py $J --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
echo "DV2 CV DONE"
$PY roundDV2_analysis.py
echo "DV2 ANALYSIS DONE"
