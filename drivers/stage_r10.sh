#!/bin/bash
# Tenth review: after round FUSE (idle host), features of round TOK, pooled outputs, then 512 runs in two pools.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until [ -f $W/runs/phaseA/bench_fuse.json ]; do sleep 30; done
echo "FUSE DONE"
D=$W/data/diving48/official
mkdir -p /data1/anon/wam_feats_DV2/in8f
[ -e $W/feats_DV2/in8f ] || ln -s /data1/anon/wam_feats_DV2/in8f $W/feats_DV2/in8f
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s WAM_FEAT_ROOT=$W/feats $PY extract.py --configs in4f,handcurshufpa_L9 --shard $s --nshards 3 --batch 4 > $W/feats/logs/extract_TOK_$s.log 2>&1 &
done
CFGS=in8f,in4f,kbsmpa_L12,wampa_L12,wamshufpa_L12,in16f+kbsmpa_L9,in16f+wampa_L9,in16f+wamshufpa_L9
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_FEAT_INT8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $W/feats_DV2/logs/extract_TOK_$s.log 2>&1 &
done
wait
grep -il traceback $W/feats/logs/extract_TOK_*.log $W/feats_DV2/logs/extract_TOK_*.log && { echo "R10 EXTRACT FAILED"; exit 1; }
echo "TOK EXTRACT DONE"
for spec in "dense 2,2,2" "dense 2,4,4" "in4f 8,2,2" "in8f 4,2,2"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=0 WAM_FEAT_ROOT=$W/feats $PY pool_feats.py --src $1 --k $2 >> $W/feats/logs/pool_TOK.log 2>&1 || { echo "POOL FAILED $1"; exit 1; }
  CUDA_VISIBLE_DEVICES=1 WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 $PY pool_feats.py --src $1 --k $2 >> $W/feats_DV2/logs/pool_TOK.log 2>&1 || { echo "POOL FAILED DV $1"; exit 1; }
done
echo "TOK POOL DONE"
$PY jobs_r10.py
$PY gpu_pool_claim.py $W/runs_DV2/jobs_r10.jsonl --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1 > $W/tmp/pool_r10_A.log 2>&1 &
# a third slot per GPU once every large bank job has been claimed
until [ $($PY -c "import json,os;J=[json.loads(l) for l in open('$W/runs_DV2/jobs_r10.jsonl')];print(sum(1 for j in J if j['big'] and not (os.path.exists(j['done']+'.claim') or os.path.exists(j['done']))))") -eq 0 ]; do sleep 60; done
$PY gpu_pool_claim.py $W/runs_DV2/jobs_r10.jsonl --gpus 0,1,2,3,4,5 --per_gpu 1 > $W/tmp/pool_r10_B.log 2>&1 &
wait
echo "R10 RUNS DONE"
