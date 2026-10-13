#!/bin/bash
# Round DV2 (eighth review), from the flow maps on: the clip set is built (build_diving2.py).
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
D=$W/data/diving48/official
export WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_RUN_ROOT=$W/runs_DV2 WAM_FEAT_INT8=1
$PY dv2_prep.py || { echo "PREP FAILED"; exit 1; }
$PY flow_maps.py --workers 32 > $WAM_FEAT_ROOT/logs/flow.log 2>&1 || { echo "FLOW FAILED"; exit 1; }
tail -n 1 $WAM_FEAT_ROOT/logs/flow.log
echo "DV2 PREP DONE"
CFGS=dense,in16f,kbsmpa_s25,wampa_s25,wamshufpa_s25,kbsmpa_L9,wampa_L9,wamshufpa_L9,kbsmpa_L12,wampa_L12,wamshufpa_L12,in16f+kbsmpa_L9,in16f+wampa_L9,in16f+wamshufpa_L9,vidtldrpa_L9,vidtldrshufpa_L9,pospa_L9,flowpa_L9,flowshufpa_L9,flowcpa_L9,flowcshufpa_L9
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $WAM_FEAT_ROOT/logs/extract_$s.log 2>&1 &
done
wait
grep -il traceback $WAM_FEAT_ROOT/logs/extract_*.log && { echo "DV2 EXTRACT FAILED"; exit 1; }
grep -h "K=" $WAM_FEAT_ROOT/logs/extract_0.log
echo "DV2 EXTRACT DONE"
$PY jobs_DV2.py
$PY gpu_pool_claim.py $W/runs_DV2/jobs_DV2.jsonl --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
echo "DV2 CV DONE"
$PY roundDV2_analysis.py
echo "DV2 ANALYSIS DONE"
