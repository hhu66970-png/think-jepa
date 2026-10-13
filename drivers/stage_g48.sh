#!/bin/bash
# Tenth review, round G48: ViT-g features on Diving48 after the round TOK runs, then 16 runs and the analysis.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
D=$W/data/diving48/official
until grep -q "R10 RETRY DONE" $W/tmp/stage_r10c.log 2>/dev/null; do sleep 60; done
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_FEAT_INT8=1 WAM_VITG=$W/repo/vjepa2/vitg_encoder.pt PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY extract.py --arch vitg --configs in12f,kbsmpa_gL9,wampa_gL9,wamshufpa_gL9 --shard $s --nshards 6 --batch 4 > $W/feats_DV2/logs/extract_G48_$s.log 2>&1 &
done
wait
grep -il traceback $W/feats_DV2/logs/extract_G48_*.log && { echo "G48 EXTRACT FAILED"; exit 1; }
echo "G48 EXTRACT DONE"
$PY jobs_g48.py
$PY gpu_pool_claim.py $W/runs_DV2/jobs_g48.jsonl --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
echo "G48 RUNS DONE"
WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_RUN_ROOT=$W/runs_DV2 $PY roundG48_analysis.py > $W/tmp/roundG48.txt 2>&1
echo "G48 ANALYSIS DONE"
