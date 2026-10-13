#!/bin/bash
# Ninth review: features of rounds DV3 (Diving48 frame frontier), VM (VideoMAE on Diving48), SEG (EgoDex), then 100 runs.
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until grep -q "LAT2 PROF DONE" $W/tmp/stage_lat2.log; do sleep 30; done
D=$W/data/diving48/official
mkdir -p /data1/anon/wam_feats_DV2/in24f
[ -e $W/feats_DV2/in24f ] || ln -s /data1/anon/wam_feats_DV2/in24f $W/feats_DV2/in24f
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s WAM_FEAT_ROOT=$W/feats $PY extract.py --configs handcursegpa_L12,handcursegshufpa_L12 --shard $s --nshards 3 --batch 4 > $W/feats/logs/extract_SEG_$s.log 2>&1 &
done
CFGS=in24f,in12f,in24f+kbsmpa_s25,in24f+wampa_s25,in24f+wamshufpa_s25,in16f+kbsmpa_s25,in16f+wampa_s25,in16f+wamshufpa_s25
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_FEAT_INT8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY extract.py --configs $CFGS --shard $s --nshards 6 --batch 4 > $W/feats_DV2/logs/extract_DV3_$s.log 2>&1 &
done
wait
grep -il traceback $W/feats/logs/extract_SEG_*.log $W/feats_DV2/logs/extract_DV3_*.log && { echo "R9 EXTRACT FAILED"; exit 1; }
echo "DV3 SEG EXTRACT DONE"
export VIDEOMAE_PATH=$W/cache/hf/hub/models--MCG-NJU--videomae-large/snapshots/12da269a02d3e1fbbb7011e610e591fea8061dca/
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_FEAT_INT8=1 $PY extract_videomae.py --configs dense,kbsmpa_s25,wampa_s25,wamshufpa_s25,kbsmpa_L9,wampa_L9,wamshufpa_L9 --shard $s --nshards 6 --batch 4 > $W/feats_DV2/logs/extract_VM_$s.log 2>&1 &
done
wait
grep -il traceback $W/feats_DV2/logs/extract_VM_*.log && { echo "VM EXTRACT FAILED"; exit 1; }
echo "VM EXTRACT DONE"
$PY jobs_r9.py
$PY gpu_pool_claim.py $W/runs_DV2/jobs_r9.jsonl --gpus 0,0,1,1,2,2,3,3,4,4,5,5 --per_gpu 1
echo "R9 CV DONE"
WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_RUN_ROOT=$W/runs_DV2 $PY roundDV3_analysis.py
WAM_SLIM=$D/slim WAM_SPLITS=$D/splits WAM_FEAT_ROOT=$W/feats_DV2 WAM_RUN_ROOT=$W/runs_DV2 $PY roundVM_analysis.py
WAM_FEAT_ROOT=$W/feats $PY roundSEG_analysis.py
WAM_FEAT_ROOT=$W/feats $PY mechanism2.py
echo "R9 ANALYSIS DONE"
