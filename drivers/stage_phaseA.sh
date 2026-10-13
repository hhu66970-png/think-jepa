#!/bin/bash
# Phase A, stage A1: extract capacity arms (all budgets), allocation metrics, validation selection.
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
C=""
for m in capc1pa capc2pa capc4pa caps2pa caps4pa caps8pa; do for b in L9 L12 s25; do C="$C,${m}_$b"; done; done
C=${C#,}
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $C --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_phaseA_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_phaseA_0.log
grep -il traceback $W/feats/logs/extract_phaseA_*.log && { echo "EXTRACT FAILED"; exit 1; }
# free shm: no-PA B1/B2 banks are finished (disk copies remain in $W/feats)
for d in b2_s25 b2_L9 b2_L12 kbsmlm_s25 kbsmlm_L9 kbsmlm_L12; do rm -rf $S/$d; done
for c in ${C//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
df -h /dev/shm | tail -1
export WAM_FEAT_ROOT=$S
J=$W/runs/select/jobs_phaseA.jsonl; rm -f $J
V=""
for m in kbsmpa capc1pa capc2pa capc4pa caps2pa caps4pa caps8pa rrecv15pa; do V="$V,${m}_L9,${m}_L12"; done
$PY make_jobs.py select --cfgs ${V#,} --inputs vis --seeds 0-9 --extra="$HP --val" --tag _val --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
echo "A1 VAL DONE"
for b in L9 L12 s25; do
  A=""; for m in kbsm kbsmpa capc1pa capc2pa capc4pa caps2pa caps4pa caps8pa rrecv15pa mrecv15pa handmultpa wampa kbsmlmpa b2pa; do A="$A,${m}_$b"; done
  CUDA_VISIBLE_DEVICES=0 $PY alloc_metrics.py --split train --ref kbsmpa --cfgs ${A#,} --out $W/feats/alloc_phaseA_$b.json 2>&1 | grep -v Warn | grep -v "tok = "
done
echo "PHASE A1 DONE"
