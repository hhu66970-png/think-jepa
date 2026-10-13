#!/bin/bash
HP="--lr 2e-4 --epochs 80"
W=$WAM_ROOT
PY=$W/conda/bin/python
S=$WAM_SHM/feats
cd $W/repo/wam_forecast
for p in $(pgrep -f stage_b1.sh); do kill $p; done
for p in $(pgrep -f "extract.py --configs kbsmlm"); do kill $p; done
sleep 3
rm -rf $W/feats/kbsmlm_s25 $W/feats/kbsmlm_L9 $W/feats/kbsmlm_L12
C=""
for m in kbsmlm wampa handmultpa mrecv15pa rrecv15pa kbsmlmpa; do for b in s25 L9 L12; do C="$C,${m}_$b"; done; done
C=${C#,}
echo "configs: $C"
export WAM_FEAT_ROOT=$W/feats
for s in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$s $PY extract.py --configs $C --shard $s --nshards 6 --batch 8 > $W/feats/logs/extract_parel_$s.log 2>&1 &
done
wait
grep -h "K=" $W/feats/logs/extract_parel_0.log
grep -il traceback $W/feats/logs/extract_parel_*.log
# free shm from finished rounds, then stage the new configs
for d in kbsm_L10g crop_d256_h0_c11 crop_d256_h1_c11 crop_raw_h0_c11 crop_raw_h1_c11 crop_raw_h0_dense crop_raw_h1_dense; do rm -rf $S/$d; done
for c in ${C//,/ }; do mkdir -p $S/$c && cp $W/feats/$c/* $S/$c/; done
df -h /dev/shm | tail -1
CUDA_VISIBLE_DEVICES=0 $PY alloc_metrics.py --split train --cfgs kbsm_L9,kbsmpa_L9,kbsmlm_L9,wampa_L9,handmultpa_L9,mrecv15pa_L9,rrecv15pa_L9,kbsmlmpa_L9 \
  --out $W/feats/alloc_parel.json 2>&1 | grep -v Warn | grep -v "tok = " &
export WAM_FEAT_ROOT=$S
J=$W/runs/test_main/jobs_parel.jsonl; rm -f $J
L9L12=$(for m in kbsmlm wampa handmultpa mrecv15pa rrecv15pa kbsmlmpa; do printf "%s_L9,%s_L12," $m $m; done)
S25=$(for m in kbsmlm wampa handmultpa mrecv15pa rrecv15pa kbsmlmpa; do printf "%s_s25," $m; done)
$PY make_jobs.py test_main --cfgs ${L9L12%,} --inputs vis --seeds 0-19 --extra="$HP" --out $J
$PY make_jobs.py test_main --cfgs ${S25%,} --inputs vis --seeds 0-9 --extra="$HP" --out $J
$PY gpu_pool.py $J --gpus 0,1,2,3,4,5 --per_gpu 3
wait
echo "== B1 (no PA) vs kbsm"
$PY pairs.py $W/runs/test_main kbsmlm_s25/vis:kbsm_s25/vis kbsmlm_L9/vis:kbsm_L9/vis kbsmlm_L12/vis:kbsm_L12/vis
echo "== PA arms vs kbsmpa"
for b in L9 L12 s25; do
  $PY pairs.py $W/runs/test_main wampa_$b/vis:kbsmpa_$b/vis handmultpa_$b/vis:kbsmpa_$b/vis mrecv15pa_$b/vis:kbsmpa_$b/vis rrecv15pa_$b/vis:kbsmpa_$b/vis kbsmlmpa_$b/vis:kbsmpa_$b/vis
done
echo "PAREL DONE"
