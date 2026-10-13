#!/bin/bash
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
until grep -q "\[pool\] finished" $W/tmp/pool_S.log; do sleep 120; done
for m in fixed30 uniform; do
  WAM_SLIM=$W/data/egodex_fps/$m/slim WAM_SPLITS=$W/data/egodex_fps/$m/splits WAM_FEAT_ROOT=$W/feats_fps/$m WAM_RUN_ROOT=$W/runs_S/$m $PY roundS_analysis.py --tag $m
done
echo "S ANALYSIS DONE"
