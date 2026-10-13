#!/bin/bash
W=$WAM_ROOT; PY=$W/conda/bin/python; cd $W/repo/wam_forecast
for m in fixed30 uniform; do
  export WAM_SLIM=$W/data/egodex_fps/$m/slim WAM_SPLITS=$W/data/egodex_fps/$m/splits WAM_FEAT_ROOT=$W/feats_fps/$m
  mkdir -p $WAM_FEAT_ROOT/logs
  $PY prep_poses.py && $PY hand_masks.py && $PY flow_maps.py --workers 24 || { echo "PREP $m FAILED"; exit 1; }
done
echo "PREP S DONE"
