#!/bin/bash
# Wait for the aria2 download, give the shards their names, then run the round DV2 stage.
W=$WAM_ROOT
until grep -q "ALL TRAIN SHARDS DONE" $W/tmp/dl_train.log; do sleep 60; done
cd $W/data/diving48/mteb
while read f h; do [ -f "$h" ] && [ ! -f "$h.aria2" ] && mv -f "$h" "$f"; done < $W/tmp/dl_map.txt
ls *.aria2 2>/dev/null && { echo "INCOMPLETE SHARDS"; exit 1; }
echo "RENAMED $(date -u)"
exec bash $W/tmp/stage_DV2.sh
