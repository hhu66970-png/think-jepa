#!/bin/bash
# Diving48 V2 official train split via aria2c (resumes the partial files left by curl).
cd $WAM_ROOT/data/diving48/mteb
until aria2c -q --continue=true --file-allocation=none -j 17 -x 8 -s 8 -k 1M --max-tries=0 --retry-wait=5 -i $WAM_ROOT/tmp/dl_urls.txt; do sleep 10; done
echo "ALL TRAIN SHARDS DONE"
