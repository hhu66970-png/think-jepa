"""Resumable GPU job pool.

jobs file: one JSON object per line: {"cmd": "...", "done": "<file that exists when finished>",
"log": "<log path>"}. Jobs whose `done` file exists are skipped, so re-running the same file
continues where it stopped. Usage:
  python gpu_pool.py jobs.jsonl --gpus 0,1,2,3,4,5 --per_gpu 2
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--per_gpu", type=int, default=1)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    todo = [j for j in jobs if not Path(j["done"]).exists()]
    print(f"[pool] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    running = {}                         # slot index -> (Popen, job, t0)
    failed = 0
    while todo or running:
        for s in range(len(slots)):
            if s not in running and todo:
                j = todo.pop(0)
                Path(j["log"]).parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[s])
                fh = open(j["log"], "w")
                p = subprocess.Popen(j["cmd"], shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env)
                running[s] = (p, j, time.time(), fh)
        time.sleep(2)
        for s in list(running):
            p, j, t0, fh = running[s]
            if p.poll() is not None:
                fh.close()
                ok = p.returncode == 0 and Path(j["done"]).exists()
                failed += not ok
                print(f"[pool] {'OK ' if ok else 'FAIL'} gpu{slots[s]} {time.time()-t0:6.0f}s  {j['log']}", flush=True)
                del running[s]
    print(f"[pool] finished, failures={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
