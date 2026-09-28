"""Download the raw 1920x1080 episode videos needed by the 2000 official clips (HF main
branch raw_videos/<task>/<id>.mp4) with 8 parallel workers; skips files already present."""
import concurrent.futures as cf
import os
import time
import urllib.request

from common import W, all_clips

REPO, ENDPOINT = "haichaozhang/cache", "https://hf-mirror.com"
OUT = W / "data" / "raw_videos"


def vid_of(key):
    return key[0], key[1].split("_", 1)[0]


def one(tv, tok):
    task, vid = tv
    dst = OUT / task / f"{vid}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/raw_videos/{task}/{vid}.mp4"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            return len(data)
        except Exception as e:
            if attempt == 4:
                print("FAIL", task, vid, type(e).__name__, e, flush=True)
                return -1
            time.sleep(3 * (attempt + 1))


def main():
    tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    vids = sorted({vid_of(k) for k in all_clips()})
    print("unique videos:", len(vids), flush=True)
    t0, got, fails = time.time(), 0, 0
    with cf.ThreadPoolExecutor(8) as ex:
        for i, n in enumerate(ex.map(lambda v: one(v, tok), vids)):
            fails += n < 0
            got += max(n, 0)
            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(vids)}] {got/1e9:.2f} GB {got/1e6/(time.time()-t0):.2f} MB/s fails={fails}", flush=True)
    print(f"RAW DONE fails={fails} total {got/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
