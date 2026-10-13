"""Second task (seventh review): Diving48 V2, the 1970 clips of its official test split (mteb/diving48v2 parquet
copy of the official release). Each clip: 32 frames at np.linspace(start_frame, end_frame, 32) of its video, resized so
that the short side is 292 (the encoder then centre crops 256), stored compressed as <out>/slim/c<label>/<vid>.npz.
All clips go into train_cache.txt (the CV pool); <out>/labels.json maps each clip to its class.
  python build_diving.py --parquet data/diving48/mteb --out data/diving48/v2test"""
import argparse
import glob
import io
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SHORT = 292


def decode(args):
    vid, data, s, e, label, out = args
    dst = Path(out) / "slim" / f"c{label:02d}" / f"{vid}.npz"
    if dst.exists():
        return str(dst), label, None
    try:
        import cv2
        from decord import VideoReader, cpu
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(data); f.flush()
            vr = VideoReader(f.name, ctx=cpu(0))
            n = len(vr)
            e = min(int(e), n - 1); s = min(max(int(s), 0), e)
            fi = np.round(np.linspace(s, e, 32)).astype(np.int64)
            imgs = vr.get_batch(fi).asnumpy()
        H, W = imgs.shape[1:3]
        sc = SHORT / min(H, W)
        nh, nw = int(round(H * sc)), int(round(W * sc))
        imgs = np.stack([cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA) for im in imgs])
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, imgs=imgs, frame_indices=fi, label=np.int64(label))
        tmp.rename(dst)
        return str(dst), label, None
    except Exception as ex:
        return str(dst), label, repr(ex)


def rows(parquet):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(parquet, "test-*.parquet"))):
        t = pq.read_table(f)
        for r in t.to_pylist():
            v = r["video"]
            data = v["bytes"] if isinstance(v, dict) else v
            yield r["vid_name"], data, r["start_frame"], r["end_frame"], int(r["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(decode, ((*r, str(out)) for r in rows(a.parquet)), chunksize=4))
    ok = sorted((d, l) for d, l, err in res if err is None)
    bad = [(d, err) for d, l, err in res if err is not None]
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "train_cache.txt").write_text("\n".join(d for d, _ in ok) + "\n")
    (out / "splits" / "test_cache.txt").write_text("")
    (out / "labels.json").write_text(json.dumps({Path(d).name: l for d, l in ok}))
    print(json.dumps(dict(clips=len(res), built=len(ok), failed=len(bad), failures=bad[:5], classes=len({l for _, l in ok}))))


if __name__ == "__main__":
    main()
