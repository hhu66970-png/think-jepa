"""Eighth review (round DV2): Diving48 V2 on its official split. Writes the original video of every clip of the official
train (15027) and test (1970) splits from the mteb/diving48v2 parquet copy, unchanged, as <out>/slim/c<label>/<vid>.mp4,
with <out>/segments.json (start and end frame of the dive), <out>/labels.json, and <out>/splits/{train,test}_cache.txt.
Frames are decoded on the fly by common.past_frames, exactly as build_diving.py stored them.
  python build_diving2.py --parquet data/diving48/mteb --out data/diving48/official"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    out = Path(a.out)
    seg, lab, split = {}, {}, {"train": [], "test": []}
    for name in ("train", "test"):
        files = sorted(glob.glob(os.path.join(a.parquet, f"{name}-*.parquet")))
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                for r in pf.read_row_group(rg).to_pylist():
                    v = r["video"]
                    data = v["bytes"] if isinstance(v, dict) else v
                    vid, label = r["vid_name"], int(r["label"])
                    fn = f"{vid}.mp4"
                    assert fn not in lab, f"duplicate clip {vid}"
                    dst = out / "slim" / f"c{label:02d}" / fn
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dst.with_suffix(".tmp")
                        tmp.write_bytes(data)
                        tmp.rename(dst)
                    seg[fn] = [int(r["start_frame"]), int(r["end_frame"])]
                    lab[fn] = label
                    split[name].append(str(dst))
            print(name, f, len(split[name]), flush=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        (out / "splits" / f"{name}_cache.txt").write_text("\n".join(sorted(split[name])) + "\n")
    (out / "segments.json").write_text(json.dumps(seg))
    (out / "labels.json").write_text(json.dumps(lab))
    print(json.dumps(dict(train=len(split["train"]), test=len(split["test"]),
                          classes_train=len({lab[Path(p).name] for p in split["train"]}),
                          classes_test=len({lab[Path(p).name] for p in split["test"]}))))


if __name__ == "__main__":
    main()
