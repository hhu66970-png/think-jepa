"""Second dataset for the round-R replication: the official EgoDex test set (test.zip).

Clips are built exactly like the part-2 cache used so far: one clip per episode, 64 frames at
np.linspace(0, T-1, 64) over the whole episode (load_supervision_from_hdf5), the same 52 joints
(NPZ_TARGET_QUERY_TFS), frames decoded with decord at 256x256 (verified to reproduce the stored
part-2 frames: mean |diff| 0.001). Fixed before any result:
  * episodes with fewer than 64 frames are skipped;
  * if more than --max_clips (2000) episodes remain, a random subset (RandomState(2027)) is used;
  * split 80 / 20 into train / test by episode with RandomState(2028).
Outputs: <out>/slim/<task>/<id>.npz, <out>/splits/{train,test}_cache.txt, <out>/build.json

  python build_testset.py --src data/egodex_test/test --out data/egodex_testset --workers 16
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/21231_data1/huhaoming_wam/ThinkJEPA")
from egodex.trajectory_dataset import NPZ_TARGET_QUERY_TFS, load_supervision_from_hdf5  # noqa: E402


def n_frames(h5):
    with h5py.File(h5, "r") as r:
        return int(r["/transforms/camera"].shape[0])


def build_one(args):
    h5, out = args
    h5 = Path(h5)
    dst = Path(out) / "slim" / h5.parent.name / f"{h5.stem}.npz"
    if dst.exists():
        return str(dst), None
    try:
        from decord import VideoReader, cpu
        sup = load_supervision_from_hdf5(str(h5), NPZ_TARGET_QUERY_TFS)
        vr = VideoReader(str(h5.with_suffix(".mp4")), ctx=cpu(0), width=256, height=256)
        fi = np.clip(sup["frame_indices"], 0, len(vr) - 1).astype(np.int64)
        imgs = vr.get_batch(fi).asnumpy()
        assert imgs.shape == (64, 256, 256, 3), imgs.shape
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, imgs=imgs, frame_indices=sup["frame_indices"], xyz_cam=sup["xyz_cam"].astype(np.float32),
                 xyz_world=sup["xyz_world"].astype(np.float32), cam_ext=sup["cam_ext"].astype(np.float32),
                 cam_int=np.asarray(sup["cam_int"], np.float32), confs=sup["confs"].astype(np.float32),
                 lang_instruct=np.asarray(str(sup["lang_instruct"])), path=np.asarray(str(h5)))
        tmp.rename(dst)
        return str(dst), None
    except Exception as e:                                  # reported, never silently dropped
        return str(dst), repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_clips", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    eps = sorted(p for p in Path(a.src).rglob("*.hdf5") if p.with_suffix(".mp4").exists())
    with ProcessPoolExecutor(a.workers) as ex:
        T = list(ex.map(n_frames, map(str, eps), chunksize=16))
    keep = [p for p, t in zip(eps, T) if t >= 64]
    if len(keep) > a.max_clips:
        idx = np.sort(np.random.RandomState(2027).choice(len(keep), a.max_clips, replace=False))
        keep = [keep[i] for i in idx]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(build_one, [(str(p), str(out)) for p in keep], chunksize=4))
    ok = [d for d, err in res if err is None]
    bad = [(d, err) for d, err in res if err is not None]
    perm = np.random.RandomState(2028).permutation(len(ok))
    n_test = len(ok) // 5
    test = sorted(ok[i] for i in perm[:n_test])
    train = sorted(ok[i] for i in perm[n_test:])
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "train_cache.txt").write_text("\n".join(train) + "\n")
    (out / "splits" / "test_cache.txt").write_text("\n".join(test) + "\n")
    tasks = sorted({Path(d).parent.name for d in ok})
    info = dict(episodes_found=len(eps), episodes_ge64=sum(t >= 64 for t in T), used=len(keep), built=len(ok),
                failed=len(bad), failures=bad[:20], train=len(train), test=len(test), tasks=len(tasks),
                frames_median=float(np.median(T)) if T else 0)
    (out / "build.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "failures"}), flush=True)
    if bad:
        print("FAILURES (first 5):", bad[:5])


if __name__ == "__main__":
    main()
