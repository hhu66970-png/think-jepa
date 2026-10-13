"""Seventh review, sampling control: the same held out episodes of the 13 part 2 tasks (official EgoDex test archive, at
least 64 frames) sampled two ways.
  uniform  64 frames at np.linspace(0, T-1, 64) over the episode, as every clip so far
  fixed30  64 consecutive frames (30 fps) centred in the episode: frames 0-31 observed, 32-63 forecast
Frames decoded with decord at 256x256, joints and cameras as build_testset.py; stored compressed.
Split: 10% of the episodes held out as test (RandomState(2029)), never used; CV runs use the rest.
  python build_fps.py --mode fixed30 --out data/egodex_fps/fixed30"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))
import egodex.trajectory_dataset as TD  # noqa: E402
from common import W  # noqa: E402

MODE = None


def fixed30(T):
    start = max(0, (T - 64) // 2)
    return start + np.arange(64)


def build_one(args):
    h5, out, mode = args
    h5 = Path(h5)
    dst = Path(out) / "slim" / h5.parent.name / f"{h5.stem}.npz"
    if dst.exists():
        return str(dst), None
    try:
        from decord import VideoReader, cpu
        if mode == "fixed30":
            TD._sample_dense_jepa_frame_indices = fixed30
        sup = TD.load_supervision_from_hdf5(str(h5), TD.NPZ_TARGET_QUERY_TFS)
        vr = VideoReader(str(h5.with_suffix(".mp4")), ctx=cpu(0), width=256, height=256)
        fi = np.clip(sup["frame_indices"], 0, len(vr) - 1).astype(np.int64)
        imgs = vr.get_batch(fi).asnumpy()
        assert imgs.shape == (64, 256, 256, 3), imgs.shape
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, imgs=imgs, frame_indices=sup["frame_indices"], xyz_cam=sup["xyz_cam"].astype(np.float32),
                            xyz_world=sup["xyz_world"].astype(np.float32), cam_ext=sup["cam_ext"].astype(np.float32),
                            cam_int=np.asarray(sup["cam_int"], np.float32), confs=sup["confs"].astype(np.float32),
                            lang_instruct=np.asarray(str(sup["lang_instruct"])), path=np.asarray(str(h5)))
        tmp.rename(dst)
        return str(dst), None
    except Exception as e:
        return str(dst), repr(e)


def n_frames(h5):
    with h5py.File(h5, "r") as r:
        return int(r["/transforms/camera"].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uniform", "fixed30"], required=True)
    ap.add_argument("--src", default=str(W / "data" / "egodex_test" / "unzipped" / "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    tasks = sorted(os.listdir(W / "data" / "egodex_slim" / "part2"))
    eps = sorted(p for t in tasks for p in (Path(a.src) / t).glob("*.hdf5") if p.with_suffix(".mp4").exists())
    with ProcessPoolExecutor(a.workers) as ex:
        T = list(ex.map(n_frames, map(str, eps), chunksize=16))
    keep = [p for p, t in zip(eps, T) if t >= 64]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(build_one, [(str(p), str(out), a.mode) for p in keep], chunksize=4))
    ok = sorted(d for d, err in res if err is None)
    bad = [(d, err) for d, err in res if err is not None]
    perm = np.random.RandomState(2029).permutation(len(ok))
    n_test = len(ok) // 10
    test = sorted(ok[i] for i in perm[:n_test]); train = sorted(ok[i] for i in perm[n_test:])
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "train_cache.txt").write_text("\n".join(train) + "\n")
    (out / "splits" / "test_cache.txt").write_text("\n".join(test) + "\n")
    info = dict(mode=a.mode, episodes=len(eps), ge64=len(keep), built=len(ok), failed=len(bad), failures=bad[:10],
                train=len(train), test=len(test), frames_median=float(np.median(T)))
    (out / "build.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
