"""Round X split: train = the 1800 part 2 training clips, test = every clip of the 13 part 2 tasks in the held out
archive (official EgoDex test set, built by build_testset.py) whose (task, episode) label does not occur in part 2.
Writes <out>/slim (symlinks; held out clips prefixed ho_) and <out>/splits/{train,test}_cache.txt.
  python build_xsplit.py   (then run with WAM_SLIM=<out>/slim WAM_SPLITS=<out>/splits)"""
import argparse
import os
from pathlib import Path

from common import W


def keys(path):
    return [(Path(l.strip()).parent.name, Path(l.strip()).name) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part2", default=str(W / "data" / "egodex_slim" / "part2"))
    ap.add_argument("--part2_splits", default=str(W / "tmp"))
    ap.add_argument("--heldout", default=str(W / "data" / "egodex_testset" / "slim"))
    ap.add_argument("--out", default=str(W / "data" / "egodex_x"))
    a = ap.parse_args()
    P2, HO, X = Path(a.part2), Path(a.heldout), Path(a.out)
    tr = keys(Path(a.part2_splits) / "train_cache.txt")
    te0 = keys(Path(a.part2_splits) / "test_cache.txt")
    seen = {(t, n.split("_")[0]) for t, n in tr + te0}
    test, excl = [], 0
    for t in sorted(os.listdir(P2)):
        for f in sorted(os.listdir(HO / t), key=lambda s: int(s.split(".")[0])):
            if (t, f.replace(".npz", "")) in seen:
                excl += 1
                continue
            test.append((t, f))
    (X / "splits").mkdir(parents=True, exist_ok=True)
    for t, n in tr:
        (X / "slim" / t).mkdir(parents=True, exist_ok=True)
        if not (X / "slim" / t / n).exists():
            (X / "slim" / t / n).symlink_to(P2 / t / n)
    for t, f in test:
        (X / "slim" / t).mkdir(parents=True, exist_ok=True)
        if not (X / "slim" / t / ("ho_" + f)).exists():
            (X / "slim" / t / ("ho_" + f)).symlink_to(HO / t / f)
    (X / "splits" / "train_cache.txt").write_text("".join(f"{X}/slim/{t}/{n}\n" for t, n in tr))
    (X / "splits" / "test_cache.txt").write_text("".join(f"{X}/slim/{t}/ho_{f}\n" for t, f in test))
    print(f"train {len(tr)} test {len(test)} excluded colliding labels {excl}")


if __name__ == "__main__":
    main()
