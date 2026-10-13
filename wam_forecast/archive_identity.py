"""Content check of the clips whose (task, episode) label occurs both in EgoDex part 2 and in the held out archive.

For every colliding label, hashes (SHA-256 of the raw array bytes) the video frames ("imgs") and the joints
("xyz_cam") of both copies and records whether the content is identical. Reads clip_tasks.json (the clip lists
of both archives); writes <out>/archive_identity.json with the full hashes and the short prefixes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import W


def find(root, t, e):
    c = list((root / t).glob(f"{e}.npz")) + list((root / t).glob(f"{e}*"))
    return c[0] if c else None


def sha(p, key):
    with np.load(p) as z:
        return hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(W / "runs" / "paper_data" / "clip_tasks.json"))
    ap.add_argument("--part2", default=str(W / "data" / "egodex_slim" / "part2"))
    ap.add_argument("--heldout", default=str(W / "data" / "egodex_testset" / "slim"))
    ap.add_argument("--out", default=str(W / "runs" / "phaseA"))
    a = ap.parse_args()
    d = json.load(open(a.tasks))
    P2, HO = Path(a.part2), Path(a.heldout)
    p2 = [(f"part2/{s}", t, e) for s in ("train", "test") for t, e in d["part2"][s]]
    ho = [(f"heldout/{s}", t, e) for s in ("train", "test") for t, e in d["heldout"][s]]
    idx = {}
    for s, t, e in ho:
        idx.setdefault((t, str(e)), []).append(s)
    rows, same, diff, miss = [], 0, 0, 0
    for s, t, e in p2:
        if (t, str(e)) not in idx:
            continue
        hs = idx[(t, str(e))]
        fa, fb = find(P2, t, str(e)), find(HO, t, str(e))
        if fa is None or fb is None:
            miss += 1
            rows.append(dict(part2_split=s, heldout_splits=hs, task=t, episode=str(e), content="missing"))
            continue
        ia, ib, pa, pb = sha(fa, "imgs"), sha(fb, "imgs"), sha(fa, "xyz_cam"), sha(fb, "xyz_cam")
        eq = ia == ib and pa == pb
        same += eq; diff += not eq
        rows.append(dict(part2_split=s, heldout_splits=hs, task=t, episode=str(e), content="SAME" if eq else "different",
                         imgs_sha=ia[:16] + "/" + ib[:16], joints_sha=pa[:16] + "/" + pb[:16],
                         imgs_sha256=[ia, ib], joints_sha256=[pa, pb]))
    out = dict(collisions=rows, identical=same, different=diff, missing=miss)
    Path(a.out, "archive_identity.json").write_text(json.dumps(out, indent=1))
    print(f"collisions {len(rows)}: identical content {same}, different {diff}, missing {miss}")


if __name__ == "__main__":
    main()
