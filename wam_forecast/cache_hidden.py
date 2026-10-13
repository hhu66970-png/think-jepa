"""Cache the dense hidden states after block L (the first merge layer) for TRAIN clips.
Output <FEAT_ROOT>/hidden_L<L>_train.npy [n,4096,1024] fp16 + rows. These are exactly the
features a relevance scorer sees at merge time (the merge runs after block L)."""
import argparse

import numpy as np
import torch

import extract as E
from common import FEAT_ROOT, all_clips, clip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    P = np.load(FEAT_ROOT / "poses.npz")
    rows = np.flatnonzero(~P["is_test"])
    keys = all_clips()
    enc = E.load_encoder()
    E.set_merge(enc, None)
    store = {}
    h = enc.blocks[a.layer].register_forward_hook(lambda m, i, o: store.__setitem__("x", o))
    out = np.lib.format.open_memmap(FEAT_ROOT / f"hidden_L{a.layer}_train.npy", "w+", np.float16,
                                    (len(rows), 4096, 1024))
    for i in range(0, len(rows), a.batch):
        r = rows[i:i + a.batch]
        u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[k]))["imgs"][:32]) for k in r]).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            enc(E.preprocess(u8), restore_dense=False)
        out[i:i + len(r)] = store["x"].half().cpu().numpy()
    out.flush()
    np.save(FEAT_ROOT / f"hidden_L{a.layer}_train_rows.npy", rows)
    h.remove()
    print("cached", out.shape)


if __name__ == "__main__":
    main()
