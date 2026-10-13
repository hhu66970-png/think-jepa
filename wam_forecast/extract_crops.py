"""Encode the tracked hand crops (crop_hands.py) with the frozen V-JEPA ViT-L and cache the
(merged) tokens, one feature config per (source, hand, schedule):
  <FEAT_ROOT>/crop_<src>_h<hand>_<sched>/tokens_*.npy, idx_*.npy, rows_*.npy
src in {raw, d256}; hand 0 = right, 1 = left. Crops are S x S already, so preprocessing is
only /255 and ImageNet normalisation (no resize or centre crop). An S=128 crop gives an
8 x 8 x 16 = 1024-token grid; RoPE rescales spatial positions to the 16-grid internally, so the
crop's location in the frame must be supplied to the read-out (see train_probe --crops).

Usage: CUDA_VISIBLE_DEVICES=0 python extract_crops.py --S 128 --scheds dense,c11 --shard 0 --nshards 6
"""
import argparse
import time

import numpy as np
import torch

import extract as E
from common import FEAT_ROOT, merge_config, write_json



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--scheds", default="dense,c11")
    ap.add_argument("--srcs", default="raw,d256")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    d = FEAT_ROOT / f"crops_S{a.S}"
    arrs = {s: np.load(d / f"{s}.npy", mmap_mode="r") for s in a.srcs.split(",")}
    N = next(iter(arrs.values())).shape[0]
    rows = [i for i in range(N) if i % a.nshards == a.shard]
    enc = E.load_encoder()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)
    tag = f"{a.shard}of{a.nshards}"
    out, K_of, t_enc = {}, {}, {}
    for w in range(0, len(rows), a.batch):
        rb = rows[w:w + a.batch]
        for src, arr in arrs.items():
            for hand in (0, 1):
                u8 = torch.from_numpy(np.ascontiguousarray(arr[rb, hand])).cuda()   # [B,32,S,S,3]
                x = (u8.float().div_(255).permute(0, 4, 1, 2, 3) - mean) / std       # [B,3,32,S,S]
                for sched in a.scheds.split(","):
                    name = f"crop_{src}_h{hand}_{sched}"
                    cfg = None if sched == "dense" else merge_config("kbsm", sched)
                    E.set_merge(enc, cfg)
                    torch.cuda.synchronize(); t0 = time.time()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        f = enc(x, restore_dense=False)
                    torch.cuda.synchronize(); t_enc[name] = t_enc.get(name, 0) + time.time() - t0
                    B, K, D = f.shape
                    if name not in out:
                        dd = FEAT_ROOT / name
                        dd.mkdir(parents=True, exist_ok=True)
                        ntok = f.shape[1] if cfg is None else (a.S // 16) ** 2 * 16
                        out[name] = dict(
                            tok=np.lib.format.open_memmap(dd / f"tokens_{tag}.npy", "w+", np.float16, (len(rows), K, D)),
                            idx=None if cfg is None else np.lib.format.open_memmap(
                                dd / f"idx_{tag}.npy", "w+", np.int16, (len(rows), ntok)))
                        K_of[name] = K
                    assert K == K_of[name]
                    out[name]["tok"][w:w + B] = f.float().cpu().numpy().astype(np.float16)
                    if cfg is not None:
                        tid, _, rep = enc.last_merge_state
                        out[name]["idx"][w:w + B] = E.rep_index(tid, rep).cpu().numpy().astype(np.int16)
        if (w // a.batch) % 20 == 0:
            print(f"[{tag}] {w + len(rb)}/{len(rows)}", flush=True)
    for name, o in out.items():
        o["tok"].flush()
        if o["idx"] is not None:
            o["idx"].flush()
        np.save(FEAT_ROOT / name / f"rows_{tag}.npy", np.asarray(rows, np.int32))
        write_json(dict(K=K_of[name], S=a.S, clips=len(rows), encoder_s_per_clip=t_enc[name] / len(rows)),
                   FEAT_ROOT / name / f"meta_{tag}.json")
        print(f"{name}: K={K_of[name]} {1000 * t_enc[name] / len(rows):.1f} ms/clip")


if __name__ == "__main__":
    main()
