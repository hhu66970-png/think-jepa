"""Round TOK (tenth review): pooled OUTPUT tokens as feature configs, so the probes read them with the compact readout.
<src>@p<kt><kh><kw>: the restored output of <src> (tokens[idx] on the 16x16x16 grid) average pooled by a (kt,kh,kw) box with
train_probe.out_pool; idx maps every cell to its pooled token. int8 sources are written int8 (per token scales, as extract.py).
  WAM_FEAT_ROOT=... python pool_feats.py --src dense --k 2,4,4"""
import argparse
import json

import numpy as np
import torch

from common import FEAT_ROOT, TOKENS_PAST, write_json
from extract import quantize_int8
from train_probe import out_pool


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--k", required=True)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    k = tuple(int(v) for v in a.k.split(","))
    name = f"{a.src}@p{k[0]}{k[1]}{k[2]}"
    sd, od = FEAT_ROOT / a.src, FEAT_ROOT / name
    od.mkdir(parents=True, exist_ok=True)
    K = TOKENS_PAST // (k[0] * k[1] * k[2])
    for tf in sorted(sd.glob("tokens_*.npy")):
        tag = tf.stem.split("_", 1)[1]
        tok = np.load(tf, mmap_mode="r")
        idxf, scf = sd / f"idx_{tag}.npy", sd / f"scale_{tag}.npy"
        idx = np.load(idxf, mmap_mode="r") if idxf.exists() else None
        sc = np.load(scf, mmap_mode="r") if scf.exists() else None
        n, D = tok.shape[0], tok.shape[2]
        int8 = sc is not None
        ot = np.lib.format.open_memmap(od / f"tokens_{tag}.npy", "w+", np.int8 if int8 else np.float16, (n, K, D))
        os_ = np.lib.format.open_memmap(od / f"scale_{tag}.npy", "w+", np.float16, (n, K)) if int8 else None
        oi = np.lib.format.open_memmap(od / f"idx_{tag}.npy", "w+", np.int16, (n, TOKENS_PAST))
        for i in range(0, n, a.chunk):
            t = torch.from_numpy(np.asarray(tok[i:i + a.chunk])).cuda()
            if int8:
                t = (t.float() * torch.from_numpy(np.asarray(sc[i:i + a.chunk])).cuda().float().unsqueeze(-1)).half()
            if idx is not None:
                ix = torch.from_numpy(np.asarray(idx[i:i + a.chunk]).astype(np.int64)).cuda()
                t = t.gather(1, ix.unsqueeze(-1).expand(-1, -1, D))
            p, pidx = out_pool(t, k)
            if int8:
                q, s = quantize_int8(p.float())
                ot[i:i + len(p)] = q.cpu().numpy(); os_[i:i + len(p)] = s.cpu().numpy()
            else:
                ot[i:i + len(p)] = p.cpu().numpy().astype(np.float16)
            oi[i:i + len(p)] = pidx.cpu().numpy().astype(np.int16)
        ot.flush(); oi.flush()
        if os_ is not None:
            os_.flush()
        np.save(od / f"rows_{tag}.npy", np.load(sd / f"rows_{tag}.npy"))
        write_json(dict(src=a.src, k=list(k), K=K, int8=int8, clips=n), od / f"meta_{tag}.json")
        print(name, tag, n, "clips", K, "tokens", flush=True)


if __name__ == "__main__":
    main()
