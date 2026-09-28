"""Row-mapping check: re-encode a few clips densely and compare with the cached dense bank."""
import numpy as np
import torch

import extract as E
from common import all_clips, clip_path
from train_probe import FeatureBank

keys = all_clips()
bank = FeatureBank("dense")
enc = E.load_encoder()
for rows in ([0, 6, 12, 18, 24, 30, 36, 42], [0, 1, 2, 3]):   # shard-0 batch vs mixed batch
    u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[r]))["imgs"][:32]) for r in rows]).cuda()
    E.set_merge(enc, None)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = enc(E.preprocess(u8), restore_dense=False).half().float().cpu()
    for b, r in enumerate(rows):
        s, j = bank.where[r]
        c = torch.from_numpy(np.asarray(bank.tok[s][j]).astype(np.float32))
        cos = torch.nn.functional.cosine_similarity(out[b], c, dim=-1)
        other = torch.from_numpy(np.asarray(bank.tok[bank.where[(r + 1) % 2000][0]][bank.where[(r + 1) % 2000][1]]).astype(np.float32))
        print(f"batch={rows[:3]}.. row {r}: max|diff| {float((out[b]-c).abs().max()):.4f}  "
              f"mean cos same {float(cos.mean()):.5f}  vs next clip {float(torch.nn.functional.cosine_similarity(out[b], other, dim=-1).mean()):.3f}")
