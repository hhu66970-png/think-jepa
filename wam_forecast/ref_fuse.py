"""Reference outputs (WAM_LEAN unset) of the round FUSE arms on 8 clips, for the equality check."""
import os, sys, torch, numpy as np
assert os.environ.get("WAM_LEAN", "0") != "1"
import extract as E
from common import all_clips, past_frames
sys.argv += []
import bench_fuse as BF
keys = all_clips()[:8]
u8 = torch.stack([torch.from_numpy(np.ascontiguousarray(past_frames(k))) for k in keys]).cuda()
x8 = E.preprocess(u8)
enc = E.load_encoder()
out = {}
for name in BF.ARMS:
    E.set_merge(enc, BF.cfg_of(name))
    base = name.split("+", 1)[0]
    xin = E.INPUT_DS[base](x8) if base in E.INPUT_DS else x8
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = enc(xin, restore_dense=False)
    st = getattr(enc, "last_merge_state", None) if BF.cfg_of(name) is not None else None
    out[name] = dict(out=o.cpu(), ids=None if st is None else st[0].cpu())
torch.save(out, sys.argv[1])
print("saved", len(out))
