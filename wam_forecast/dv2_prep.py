"""Round DV2 preparation: an empty hand mask file (Diving48 has no joints) and a decoding check against round DV."""
import numpy as np

from common import FEAT_ROOT, W, all_clips, past_frames

keys = all_clips()
n = len(keys)
print("clips", n)
FEAT_ROOT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(FEAT_ROOT / "hand_masks.npz", mask=np.zeros((n, 16, 16, 16), bool))
# the stored round DV frames of the same test clips must equal the on the fly decode
old = W / "data" / "diving48" / "v2test" / "slim"
diffs = []
for k in [k for k in keys if (old / k[0] / k[1].replace(".mp4", ".npz")).exists()][:8]:
    with np.load(old / k[0] / k[1].replace(".mp4", ".npz")) as z:
        a = z["imgs"][:32].astype(np.int16)
    b = past_frames(k).astype(np.int16)
    diffs.append(int(np.abs(a - b).max()) if a.shape == b.shape else -1)
print("decode check, max abs pixel difference on 8 test clips:", diffs)
assert all(d == 0 for d in diffs), diffs
