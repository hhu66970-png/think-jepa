"""Sanity tests for proportional attention (PA) in the merged V-JEPA encoder.

T1 (mathematical identity, one real encoder block, fp32): a token of size 2 with key bias
   log 2 must give exactly the same block output as the unmerged sequence in which that token
   appears twice (same RoPE position id) with no bias. This is the defining property of ToMe
   proportional attention; without the bias the two differ.
T2 (token size matters): perturbing token_size changes the block output only when PA is on.
T3 (end-to-end): with prop_attn=False the patched encoder reproduces the HEAD (pre-PA)
   vision_transformer.py bit-exactly; with prop_attn=True the output changes.
T4 (guard): PA must not silently switch off under activation checkpointing.

  CUDA_VISIBLE_DEVICES=0 python test_prop_attn.py
"""
import importlib.util
import subprocess
import sys

import numpy as np
import torch

import extract as E
from common import REPO, all_clips, clip_path, merge_config

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
enc = E.load_encoder().float()
blk = enc.blocks[14]
ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")


# ---------------- T1: size-2 token with log 2 bias == duplicated token, no bias -------------
N, D = 256, enc.embed_dim
ids = torch.randperm(4096, device="cuda")[:N].sort().values[None]   # arbitrary grid positions
x = torch.randn(1, N, D, device="cuda")
j = 17
x_dup = torch.cat([x, x[:, j:j + 1]], 1)
ids_dup = torch.cat([ids, ids[:, j:j + 1]], 1)
size = torch.ones(1, N, device="cuda"); size[0, j] = 2.0
bias = size.log()[:, None, None, :]
with torch.no_grad():
    y_pa = blk(x, mask=ids, attn_mask=bias, T=16, H_patches=16, W_patches=16)
    y_nopa = blk(x, mask=ids, attn_mask=None, T=16, H_patches=16, W_patches=16)
    y_dup = blk(x_dup, mask=ids_dup, attn_mask=None, T=16, H_patches=16, W_patches=16)[:, :N]
e_pa = float((y_pa - y_dup).abs().max()); e_no = float((y_nopa - y_dup).abs().max())
check("T1 PA == duplicated tokens", e_pa < 1e-3 and e_no > 100 * e_pa,
      f"max|PA - dup| = {e_pa:.2e}, max|noPA - dup| = {e_no:.2e}")

# ---------------- T2: token size changes the output only through PA ------------------------
size2 = torch.ones(1, N, device="cuda"); size2[0, :64] = torch.randint(2, 40, (64,), device="cuda").float()
with torch.no_grad():
    y_a = blk(x, mask=ids, attn_mask=size2.log()[:, None, None, :], T=16, H_patches=16, W_patches=16)
    y_b = blk(x, mask=ids, attn_mask=torch.zeros_like(size2)[:, None, None, :], T=16, H_patches=16, W_patches=16)
d = float((y_a - y_b).abs().max()); d0 = float((y_b - y_nopa).abs().max())
check("T2 size -> output (PA on)", d > 1e-2, f"max|out(size) - out(size=1)| = {d:.3e}")
check("T2 zero bias == no bias", d0 < 1e-4, f"max|zero-bias - None| = {d0:.2e}")

# ---------------- T3: end-to-end on 4 real clips (fp16 autocast, as in extraction) ----------
enc = E.load_encoder()          # fresh fp16-autocast encoder, identical to extract.py
keys = [k for k in all_clips() if clip_path(k).exists()][:4]
u8 = torch.stack([torch.from_numpy(np.load(clip_path(k))["imgs"][:32]) for k in keys]).cuda()
xin = E.preprocess(u8)


def run(model, cfg):
    E.set_merge(model, cfg)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(xin, restore_dense=False).float()
    st = getattr(model, "last_merge_state", None)      # absent in the HEAD module
    return out, None if st is None else st[1].float()


out_k, size_k = run(enc, merge_config("kbsm", "L9"))
out_pa, size_pa = run(enc, merge_config("kbsmpa", "L9"))
print(f"     L9 final group sizes: mean {float(size_k.mean()):.1f}, max {float(size_k.max()):.0f}")
check("T3 prop_attn changes encoder output", float((out_k - out_pa).abs().max()) > 1e-2,
      f"max|kbsm - kbsmpa| = {float((out_k - out_pa).abs().max()):.3e}")

src = subprocess.run(["git", "-C", str(REPO), "show", "ee09031:vjepa2/src/models/vision_transformer.py"],
                     capture_output=True, text=True, check=True).stdout
path = REPO / "vjepa2/src/models/_vt_head_tmp.py"
path.write_text(src)
try:
    spec = importlib.util.spec_from_file_location("src.models._vt_head_tmp", path)
    vt_head = importlib.util.module_from_spec(spec); sys.modules[spec.name] = vt_head
    spec.loader.exec_module(vt_head)
    head = vt_head.vit_large_rope(img_size=(256, 256), num_frames=64)
    head.load_state_dict(enc.state_dict(), strict=True)
    head = head.cuda().eval()
    out_h, _ = run(head, merge_config("kbsm", "L9"))
    check("T3 prop_attn=False == original (ee09031) encoder (bit-exact)", torch.equal(out_h, out_k),
          f"max|orig - patched| = {float((out_h - out_k).abs().max()):.1e}")
finally:
    path.unlink()

# ---------------- T4: checkpointing guard -------------------------------------------------
E.set_merge(enc, merge_config("kbsmpa", "L9"))
enc.use_activation_checkpointing = True
try:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        enc(xin[:1], restore_dense=False)
    check("T4 checkpointing + prop_attn raises", False, "silently ran without PA")
except (NotImplementedError, RuntimeError) as e:
    check("T4 checkpointing + prop_attn raises", "prop_attn" in str(e), str(e)[:80])
enc.use_activation_checkpointing = False

print("ALL PASS" if ok else "SOME TESTS FAILED")
sys.exit(0 if ok else 1)
