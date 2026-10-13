"""Unit test for round E. Asserts that (1) the hand written VideoMAE layer (extract_videomae.block) reproduces
VideoMAEModel on real clips, (2) every merge config runs end to end with the expected 119 finite tokens, and
(3) the checkpoint is the pinned revision. Writes runs/phaseA/test_videomae.json; exits nonzero on any failure.
  VIDEOMAE_PATH=<snapshot dir> python test_videomae.py"""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import VideoMAEModel

import extract_videomae as E
from common import FEAT_ROOT, W, all_clips, clip_path, merge_config, T_PAST, write_json
from src.models.vision_transformer import _build_token_merger
from src.models.utils.token_merge import normalize_merge_config

REVISION = "12da269a02d3e1fbbb7011e610e591fea8061dca"          # MCG-NJU/videomae-large on the Hugging Face hub
SHA256 = "243706355bfd0f9399af7121df22a6d75286d5e03eaf820cad5b6c5570b3efe3"  # model.safetensors
TOL_REL = 1e-3                                                  # max |diff| / mean |ref|, fp32
N_TOKENS = 119                                                  # 1568 tokens, 25% merged at each of 9 layers

path = Path(os.environ["VIDEOMAE_PATH"])
sha = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
log = dict(revision_dir=path.name, sha256=sha, torch=torch.__version__, transformers=transformers.__version__,
           python=platform.python_version(), gpu=torch.cuda.get_device_name(0), checks={})
ok = True


def check(name, cond, **info):
    global ok
    log["checks"][name] = dict(passed=bool(cond), **info)
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


check("pinned revision", path.name == REVISION and sha == SHA256, revision=path.name, sha256=sha)
m = VideoMAEModel.from_pretrained(path).cuda().eval()
keys = all_clips()
rows = (0, 1, 2, 3)
u8 = torch.stack([torch.from_numpy(np.load(clip_path(keys[r]))["imgs"][:T_PAST]) for r in rows]).cuda()
px = E.preprocess(u8)
with torch.no_grad():
    ref = m(pixel_values=px).last_hidden_state
    mine, _, _ = E.encode(m, px, None, None)
    mine = m.layernorm(mine) if m.layernorm is not None else mine
    d, scale = float((ref - mine).abs().max()), float(ref.abs().mean())
    check("dense layer equals VideoMAEModel", d / scale < TOL_REL, clips=len(rows), max_abs_diff=d,
          mean_abs_ref=scale, rel=d / scale, tol_rel=TOL_REL, tokens=int(ref.shape[1]))
    for name in ("kbsm_L9", "kbsmpa_L9", "wampa_L9", "handmultpa_L9"):
        cfg = merge_config(*name.split("_", 1))
        ext = None
        if name.startswith("hand"):                             # observed hand mask, as in extract_videomae.main
            H = np.load(FEAT_ROOT / "hand_masks.npz")["mask"].reshape(len(keys), -1)
            ext = E.hand_relevance(torch.from_numpy(H[list(rows)]).cuda())
        x, _, _ = E.encode(m, px, cfg, _build_token_merger(normalize_merge_config(cfg)), ext)
        check(f"{name} runs", x.shape[1] == N_TOKENS and bool(torch.isfinite(x).all()),
              tokens=int(x.shape[1]), finite=bool(torch.isfinite(x).all()))
write_json(log, W / "runs" / "phaseA" / "test_videomae.json")
sys.exit(0 if ok else 1)
