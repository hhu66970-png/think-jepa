"""Round E: a second encoder family. Frozen VideoMAE ViT-L (self-supervised, absolute sinusoidal positions,
no rotary embedding), with the SAME merger implementation as for V-JEPA 2 (token_merge_diagnostics), and
cache its final tokens in the format of extract.py, so that train_probe.py runs unchanged.

Input: the observed frames 0..31, every second frame (16 frames), resized 256 -> 224, ImageNet normalisation.
VideoMAE grid: 8 x 14 x 14 = 1568 tokens (tubelet 2, patch 16). Merging uses the head-mean attention key of
the merge layer (no rotary embedding, so K-BSM and ToMe coincide) at layers 12..20 with r = 25% per layer
(1568 -> 119 tokens after integer rounding, 7.6% of the input against 309 / 4096 = 7.5%), and proportional attention when requested.
The probe reads the 16 x 16 x 16 grid of V-JEPA: every probe cell points to the VideoMAE cell that covers it
(t // 2, floor(h * 14 / 16), floor(w * 14 / 16)), so idx_<s>.npy maps the 4096 probe cells to token rows.

  CUDA_VISIBLE_DEVICES=0 python extract_videomae.py --configs dense,kbsmpa_L9 --shard 0 --nshards 2
Writes <FEAT_ROOT>/vmae_<cfg>/ (tokens, idx, rows, meta), as extract.py.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import FEAT_ROOT, REPO, T_PAST, TOKENS_PAST, all_clips, clip_path, merge_config, past_frames, write_json

sys.path[:0] = [str(REPO), str(REPO / "vjepa2")]
from src.models.vision_transformer import _build_token_merger  # noqa: E402
from src.models.utils.token_merge import init_token_merge_state, normalize_merge_config  # noqa: E402
from src.models.utils import token_merge_diagnostics as TMD  # noqa: E402

VM_PATH = os.environ.get("VIDEOMAE_PATH", "")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
TG, HG, WG = 8, 14, 14


def probe_map():
    """[4096] index of the VideoMAE cell (t8, h14, w14) covering each probe cell (t16, h16, w16)."""
    t, h, w = torch.meshgrid(torch.arange(16), torch.arange(16), torch.arange(16), indexing="ij")
    return ((t // 2) * HG * WG + (h * HG // 16) * WG + (w * WG // 16)).reshape(-1)


PMAP = probe_map()


def preprocess(u8):
    """[B,32,256,256,3] uint8 -> [B,16,3,224,224] (every second observed frame)."""
    x = u8[:, ::2].float().div_(255.0).permute(0, 1, 4, 2, 3)
    B, T, C, H, W = x.shape
    x = F.interpolate(x.reshape(B * T, C, H, W), size=(224, 224), mode="bilinear", align_corners=False,
                      antialias=True).reshape(B, T, C, 224, 224)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def block(layer, x, bias=None, want_key=False):
    """One VideoMAE layer written out, so that proportional attention (a per-key logit bias) can be added
    and the head-mean key returned. Identical to VideoMAELayer.forward when bias is None."""
    sa = layer.attention.attention
    h = layer.layernorm_before(x)
    qb = getattr(sa, "q_bias", None)
    vb = getattr(sa, "v_bias", None)
    q = F.linear(h, sa.query.weight, qb if qb is not None else sa.query.bias)
    k = F.linear(h, sa.key.weight, sa.key.bias)
    v = F.linear(h, sa.value.weight, vb if vb is not None else sa.value.bias)
    B, N, D = q.shape
    nh = sa.num_attention_heads
    dh = D // nh
    q, k, v = (t.view(B, N, nh, dh).transpose(1, 2) for t in (q, k, v))
    a = (q @ k.transpose(-1, -2)) * dh ** -0.5
    if bias is not None:
        a = a + bias[:, None, None, :].to(a.dtype)
    o = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
    x = x + layer.attention.output.dense(o)
    y = layer.output.dense(layer.intermediate(layer.layernorm_after(x)))
    x = x + y
    key = k.mean(1) if want_key else None                  # [B, N, dh] head mean, as the V-JEPA stash
    return x, key


INT8 = os.environ.get("WAM_FEAT_INT8", "0") == "1"


def quantize_int8(f):
    """[B,K,D] float -> int8 tokens and fp16 per token scales (symmetric absmax over D), as extract.py."""
    sc = f.abs().amax(-1).clamp_min(1e-8) / 127.0
    q = torch.round(f / sc.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return q, sc.half()


def hand_relevance(mask16):
    """[B, 4096] observed-hand mask on the probe grid -> [B, 1568] on the VideoMAE grid (max over the
    probe cells each VideoMAE cell covers)."""
    B = mask16.shape[0]
    out = torch.zeros(B, TG * HG * WG, device=mask16.device)
    out.scatter_reduce_(1, PMAP.to(mask16.device).unsqueeze(0).expand(B, -1), mask16.float(), reduce="amax")
    return out


def encode(model, px, cfg, merger, ext=None):
    x = model.embeddings(px, None)                         # [B, 1568, D], patch + sinusoidal positions
    B, N, D = x.shape
    if cfg is None:
        for layer in model.encoder.layer:
            x, _ = block(layer, x)
        ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x, ids, ids
    layers = [int(v) for v in str(cfg["merge_layers"]).split(",")]
    pa = bool(cfg.get("prop_attn", False))
    token_ids, token_size, rep = init_token_merge_state(B, N, x.device, x.dtype)
    seen = False
    if ext is not None:
        TMD.set_external_relevance(ext)
    for i, layer in enumerate(model.encoder.layer):
        bias = token_size.float().clamp_min(1.0).log() if (pa and seen) else None
        x, key = block(layer, x, bias, want_key=i in layers)
        if i in layers:
            merger._block_index, merger._num_blocks = i, len(model.encoder.layer)
            x, token_ids, token_size, rep, _ = merger(x=x, token_ids=token_ids, token_size=token_size,
                                                      rep_for_orig=rep, t_grid=TG, h_grid=HG, w_grid=WG,
                                                      attn_key=key)
            seen = True
    return x, token_ids, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from transformers import VideoMAEModel
    model = VideoMAEModel.from_pretrained(VM_PATH).cuda().eval()
    norm = model.layernorm if model.layernorm is not None else torch.nn.Identity()
    torch.backends.cuda.matmul.allow_tf32 = True
    keys = all_clips()
    rows = [i for i in range(len(keys)) if i % a.nshards == a.shard and clip_path(keys[i]).exists()]
    if a.limit:
        rows = rows[: a.limit]
    cfgs = [(n, None if n == "dense" else merge_config(*n.split("_", 1))) for n in a.configs.split(",")]
    mergers = {n: (_build_token_merger(normalize_merge_config(c)) if c else None) for n, c in cfgs}
    H = np.load(FEAT_ROOT / "hand_masks.npz")["mask"].reshape(len(keys), -1)
    tag = f"{a.shard}of{a.nshards}"
    out, K_of, t_enc, written = {}, {}, {n: 0.0 for n, _ in cfgs}, 0
    pm = PMAP.cuda()
    for s in range(0, len(rows), a.batch):
        rb = rows[s:s + a.batch]
        u8 = []
        for r in rb:
            u8.append(torch.from_numpy(np.ascontiguousarray(past_frames(keys[r]))))
        px = preprocess(torch.stack(u8).cuda())
        for name, cfg in cfgs:
            ext = None
            if cfg is not None and cfg.get("relevance_source") == "external":
                ext = hand_relevance(torch.from_numpy(H[rb]).cuda())
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                x, token_ids, rep = encode(model, px, cfg, mergers[name], ext)
                x = norm(x)
            torch.cuda.synchronize(); t_enc[name] += time.time() - t0
            B, K, D = x.shape
            # row of each original VideoMAE cell's representative, then of each probe cell
            pos = torch.empty(B, TG * HG * WG, dtype=torch.long, device=x.device)
            pos.scatter_(1, token_ids, torch.arange(K, device=x.device).expand(B, -1))
            idx = pos.gather(1, rep).gather(1, pm.unsqueeze(0).expand(B, -1))   # [B, 4096]
            if name not in out:
                d = FEAT_ROOT / f"vmae_{name}"
                d.mkdir(parents=True, exist_ok=True)
                out[name] = dict(tok=np.lib.format.open_memmap(d / f"tokens_{tag}.npy", "w+", np.int8 if INT8 else np.float16, (len(rows), K, D)),
                                 idx=np.lib.format.open_memmap(d / f"idx_{tag}.npy", "w+", np.int16, (len(rows), TOKENS_PAST)),
                                 scale=np.lib.format.open_memmap(d / f"scale_{tag}.npy", "w+", np.float16, (len(rows), K)) if INT8 else None)
                K_of[name] = K
            assert K == K_of[name]
            sl = slice(written, written + B)
            if INT8:
                q, scl = quantize_int8(x.float())
                out[name]["tok"][sl] = q.cpu().numpy(); out[name]["scale"][sl] = scl.cpu().numpy()
            else:
                out[name]["tok"][sl] = x.float().cpu().numpy().astype(np.float16)
            out[name]["idx"][sl] = idx.cpu().numpy().astype(np.int16)
        written += len(rb)
        if written % 80 < a.batch:
            print(f"[{tag}] {written}/{len(rows)} clips", flush=True)
    for name, cfg in cfgs:
        out[name]["tok"].flush(); out[name]["idx"].flush()
        if out[name].get("scale") is not None:
            out[name]["scale"].flush()
        d = FEAT_ROOT / f"vmae_{name}"
        np.save(d / f"rows_{tag}.npy", np.asarray(rows, np.int32))
        write_json(dict(config=cfg, K=K_of[name], clips=len(rows), encoder="videomae-large",
                        encoder_s_per_clip=t_enc[name] / max(1, len(rows)), batch=a.batch), d / f"meta_{tag}.json")
        print(f"vmae_{name}: K={K_of[name]}  encoder {1000 * t_enc[name] / len(rows):.1f} ms/clip", flush=True)


if __name__ == "__main__":
    main()
