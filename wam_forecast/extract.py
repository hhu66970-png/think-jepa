"""Run the frozen V-JEPA ViT-L on the observed past (frames 0..31) once per merge config
and cache the final-layer tokens in compressed form.

Per config and shard, under <FEAT_ROOT>/<cfg>/:
  tokens_<s>.npy  [n, K, 1024] fp16   final LayerNorm output of the K surviving tokens
  idx_<s>.npy     [n, 4096]   int16   for every original token, the row of its representative
                                      (absent for dense, where idx is the identity)
  rows_<s>.npy    [n]         int32   row of each clip in all_clips()
Restoring dense = tokens[idx], which is exactly what restore_dense_tokens() returns.

Usage: CUDA_VISIBLE_DEVICES=0 python extract.py --configs dense,kbsm_s25,wam_s25 --shard 0 --nshards 6
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (FEAT_ROOT, REPO, T_PAST, TOKENS_PAST, VITG, VITL, all_clips, cfg_dir,
                    clip_path, merge_config, write_json)

sys.path[:0] = [str(REPO), str(REPO / "vjepa2")]
from src.models.vision_transformer import (_build_token_merger, vit_giant_xformers,  # noqa: E402
                                           vit_large_rope)
from src.models.utils.token_merge import normalize_merge_config  # noqa: E402
from src.models.utils import token_merge_diagnostics as TMD  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
SHORT_SIDE = int(256.0 / 224 * 256)   # 292, as build_dense_jepa_video_transform(256)


def preprocess(u8):
    """[B,T,H,W,3] uint8 -> [B,3,T,256,256]; identical to VideoObservationAdapter
    (bilinear resize of the short side to 292, no antialias, centre crop 256, ImageNet norm)."""
    x = u8.float().div_(255.0).permute(0, 1, 4, 2, 3)            # B,T,3,H,W
    B, T, C, H, Wd = x.shape
    x = x.reshape(B * T, C, H, Wd)
    if H <= Wd:
        nh, nw = SHORT_SIDE, int(round(Wd * SHORT_SIDE / H))
    else:
        nh, nw = int(round(H * SHORT_SIDE / Wd)), SHORT_SIDE
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=False)
    top, left = int(round((nh - 256) / 2.0)), int(round((nw - 256) / 2.0))
    x = x[:, :, top:top + 256, left:left + 256].reshape(B, T, C, 256, 256)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.permute(0, 2, 1, 3, 4).contiguous()


def load_encoder(arch="vitl"):
    if arch == "vitg":
        enc = vit_giant_xformers(img_size=(256, 256), num_frames=64, use_rope=True)
        ckpt = VITG
    else:
        enc = vit_large_rope(img_size=(256, 256), num_frames=64)
        ckpt = VITL
    sd = torch.load(ckpt, weights_only=True, map_location="cpu")["encoder"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    msg = enc.load_state_dict(sd, strict=False)
    assert not msg.missing_keys, msg.missing_keys
    return enc.cuda().eval()


def set_merge(enc, cfg):
    enc.merge_config = normalize_merge_config(cfg)
    enc.token_merger = _build_token_merger(enc.merge_config) if enc.merge_config.enabled else None


def rep_index(token_ids, rep_for_orig):
    """Row of each original token's representative inside the compressed array."""
    B, K = token_ids.shape
    pos = torch.empty(B, rep_for_orig.shape[1], dtype=torch.long, device=token_ids.device)
    pos.scatter_(1, token_ids, torch.arange(K, device=token_ids.device).expand(B, -1))
    return pos.gather(1, rep_for_orig)


class PastClips(torch.utils.data.Dataset):
    def __init__(self, rows, keys):
        self.rows, self.keys = rows, keys

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with np.load(clip_path(self.keys[r])) as z:
            imgs = z["imgs"][:T_PAST]
        return r, torch.from_numpy(np.ascontiguousarray(imgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma list: dense | <method>_<sched>")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N clips of the shard")
    ap.add_argument("--arch", choices=["vitl", "vitg"], default="vitl")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True          # as the original pipeline
    keys = all_clips()
    rows = [i for i in range(len(keys)) if i % a.nshards == a.shard and clip_path(keys[i]).exists()]
    if a.limit:
        rows = rows[: a.limit]
    cfgs = []
    for name in a.configs.split(","):
        if name == "dense":
            cfgs.append((name, None))
        else:
            method, sched = name.split("_", 1)
            cfgs.append((name, merge_config(method, sched)))

    ext_rel = None
    if any(c is not None and c.get("relevance_source") == "external"
           and not c.get("external_map", "mask").startswith("lattice") for _, c in cfgs):
        # E2 prior: binary hand mask from the observed past (hand_masks.py); within the
        # mask, ties are broken by position.
        H = np.load(FEAT_ROOT / "hand_masks.npz")
        ext_rel = {k: torch.from_numpy(H[k].reshape(len(keys), -1).astype(np.float32))
                   for k in H.files}
    if any(c is not None and c.get("relevance_source") == "scorer" for _, c in cfgs):
        from fit_scorer import Scorer
        ck = torch.load(FEAT_ROOT / "scorer_L12.pt", map_location="cpu")
        sc = Scorer()
        sc.load_state_dict(ck["state"])
        TMD.set_scorer(sc.cuda().eval())
    b2_mods = {}
    for _, c in cfgs:
        if c is not None and c.get("matcher_file") and c["matcher_file"] not in b2_mods:
            from fit_matcher import Matcher
            sd = torch.load(FEAT_ROOT / c["matcher_file"], map_location="cpu")["matchers"]
            mods = []
            for st in sd:
                mk = Matcher(); mk.net.load_state_dict(st); mods.append(mk.cuda().eval())
            b2_mods[c["matcher_file"]] = mods
    if any(c is not None and c.get("bsm_match_metric") == "learned" and not c.get("matcher_file")
           for _, c in cfgs):
        from fit_matcher import Matcher
        mk = Matcher()
        mk.load_state_dict(torch.load(FEAT_ROOT / "matcher_L12.pt", map_location="cpu")["state"])
        TMD.set_matcher(mk.cuda().eval())
    enc = load_encoder(a.arch)
    dl = torch.utils.data.DataLoader(PastClips(rows, keys), batch_size=a.batch,
                                     num_workers=6, shuffle=False, pin_memory=True)
    tag = f"{a.shard}of{a.nshards}"
    out, K_of, t_enc = {}, {}, {n: 0.0 for n, _ in cfgs}
    written = 0
    for r_b, u8 in dl:
        x = preprocess(u8.cuda(non_blocking=True))
        for name, cfg in cfgs:
            if cfg is not None and cfg.get("matcher_file"):
                # B2: per-layer learned matchers, mapped to this schedule's merge layers
                ml = [int(v) for v in str(cfg["merge_layers"]).split(",")]
                G = b2_mods[cfg["matcher_file"]]
                TMD.set_matcher([G[min(max(l - 12, 0), len(G) - 1)] for l in ml])
            if cfg is not None and cfg.get("relevance_source") == "external":
                emap = cfg.get("external_map", "mask")
                if emap.startswith("lattice"):
                    # content-free anchors: every k-th row and column of every tubelet
                    k = int(emap[len("lattice"):])
                    lat = torch.zeros(16, 16, 16)
                    lat[:, ::k, ::k] = 1.0
                    TMD.set_external_relevance(lat.reshape(1, -1).expand(len(r_b), -1).contiguous().cuda())
                else:
                    TMD.set_external_relevance(ext_rel[emap][r_b.long()].cuda())
            set_merge(enc, cfg)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                feats = enc(x, restore_dense=False)
            torch.cuda.synchronize()
            t_enc[name] += time.time() - t0
            B, K, D = feats.shape
            if name not in out:
                d = FEAT_ROOT / cfg_dir(name, a.arch)
                d.mkdir(parents=True, exist_ok=True)
                out[name] = dict(
                    tok=np.lib.format.open_memmap(d / f"tokens_{tag}.npy", "w+", np.float16, (len(rows), K, D)),
                    idx=None if cfg is None else np.lib.format.open_memmap(
                        d / f"idx_{tag}.npy", "w+", np.int16, (len(rows), TOKENS_PAST)))
                K_of[name] = K
            assert K == K_of[name], f"{name}: token count changed {K_of[name]} -> {K}"
            sl = slice(written, written + B)
            out[name]["tok"][sl] = feats.float().cpu().numpy().astype(np.float16)
            if cfg is not None:
                token_ids, _, rep_for_orig = enc.last_merge_state
                out[name]["idx"][sl] = rep_index(token_ids, rep_for_orig).cpu().numpy().astype(np.int16)
        written += len(r_b)
        if written % 80 < a.batch:
            print(f"[{tag}] {written}/{len(rows)} clips", flush=True)
    for name, cfg in cfgs:
        o = out[name]
        o["tok"].flush()
        if o["idx"] is not None:
            o["idx"].flush()
        np.save(FEAT_ROOT / cfg_dir(name, a.arch) / f"rows_{tag}.npy", np.asarray(rows, np.int32))
        write_json(dict(config=cfg, K=K_of[name], clips=len(rows),
                        encoder_s_per_clip=t_enc[name] / max(1, len(rows)), batch=a.batch,
                        arch=a.arch),
                   FEAT_ROOT / cfg_dir(name, a.arch) / f"meta_{tag}.json")
        print(f"{name}: K={K_of[name]}  encoder {1000*t_enc[name]/len(rows):.1f} ms/clip")


if __name__ == "__main__":
    main()
