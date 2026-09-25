"""viz_attn_entropy.py  ---  Motivation evidence C5.

Goal: show that redundancy GROWS WITH DEPTH. Two complementary per-layer
curves, both extracted from the V-JEPA2 ViT-L RoPE encoder (NO token merge):

  (A) REPRESENTATION RANK PROXY  (default, always available)
      For each layer's token matrix X in R^{N x D} we compute the singular
      values of the centred X and report:
        * participation ratio  PR = (sum s^2)^2 / sum s^4   (effective #dims)
        * effective rank       exp(H(p)), p_i = s_i^2 / sum s_j^2  (spectral entropy)
        * normalised variants PR/D and erank/D, and #components for 90/95/99%
      Low effective rank in late layers == tokens live in a low-dim subspace ==
      representational redundancy. This is a faithful, SDPA-safe proxy for
      "attention has little left to attend to" and needs no attention matrix.

  (B) TRUE ATTENTION ENTROPY      (optional: --true_attn)
      Mean per-row Shannon entropy of the softmax attention map, averaged over
      heads/queries, per layer. REQUIRES the attention matrix, which SDPA does
      NOT expose (RoPEAttention sets attn=None under F.scaled_dot_product_
      attention). We therefore (a) build the model with use_sdpa=False so the
      explicit softmax path runs, AND (b) recompute the attention map from the
      captured attention-input by replaying the module's qkv + RoPE. Lower
      entropy in late layers == attention concentrates == redundancy.

Output: per-layer JSON + a multi-panel figure (PR / effective-rank / optional
attention-entropy curves vs depth).

------------------------------------------------------------------------------
DEPENDENCIES
    torch, numpy, matplotlib (matplotlib only for --plot)
    ThinkJEPA ``src/`` importable (run from repo root containing src/).

USAGE (GPU box)
    python viz_attn_entropy.py \
        --harness /ABS/PATH/run_token_merge_pca_experiment.py \
        --ckpt vjepa2/vitl.pt \
        --npz /ABS/PATH/clip.npz \
        --out /ABS/PATH/out_dir \
        --plot
    # add --true_attn to also compute real attention entropy (slower, rebuilds
    # the model with use_sdpa=False). On 8192 tokens the NxN map is ~268M floats
    # per head; use --attn_max_tokens to subsample query rows.

We reuse the harness's build_model / load_video. For --true_attn we re-build via
vision_transformer.vit_large_rope with use_sdpa=False using the SAME kwargs as
harness.build_model (kept in sync below) and reload the same checkpoint.

------------------------------------------------------------------------------
API TOUCHPOINTS TO VERIFY ON GPU (best-effort; confirm before trusting):
  * Layer L (1-based) == output of block index L-1. ViT-L has 24 blocks.
  * Representation proxy: forward hook on model.blocks[i] -> output [B, N, D].
    Confirmed by Block.forward returning the hidden state.
  * use_sdpa flag: RoPEAttention(use_sdpa=False) materialises
    attn = softmax((q@k^T)*scale). We pass use_sdpa=False at BUILD time for the
    --true_attn pass (cannot be toggled per-forward in this code version).
  * TRUE-ATTENTION recompute (fragile): we hook each block's .attn
    (RoPEAttention) PRE-forward to capture its input x_in = norm1(x) [B,N,C],
    then replay: qkv -> split heads -> rotate_queries_or_keys on the d/h/w
    sub-bands using the dense positional ids (mask=arange(T*H*W) separated by
    RoPEAttention.separate_positions) -> attn = softmax((q@k^T)*scale). This
    mirrors RoPEAttention.forward EXACTLY for the unmasked dense case. VERIFY:
    head_dim sub-band widths self.attn.d_dim/h_dim/w_dim, the scale
    self.attn.scale, and that rotate_queries_or_keys is importable from
    src.models.utils.modules. If the import path differs, adjust _import_rope().
  * Grid dims: t=num_frames//tubelet(2), h=w=img_size//patch.
"""
import argparse
import importlib.util
import math
import os
import sys

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except Exception as exc:
    np = None
    torch = None
    F = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def load_harness(harness_path):
    harness_path = os.path.abspath(harness_path)
    repo_root = harness_path
    for _ in range(6):
        repo_root = os.path.dirname(repo_root)
        if os.path.isdir(os.path.join(repo_root, "src")):
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            break
    spec = importlib.util.spec_from_file_location("tm_harness", harness_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_rope():
    """Return rotate_queries_or_keys from the ThinkJEPA modules (API touchpoint)."""
    from src.models.utils.modules import rotate_queries_or_keys
    return rotate_queries_or_keys


def grid_dims(num_frames, img_size, patch_size, tubelet_size=2):
    return (int(num_frames // tubelet_size),
            int(img_size // patch_size), int(img_size // patch_size))


# --------------------------------------------------------------------------
# (A) representation rank proxy
# --------------------------------------------------------------------------
def rank_stats(feat):
    """feat: [N, D] -> participation ratio, effective rank, ncomp thresholds."""
    x = feat.float()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)            # [min(N,D)]
    var = s ** 2
    total = var.sum().clamp_min(1e-12)
    p = var / total
    # participation ratio (effective number of dimensions)
    pr = float((var.sum() ** 2 / (var ** 2).sum().clamp_min(1e-12)).item())
    # spectral (effective) rank = exp(entropy of normalised eigenvalues)
    ent = float(-(p * p.clamp_min(1e-12).log()).sum().item())
    erank = float(math.exp(ent))
    cum = torch.cumsum(p, dim=0)
    D = int(s.numel())
    out = {
        "dim": D,
        "participation_ratio": pr,
        "participation_ratio_norm": pr / D,
        "effective_rank": erank,
        "effective_rank_norm": erank / D,
        "spectral_entropy": ent,
    }
    for th in (0.90, 0.95, 0.99):
        out[f"ncomp_{int(th*100)}pct"] = int(min(D, int((cum < th).sum().item()) + 1))
    return out


class BlockOutputCollector:
    def __init__(self, model, block_indices):
        self.captured = {}
        self._handles = []
        for bi in block_indices:
            self._handles.append(
                model.blocks[bi].register_forward_hook(self._make(bi)))

    def _make(self, bi):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            self.captured[bi] = t.detach()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


# --------------------------------------------------------------------------
# (B) true attention entropy via RoPE replay (fragile; see header touchpoint)
# --------------------------------------------------------------------------
class AttnInputCollector:
    """Capture the INPUT to each block's RoPEAttention (i.e. norm1(x))."""
    def __init__(self, model, block_indices):
        self.captured = {}
        self._handles = []
        for bi in block_indices:
            attn = model.blocks[bi].attn
            self._handles.append(attn.register_forward_pre_hook(self._make(bi)))

    def _make(self, bi):
        def pre_hook(_module, args):
            # RoPEAttention.forward(self, x, mask=None, attn_mask=None, T,H,W)
            x_in = args[0]
            self.captured[bi] = x_in.detach()
        return pre_hook

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def attention_entropy_for_block(attn_module, x_in, T, H, W, rotate_fn,
                                max_tokens=None, seed=0):
    """Replay RoPEAttention to get the softmax map, return mean row entropy.

    Mirrors RoPEAttention.forward for the dense, unmasked case. Returns mean
    Shannon entropy (nats) of attention rows averaged over heads and (sampled)
    query positions, plus the normalised entropy (divided by ln N).
    """
    B, N, C = x_in.shape
    nh = attn_module.num_heads
    qkv = attn_module.qkv(x_in).unflatten(-1, (3, nh, -1)).permute(2, 0, 3, 1, 4)
    q, k, _v = qkv[0], qkv[1], qkv[2]          # [B, nh, N, head_dim]

    # dense positional ids -> separate into depth/height/width components.
    # We replicate RoPEAttention.forward's UNMASKED branch EXACTLY: mask is a
    # bare 1-D arange [N]; separate_positions returns 1-D [N] tensors that
    # rotate_queries_or_keys broadcasts (einsum "..., f -> ... f") against the
    # [B, nh, N, head_dim] q/k. Keeping pos 1-D (not [B,nh,N]) matches the source
    # byte-for-byte and avoids any broadcasting drift.
    mask = torch.arange(int(T * H * W), device=x_in.device)
    d_mask, h_mask, w_mask = attn_module.separate_positions(mask, H, W)

    s = 0
    qd = rotate_fn(q[..., s:s + attn_module.d_dim], pos=d_mask)
    kd = rotate_fn(k[..., s:s + attn_module.d_dim], pos=d_mask)
    s += attn_module.d_dim
    qh = rotate_fn(q[..., s:s + attn_module.h_dim], pos=h_mask)
    kh = rotate_fn(k[..., s:s + attn_module.h_dim], pos=h_mask)
    s += attn_module.h_dim
    qw = rotate_fn(q[..., s:s + attn_module.w_dim], pos=w_mask)
    kw = rotate_fn(k[..., s:s + attn_module.w_dim], pos=w_mask)
    s += attn_module.w_dim
    if s < attn_module.head_dim:
        q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)

    # optionally subsample query rows to bound the NxN cost
    if max_tokens is not None and max_tokens < N:
        gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed))
        qsel = torch.randperm(N, generator=gen)[:max_tokens].to(q.device)
        q = q.index_select(2, qsel)

    # accumulate entropy over heads in fp32; chunk queries to cap memory
    scale = attn_module.scale
    Nq = q.shape[2]
    ent_sum = 0.0
    count = 0
    chunk = 1024
    lnN = math.log(N)
    for c0 in range(0, Nq, chunk):
        qc = q[:, :, c0:c0 + chunk, :]                       # [B,nh,c,hd]
        logits = (qc @ k.transpose(-2, -1)) * scale          # [B,nh,c,N]
        p = logits.softmax(dim=-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)    # [B,nh,c]
        ent_sum += float(ent.sum().item())
        count += int(ent.numel())
    mean_ent = ent_sum / max(1, count)
    return {"attn_entropy_nats": mean_ent,
            "attn_entropy_norm": mean_ent / lnN,
            "num_query_rows": int(Nq), "num_keys": int(N)}


def build_no_sdpa_model(harness, vision_transformer, ckpt, num_frames, img_size,
                        patch_size, device):
    """Rebuild ViT-L RoPE with use_sdpa=False (kwargs kept in sync with
    harness.build_model) and load the same checkpoint, merge DISABLED."""
    merge_config = {
        "enabled": False, "merge_layers": (), "merge_ratio": 0.0,
        "strategy": "local_2x2_same_time_vec", "receiver": "max_norm",
        "restore_dense": False, "profile": False, "importance_source": "none",
        "bsm_match_metric": "key",
    }
    model = vision_transformer.vit_large_rope(
        img_size=(img_size, img_size), num_frames=num_frames, patch_size=patch_size,
        tubelet_size=2, out_layers=None, use_sdpa=False, use_silu=False,
        wide_silu=True, uniform_power=False, merge_config=merge_config)
    try:
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(str(ckpt), map_location="cpu")
    if isinstance(blob, dict) and "encoder" in blob:
        state = blob["encoder"]
    elif isinstance(blob, dict) and "model" in blob:
        state = blob["model"]
    else:
        state = blob
    clean = {k.replace("module.", "").replace("backbone.", ""): v
             for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description="C5 per-layer redundancy: rank "
                                             "proxy + optional true attn entropy.")
    ap.add_argument("--harness", required=True)
    ap.add_argument("--ckpt", default="vjepa2/vitl.pt")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--layers", default="",
                    help="1-based layers to probe; empty => ALL blocks")
    ap.add_argument("--layers_zero_based", action="store_true")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--true_attn", action="store_true",
                    help="also compute real attention entropy (rebuilds model "
                         "with use_sdpa=False; slower)")
    ap.add_argument("--attn_max_tokens", type=int, default=2048,
                    help="subsample this many query rows for the NxN attn map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"torch/numpy import failed: {_IMPORT_ERROR}")

    os.makedirs(args.out, exist_ok=True)
    harness = load_harness(args.harness)
    t_grid, h_grid, w_grid = grid_dims(args.num_frames, args.img_size, args.patch_size)

    model = harness.build_model(
        args.ckpt, args.num_frames, args.img_size, args.patch_size,
        initial_strategy="local_2x2_same_time_vec", device=args.device)
    model.merge_config.enabled = False
    n_blocks = len(model.blocks)

    if args.layers.strip():
        nums = [int(v) for v in args.layers.split(",") if v.strip()]
        block_idx = nums if args.layers_zero_based else [n - 1 for n in nums]
    else:
        block_idx = list(range(n_blocks))
        nums = block_idx if args.layers_zero_based else [b + 1 for b in block_idx]

    video, total_frames = harness.load_video(
        args.npz, args.num_frames, args.img_size, args.device)
    print(f"[INFO] clip={args.npz} src_frames={total_frames} input={tuple(video.shape)} "
          f"blocks={n_blocks} probing={block_idx} grid t={t_grid} h={h_grid} w={w_grid}")

    # ---- (A) representation rank proxy ----
    coll = BlockOutputCollector(model, block_idx)
    with torch.no_grad():
        _ = model(video, return_merge_info=False, restore_dense=False)
    coll.remove()

    per_layer = {}
    for ln, bi in zip(nums, block_idx):
        st = rank_stats(coll.captured[bi][0])
        per_layer[str(ln)] = {"rank_proxy": st}
        print(f"  layer {ln:>2} (blk {bi}): PR={st['participation_ratio']:.1f} "
              f"erank={st['effective_rank']:.1f} "
              f"ncomp95={st['ncomp_95pct']} / D={st['dim']}")

    # ---- (B) optional true attention entropy ----
    if args.true_attn:
        from src.models import vision_transformer
        rotate_fn = _import_rope()
        amodel = build_no_sdpa_model(
            harness, vision_transformer, args.ckpt, args.num_frames,
            args.img_size, args.patch_size, args.device)
        acoll = AttnInputCollector(amodel, block_idx)
        with torch.no_grad():
            _ = amodel(video, return_merge_info=False, restore_dense=False)
        acoll.remove()
        for ln, bi in zip(nums, block_idx):
            x_in = acoll.captured[bi]
            ae = attention_entropy_for_block(
                amodel.blocks[bi].attn, x_in, t_grid, h_grid, w_grid, rotate_fn,
                max_tokens=args.attn_max_tokens, seed=args.seed)
            per_layer[str(ln)]["attn_entropy"] = ae
            print(f"  layer {ln:>2} (blk {bi}): attn_entropy="
                  f"{ae['attn_entropy_nats']:.3f} nats "
                  f"(norm {ae['attn_entropy_norm']:.3f})")

    import json
    results = {
        "meta": {
            "npz": args.npz, "ckpt": args.ckpt, "num_frames": args.num_frames,
            "img_size": args.img_size, "patch_size": args.patch_size,
            "grid": [t_grid, h_grid, w_grid], "n_blocks": n_blocks,
            "layers": nums, "block_indices": block_idx,
            "true_attn": bool(args.true_attn),
            "attn_max_tokens": args.attn_max_tokens, "seed": args.seed,
        },
        "per_layer": per_layer,
    }
    jp = os.path.join(args.out, "attn_entropy_stats.json")
    with open(jp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {jp}")

    if args.plot:
        _render(results, args.out)


def _render(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results["per_layer"].keys(), key=lambda s: int(s))
    xs = [int(l) for l in layers]
    pr = [results["per_layer"][l]["rank_proxy"]["participation_ratio"] for l in layers]
    er = [results["per_layer"][l]["rank_proxy"]["effective_rank"] for l in layers]
    nc = [results["per_layer"][l]["rank_proxy"]["ncomp_95pct"] for l in layers]
    has_attn = all("attn_entropy" in results["per_layer"][l] for l in layers) and \
        results["meta"].get("true_attn")

    npanel = 2 + (1 if has_attn else 0)
    fig, axes = plt.subplots(1, npanel, figsize=(5.2 * npanel, 4.2), squeeze=False)
    ax = axes[0]
    ax[0].plot(xs, pr, marker="o", label="participation ratio")
    ax[0].plot(xs, er, marker="s", label="effective rank")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("effective #dims")
    ax[0].set_title("Representation rank vs depth\n(lower late = more redundant)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(xs, nc, marker="d", color="tab:red")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("# PCA comps for 95% var")
    ax[1].set_title("PCA components to 95% variance vs depth")
    ax[1].grid(alpha=0.3)

    if has_attn:
        ae = [results["per_layer"][l]["attn_entropy"]["attn_entropy_norm"] for l in layers]
        ax[2].plot(xs, ae, marker="^", color="tab:purple")
        ax[2].set_xlabel("layer"); ax[2].set_ylabel("normalised attn entropy")
        ax[2].set_title("True attention entropy vs depth\n(lower late = concentrated)")
        ax[2].grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "attn_entropy_by_depth.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {p}")


if __name__ == "__main__":
    main()
