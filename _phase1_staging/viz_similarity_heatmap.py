"""viz_similarity_heatmap.py  ---  Motivation evidence C2.

Goal: show *spatial redundancy is high while temporal structure differs*. We
extract per-layer token representations from the V-JEPA2 ViT-L RoPE video
encoder (NO token merge), compute token-token cosine similarity, and group the
similarities by the (t, h, w) grid relationship between the two tokens:

  (a) same-frame spatial NEIGHBOURS  (same t, |dh|+|dw| == 1)      -> "can merge"
  (b) cross-frame SAME position      (same h,w, different t)
  (c) cross-frame DIFFERENT position (different t, different (h,w))

For several layers (default 4,8,12,16,20,24) we report the mean / std /
percentiles of each group, plus a small same-frame spatial-offset similarity
HEATMAP (avg cosine vs (dh, dw) offset). The per-layer curves + offset heatmaps
are the figure: same-frame adjacency stays highly similar (mergeable) and the
cross-frame distributions look structurally different from the spatial one.

------------------------------------------------------------------------------
DEPENDENCIES
    torch, numpy, matplotlib   (matplotlib only needed if --plot)
    The ThinkJEPA source tree must be importable, i.e. run from the repo root
    that contains ``src/`` (same place the encoder harness is run), e.g.:
        cd /root/autodl-tmp/thinkjepa-work/<repo-with-src>
    so that ``src.models.vision_transformer`` and the merge utils import.

USAGE (on a GPU box; this file only does py_compile locally)
    python viz_similarity_heatmap.py \
        --harness /ABS/PATH/run_token_merge_pca_experiment.py \
        --ckpt vjepa2/vitl.pt \
        --npz /ABS/PATH/clip.npz \
        --layers 4,8,12,16,20,24 \
        --out /ABS/PATH/out_dir \
        --plot

We reuse the harness's ``build_model`` and ``load_video`` verbatim (imported by
path) so model construction / checkpoint loading / input preprocessing are
byte-identical to the validated encoder pipeline. Token merge is DISABLED here:
we want the dense 8192-token grid so every token id maps cleanly to (t,h,w).

------------------------------------------------------------------------------
API TOUCHPOINTS TO VERIFY ON GPU  (best-effort here; confirm before trusting):
  * We capture per-block output via a forward hook on ``model.blocks[i]``
    (nn.ModuleList of Block; Block.forward returns the post-block hidden state
    [B, N, D]).  -- confirmed in src/models/utils/modules.py Block.forward.
  * "layer L" is interpreted as the output of block index L-1 (1-based layer
    numbering -> 0-based block index). Layer 24 == blocks[23] for ViT-L (24
    blocks). If you prefer 0-based, pass --layers_zero_based.
  * With merge disabled, blocks receive mask=None and token ordering is the
    canonical arange, so token position p == token id p == (t,h,w) via
    ids_to_coords(p, h_grid, w_grid).  -- confirmed: vision_transformer.forward
    passes mask=token_ids only when merge_enabled.
  * Grid dims: t_grid = num_frames // tubelet_size, h_grid = w_grid =
    img_size // patch_size. For the defaults (64 frames, tubelet 2, 256px,
    patch16) that is t=32, h=w=16 -> 8192 tokens.
"""
import argparse
import importlib.util
import os
import sys

try:
    import numpy as np
    import torch
except Exception as exc:  # keep py_compile happy even without torch installed
    np = None
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# --------------------------------------------------------------------------
# Import build_model / load_video from the harness so loading stays identical.
# --------------------------------------------------------------------------
def load_harness(harness_path):
    """Import the encoder harness module by absolute path and return it."""
    harness_path = os.path.abspath(harness_path)
    repo_root = harness_path
    # walk up until we find a dir that contains 'src' (the import root)
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


def grid_dims(num_frames, img_size, patch_size, tubelet_size=2):
    t_grid = num_frames // tubelet_size
    h_grid = img_size // patch_size
    w_grid = img_size // patch_size
    return int(t_grid), int(h_grid), int(w_grid)


def coords_for_ids(num_tokens, h_grid, w_grid):
    """Return (t, h, w) LongTensors for token positions 0..num_tokens-1.

    Mirrors token_merge.ids_to_coords but kept local so the script is
    self-contained for the dense (merge-disabled) grid.
    """
    tpf = int(h_grid * w_grid)
    ids = torch.arange(num_tokens)
    t = ids // tpf
    rem = ids - t * tpf
    h = rem // int(w_grid)
    w = rem - h * int(w_grid)
    return t.long(), h.long(), w.long()


# --------------------------------------------------------------------------
# Hook helper: grab outputs of selected blocks in ONE forward pass.
# --------------------------------------------------------------------------
class BlockOutputCollector:
    def __init__(self, model, block_indices):
        self.captured = {}
        self._handles = []
        blocks = model.blocks
        for bi in block_indices:
            if bi < 0 or bi >= len(blocks):
                raise IndexError(f"block index {bi} out of range (0..{len(blocks)-1})")
            self._handles.append(blocks[bi].register_forward_hook(self._make(bi)))

    def _make(self, bi):
        def hook(_module, _inp, out):
            # Block.forward returns the hidden state tensor [B, N, D].
            t = out[0] if isinstance(out, (tuple, list)) else out
            self.captured[bi] = t.detach()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


# --------------------------------------------------------------------------
# Similarity grouping for ONE layer's tokens.
# --------------------------------------------------------------------------
def grouped_similarity_stats(feat, t, h, w, *, max_pairs_per_group=200000,
                             seed=0, offset_radius=4):
    """feat: [N, D] (single sample). t,h,w: [N] grid coords.

    Returns a dict of per-group similarity summaries and a same-frame spatial
    offset heatmap (mean cosine as a function of (dh, dw) within a frame).

    To stay tractable on 8192 tokens (~33M pairs) we SAMPLE token pairs per
    group rather than materialising the full NxN matrix. Sampling is uniform
    over the relevant index sets and reproducible via ``seed``.
    """
    device = feat.device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    fn = torch.nn.functional.normalize(feat.float(), dim=-1, eps=1e-6)  # [N, D]
    N = fn.shape[0]
    tpf_t, tpf_h, tpf_w = t.to(device), h.to(device), w.to(device)

    def sample_pairs(mask_fn, count):
        """Rejection-sample index pairs (i, j), i != j, satisfying mask_fn.

        Robust against *rare* groups (e.g. same-frame 4-neighbours, whose
        acceptance rate is ~4/(t_grid*h*w)). The previous version drew only
        ``need*3`` candidates per try and shrank that pool as ``need`` fell, so
        a rare group could accept 0 pairs across all tries and silently return
        an EMPTY group (the "groups=0" symptom). We instead keep a generous,
        fixed minimum candidate batch and a larger try budget so rare-but-
        present relations are reliably sampled; counts still cap at ``count``.
        """
        got_i, got_j, got = [], [], 0
        # candidate batch never collapses below this, regardless of remaining need
        batch = max(int(count) * 3, 1_000_000)
        tries = 0
        while got < count and tries < 200:
            tries += 1
            bi = torch.randint(0, N, (batch,), generator=g)
            bj = torch.randint(0, N, (batch,), generator=g)
            keep = mask_fn(bi, bj) & (bi != bj)
            bi, bj = bi[keep], bj[keep]
            if bi.numel():
                got_i.append(bi)
                got_j.append(bj)
                got += int(bi.numel())
        if not got_i:
            return None
        ii = torch.cat(got_i)[:count]
        jj = torch.cat(got_j)[:count]
        return ii, jj

    tc, hc, wc = t.cpu(), h.cpu(), w.cpu()

    def cos_of(pairs):
        if pairs is None:
            return None
        ii, jj = pairs
        a = fn.index_select(0, ii.to(device))
        b = fn.index_select(0, jj.to(device))
        return (a * b).sum(-1).detach().cpu().numpy()

    # (a) same frame, 4-neighbour adjacency (|dh|+|dw| == 1)
    def m_neigh(i, j):
        return (tc[i] == tc[j]) & ((hc[i] - hc[j]).abs() + (wc[i] - wc[j]).abs() == 1)

    # (b) cross frame, identical (h, w)
    def m_cross_same(i, j):
        return (tc[i] != tc[j]) & (hc[i] == hc[j]) & (wc[i] == wc[j])

    # (c) cross frame, different (h, w)
    def m_cross_diff(i, j):
        return (tc[i] != tc[j]) & ((hc[i] != hc[j]) | (wc[i] != wc[j]))

    # extra reference group: same frame but NOT adjacent (far spatial)
    def m_same_far(i, j):
        return (tc[i] == tc[j]) & ((hc[i] - hc[j]).abs() + (wc[i] - wc[j]).abs() > 1)

    groups = {
        "same_frame_neighbour": cos_of(sample_pairs(m_neigh, max_pairs_per_group)),
        "same_frame_nonadjacent": cos_of(sample_pairs(m_same_far, max_pairs_per_group)),
        "cross_frame_same_pos": cos_of(sample_pairs(m_cross_same, max_pairs_per_group)),
        "cross_frame_diff_pos": cos_of(sample_pairs(m_cross_diff, max_pairs_per_group)),
    }

    def summarize(arr):
        if arr is None or arr.size == 0:
            return None
        return {
            "n_pairs": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "hist_counts": np.histogram(arr, bins=40, range=(-1.0, 1.0))[0].tolist(),
        }

    summary = {k: summarize(v) for k, v in groups.items()}

    # same-frame spatial-offset heatmap: mean cosine vs (dh, dw), |d|<=radius.
    # We exploit the regular grid: for each offset we compare every in-frame
    # token to its shifted partner (vectorised over the [t,h,w] grid).
    R = int(offset_radius)
    feat_grid = fn.reshape(int(tpf_t.max().item()) + 1, int(hc.max().item()) + 1,
                           int(wc.max().item()) + 1, fn.shape[-1])
    Tn, Hn, Wn, _ = feat_grid.shape
    heat = np.full((2 * R + 1, 2 * R + 1), np.nan, dtype=np.float64)
    for dh in range(-R, R + 1):
        for dw in range(-R, R + 1):
            h0, h1 = max(0, -dh), min(Hn, Hn - dh)
            w0, w1 = max(0, -dw), min(Wn, Wn - dw)
            if h1 <= h0 or w1 <= w0:
                continue
            a = feat_grid[:, h0:h1, w0:w1, :]
            b = feat_grid[:, h0 + dh:h1 + dh, w0 + dw:w1 + dw, :]
            cs = (a * b).sum(-1)  # [T, h', w'] cosine (already unit norm)
            heat[dh + R, dw + R] = float(cs.mean().item())

    return {"groups": summary, "offset_heatmap": heat.tolist(),
            "offset_radius": R}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="C2 similarity heatmap / per-layer "
                                             "same-frame vs cross-frame cosine.")
    ap.add_argument("--harness", required=True,
                    help="abs path to run_token_merge_pca_experiment.py (for "
                         "build_model/load_video reuse)")
    ap.add_argument("--ckpt", default="vjepa2/vitl.pt")
    ap.add_argument("--npz", required=True, help="abs path to a clip .npz (key 'imgs')")
    ap.add_argument("--layers", default="4,8,12,16,20,24",
                    help="1-based layer numbers (output of that block)")
    ap.add_argument("--layers_zero_based", action="store_true",
                    help="treat --layers as 0-based block indices instead")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--max_pairs_per_group", type=int, default=200000)
    ap.add_argument("--offset_radius", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--plot", action="store_true", help="also render PNG figures")
    args = ap.parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"torch/numpy import failed: {_IMPORT_ERROR}")

    os.makedirs(args.out, exist_ok=True)
    harness = load_harness(args.harness)

    # Build model WITH merge disabled (dense 8192-token grid).
    model = harness.build_model(
        args.ckpt, args.num_frames, args.img_size, args.patch_size,
        initial_strategy="local_2x2_same_time_vec", device=args.device,
    )
    model.merge_config.enabled = False  # force true dense forward (no merging)

    video, total_frames = harness.load_video(
        args.npz, args.num_frames, args.img_size, args.device)
    t_grid, h_grid, w_grid = grid_dims(args.num_frames, args.img_size, args.patch_size)
    print(f"[INFO] clip={args.npz} src_frames={total_frames} input={tuple(video.shape)} "
          f"grid t={t_grid} h={h_grid} w={w_grid}")

    layer_nums = [int(v) for v in args.layers.split(",") if v.strip()]
    block_idx = layer_nums if args.layers_zero_based else [n - 1 for n in layer_nums]

    collector = BlockOutputCollector(model, block_idx)
    with torch.no_grad():
        # We do not need the model output, only the hooked block outputs.
        _ = model(video, return_merge_info=False, restore_dense=False)
    collector.remove()

    # coords for the dense grid (token position == token id)
    any_feat = next(iter(collector.captured.values()))
    N = any_feat.shape[1]
    expected = t_grid * h_grid * w_grid
    if N != expected:
        print(f"[WARN] captured N={N} != expected dense {expected}; coords assume "
              f"row-major (t,h,w). Check merge really disabled.")
    t_idx, h_idx, w_idx = coords_for_ids(N, h_grid, w_grid)

    results = {
        "meta": {
            "npz": args.npz, "ckpt": args.ckpt, "num_frames": args.num_frames,
            "img_size": args.img_size, "patch_size": args.patch_size,
            "grid": [t_grid, h_grid, w_grid], "num_tokens": int(N),
            "layers_requested": layer_nums, "block_indices": block_idx,
            "layers_zero_based": bool(args.layers_zero_based),
            "max_pairs_per_group": args.max_pairs_per_group,
            "offset_radius": args.offset_radius, "seed": args.seed,
        },
        "per_layer": {},
    }

    for ln, bi in zip(layer_nums, block_idx):
        feat = collector.captured[bi][0]  # [N, D] sample 0
        stats = grouped_similarity_stats(
            feat, t_idx, h_idx, w_idx,
            max_pairs_per_group=args.max_pairs_per_group,
            seed=args.seed, offset_radius=args.offset_radius)
        results["per_layer"][str(ln)] = stats
        gm = {k: (v["mean"] if v else None) for k, v in stats["groups"].items()}
        print(f"  layer {ln:>2} (block {bi}): "
              f"neighbour={gm['same_frame_neighbour']} "
              f"cross_same={gm['cross_frame_same_pos']} "
              f"cross_diff={gm['cross_frame_diff_pos']}")

    import json
    json_path = os.path.join(args.out, "similarity_heatmap_stats.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {json_path}")

    if args.plot:
        _render_plots(results, args.out)


def _render_plots(results, out_dir):
    import json  # noqa: F401  (results already in memory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results["per_layer"].keys(), key=lambda s: int(s))
    group_keys = ["same_frame_neighbour", "same_frame_nonadjacent",
                  "cross_frame_same_pos", "cross_frame_diff_pos"]

    # (1) per-layer mean similarity curves with p10-p90 band
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [int(l) for l in layers]
    for gk in group_keys:
        means, lo, hi = [], [], []
        for l in layers:
            s = results["per_layer"][l]["groups"][gk]
            means.append(s["mean"] if s else float("nan"))
            lo.append(s["p10"] if s else float("nan"))
            hi.append(s["p90"] if s else float("nan"))
        ax.plot(xs, means, marker="o", label=gk)
        ax.fill_between(xs, lo, hi, alpha=0.12)
    ax.set_xlabel("layer")
    ax.set_ylabel("token-token cosine similarity")
    ax.set_title("Same-frame vs cross-frame similarity by depth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "similarity_by_depth.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # (2) same-frame spatial-offset heatmaps, one panel per layer
    n = len(layers)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for k, l in enumerate(layers):
        ax = axes[k // cols][k % cols]
        heat = np.array(results["per_layer"][l]["offset_heatmap"], dtype=float)
        im = ax.imshow(heat, origin="lower", cmap="viridis", vmin=-1.0, vmax=1.0)
        R = results["per_layer"][l]["offset_radius"]
        ax.set_xticks([0, R, 2 * R]); ax.set_xticklabels([-R, 0, R])
        ax.set_yticks([0, R, 2 * R]); ax.set_yticklabels([-R, 0, R])
        ax.set_title(f"layer {l}: same-frame cos vs (dh,dw)")
        ax.set_xlabel("dw"); ax.set_ylabel("dh")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "same_frame_offset_heatmap.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {p1}\n[SAVED] {p2}")


if __name__ == "__main__":
    main()
