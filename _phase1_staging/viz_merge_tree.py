"""viz_merge_tree.py  ---  Motivation evidence C4.

Goal: show WHICH patches Gradual K-BSM merges away. We run the encoder with
strategy ``bsm_ksim_gradual_vec`` over merge_layers (12,14,16,18,20),
restore_dense=True, capture the final source->receiver token mapping, reverse
it onto the 16x16 spatial grid, and overlay it on the original video frames:

  * MERGED-AWAY patches (a source token absorbed into some receiver) -> red tint
  * SURVIVOR patches (still its own token / a receiver)              -> green tint

The thesis: merged-away patches concentrate on background / low-texture /
static regions, while hands / edges / motion regions are preserved.

------------------------------------------------------------------------------
DEPENDENCIES
    torch, numpy, matplotlib
    ThinkJEPA ``src/`` importable (run from repo root containing src/), so the
    diagnostic merger ``bsm_ksim_gradual_vec`` is available.

USAGE (GPU box)
    python viz_merge_tree.py \
        --harness /ABS/PATH/run_token_merge_pca_experiment.py \
        --ckpt vjepa2/vitl.pt \
        --npz /ABS/PATH/clip.npz \
        --merge_layers 12,14,16,18,20 \
        --r_per_layer 0.15 \
        --frames 0,8,16,24,31 \
        --out /ABS/PATH/out_dir

We reuse the harness's build_model / load_video / apply_merge_config so that
model construction, checkpoint load, input preprocessing and the K-BSM config
path are byte-identical to the validated benchmark runner.

------------------------------------------------------------------------------
API TOUCHPOINTS TO VERIFY ON GPU  (best-effort; confirm before trusting):
  * HOW WE GET THE MERGE MAP (load-bearing):
    The encoder forward calls ``self.token_merger(...)`` and the merger returns
    a 5-tuple ``(x_new, token_ids, token_size, rep_for_orig, info)`` (confirmed:
    DiagnosticTokenMerger._forward_bsm + LocalTokenMerger.forward). We register
    a forward hook on ``model.token_merger`` that records ``token_ids`` (elem 1)
    and ``rep_for_orig`` (elem 3) AFTER EVERY merge layer; the LAST capture is
    the final mapping. ``rep_for_orig`` is [B, original_num_tokens]: for each
    ORIGINAL token id it stores the id of the surviving token it now maps to.
    -> merged_away(orig) == (rep_for_orig[orig] != orig).
    -> survivor(orig)    == (orig is present in final token_ids), equivalently
                            rep_for_orig[orig] == orig.
    VERIFY: that the hook fires once per merge layer and that elements 1 & 3 are
    token_ids / rep_for_orig respectively (matches the 5-tuple order in
    token_merge.py). If the encoder version wraps the merger differently, adapt
    the hook to read the same two tensors.
  * apply_merge_config(..., strategy='bsm_ksim_gradual_vec', restore_dense=True)
    rebinds DiagnosticTokenMerger (harness.ensure_token_merger). We then run
    model(video, return_merge_info=True, restore_dense=True).
  * merge_ratio semantics: float<1 => per-layer ratio of CURRENT tokens (capped
    at 0.25/layer inside _forward_bsm). We pass --r_per_layer straight through.
  * token id -> (t,h,w): ids_to_coords(id, h_grid, w_grid); patch (h,w) covers
    original-frame pixels [h*patch:(h+1)*patch, w*patch:(w+1)*patch]. A grid
    time-step t corresponds to tubelet_size source frames (t*2, t*2+1); we map a
    requested SOURCE frame f -> grid t = f // tubelet_size for the overlay.
"""
import argparse
import importlib.util
import os
import sys

try:
    import numpy as np
    import torch
except Exception as exc:
    np = None
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def grid_dims(num_frames, img_size, patch_size, tubelet_size=2):
    return (int(num_frames // tubelet_size),
            int(img_size // patch_size), int(img_size // patch_size))


def ids_to_coords_np(ids, h_grid, w_grid):
    tpf = int(h_grid * w_grid)
    ids = np.asarray(ids)
    t = ids // tpf
    rem = ids - t * tpf
    h = rem // int(w_grid)
    w = rem - h * int(w_grid)
    return t, h, w


class MergeStateCollector:
    """Forward hook on model.token_merger capturing (token_ids, rep_for_orig)
    after each merge layer. The merger returns a 5-tuple; see header touchpoint.
    """
    def __init__(self, merger):
        self.steps = []  # list of (token_ids[B,N], rep_for_orig[B,Norig])
        self._handle = merger.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        # out = (x_new, token_ids, token_size, rep_for_orig, info)
        if not isinstance(out, (tuple, list)) or len(out) < 5:
            return
        token_ids = out[1]
        rep_for_orig = out[3]
        self.steps.append((token_ids.detach().clone(),
                           rep_for_orig.detach().clone()))

    def remove(self):
        self._handle.remove()


def load_raw_frames(npz_path, num_frames, img_size, patch_size):
    """Return (frames_uint8 [t_grid, img_size, img_size, 3], grid_dims).

    Reproduces the harness frame SELECTION (np.linspace over the clip) and
    resize so overlay pixels align with the tokens the encoder actually saw,
    then averages each tubelet (tubelet_size consecutive sampled frames) down to
    the t_grid time-steps the encoder produces. We keep RAW (un-normalised)
    pixels for display.
    """
    tubelet = 2
    d = np.load(npz_path, allow_pickle=True)
    imgs = d["imgs"]  # [F, H, W, 3] uint8
    total = int(imgs.shape[0])
    idx = np.linspace(0, total - 1, num_frames).round().astype(int)
    sel = imgs[idx].astype(np.float32)  # [num_frames, H, W, 3]
    # resize to img_size with the same bilinear op as load_video (via torch)
    tns = torch.from_numpy(sel).permute(0, 3, 1, 2) / 255.0
    if tns.shape[-1] != img_size or tns.shape[-2] != img_size:
        tns = torch.nn.functional.interpolate(
            tns, size=(img_size, img_size), mode="bilinear", align_corners=False)
    sel = (tns.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # average tubelets -> [t_grid, img_size, img_size, 3] to match grid time-step
    t_grid = num_frames // tubelet
    sel = sel[: t_grid * tubelet].reshape(t_grid, tubelet, img_size, img_size, 3)
    frames = sel.mean(axis=1).clip(0, 255).astype(np.uint8)
    return frames, total


def main():
    ap = argparse.ArgumentParser(description="C4 merge-tree overlay: which "
                                             "patches K-BSM merges away.")
    ap.add_argument("--harness", required=True)
    ap.add_argument("--ckpt", default="vjepa2/vitl.pt")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--merge_layers", default="12,14,16,18,20")
    ap.add_argument("--r_per_layer", type=float, default=0.15,
                    help="per-layer merge ratio (float<1, capped 0.25/layer)")
    ap.add_argument("--frames", default="0,8,16,24,31",
                    help="GRID time-steps to render (0..t_grid-1)")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--bsm_match_metric", default="key", choices=["key", "feature"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"torch/numpy import failed: {_IMPORT_ERROR}")

    os.makedirs(args.out, exist_ok=True)
    harness = load_harness(args.harness)
    merge_layers = [int(v) for v in args.merge_layers.split(",") if v.strip()]
    t_grid, h_grid, w_grid = grid_dims(args.num_frames, args.img_size, args.patch_size)

    model = harness.build_model(
        args.ckpt, args.num_frames, args.img_size, args.patch_size,
        initial_strategy="bsm_ksim_gradual_vec", device=args.device)

    # Configure K-BSM exactly like the benchmark runner (restore_dense=True).
    harness.apply_merge_config(
        model, enabled=True, strategy="bsm_ksim_gradual_vec",
        merge_layers=merge_layers, merge_ratio=float(args.r_per_layer),
        restore_dense=True, bsm_match_metric=args.bsm_match_metric)

    video, total_frames = harness.load_video(
        args.npz, args.num_frames, args.img_size, args.device)
    print(f"[INFO] clip={args.npz} src_frames={total_frames} input={tuple(video.shape)} "
          f"merge_layers={merge_layers} r/layer={args.r_per_layer} "
          f"grid t={t_grid} h={h_grid} w={w_grid}")

    collector = MergeStateCollector(model.token_merger)
    with torch.no_grad():
        _out, merge_infos = model(video, return_merge_info=True, restore_dense=True)
    collector.remove()

    if not collector.steps:
        raise RuntimeError(
            "No merge-state captured. The token_merger hook never fired or the "
            "return tuple layout differs. See the API TOUCHPOINTS in the header "
            "and adapt MergeStateCollector to read token_ids / rep_for_orig.")

    final_token_ids, final_rep = collector.steps[-1]
    final_token_ids = final_token_ids[0].cpu().numpy()      # [N_final]
    final_rep = final_rep[0].cpu().numpy()                  # [N_orig]
    n_orig = int(final_rep.shape[0])
    expected = t_grid * h_grid * w_grid
    if n_orig != expected:
        print(f"[WARN] n_orig={n_orig} != expected {expected}; check grid.")

    # merged-away mask per ORIGINAL token: rep maps it to a *different* id.
    orig_ids = np.arange(n_orig)
    merged_away = final_rep != orig_ids                     # bool [N_orig]
    survivor_set = set(final_token_ids.tolist())
    survivor = np.array([i in survivor_set for i in orig_ids])  # bool [N_orig]
    # receivers = survivors that absorbed >=1 source
    recv_ids, recv_counts = np.unique(final_rep[merged_away], return_counts=True)
    receiver_load = dict(zip(recv_ids.tolist(), recv_counts.tolist()))

    # per-layer trajectory summary (counts) for the figure caption / json
    traj = harness.layer_trajectory(merge_infos, original_tokens=n_orig)

    tt, hh, ww = ids_to_coords_np(orig_ids, h_grid, w_grid)

    # save the raw masks for downstream / reproducibility
    import json
    summary = {
        "meta": {
            "npz": args.npz, "ckpt": args.ckpt, "merge_layers": merge_layers,
            "r_per_layer": args.r_per_layer, "grid": [t_grid, h_grid, w_grid],
            "num_tokens_original": n_orig,
            "num_tokens_final": int(final_token_ids.shape[0]),
            "frac_merged_away": float(merged_away.mean()),
            "bsm_match_metric_requested": args.bsm_match_metric,
        },
        "layer_trajectory": traj,
        "num_receivers": int(len(receiver_load)),
    }
    with open(os.path.join(args.out, "merge_tree_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.out, "merge_tree_masks.npz"),
             merged_away=merged_away, survivor=survivor,
             rep_for_orig=final_rep, final_token_ids=final_token_ids,
             t=tt, h=hh, w=ww)
    print(f"[INFO] merged_away fraction={merged_away.mean():.3f} "
          f"survivors={survivor.sum()} receivers={len(receiver_load)}")

    # overlays
    want_frames = [int(v) for v in args.frames.split(",") if v.strip()]
    want_frames = [f for f in want_frames if 0 <= f < t_grid]
    frames_rgb, _ = load_raw_frames(args.npz, args.num_frames, args.img_size, args.patch_size)
    _render_overlays(frames_rgb, merged_away, receiver_load, tt, hh, ww,
                     h_grid, w_grid, args.patch_size, want_frames, args.out, traj)


def _render_overlays(frames_rgb, merged_away, receiver_load, tt, hh, ww,
                     h_grid, w_grid, patch, want_frames, out_dir, traj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(want_frames)
    if n == 0:
        print("[WARN] no valid --frames to render")
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)

    for k, ft in enumerate(want_frames):
        ax = axes[k // cols][k % cols]
        ax.imshow(frames_rgb[ft])
        sel = tt == ft
        ids_here = np.where(sel)[0]
        for tid in ids_here:
            h, w = int(hh[tid]), int(ww[tid])
            y0, x0 = h * patch, w * patch
            if merged_away[tid]:
                # source token absorbed elsewhere -> red overlay
                ax.add_patch(Rectangle((x0, y0), patch, patch, linewidth=0,
                                       facecolor=(1, 0, 0, 0.42)))
            else:
                load = receiver_load.get(int(tid), 0)
                if load > 0:
                    # receiver that absorbed sources -> green, brighter = more
                    a = min(0.6, 0.2 + 0.12 * load)
                    ax.add_patch(Rectangle((x0, y0), patch, patch, linewidth=0,
                                           facecolor=(0, 1, 0, a)))
                # plain survivor (load 0): no overlay (show original pixels)
        ax.set_title(f"grid t={ft}  (red=merged-away, green=receiver)")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("K-BSM merge map: merged-away (background/static) vs survivors",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, "merge_tree_overlay.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # also a compact per-frame "merged-away density" heatmap (16x16)
    fig2, axes2 = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False)
    for k, ft in enumerate(want_frames):
        ax = axes2[k // cols][k % cols]
        grid = np.zeros((h_grid, w_grid), dtype=float)
        sel = tt == ft
        for tid in np.where(sel)[0]:
            grid[int(hh[tid]), int(ww[tid])] = 1.0 if merged_away[tid] else 0.0
        ax.imshow(grid, origin="upper", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"t={ft}: merged-away grid")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, rows * cols):
        axes2[k // cols][k % cols].axis("off")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "merge_tree_grid.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"[SAVED] {p}\n[SAVED] {p2}")


if __name__ == "__main__":
    main()
