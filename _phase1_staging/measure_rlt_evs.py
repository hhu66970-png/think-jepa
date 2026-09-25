"""Measure STATIC BACKGROUND TEMPORAL REDUNDANCY on EgoDex clips for ThinkJEPA
(V-JEPA2 ViT-L video encoder).

Goal B remaining piece: quantify how much temporal redundancy RLT (run-length
tokenization) / EVS (efficient video sampling) can SAFELY cut from the dense
patch-embed token grid, WHERE the cut tokens live (static background vs hand-
motion regions), and (step 4, optional) the fidelity cost of pushing the cut
token set through the rest of the transformer vs the dense forward.

Everything is computed from a REAL encoder forward on actual clips. No estimated
numbers. RLT/EVS savings are content-dependent; we report the real cut on these
EgoDex clips (hand-motion regions are EXPECTED not to compress).

Reuses the existing harness (tools/run_token_merge_pca_experiment.py):
  - build_model  : vit_large_rope + load vitl.pt
  - load_video   : npz -> [1, 3, T, H, W] normalized
RLT/EVS implementations come from _phase1_staging/temporal_premerge.py
  - rlt_premerge (ragged: target_ratio=None -> per-sample run counts)
  - evs_static_prune (ragged: target_ratio=None -> per-sample kept counts)

Grid (64 frames / 256 / patch16 / tubelet2): t_grid=32, h_grid=w_grid=16,
N=8192 tokens. Time is the SLOW axis: flat id = t*256 + h*16 + w, so reshape
[B, t, hw, D] puts adjacent time steps of a fixed (h,w) at stride hw.

Usage (on GPU box):
  export PYTHONPATH=.:cache_train:vjepa2:vjepa2/src CUDA_VISIBLE_DEVICES=0
  python _phase1_staging/measure_rlt_evs.py \
      --clips <glob> --checkpoint vjepa2/vitl.pt --do_step4 \
      --out /root/autodl-tmp/thinkjepa-work/tiny-cache/measure_rlt_evs_out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Import harness helpers + RLT/EVS. PYTHONPATH must include repo root + vjepa2.
# ---------------------------------------------------------------------------
def _import_helpers(staging_dir):
    sys.path.insert(0, os.path.abspath("tools"))
    sys.path.insert(0, os.path.abspath(staging_dir))
    from run_token_merge_pca_experiment import build_model, load_video  # noqa
    from temporal_premerge import rlt_premerge, evs_static_prune  # noqa
    return build_model, load_video, rlt_premerge, evs_static_prune


# ---------------------------------------------------------------------------
# Step 1: adjacent-in-time cosine per spatial location.
# ---------------------------------------------------------------------------
@torch.no_grad()
def adjacent_time_cosine(tokens, t_grid, h_grid, w_grid, eps=1e-6):
    """tokens [B,N,D] -> adjacent-in-time cosine [B, t-1, hw] and stats.

    Returns (adj_cos[B,t-1,hw], dict_of_stats). Stats are over ALL (b,t-1,hw)
    adjacent pairs: mean/median/percentiles + fraction of pairs above common
    similarity cuts (the same cuts RLT will use).
    """
    B, N, D = tokens.shape
    hw = h_grid * w_grid
    tok_grid = tokens.reshape(B, t_grid, hw, D).float()
    normed = torch.nn.functional.normalize(tok_grid, dim=-1, eps=eps)
    adj = (normed[:, 1:] * normed[:, :-1]).sum(-1)  # [B, t-1, hw]
    flat = adj.reshape(-1).cpu()
    q = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    pct = torch.quantile(flat, q).tolist()
    stats = {
        "n_pairs": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "percentiles": {f"p{int(qq*100)}": float(v) for qq, v in zip(q.tolist(), pct)},
        "frac_ge": {
            f"{c:.2f}": float((flat >= c).float().mean())
            for c in (0.80, 0.85, 0.90, 0.95, 0.99)
        },
    }
    return adj, stats


# ---------------------------------------------------------------------------
# Step 2-3: RLT token-cut + spatial distribution of the cut.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rlt_sweep(rlt_premerge, tokens, t_grid, h_grid, w_grid, thresholds):
    """Run rlt_premerge (ragged) at each sim threshold. Returns list of records.

    Token-cut % = 1 - num_runs/N. Also returns per-location run counts so the
    caller can map WHERE the cut happens (a location with 1 run = fully static =
    fully cut except 1 token; 32 runs = fully dynamic = no cut)."""
    B, N, D = tokens.shape
    hw = h_grid * w_grid
    recs = []
    runs_per_loc_by_thr = {}
    for thr in thresholds:
        _, _, size_list, meta = rlt_premerge(
            tokens, t_grid, h_grid, w_grid,
            sim_threshold=thr, target_ratio=None, rep_position="start",
        )
        nruns = meta["num_runs_per_sample"].float()  # [B]
        # strongest correctness check from INTEGRATION doc: every original frame
        # covered by exactly one run -> per-sample run-length sum == N.
        size_sum_ok = all(int(s.sum().item()) == N for s in size_list)
        runs_per_loc = meta["runs_per_loc"].float()  # [B, hw]
        runs_per_loc_by_thr[f"{thr:.2f}"] = runs_per_loc.cpu()
        cut_frac = 1.0 - (nruns / N)
        recs.append({
            "sim_threshold": float(thr),
            "tokens_before": int(N),
            "tokens_after_mean": float(nruns.mean()),
            "tokens_cut_pct_mean": float(cut_frac.mean() * 100.0),
            "tokens_cut_pct_per_clip": [float((1.0 - r / N) * 100.0) for r in nruns.tolist()],
            "size_sum_eq_N_ok": bool(size_sum_ok),
            "merge_cos_mean": meta.get("merge_cos_mean"),
            "merge_cos_min": meta.get("merge_cos_min"),
        })
    return recs, runs_per_loc_by_thr


@torch.no_grad()
def evs_sweep(evs_static_prune, tokens, t_grid, h_grid, w_grid, thresholds, metric="l2_rel"):
    """Run evs_static_prune (ragged) at each diff threshold. Returns records +
    per-location kept counts (for spatial distribution)."""
    B, N, D = tokens.shape
    hw = h_grid * w_grid
    recs = []
    kept_per_loc_by_thr = {}
    for thr in thresholds:
        tok_list, ids_list, _, meta = evs_static_prune(
            tokens, t_grid, h_grid, w_grid,
            diff_threshold=thr, metric=metric, target_ratio=None,
            keep_first_frame=True,
        )
        kept = meta["kept_per_sample"].float()  # [B]
        # kept tokens per location: recompute from ids (id % hw == loc).
        kept_per_loc = torch.zeros(B, hw)
        for b in range(B):
            loc = (ids_list[b] % hw).cpu()
            kept_per_loc[b].scatter_add_(0, loc, torch.ones_like(loc, dtype=torch.float32))
        kept_per_loc_by_thr[f"{thr:.3f}"] = kept_per_loc
        cut_frac = 1.0 - (kept / N)
        recs.append({
            "diff_threshold": float(thr),
            "metric": metric,
            "tokens_before": int(N),
            "tokens_after_mean": float(kept.mean()),
            "tokens_cut_pct_mean": float(cut_frac.mean() * 100.0),
            "tokens_cut_pct_per_clip": [float((1.0 - k / N) * 100.0) for k in kept.tolist()],
            "kept_fraction": meta.get("kept_fraction"),
        })
    return recs, kept_per_loc_by_thr


# ---------------------------------------------------------------------------
# Step 3: classify locations as static-background vs dynamic-hand and report
# what fraction of the cut tokens are background.
# ---------------------------------------------------------------------------
@torch.no_grad()
def spatial_distribution(adj_cos, runs_per_loc, h_grid, w_grid, sim_threshold,
                         static_cell_cut=0.90):
    """adj_cos [B,t-1,hw]; runs_per_loc [B,hw] at the given RLT threshold.

    A spatial cell is 'static background' if its MEAN adjacent-time cosine
    (over the clip) >= static_cell_cut. We then report:
      - fraction of cells that are static (background area share)
      - of the tokens RLT CUT, what fraction came from static cells
        (cut_in_loc = t_grid - runs_in_loc tokens removed there)
      - a coarse spatial map (h x w) of mean adjacent cosine + runs (clip 0).
    """
    adj_cos = adj_cos.cpu()
    runs_per_loc = runs_per_loc.cpu()
    B, tm1, hw = adj_cos.shape
    t_grid = tm1 + 1
    cell_mean_cos = adj_cos.mean(dim=1)  # [B, hw] mean over time per location
    is_static_cell = cell_mean_cos >= static_cell_cut  # [B, hw]
    # tokens cut at each loc = t_grid - runs_in_loc (runs collapse t_grid -> runs)
    cut_per_loc = (float(t_grid) - runs_per_loc)  # [B, hw]
    total_cut = cut_per_loc.sum(dim=1).clamp_min(1.0)  # [B]
    cut_from_static = (cut_per_loc * is_static_cell.float()).sum(dim=1)  # [B]
    frac_cut_from_static = (cut_from_static / total_cut)  # [B]
    # spatial maps for clip 0 (reshape hw -> h x w)
    map_cos = cell_mean_cos[0].reshape(h_grid, w_grid).cpu().tolist()
    map_runs = runs_per_loc[0].reshape(h_grid, w_grid).cpu().tolist()
    return {
        "static_cell_cut": float(static_cell_cut),
        "sim_threshold_for_runs": float(sim_threshold),
        "background_area_frac_mean": float(is_static_cell.float().mean()),
        "background_area_frac_per_clip": [float(x) for x in is_static_cell.float().mean(dim=1).tolist()],
        "frac_of_cut_tokens_from_background_mean": float(frac_cut_from_static.mean()),
        "frac_of_cut_tokens_from_background_per_clip": [float(x) for x in frac_cut_from_static.tolist()],
        "mean_runs_static_cells": float(runs_per_loc[is_static_cell].mean()) if bool(is_static_cell.any()) else None,
        "mean_runs_dynamic_cells": float(runs_per_loc[~is_static_cell].mean()) if bool((~is_static_cell).any()) else None,
        "clip0_mean_adjcos_map_hxw": map_cos,
        "clip0_runs_map_hxw": map_runs,
    }


# ---------------------------------------------------------------------------
# Step 4 (optional): push cut tokens through the rest of the transformer with
# correct RoPE (mask=token_ids), restore to dense, compare cos vs dense forward.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _blocks_forward(model, x, token_ids, T, H_patches, W_patches):
    """Run patch-embed output x [B,N,D] through ALL blocks + final norm, with
    RoPE driven by token_ids (passed as `mask`). Mirrors VisionTransformer.forward
    with merge disabled but variable token set. Returns normed [B,N,D]."""
    for blk in model.blocks:
        x = blk(x, mask=token_ids, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
    if model.norm is not None:
        x = model.norm(x)
    return x


@torch.no_grad()
def step4_fidelity(model, restore_fn, ids_to_coords, premerge_fn, kind, tokens,
                   t_grid, h_grid, w_grid, thr, T, eps=1e-6, **pm_kwargs):
    """Compute cos_vs_dense for one (method, threshold). Dense and merged use the
    SAME block path (apples-to-apples); only the token set / ids differ.

    Returns dict with token-cut %, cos (mean over restored tokens), rel_l2.
    """
    B, N, D = tokens.shape
    device = tokens.device
    # dense reference (token_ids = arange N)
    dense_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).contiguous()
    dense_out = _blocks_forward(model, tokens.clone(), dense_ids, T, h_grid, w_grid)

    # CONTROL: run the FULL dense token set through the SAME merged-path code
    # (K=N, arange ids) -> must reproduce dense_out exactly (cos==1). Isolates any
    # bug in the block path / cosine from the genuine run-collapse loss.
    ctrl_cos = float(torch.nn.functional.cosine_similarity(
        _blocks_forward(model, tokens.clone(), dense_ids, T, h_grid, w_grid).float(),
        dense_out.float(), dim=-1).mean())

    # merged path: ragged would be variable-length per clip; for the block forward
    # we need rectangular [B,K,D]. With B==1 ragged is already rectangular; we run
    # PER-CLIP to support B>1 safely.
    cos_list, rell2_list, cut_list, preblock_cos_list = [], [], [], []
    for b in range(B):
        tb = tokens[b:b + 1]
        if kind == "rlt":
            t_new, id_new, sz_new, meta = premerge_fn(
                tb, t_grid, h_grid, w_grid, sim_threshold=thr,
                target_ratio=None, rep_position="start")
        else:
            t_new, id_new, sz_new, meta = premerge_fn(
                tb, t_grid, h_grid, w_grid, diff_threshold=thr,
                metric=pm_kwargs.get("metric", "l2_rel"), target_ratio=None,
                keep_first_frame=True)
        # ragged returns lists of length 1
        xt = t_new[0].unsqueeze(0).to(tb.dtype)         # [1, K, D]
        ids = id_new[0].unsqueeze(0).contiguous()        # [1, K]
        K = xt.shape[1]
        rep = meta["rep_for_orig"]
        # premerge ran on the SINGLE clip tb, so meta is batch-1: always row 0.
        rep_b = rep[0].unsqueeze(0) if isinstance(rep, list) else rep[0:1]
        # PRE-BLOCK diagnostic: restore the merged tokens at the PATCH-EMBED level
        # (no transformer) and compare to the dense patch-embed. Isolates the pure
        # run-mean / broadcast loss from the downstream transformer divergence.
        restored_pre = restore_fn(xt, ids, rep_b, N)  # [1, N, D]
        pre_c = torch.nn.functional.cosine_similarity(
            restored_pre.float(), tb.float(), dim=-1).mean()
        preblock_cos_list.append(float(pre_c))

        merged_out = _blocks_forward(model, xt, ids, T, h_grid, w_grid)  # [1,K,D]
        # restore to dense [1, N, D]
        restored = restore_fn(merged_out, ids, rep_b, N)  # [1, N, D]
        dref = dense_out[b:b + 1]
        c = torch.nn.functional.cosine_similarity(
            restored.float(), dref.float(), dim=-1).mean()
        rl = (torch.linalg.norm((restored - dref).float(), dim=-1) /
              (torch.linalg.norm(dref.float(), dim=-1) + eps)).mean()
        cos_list.append(float(c))
        rell2_list.append(float(rl))
        cut_list.append(float((1.0 - K / N) * 100.0))
    return {
        "method": kind,
        "threshold": float(thr),
        "tokens_cut_pct_mean": float(np.mean(cut_list)),
        "cos_vs_dense_mean": float(np.mean(cos_list)),
        "cos_vs_dense_per_clip": cos_list,
        "rel_l2_vs_dense_mean": float(np.mean(rell2_list)),
        "rel_l2_per_clip": rell2_list,
        "preblock_cos_vs_dense_mean": float(np.mean(preblock_cos_list)),
        "control_fullset_cos": ctrl_cos,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="glob for *.npz clips")
    ap.add_argument("--checkpoint", default="vjepa2/vitl.pt")
    ap.add_argument("--staging_dir", default="_phase1_staging")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rlt_thresholds", default="0.95,0.90,0.85,0.80")
    ap.add_argument("--evs_thresholds", default="0.02,0.05,0.10")
    ap.add_argument("--evs_metric", default="l2_rel", choices=["l2_rel", "cosine"])
    ap.add_argument("--static_cell_cut", type=float, default=0.90)
    ap.add_argument("--do_step4", action="store_true")
    ap.add_argument("--step4_rlt_thr", default="0.95,0.90,0.85")
    ap.add_argument("--step4_evs_thr", default="0.02,0.05,0.10")
    ap.add_argument("--out", default="measure_rlt_evs_out.json")
    args = ap.parse_args()

    rlt_thrs = [float(x) for x in args.rlt_thresholds.split(",") if x.strip()]
    evs_thrs = [float(x) for x in args.evs_thresholds.split(",") if x.strip()]

    build_model, load_video, rlt_premerge, evs_static_prune = _import_helpers(args.staging_dir)
    from src.models.utils.token_merge import restore_dense_tokens, ids_to_coords  # PYTHONPATH=vjepa2/src

    clips = sorted(glob.glob(args.clips))
    if not clips:
        raise SystemExit(f"no clips matched {args.clips!r}")
    print(f"[INFO] {len(clips)} clip(s):")
    for c in clips:
        print("   ", c)

    model = build_model(args.checkpoint, args.num_frames, args.img_size,
                        args.patch_size, "local_2x2_same_time_vec", args.device)
    T = args.num_frames // 2  # tubelet 2 -> t_grid
    h_grid = w_grid = args.img_size // args.patch_size

    per_clip = []
    # accumulate patch-embed tokens (B=1 each); we process clips one at a time to
    # keep memory low but report both per-clip and aggregate.
    all_tokens = []
    for path in clips:
        video, total = load_video(path, args.num_frames, args.img_size, args.device)
        with torch.no_grad():
            tok = model.patch_embed(video)  # [1, N, D]
        N = tok.shape[1]
        exp = T * h_grid * w_grid
        assert N == exp, f"expected N={exp} got {N} for {path}"
        all_tokens.append(tok)
        per_clip.append({"clip": os.path.basename(path), "total_frames": int(total),
                         "tokens": int(N), "dim": int(tok.shape[-1])})
        print(f"[INFO] {os.path.basename(path)}: patch_embed -> {tuple(tok.shape)} "
              f"(t={T},h={h_grid},w={w_grid}), src_frames={total}")

    tokens = torch.cat(all_tokens, dim=0)  # [B, N, D]
    B, N, D = tokens.shape

    # ---- Step 1: adjacent-time cosine ----
    adj_cos, adj_stats = adjacent_time_cosine(tokens, T, h_grid, w_grid)
    print(f"[STEP1] adjacent-time cosine: mean={adj_stats['mean']:.4f} "
          f"p50={adj_stats['percentiles']['p50']:.4f} "
          f">=0.90 frac={adj_stats['frac_ge']['0.90']:.4f}")

    # ---- Step 2: RLT + EVS sweeps ----
    rlt_recs, runs_per_loc_by_thr = rlt_sweep(rlt_premerge, tokens, T, h_grid, w_grid, rlt_thrs)
    evs_recs, kept_per_loc_by_thr = evs_sweep(evs_static_prune, tokens, T, h_grid, w_grid,
                                              evs_thrs, metric=args.evs_metric)
    print("[STEP2] RLT token-cut % vs sim_threshold:")
    for r in rlt_recs:
        print(f"   sim>={r['sim_threshold']:.2f}: cut={r['tokens_cut_pct_mean']:.1f}% "
              f"(tokens {r['tokens_before']}->{r['tokens_after_mean']:.0f}) "
              f"merge_cos_mean={r['merge_cos_mean']} size_ok={r['size_sum_eq_N_ok']}")
    print("[STEP2] EVS token-cut % vs diff_threshold:")
    for r in evs_recs:
        print(f"   diff<{r['diff_threshold']:.3f} ({r['metric']}): cut={r['tokens_cut_pct_mean']:.1f}% "
              f"(tokens {r['tokens_before']}->{r['tokens_after_mean']:.0f})")

    # ---- Step 3: spatial distribution at sim>=0.90 (and report 0.95 too) ----
    spatial = {}
    for thr in (0.90, 0.95):
        key = f"{thr:.2f}"
        if key in runs_per_loc_by_thr:
            spatial[key] = spatial_distribution(
                adj_cos, runs_per_loc_by_thr[key], h_grid, w_grid, thr,
                static_cell_cut=args.static_cell_cut)
    if "0.90" in spatial:
        s = spatial["0.90"]
        print(f"[STEP3] @sim>=0.90: background area={s['background_area_frac_mean']*100:.1f}% of cells; "
              f"of cut tokens, {s['frac_of_cut_tokens_from_background_mean']*100:.1f}% are background; "
              f"runs static={s['mean_runs_static_cells']} dynamic={s['mean_runs_dynamic_cells']}")

    result = {
        "config": {
            "clips": clips, "checkpoint": args.checkpoint,
            "num_frames": args.num_frames, "img_size": args.img_size,
            "patch_size": args.patch_size, "t_grid": T,
            "h_grid": h_grid, "w_grid": w_grid, "N": int(N), "dim": int(D),
            "rlt_thresholds": rlt_thrs, "evs_thresholds": evs_thrs,
            "evs_metric": args.evs_metric, "static_cell_cut": args.static_cell_cut,
        },
        "per_clip": per_clip,
        "step1_adjacent_time_cosine": adj_stats,
        "step2_rlt_sweep": rlt_recs,
        "step2_evs_sweep": evs_recs,
        "step3_spatial_distribution": spatial,
    }

    # ---- Step 4 (optional) ----
    if args.do_step4:
        print("[STEP4] fidelity cos_vs_dense (cut tokens through remaining blocks + RoPE):")
        s4 = []
        try:
            for thr in [float(x) for x in args.step4_rlt_thr.split(",") if x.strip()]:
                rec = step4_fidelity(model, restore_dense_tokens, ids_to_coords,
                                     rlt_premerge, "rlt", tokens, T, h_grid, w_grid, thr, T)
                s4.append(rec)
                print(f"   RLT sim>={thr:.2f}: cut={rec['tokens_cut_pct_mean']:.1f}% "
                      f"cos_vs_dense={rec['cos_vs_dense_mean']:.4f} (preblock={rec['preblock_cos_vs_dense_mean']:.4f}, "
                      f"ctrl={rec['control_fullset_cos']:.4f}) rel_l2={rec['rel_l2_vs_dense_mean']:.4f}")
            for thr in [float(x) for x in args.step4_evs_thr.split(",") if x.strip()]:
                rec = step4_fidelity(model, restore_dense_tokens, ids_to_coords,
                                     evs_static_prune, "evs", tokens, T, h_grid, w_grid, thr, T,
                                     metric=args.evs_metric)
                s4.append(rec)
                print(f"   EVS diff<{thr:.3f}: cut={rec['tokens_cut_pct_mean']:.1f}% "
                      f"cos_vs_dense={rec['cos_vs_dense_mean']:.4f} (preblock={rec['preblock_cos_vs_dense_mean']:.4f}, "
                      f"ctrl={rec['control_fullset_cos']:.4f}) rel_l2={rec['rel_l2_vs_dense_mean']:.4f}")
            result["step4_fidelity_vs_dense"] = s4
            result["step4_status"] = "ok"
        except Exception as e:  # pragma: no cover
            import traceback
            result["step4_status"] = f"FAILED: {e}"
            result["step4_traceback"] = traceback.format_exc()
            print(f"[STEP4] FAILED: {e}")
            traceback.print_exc()
    else:
        result["step4_status"] = "skipped (no --do_step4)"

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DONE] wrote {args.out}")


if __name__ == "__main__":
    main()
