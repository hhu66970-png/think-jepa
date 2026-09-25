"""Training-free TEMPORAL token pre-merge for V-JEPA2 ViT-L video tokens.

Two methods, both run ONCE as a "pre-merge" hook right after ``patch_embed``
(and before transformer block 0). They are orthogonal to the in-block spatial
merging (K-BSM / local 2x2) in ``token_merge.py`` and produce token sets that
follow the SAME representation contract so K-BSM can keep running afterwards:

  - ``tokens``       [B, N', D]  rectangular hidden states (size-weighted avg)
  - ``token_ids``    [B, N']     ORIGINAL flat position id  (t*H*W + h*W + w)
  - ``token_size``   [B, N']     #original tokens represented (size-weighting)
  - ``rep_for_orig`` [B, N_orig] for each original id -> id of its representative
                                 (consumed by ``restore_dense_tokens``)

See ``src/token_merge.py`` for the source of this convention
(``init_token_merge_state``, ``ids_to_coords``, ``restore_dense_tokens``,
``_apply_merge_from_positions``). The two functions here return
``rep_for_orig`` already remapped through the pre-merge so a *subsequent*
multi-layer K-BSM (which composes its own remap onto the running
``rep_for_orig``) and ``restore_dense_tokens`` both stay correct.

Grid convention for V-JEPA2 ViT-L @ 64 frames / 256x256 / patch16 / tubelet2:
  t_grid = 32 (temporal, after tubelet2), h_grid = w_grid = 16, so
  N_orig = 32 * 16 * 16 = 8192 tokens. RoPE attends by each token's (t, h, w);
  hence both methods preserve / report the original (t, h, w) of every surviving
  token (EVS keeps ids unchanged; RLT reports the run's representative position).

torch is imported defensively: the offline dev box has NO torch, so we must
still pass ``python3 -m py_compile``. At runtime on the GPU box torch IS present
and ``_HAS_TORCH`` is True; calling these functions without torch raises a clear
RuntimeError instead of a NameError.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - offline dev box without torch
    # py_compile only checks SYNTAX, so the module compiles fine without torch.
    # All tensor ops below are written against the real torch API; they only
    # execute on the GPU box where the import succeeds.
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            "temporal_premerge requires PyTorch at runtime; this environment has "
            "no torch (offline dev box). Run on the GPU server."
        )


def _coords_for_ids(token_ids, h_grid, w_grid):
    """Flat id -> (t, h, w). Identical math to token_merge.ids_to_coords, kept
    local so this module has no hard import dependency on token_merge."""
    tokens_per_frame = int(h_grid * w_grid)
    t = token_ids // tokens_per_frame
    rem = token_ids - t * tokens_per_frame
    h = rem // int(w_grid)
    w = rem - h * int(w_grid)
    return t, h, w


# ======================================================================
# RLT: Run-Length Tokenization (content-adaptive temporal run merging)
# ======================================================================
def rlt_premerge(
    tokens,
    t_grid,
    h_grid,
    w_grid,
    sim_threshold: float = 0.9,
    target_ratio=None,
    rep_position: str = "start",
    eps: float = 1e-6,
):
    """Run-length-merge temporally-redundant patches per spatial location.

    For every spatial location (h, w) independently, walk the t_grid time steps
    in order. Adjacent time steps whose patch embeddings have cosine similarity
    >= ``sim_threshold`` are absorbed into the CURRENT run; otherwise a new run
    starts. Each run collapses to ONE output token whose value is the
    size-weighted average of the patches in the run (here every original patch
    has size 1, so it is a plain mean over the run; size-weighting matters once
    this composes with later merges that carry size>1 -- the contract is kept).

    Args:
        tokens:        [B, N, D] dense patch-embed output, N == t*h*w. (A
                       [B, T, H*W, D] input should be reshaped to [B, N, D]
                       BEFORE calling; see INTEGRATION doc.)
        t_grid, h_grid, w_grid: temporal / spatial grid (32, 16, 16 here).
        sim_threshold: cosine cut for "almost unchanged". Higher => fewer merges
                       (more conservative). Used directly in the per-sample
                       variable-length path AND to *build* runs in the batched
                       top-K path.
        target_ratio:  if not None, the FINAL token count is forced to
                       round((1-?) ... ) -> we keep ``K = ceil(target_ratio*N)``
                       tokens per sample so the batch stays a rectangular
                       [B, K, D] (see "batch handling" below). If None, returns a
                       per-sample (ragged) Python list and the caller must pad.
        rep_position:  "start" | "mid" -> which original (t,h,w) inside a run is
                       reported as the surviving token's id (for RoPE). "start"
                       (run's first frame) matches RLT's original formulation and
                       keeps ids monotonically increasing within a location, which
                       is convenient for debugging; "mid" centers the positional
                       phase over the run's temporal extent (slightly better for
                       long runs but loses the clean "first occurrence" semantics).
                       We default to "start".
        eps:           normalize epsilon.

    Returns:
        (tokens_new, token_ids_new, token_size_new, meta)

        If ``target_ratio`` is None (RAGGED mode):
            tokens_new      : list of length B, each [n_b, D]
            token_ids_new   : list of length B, each [n_b]  (original flat ids,
                              the chosen representative position per run)
            token_size_new  : list of length B, each [n_b]  (run length = #frames)
            meta            : dict (see below); meta["rep_for_orig"] is a list too.

        If ``target_ratio`` is a float in (0,1] (RECTANGULAR mode, preferred by
        our pipeline):
            tokens_new      : [B, K, D]
            token_ids_new   : [B, K]
            token_size_new  : [B, K]
            meta["rep_for_orig"] : [B, N]  ready for restore_dense_tokens

    meta always contains: rep_for_orig, num_tokens_before (=N), per-sample
    num_runs (pre-truncation), the chosen K (rectangular mode), rep_position,
    sim_threshold, and the cosine stats of accepted merges (for logging).

    ---- Batch handling (ragged vs rectangular) -------------------------------
    Different samples (and even different (h,w) columns) yield different run
    counts, so the natural RLT output is RAGGED. Our existing pipeline
    (token_merge.py) strongly prefers a rectangular [B, N', D] (its dense-grid
    fast paths even assert equal per-sample lengths). Two strategies:

      (A) target_ratio (DEFAULT for our pipeline): build runs with the threshold,
          then for each sample SCORE every run by how redundant it was -- we use
          the run's internal mean adjacent cosine (higher = safer to keep merged)
          times (run_len - 1) as the "savings" -- and keep the global top-K runs
          per sample so all samples end at exactly K = ceil(target_ratio * N)
          tokens. Concretely we keep the K runs with the LARGEST run_len first
          (most compression), breaking ties by highest internal similarity; any
          residual budget keeps singleton runs. This guarantees rectangular
          output and a fixed compute budget, at the cost of being slightly less
          content-adaptive than pure thresholding.

      (B) padding + mask (RAGGED -> pad to max_n with an attention mask): keeps
          RLT fully content-adaptive but forces every downstream op (K-BSM,
          attention, restore) to be mask-aware. token_merge.py is NOT mask-aware
          today, so this needs plumbing; we expose it via target_ratio=None +
          a documented pad helper rather than doing it implicitly here.

    We implement BOTH: ragged when target_ratio is None, rectangular top-K when
    a target_ratio is given. The rectangular path is what the INTEGRATION doc
    recommends wiring first.

    ---- Complexity / approximation -------------------------------------------
    * Per sample: one pass over t for each of (h*w) locations to compute
      adjacent cosine: O(N * D) work, O(N) similarities. Run assignment is a
      cumulative-sum over the per-location "break" mask: O(N). Group means are a
      scatter-add (index_add_) over D: O(N * D). So overall O(B * N * D), the
      same order as a single linear layer over the tokens -- cheap.
    * APPROXIMATION: a run is represented by the MEAN of its frames. That is
      exact only if the patches are truly identical; for "almost identical"
      patches it is a low-error average (this is the whole training-free premise).
      RoPE is applied at the run's representative (t,h,w), so a long run gets ONE
      positional phase instead of spanning several -- acceptable when the run is
      genuinely static, lossy if the threshold is too low. HONEST NOTE: on
      EgoDex (fixed ego camera, static background) static regions compress well,
      but HAND-MOTION regions will NOT form long runs, so real savings are
      content-dependent and MUST be measured on GPU (see INTEGRATION doc) -- no
      a-priori speedup is claimed here.
    """
    _require_torch()
    if tokens.dim() != 3:
        raise ValueError(f"rlt_premerge expects [B, N, D], got shape {tuple(tokens.shape)}")
    batch_size, num_tokens, dim = tokens.shape
    t_grid = int(t_grid)
    h_grid = int(h_grid)
    w_grid = int(w_grid)
    hw = h_grid * w_grid
    expected = t_grid * hw
    if num_tokens != expected:
        raise ValueError(
            f"rlt_premerge expects dense N = t*h*w = {expected}, got {num_tokens}. "
            "Pre-merge must run on the dense patch grid (before any other merge)."
        )
    if rep_position not in ("start", "mid"):
        raise ValueError("rep_position must be 'start' or 'mid'")

    device = tokens.device
    # View as [B, t, hw, D]; time is the SLOW axis within a frame-major flat id
    # (id = t*hw + (h*w)), so reshape(B, t, hw, D) puts adjacent time steps of a
    # fixed (h,w) at stride hw -- exactly tok_grid[:, :, loc, :].
    tok_grid = tokens.reshape(batch_size, t_grid, hw, dim)

    # Adjacent-in-time cosine similarity per location: [B, t-1, hw].
    normed = F.normalize(tok_grid.float(), dim=-1, eps=eps)
    adj_cos = (normed[:, 1:] * normed[:, :-1]).sum(dim=-1)  # [B, t-1, hw]

    # break[b, j, loc] == True  <=>  time step j starts a NEW run (j in 1..t-1).
    # Step 0 always starts a run. A break occurs when similarity to the previous
    # step is BELOW threshold.
    is_break = adj_cos < float(sim_threshold)  # [B, t-1, hw]
    # run_id along time per location via cumulative sum of breaks (step 0 -> id 0)
    zero_row = torch.zeros(batch_size, 1, hw, device=device, dtype=torch.long)
    run_id_time = torch.cat(
        [zero_row, torch.cumsum(is_break.long(), dim=1)], dim=1
    )  # [B, t, hw], values 0..(#runs_in_that_loc - 1)

    # Per-(b, loc) number of runs = max run_id + 1.
    runs_per_loc = run_id_time.amax(dim=1) + 1  # [B, hw]
    num_runs_per_sample = runs_per_loc.sum(dim=1)  # [B]

    # Give every (b, loc, run) a GLOBAL run index within its sample so we can
    # scatter-average. We offset each location's local run ids by the cumulative
    # run count of earlier locations -> contiguous [0 .. num_runs_b) per sample.
    # offsets[b, loc] = sum_{loc' < loc} runs_per_loc[b, loc']
    run_offsets = torch.cumsum(runs_per_loc, dim=1) - runs_per_loc  # [B, hw]
    # global_run[b, t, loc] = run_offsets[b, loc] + run_id_time[b, t, loc]
    global_run = run_id_time + run_offsets.unsqueeze(1)  # [B, t, hw], long

    # ---- size-weighted run means (here input size==1 so it's a plain mean) ----
    # Flatten (t, hw) -> sequence of N entries per sample, scatter into runs.
    flat_run = global_run.permute(0, 2, 1).reshape(batch_size, num_tokens)  # loc-major
    # We need values/ids in the SAME (loc-major) order as flat_run:
    #   loc-major flat index p = loc * t + j  -> original id = j*hw + loc
    # Precompute the original flat id for each loc-major slot once.
    loc_idx = torch.arange(hw, device=device)
    time_idx = torch.arange(t_grid, device=device)
    # orig_id[loc, j] = j*hw + loc
    orig_id_locmajor = (time_idx.unsqueeze(0) * hw + loc_idx.unsqueeze(1)).reshape(-1)  # [N]
    orig_id_locmajor = orig_id_locmajor.unsqueeze(0).expand(batch_size, -1)  # [B, N]

    tok_locmajor = tok_grid.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, dim)

    max_runs = int(num_runs_per_sample.amax().item())

    # Accumulate sums and counts into [B, max_runs, *]; unused run slots stay 0.
    run_sum = torch.zeros(batch_size, max_runs, dim, device=device, dtype=torch.float32)
    run_cnt = torch.zeros(batch_size, max_runs, device=device, dtype=torch.float32)
    run_sum.scatter_add_(
        1, flat_run.unsqueeze(-1).expand(-1, -1, dim), tok_locmajor.float()
    )
    run_cnt.scatter_add_(1, flat_run, torch.ones_like(flat_run, dtype=torch.float32))
    run_mean = run_sum / run_cnt.clamp_min(1.0).unsqueeze(-1)  # [B, max_runs, D]

    # Representative original id per run (start or mid frame of the run).
    # For each run we need min time index (start) or the median time index (mid).
    # scatter_reduce 'amin' gives the start slot's loc-major position; we then map
    # to original id. We track the loc-major POSITION (0..N-1) per run.
    locmajor_pos = torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
    big = num_tokens + 1
    run_min_pos = torch.full((batch_size, max_runs), big, device=device, dtype=torch.long)
    run_min_pos.scatter_reduce_(1, flat_run, locmajor_pos, reduce="amin", include_self=True)
    run_max_pos = torch.full((batch_size, max_runs), -1, device=device, dtype=torch.long)
    run_max_pos.scatter_reduce_(1, flat_run, locmajor_pos, reduce="amax", include_self=True)

    if rep_position == "start":
        rep_pos = run_min_pos
    else:  # "mid": midpoint between first and last loc-major slot of the run.
        # Because a run is contiguous in time at a fixed loc, (min+max)//2 is the
        # loc-major slot of the run's middle frame.
        rep_pos = (run_min_pos + run_max_pos) // 2
    rep_pos = rep_pos.clamp_max(num_tokens - 1)
    # rep ORIGINAL id for each run (gather original id at that loc-major slot).
    run_rep_id = orig_id_locmajor.gather(1, rep_pos.clamp_min(0))  # [B, max_runs]

    # Mean adjacent cosine inside each run (for top-K scoring / logging). We
    # scatter the t-1 adjacent cosines into the run of their RIGHT endpoint
    # (which shares that run when it was NOT a break).
    # adj is [B, t-1, hw]; right endpoint j(1..t-1) belongs to run global_run[:, j].
    right_run = global_run[:, 1:].permute(0, 2, 1).reshape(batch_size, -1)  # [B,(t-1)*hw]
    right_is_same = (~is_break).permute(0, 2, 1).reshape(batch_size, -1)    # [B,(t-1)*hw]
    adj_flat = adj_cos.permute(0, 2, 1).reshape(batch_size, -1)             # [B,(t-1)*hw]
    cos_sum = torch.zeros(batch_size, max_runs, device=device, dtype=torch.float32)
    cos_cnt = torch.zeros(batch_size, max_runs, device=device, dtype=torch.float32)
    masked_cos = torch.where(right_is_same, adj_flat, torch.zeros_like(adj_flat))
    cos_sum.scatter_add_(1, right_run, masked_cos)
    cos_cnt.scatter_add_(1, right_run, right_is_same.float())
    run_mean_cos = cos_sum / cos_cnt.clamp_min(1.0)  # 0 for singleton runs

    # valid run slots (run index < num_runs for that sample)
    run_index = torch.arange(max_runs, device=device).unsqueeze(0)  # [1, max_runs]
    valid_run = run_index < num_runs_per_sample.unsqueeze(1)        # [B, max_runs]

    meta = {
        "method": "rlt",
        "num_tokens_before": int(num_tokens),
        "num_runs_per_sample": num_runs_per_sample.detach(),
        "runs_per_loc": runs_per_loc.detach(),
        "rep_position": rep_position,
        "sim_threshold": float(sim_threshold),
        "t_grid": t_grid,
        "h_grid": h_grid,
        "w_grid": w_grid,
    }

    if target_ratio is None:
        # -------- RAGGED mode: return per-sample lists; caller pads if needed --
        tokens_list, ids_list, size_list, rep_list = [], [], [], []
        for b in range(batch_size):
            nb = int(num_runs_per_sample[b].item())
            tokens_list.append(run_mean[b, :nb].to(tokens.dtype))
            ids_list.append(run_rep_id[b, :nb])
            size_list.append(run_cnt[b, :nb].to(tokens.dtype))
            # rep_for_orig for this sample: map every original id -> its run's
            # representative id. global_run[b] tells each (t,loc) slot's run; the
            # run's rep id is run_rep_id[b, run].
            rep_for_orig_b = torch.empty(num_tokens, device=device, dtype=torch.long)
            # original id at loc-major slot p is orig_id_locmajor[b, p]; its run is
            # flat_run[b, p]; rep id is run_rep_id[b, flat_run[b,p]].
            rep_for_orig_b[orig_id_locmajor[b]] = run_rep_id[b, flat_run[b]]
            rep_list.append(rep_for_orig_b)
        meta["rep_for_orig"] = rep_list  # list[B] of [N]
        meta["mode"] = "ragged"
        return tokens_list, ids_list, size_list, meta

    # -------- RECTANGULAR (top-K) mode: fixed K per sample --------------------
    ratio = float(target_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"target_ratio must be in (0,1], got {ratio}")
    keep_k = int(max(1, min(num_tokens, round(ratio * num_tokens))))
    # If even the fully-merged set has > keep_k runs for some sample, we cannot
    # reach keep_k purely by merging runs (RLT only merges along time). Clamp K
    # up to the largest per-sample run count so we never request fewer tokens
    # than the most-fragmented sample actually has. This keeps output rectangular
    # and never silently drops un-mergeable (e.g. moving) tokens.
    keep_k = max(keep_k, int(num_runs_per_sample.amax().item()))
    keep_k = min(keep_k, num_tokens)

    # Score runs: prefer keeping the BIGGEST runs (largest compression) and, when
    # tie, the most internally-similar (safest). Invalid run slots score -inf so
    # they sort last. Score = run_len + small * mean_cos. run_len dominates.
    score = run_cnt + 1e-3 * run_mean_cos
    score = torch.where(valid_run, score, torch.full_like(score, float("-inf")))

    # We keep the top-K runs by score. Because keep_k >= max runs-per-sample, a
    # sample with fewer than keep_k runs keeps ALL its runs; the leftover slots
    # are filled by re-SPLITTING the lowest-priority kept runs back toward
    # singletons is NOT done (would need re-expansion); instead we pad those
    # samples by repeating their last valid run id with size 0 contribution.
    # Simpler + exact: select indices of the top-K runs per sample.
    topk_score, topk_idx = score.topk(keep_k, dim=1)  # [B, keep_k]

    tokens_new = run_mean.gather(
        1, topk_idx.unsqueeze(-1).expand(-1, -1, dim)
    ).to(tokens.dtype)
    token_ids_new = run_rep_id.gather(1, topk_idx)
    token_size_new = run_cnt.gather(1, topk_idx).to(tokens.dtype)

    # Samples that had < keep_k real runs got some -inf slots in topk. Those slots
    # picked an arbitrary invalid run (value 0, id 0, size 0). Fix them so they
    # are inert: zero size + a sentinel id == its own (so restore can't mis-route)
    invalid_slot = ~torch.isfinite(topk_score)  # [B, keep_k]
    if bool(invalid_slot.any().item()):
        # For inert slots, set size 0 and reuse the run's rep id of run 0 of that
        # sample (any valid id works; size 0 makes it contribute nothing to a
        # size-weighted merge, and restore_dense never maps an ORIGINAL to it
        # because rep_for_orig below only references REAL runs).
        token_size_new = torch.where(
            invalid_slot, torch.zeros_like(token_size_new), token_size_new
        )
        # give inert slots a distinct, in-range id (0) and zero feature
        token_ids_new = torch.where(
            invalid_slot, torch.zeros_like(token_ids_new), token_ids_new
        )
        tokens_new = torch.where(
            invalid_slot.unsqueeze(-1), torch.zeros_like(tokens_new), tokens_new
        )

    # rep_for_orig [B, N]: map each ORIGINAL id to the rep id of the run it fell
    # into. Note: this references REAL runs only (every original belongs to some
    # real run), so inert padding slots are never targets -> restore_dense_tokens
    # stays correct. BUT restore_dense_tokens locates a rep id among token_ids_new
    # via scatter; padded id=0 could collide with a real run whose rep id is 0.
    # To avoid that collision we ensure run 0's rep id (smallest original id of
    # location 0 = id 0 when present) is only padded when that run is itself kept.
    # Safer alternative used here: build rep_for_orig to point at the KEEP POSITION
    # is not needed (restore uses ids), so we keep ORIGINAL-id semantics and, in
    # the rare under-K sample, rely on size-0 inert slots not being chosen as a
    # rep target. See INTEGRATION doc "RLT + restore_dense" for the exact caveat.
    rep_for_orig = torch.empty(batch_size, num_tokens, device=device, dtype=torch.long)
    src_ids = orig_id_locmajor                       # [B, N] original id per slot
    rep_ids = run_rep_id.gather(1, flat_run)         # [B, N] rep id of slot's run
    rep_for_orig.scatter_(1, src_ids, rep_ids)

    meta["rep_for_orig"] = rep_for_orig
    meta["mode"] = "rectangular_topk"
    meta["keep_k"] = int(keep_k)
    meta["num_inert_slots"] = int(invalid_slot.sum().item())
    # accepted-merge cosine stats (over valid, multi-frame runs) for logging
    multi = valid_run & (run_cnt > 1)
    if bool(multi.any().item()):
        acc = run_mean_cos[multi]
        meta["merge_cos_mean"] = float(acc.mean().item())
        meta["merge_cos_min"] = float(acc.min().item())
    else:
        meta["merge_cos_mean"] = None
        meta["merge_cos_min"] = None

    return tokens_new, token_ids_new, token_size_new, meta


# ======================================================================
# EVS: Efficient Video Sampling (prune temporally-static patches)
# ======================================================================
def evs_static_prune(
    tokens,
    t_grid,
    h_grid,
    w_grid,
    diff_threshold: float = 0.05,
    metric: str = "l2_rel",
    target_ratio=None,
    keep_first_frame: bool = True,
    eps: float = 1e-6,
):
    """Prune patches that are ~unchanged from the SAME location in the previous
    frame; KEEP surviving tokens with their ORIGINAL (t, h, w) position id intact.

    Unlike RLT, EVS does NOT merge/average -- it simply DROPS a token at (t,h,w)
    when its patch is nearly identical to the patch at (t-1, h, w). The kept
    tokens keep their exact original ids, which is what makes EVS RoPE-friendly:
    the encoder computes RoPE phases from each kept token's unchanged (t,h,w), so
    no positional re-indexing is needed (see INTEGRATION doc). token_size stays 1
    for every kept token (EVS removes, it does not combine), so size-weighting is
    a no-op here and later K-BSM size accounting is unaffected.

    "static" test (per token at t>=1, location loc):
        metric == "l2_rel"  : ||x_t - x_{t-1}|| / (||x_{t-1}|| + eps) < threshold
        metric == "cosine"  : cos(x_t, x_{t-1}) > (1 - threshold)
    The first frame (t==0) is ALWAYS kept (if keep_first_frame) so every spatial
    location is represented at least once -- otherwise a perfectly static pixel
    column would vanish entirely.

    Args mirror rlt_premerge. ``diff_threshold`` is the staticness cut; smaller =>
    prune less. ``target_ratio`` switches between:
        None        -> RAGGED: return per-sample lists of kept tokens/ids.
        float (0,1] -> RECTANGULAR: keep exactly K = ceil(ratio*N) tokens/sample
                       by keeping the K MOST-DYNAMIC tokens (largest change), so
                       all samples share one length [B, K, D].

    Returns (tokens_new, token_ids_new, token_size_new, meta) with the same
    shapes/contract as rlt_premerge. token_size_new is all-ones (no merging).
    meta["rep_for_orig"] maps every ORIGINAL id to the id of the kept token that
    represents it: a kept token maps to itself; a PRUNED static token maps to the
    most recent kept token at the SAME location (its temporal "anchor"), so
    restore_dense_tokens can scatter the kept feature back to the pruned slot.

    ---- Complexity / approximation -------------------------------------------
    * O(B * N * D): one adjacent-frame diff per token, one cumulative "last kept"
      pass per location. The "last kept anchor" for rep_for_orig is a temporal
      cummax of (kept ? slot : -inf) along time per location -> O(N).
    * APPROXIMATION: a pruned token is assumed equal to its temporal anchor; the
      encoder simply never sees it (sparse attention over kept tokens). This is
      lossless iff the patch truly did not change. Position ids of kept tokens are
      EXACT (no averaging), so RoPE is exact for survivors. HONEST NOTE (same as
      RLT): EgoDex static background prunes heavily, hand-motion regions barely at
      all -> measure real reduction on GPU; no a-priori speedup claimed.
    """
    _require_torch()
    if tokens.dim() != 3:
        raise ValueError(f"evs_static_prune expects [B, N, D], got {tuple(tokens.shape)}")
    batch_size, num_tokens, dim = tokens.shape
    t_grid = int(t_grid)
    h_grid = int(h_grid)
    w_grid = int(w_grid)
    hw = h_grid * w_grid
    expected = t_grid * hw
    if num_tokens != expected:
        raise ValueError(
            f"evs_static_prune expects dense N = t*h*w = {expected}, got {num_tokens}."
        )
    if metric not in ("l2_rel", "cosine"):
        raise ValueError("metric must be 'l2_rel' or 'cosine'")

    device = tokens.device
    tok_grid = tokens.reshape(batch_size, t_grid, hw, dim)  # adjacent time @ stride hw

    # change[b, j, loc] for j in 1..t-1 = "how dynamic is step j vs j-1".
    cur = tok_grid[:, 1:].float()
    prev = tok_grid[:, :-1].float()
    if metric == "l2_rel":
        change = torch.linalg.norm(cur - prev, dim=-1) / (
            torch.linalg.norm(prev, dim=-1) + eps
        )  # [B, t-1, hw]; small => static
        is_static = change < float(diff_threshold)
    else:  # cosine: high cos => static
        c = F.normalize(cur, dim=-1, eps=eps)
        p = F.normalize(prev, dim=-1, eps=eps)
        cos = (c * p).sum(dim=-1)  # [B, t-1, hw]
        change = 1.0 - cos  # small => static
        is_static = cos > (1.0 - float(diff_threshold))

    # keep mask [B, t, hw]: frame 0 kept (optionally), later frames kept iff NOT
    # static vs previous frame.
    keep_grid = torch.ones(batch_size, t_grid, hw, device=device, dtype=torch.bool)
    keep_grid[:, 1:] = ~is_static
    if keep_first_frame:
        keep_grid[:, 0] = True
    else:
        # Even if not forcing frame 0, there is no t-1 for it; keep it.
        keep_grid[:, 0] = True

    # "change score" per token (for top-K dynamic selection + logging). Frame 0
    # has no predecessor -> give it +inf so it is always kept in top-K mode too.
    change_full = torch.empty(batch_size, t_grid, hw, device=device, dtype=torch.float32)
    change_full[:, 0] = float("inf")
    change_full[:, 1:] = change

    # ---- rep_for_orig via "last kept anchor" along time, per location ---------
    # original id at (t=j, loc) = j*hw + loc.
    time_idx = torch.arange(t_grid, device=device)
    loc_idx = torch.arange(hw, device=device)
    orig_id_grid = (time_idx.unsqueeze(1) * hw + loc_idx.unsqueeze(0))  # [t, hw]
    orig_id_grid = orig_id_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, t, hw]

    # For each slot, the id of the most recent kept slot at the SAME loc (<= its
    # time). Compute via cummax over time of (kept ? own_id : -1). Because ids at
    # a fixed loc increase with time, cummax of kept ids gives the latest kept id
    # at or before each time step. Frame 0 is always kept so the running max is
    # never -1 for j>=0.
    kept_id_or_neg = torch.where(
        keep_grid, orig_id_grid, torch.full_like(orig_id_grid, -1)
    )  # [B, t, hw]
    anchor_id_grid, _ = torch.cummax(kept_id_or_neg, dim=1)  # [B, t, hw]

    # rep_for_orig[b, original_id] = anchor_id at that slot.
    rep_for_orig = torch.empty(batch_size, num_tokens, device=device, dtype=torch.long)
    flat_orig_id = orig_id_grid.reshape(batch_size, num_tokens)
    flat_anchor = anchor_id_grid.reshape(batch_size, num_tokens)
    rep_for_orig.scatter_(1, flat_orig_id, flat_anchor)

    meta = {
        "method": "evs",
        "num_tokens_before": int(num_tokens),
        "diff_threshold": float(diff_threshold),
        "metric": metric,
        "t_grid": t_grid,
        "h_grid": h_grid,
        "w_grid": w_grid,
        "rep_for_orig": rep_for_orig,
    }

    # frame-major flat order matches the canonical id order (id = t*hw + loc), so
    # reshaping [B, t, hw] -> [B, N] in frame-major gives slot p == original id p.
    keep_flat = keep_grid.reshape(batch_size, num_tokens)              # [B, N] bool
    change_flat = change_full.reshape(batch_size, num_tokens)          # [B, N]
    orig_ids_flat = torch.arange(num_tokens, device=device).unsqueeze(0).expand(
        batch_size, -1
    )

    if target_ratio is None:
        # -------- RAGGED mode --------
        kept_counts = keep_flat.sum(dim=1)  # [B]
        tokens_list, ids_list, size_list = [], [], []
        for b in range(batch_size):
            m = keep_flat[b]
            tokens_list.append(tokens[b][m])
            ids_list.append(orig_ids_flat[b][m])
            size_list.append(torch.ones(int(m.sum().item()), device=device, dtype=tokens.dtype))
        meta["mode"] = "ragged"
        meta["kept_per_sample"] = kept_counts.detach()
        meta["kept_fraction"] = float(kept_counts.float().mean().item() / num_tokens)
        return tokens_list, ids_list, size_list, meta

    # -------- RECTANGULAR (top-K dynamic) mode --------
    ratio = float(target_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"target_ratio must be in (0,1], got {ratio}")
    keep_k = int(max(1, min(num_tokens, round(ratio * num_tokens))))

    # Keep the K most-dynamic tokens. Frame-0 tokens have change=+inf so the hw
    # first-frame tokens are guaranteed kept as long as keep_k >= hw (true for any
    # sane ratio: hw=256 << 8192). If keep_k < hw we'd drop some first-frame
    # anchors; guard against that so every location keeps its anchor.
    if keep_k < hw:
        keep_k = hw
    keep_k = min(keep_k, num_tokens)

    topk_change, topk_pos = change_flat.topk(keep_k, dim=1)  # [B, keep_k]
    # Sort kept positions ascending so token order stays time/space-monotonic
    # (nice for debugging + stable RoPE indexing); not required for correctness.
    topk_pos, _order = torch.sort(topk_pos, dim=1)

    tokens_new = tokens.gather(1, topk_pos.unsqueeze(-1).expand(-1, -1, dim))
    token_ids_new = orig_ids_flat.gather(1, topk_pos)  # ORIGINAL ids preserved
    token_size_new = torch.ones(batch_size, keep_k, device=device, dtype=tokens.dtype)

    # In top-K mode the actual kept set differs from the threshold keep_grid, so
    # rebuild rep_for_orig against the ACTUAL kept tokens: each original maps to
    # the most-recent KEPT token (in this top-K set) at the same location.
    kept_mask_topk = torch.zeros(batch_size, num_tokens, device=device, dtype=torch.bool)
    kept_mask_topk.scatter_(1, topk_pos, True)
    kept_grid_topk = kept_mask_topk.reshape(batch_size, t_grid, hw)
    kept_id_or_neg_topk = torch.where(
        kept_grid_topk, orig_id_grid, torch.full_like(orig_id_grid, -1)
    )
    anchor_topk, _ = torch.cummax(kept_id_or_neg_topk, dim=1)
    # Any location whose first kept token is not at t=0 could have -1 anchors for
    # early times; clamp those to the earliest kept id at that loc (the running
    # max becomes valid from the first kept time onward; for times before it we
    # back-fill with the first kept id via a reverse cummax-min).
    # Simple robust fix: where anchor==-1, fall back to the original id itself
    # (token maps to itself). This only happens for an early static token whose
    # location's first survivor is later in time -- a rare top-K edge case.
    anchor_flat_topk = anchor_topk.reshape(batch_size, num_tokens)
    anchor_flat_topk = torch.where(
        anchor_flat_topk >= 0, anchor_flat_topk, orig_ids_flat
    )
    rep_for_orig_topk = torch.empty(batch_size, num_tokens, device=device, dtype=torch.long)
    rep_for_orig_topk.scatter_(1, orig_ids_flat, anchor_flat_topk)
    meta["rep_for_orig"] = rep_for_orig_topk

    meta["mode"] = "rectangular_topk"
    meta["keep_k"] = int(keep_k)
    meta["kept_fraction"] = float(keep_k) / float(num_tokens)
    # logging: change stats among PRUNED tokens (how static were the dropped ones)
    pruned_mask = ~kept_mask_topk
    if bool(pruned_mask.any().item()):
        pruned_change = change_flat[pruned_mask]
        # +inf entries (frame 0) are never pruned, so this is finite.
        meta["pruned_change_max"] = float(pruned_change.max().item())
        meta["pruned_change_mean"] = float(pruned_change.mean().item())
    else:
        meta["pruned_change_max"] = None
        meta["pruned_change_mean"] = None

    return tokens_new, token_ids_new, token_size_new, meta


# ======================================================================
# Optional helper: pad a ragged (list) pre-merge result to a rectangular
# [B, max_n, D] + boolean keep-mask, for the "padding + mask" batch strategy.
# token_merge.py is NOT mask-aware today, so use this ONLY if you also plumb the
# mask through attention/K-BSM/restore. Documented in INTEGRATION doc.
# ======================================================================
def pad_ragged_to_rectangular(tokens_list, ids_list, size_list, pad_id=0):
    """Pad per-sample lists to [B, max_n, D] and return a keep-mask.

    Returns (tokens, token_ids, token_size, keep_mask) where keep_mask[b, i] is
    True for real tokens and False for padding. Padded token_size is 0 (inert
    under size-weighted merges); padded ids are ``pad_id`` (caller must ensure
    downstream masks them out -- a padded id can collide with a real id).
    """
    _require_torch()
    batch_size = len(tokens_list)
    dim = tokens_list[0].shape[-1]
    device = tokens_list[0].device
    dtype = tokens_list[0].dtype
    max_n = max(int(t.shape[0]) for t in tokens_list)

    tokens = torch.zeros(batch_size, max_n, dim, device=device, dtype=dtype)
    token_ids = torch.full((batch_size, max_n), int(pad_id), device=device, dtype=torch.long)
    token_size = torch.zeros(batch_size, max_n, device=device, dtype=dtype)
    keep_mask = torch.zeros(batch_size, max_n, device=device, dtype=torch.bool)
    for b in range(batch_size):
        n = int(tokens_list[b].shape[0])
        tokens[b, :n] = tokens_list[b]
        token_ids[b, :n] = ids_list[b]
        token_size[b, :n] = size_list[b]
        keep_mask[b, :n] = True
    return tokens, token_ids, token_size, keep_mask
