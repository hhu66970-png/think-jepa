"""Diagnostic / research-only token merger: home of the Gradual K-BSM strategy.

``DiagnosticTokenMerger`` subclasses ``LocalTokenMerger`` and adds exactly one
extra strategy: ``bsm_ksim_gradual_vec`` (global Bipartite Soft Matching on
post-RoPE attention-Key cosine, gradual multi-layer, size-weighted; the verified
encoder-speedup winner). A/B/C behaviour is inherited unchanged from the base.

Kept out of the main path: the encoder (``vision_transformer.py``) /
``scripts/train.sh`` never instantiate this subclass unless
``bsm_ksim_gradual_vec`` is explicitly requested.

History: the No-Go strategies B2 (``local_keep_then_merge_vec``) and
C2 (``local_2x2_similarity_gated_importance_vec``), plus the dead-end research
knobs (temporal partition, RLT pre-merge, norm protection), were removed
2026-05-31 after experiments confirmed they offered no Pareto improvement.
"""

import math
import os as _os_   # WAM_SUPPLEMENT_PATCH_20260729

import torch
import torch.nn.functional as F

from src.models.utils.token_merge import LocalTokenMerger, compute_importance

# Per-batch external relevance [B, t*h*w] set by the caller right before a forward pass
# (relevance_source="external"), e.g. the projection of the OBSERVED past hand poses.
EXTERNAL_RELEVANCE = None


def set_external_relevance(rel):
    global EXTERNAL_RELEVANCE
    EXTERNAL_RELEVANCE = rel


# Learned matching embedding (bsm_match_metric="learned", direction B): a module mapping the
# current hidden tokens [B, N, D] to matching features [B, N, d]; replaces the Key cosine.
MATCHER = None


def set_matcher(module):
    global MATCHER
    MATCHER = module


# Learned relevance scorer (relevance_source="scorer"): a module mapping the dense hidden
# tokens at the first merge layer [B, N, D] to per-token scores [B, N].
SCORER = None


def set_scorer(module):
    global SCORER
    SCORER = module


class DiagnosticTokenMerger(LocalTokenMerger):
    """``LocalTokenMerger`` extended with the Gradual K-BSM strategy.

    A/B/C behaviour is inherited unchanged from the base class; this subclass
    adds only ``bsm_ksim_gradual_vec`` (the verified encoder-speedup winner).
    """

    VECTORIZED_STRATEGIES = LocalTokenMerger.VECTORIZED_STRATEGIES + (
        "bsm_ksim_gradual_vec",  # Gradual K-BSM (grid-agnostic, multi-layer)
        "bsm_pitome_gradual_vec",  # NEW: PiToMe energy-score partition
        "bsm_taware_gradual_vec",  # WAM: K-BSM + task-relevance (motion) gate
    )
    NO_PYTHON_FALLBACK_STRATEGIES = LocalTokenMerger.NO_PYTHON_FALLBACK_STRATEGIES + (
        "bsm_ksim_gradual_vec",  # must never hit the 2x2 Python fallback path
        "bsm_pitome_gradual_vec",  # NEW: never hit the 2x2 Python fallback
        "bsm_taware_gradual_vec",  # WAM: never hit the 2x2 Python fallback
    )

    def _method_name(self):
        if self.config.strategy == "bsm_ksim_gradual_vec":
            return "BSM_ksim_gradual"
        if self.config.strategy == "bsm_pitome_gradual_vec":
            return "BSM_pitome_energy"
        if self.config.strategy == "bsm_taware_gradual_vec":
            return "BSM_taware_motion"
        return super()._method_name()

    # Gradual K-BSM: global bipartite soft matching (training-free, SDPA-safe).
    # Grid-agnostic; runs on compressed token sets across MULTIPLE layers. NEVER
    # routes through the dense 2x2 _forward_vectorized path (no _reshape_2x2_cells,
    # no compute_importance, no flat_cell_positions), so compressed inputs at
    # layers > L_start are handled natively. Reached only from
    # LocalTokenMerger.forward's bsm dispatch (subclass-only strategy).
    # ------------------------------------------------------------------
    @staticmethod
    def _capacity_select(scores, r, size_a, size_b, cap_count, cap_abs, max_rounds=16):
        """Capacity-constrained K-BSM edge selection (balanced group sizes).

        Same similarity rule as K-BSM: every source proposes to its most similar receiver and
        edges are accepted greedily in descending similarity. An edge is rejected when its
        receiver would exceed cap_count sources in this layer or cap_abs total size; a rejected
        source re-proposes to its best NON-full receiver in the next round. Exactly r edges
        per sample are returned; if the caps cannot be met (rare), the remaining slots are
        filled with the best unconstrained edges and counted in cap_forced.
        With no binding cap this selects the same edge set as top-r (up to exact score ties).
        """
        B, Na, Nb = scores.shape
        dev = scores.device
        ninf = float("-inf")
        chosen = torch.zeros(B, Na, dtype=torch.bool, device=dev)
        dest = torch.zeros(B, Na, dtype=torch.long, device=dev)
        load = torch.zeros(B, Nb, device=dev)
        mass = size_b.clone()
        need = torch.full((B,), int(r), dtype=torch.long, device=dev)
        pos = torch.arange(Na, device=dev).expand(B, -1)
        rounds = 0
        for rounds in range(1, max_rounds + 1):
            s = scores.float().masked_fill(chosen.unsqueeze(-1), ninf)
            if cap_count > 0:
                s = s.masked_fill((load >= cap_count).unsqueeze(1), ninf)
            if cap_abs > 0.0:
                s = s.masked_fill(mass.unsqueeze(1) + size_a.unsqueeze(-1) > cap_abs, ninf)
            bs, bb = s.max(dim=2)                                           # [B, Na]
            order = bs.argsort(dim=1, descending=True, stable=True)        # score order
            valid = torch.isfinite(bs.gather(1, order))
            recv = torch.where(valid, bb.gather(1, order), torch.full_like(order, Nb))
            w = size_a.gather(1, order) * valid
            # per-receiver running count / size in score order (segmented cumsum)
            perm = recv.argsort(dim=1, stable=True)
            key = recv.gather(1, perm)
            one = valid.gather(1, perm).float()
            ww = w.gather(1, perm)
            c1, cw = one.cumsum(1), ww.cumsum(1)
            start = torch.ones_like(key, dtype=torch.bool)
            start[:, 1:] = key[:, 1:] != key[:, :-1]
            sidx = torch.where(start, pos, torch.zeros_like(pos)).cummax(1).values
            occ = c1 - (c1 - one).gather(1, sidx)
            wsum = cw - (cw - ww).gather(1, sidx)
            kc = key.clamp(max=Nb - 1)
            ok_p = one > 0
            if cap_count > 0:
                ok_p &= load.gather(1, kc) + occ <= cap_count
            if cap_abs > 0.0:
                ok_p &= mass.gather(1, kc) + wsum <= cap_abs
            ok = torch.zeros_like(ok_p).scatter_(1, perm, ok_p)           # back to score order
            take_o = ok & (ok.long().cumsum(1) <= need.unsqueeze(1))
            take = torch.zeros_like(take_o).scatter_(1, order, take_o)    # A-local
            chosen |= take
            dest = torch.where(take, bb, dest)
            tb = bb.masked_fill(~take, 0)
            load.scatter_add_(1, tb, take.float())
            mass.scatter_add_(1, tb, size_a * take)
            need = need - take.sum(1)
            if int(need.max()) == 0:
                break
        forced = int(need.sum())
        if forced > 0:        # caps infeasible: fill with the best remaining unconstrained edges
            s = scores.float().masked_fill(chosen.unsqueeze(-1), ninf)
            bs, bb = s.max(dim=2)
            order = bs.argsort(dim=1, descending=True, stable=True)
            take_o = pos < need.unsqueeze(1)
            take = torch.zeros_like(take_o).scatter_(1, order, take_o)
            chosen |= take
            dest = torch.where(take, bb, dest)
        # deterministic ascending source order
        edge_a = torch.where(chosen, Na - pos, torch.zeros_like(pos)).topk(int(r), dim=1).values
        edge_a = Na - edge_a
        edge_b = dest.gather(1, edge_a)
        return edge_a, edge_b, dict(cap_rounds=rounds, cap_forced=forced)

    @torch.no_grad()
    def _forward_bsm(self, x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid, attn_key):
        batch_size, num_tokens, dim = x.shape

        # r = tokens to remove THIS layer. Same per-layer ratio cap (<=25%) as the
        # 2x2 path; also cap at floor((N-1)/2) so even the alternating A/B split
        # always has a partner for every kept A-token and >=1 token survives.
        ratio = max(0.0, min(self.config.merge_ratio, 0.25))
        r = int(math.floor(num_tokens * ratio))
        r = min(r, (num_tokens - 1) // 2)
        if r <= 0 or num_tokens < 2:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )

        # -- Matching metric (UNCHANGED from K-BSM) --------------------------
        want_key = str(getattr(self.config, "bsm_match_metric", "key")) in ("key", "key_prerope")
        if str(getattr(self.config, "bsm_match_metric", "key")) == "learned":
            if MATCHER is None:
                raise RuntimeError("bsm_match_metric='learned' but no matcher was set")
            # per-layer matchers (B2): a list indexed by the merge call within this forward;
            # the first call of a forward is recognised by the still-dense token set.
            if num_tokens == rep_for_orig.shape[1]:
                self._merge_call = 0
            else:
                self._merge_call = getattr(self, "_merge_call", 0) + 1
            mod = MATCHER
            if isinstance(MATCHER, (list, tuple)):
                mod = MATCHER[min(self._merge_call, len(MATCHER) - 1)]
            with torch.autocast("cuda", enabled=False):
                metric_src = mod(x.float()).float()
            match_metric = "learned"
            fallback_reason = None
        elif (
            want_key
            and attn_key is not None
            and attn_key.dim() == 3
            and attn_key.shape[0] == batch_size
            and attn_key.shape[1] == num_tokens
        ):
            metric_src = attn_key.float()
            match_metric = str(getattr(self.config, "bsm_match_metric", "key"))
            fallback_reason = None
        else:
            metric_src = x.float()
            match_metric = "feature_fallback"
            if not want_key:
                fallback_reason = "feature_metric_requested"
            elif attn_key is None:
                fallback_reason = "attn_key_unavailable"
            else:
                fallback_reason = "attn_key_shape_mismatch"
        # Neutralize any non-finite metric row at the source so neither the
        # cosine score bmm nor the PiToMe energy bmm can produce nan/inf; a
        # genuine zero row then normalizes to a finite ~0 vector (eps guard).
        # No-op for the finite metrics K-BSM always sees -> K-BSM unchanged.
        metric_src = torch.nan_to_num(metric_src, nan=0.0, posinf=0.0, neginf=0.0)
        metric = F.normalize(metric_src, dim=-1, eps=1e-6)  # [B, N, d]

        # -- Bipartite partition over the CURRENT token order ----------------
        # K-BSM: ToMe even/odd. PiToMe: energy-score split. Both yield
        # 1-D LongTensors a_idx [Na] / b_idx [Nb] of ABSOLUTE positions in the
        # current [0, num_tokens) order, disjoint and covering all tokens. We
        # also surface a per-A "energy" tensor for info/debug (None for K-BSM).
        use_pitome = (self.config.strategy == "bsm_pitome_gradual_vec")
        partition_mode = str(getattr(self.config, "bsm_partition", "positional"))
        taware = self.config.strategy == "bsm_taware_gradual_vec"
        part_mode = str(getattr(self.config, "relevance_partition", "none")) if taware else "none"
        rel_part = part_mode in ("recv", "excl")
        rel_anchor = float(getattr(self.config, "relevance_anchor", 0.0)) if taware else 0.0
        lam_cfg = float(getattr(self.config, "relevance_lambda", 1.0)) if taware else 0.0
        rel_cur = None
        if taware and (lam_cfg > 0.0 or rel_part or rel_anchor > 0.0):
            rel_cur = self._wam_relevance(x, token_ids, rep_for_orig,
                                          int(t_grid), int(h_grid), int(w_grid))  # [B,N] in [0,1]
        a_valid_pad = None   # [B, Na] validity of padded per-sample A rows (recv partition)
        b_valid_pad = None   # [B, Nb]
        if rel_part:
            # Even/odd split, except that relevant tokens (rel > 0.5) are moved from A to B:
            # they can still absorb similar neighbours but are never removed. Per-sample
            # counts differ, so indices are padded and padded rows/columns are masked.
            # "excl" removes relevant tokens from both sides: they pass through untouched
            # and the remaining tokens are matched among themselves (masked re-matching).
            pos = torch.arange(num_tokens, device=x.device)
            # protect at most relevance_quota of the CURRENT tokens (highest relevance first,
            # only tokens with relevance > 0), so the scheduled budget stays reachable.
            n_prot = int(math.floor(float(self.config.relevance_quota) * num_tokens))
            prot = torch.zeros_like(rel_cur, dtype=torch.bool)
            if n_prot > 0:
                prot.scatter_(1, rel_cur.topk(n_prot, dim=1).indices, True)
                prot &= rel_cur > 0
            even = (pos.remainder(2) == 0).unsqueeze(0)
            in_a = even & ~prot                                                  # [B, N]
            in_b = ~in_a if part_mode == "recv" else (~even & ~prot)
            cnt_a, cnt_b = in_a.sum(1), in_b.sum(1)
            order_a = torch.argsort((~in_a).to(torch.int8), dim=1, stable=True)  # A first, by position
            order_b = torch.argsort((~in_b).to(torch.int8), dim=1, stable=True)  # B first, by position
            na_max = int(cnt_a.max().item())
            nb_max = max(1, int(cnt_b.max().item()))
            a_idx = order_a[:, :na_max].contiguous()
            b_idx = order_b[:, :nb_max].contiguous()
            ar = torch.arange(na_max, device=x.device).unsqueeze(0)
            br = torch.arange(nb_max, device=x.device).unsqueeze(0)
            a_valid_pad = ar < cnt_a.unsqueeze(1)
            b_valid_pad = br < cnt_b.unsqueeze(1)
            energy_a = None
            partition_name = f"relevance_{part_mode}"
        elif use_pitome:
            a_idx, b_idx, energy_a = self._bsm_energy_partition(metric, num_tokens)
            partition_name = "energy"
        else:
            # positional even/odd (default K-BSM). "temporal" partition from the
            # harness is treated as positional here — use merge_axis="temporal"
            # for cross-frame-only merging instead (cleaner + grid-correct).
            pos = torch.arange(num_tokens, device=x.device)
            a_idx = pos[0::2]
            b_idx = pos[1::2]
            energy_a = None
            partition_name = "positional" if partition_mode != "temporal" else "positional_from_temporal_req"

        na = a_idx.shape[-1]
        nb = b_idx.shape[-1]
        # r cannot exceed #A-tokens (each A contributes <=1 edge).
        r = min(r, na)
        if r <= 0 or nb == 0:
            return x, token_ids, token_size, rep_for_orig, self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )

        # Gather A/B metric rows. a_idx/b_idx are 1-D [Na]/[Nb] for K-BSM
        # (even/odd, shared across the batch) but per-sample [B, Na]/[B, Nb] for
        # PiToMe (energy ranking differs per sample). Handle both shapes inline so
        # this method is correct as-written for either partition.
        # Bugfix: gather along the METRIC's own feature width (post-RoPE key
        # head_dim, e.g. 64) -- NOT x's hidden dim (1024). They are equal only on
        # the feature_fallback path (metric == x); with the key metric they differ,
        # so expanding to `dim` mis-sized the gather index and raised
        # "Size does not match at dimension 2". K-BSM took the index_select branch
        # and never hit this, which is why it surfaced only once PiToMe used key.
        d_metric = metric.shape[-1]
        if a_idx.dim() == 2:
            a_metric = metric.gather(1, a_idx.unsqueeze(-1).expand(-1, -1, d_metric))  # [B, Na, d]
            b_metric = metric.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, d_metric))  # [B, Nb, d]
        else:
            a_metric = metric.index_select(1, a_idx)   # [B, Na, d]
            b_metric = metric.index_select(1, b_idx)   # [B, Nb, d]
        scores = torch.bmm(a_metric, b_metric.transpose(1, 2))   # [B, Na, Nb] cosine
        if a_valid_pad is not None:
            scores = scores.masked_fill(~b_valid_pad.unsqueeze(1), float("-inf"))
            scores = scores.masked_fill(~a_valid_pad.unsqueeze(2), float("-inf"))

        # -- Axis constraint mask (NEW) --------------------------------------
        # Applied AFTER scores, BEFORE the per-A argmax. Frames come from the
        # ORIGINAL token id (id // (h_grid*w_grid)), verified above.
        axis = str(getattr(self.config, "merge_axis", "free"))
        a_valid_row = None  # [B, Na] bool; an A-row with no allowed B becomes False
        if axis in ("spatial", "temporal"):
            scores, a_valid_row = self._bsm_axis_mask(
                scores, token_ids, a_idx, b_idx, int(h_grid), int(w_grid), axis
            )

        # each A-token -> most similar ALLOWED B-token
        best_sim, best_b_local = scores.max(dim=2)   # [B, Na], [B, Na]

        # If an A-row was fully masked (-inf) by the axis constraint, it has no
        # valid edge: its best_sim is -inf. We must keep it OUT of the top-r so
        # we never "select" a -inf edge. Because best_sim==-inf sorts to the
        # bottom, topk will only pick a masked row if there are fewer than r rows
        # with a finite edge. So we cap r at the per-sample count of valid rows
        # (min over the batch keeps r a single scalar -> rectangular [B,N-r,D]).
        if a_valid_pad is not None:
            a_valid_row = a_valid_pad if a_valid_row is None else (a_valid_row & a_valid_pad)
        if a_valid_row is not None:
            valid_rows_per_sample = a_valid_row.sum(dim=1)            # [B]
            r_eff = int(min(int(r), int(valid_rows_per_sample.min().item())))
        else:
            r_eff = int(r)
        if r_eff <= 0:
            info = self._info(
                x, x, 0, 0, selected_scores=None, implementation="vectorized_bsm",
                num_accepted=0,
            )
            info["bsm_partition"] = partition_name
            info["merge_axis"] = axis
            if fallback_reason is not None:
                info["fallback_reason"] = fallback_reason
            return x, token_ids, token_size, rep_for_orig, info
        r = r_eff

        # -- WAM task-relevance gate (NEW; bsm_taware_gradual_vec only) -------
        # Down-weight the merge PRIORITY of any edge touching a high-relevance
        # (e.g. high-motion / task-relevant) token, so such tokens are dropped
        # LAST. Partner CHOICE (best_b_local) is UNCHANGED — only which edges the
        # top-r selects changes. relevance_lambda=0 / strategy!=taware => identical
        # to K-BSM (rank_score == best_sim).
        rank_score = best_sim
        rel_source_used = None
        if self.config.strategy == "bsm_taware_gradual_vec":
            lam = lam_cfg
            if rel_cur is not None:
                rel_source_used = str(getattr(self.config, "relevance_source", "motion") or "motion")
            if lam > 0.0:
                rel = rel_cur
                rel_source_used = str(getattr(self.config, "relevance_source", "motion") or "motion")
                pw = float(getattr(self.config, "relevance_power", 1.0))
                gate_form = str(getattr(self.config, "relevance_gate", "mult"))

                def _ab(t):   # per-current-token tensor -> (A rows, chosen B of each A row)
                    if a_idx.dim() == 2:
                        ta, tb = t.gather(1, a_idx), t.gather(1, b_idx)
                    else:
                        ta, tb = t.index_select(1, a_idx), t.index_select(1, b_idx)
                    return ta, tb.gather(1, best_b_local)

                if gate_form == "quota":
                    # hard protection: the top-q fraction of CURRENT tokens by relevance
                    # (ties broken by position) can neither be merged away nor absorb.
                    n_prot = int(math.floor(float(self.config.relevance_quota) * num_tokens))
                    prot = torch.zeros_like(rel, dtype=torch.bool)
                    if n_prot > 0:
                        prot.scatter_(1, rel.topk(n_prot, dim=1).indices, True)
                    pa, pb = _ab(prot)
                    blocked = pa | pb
                    rank_score = best_sim.masked_fill(blocked, float("-inf"))
                    r = int(min(int(r), int((~blocked & torch.isfinite(best_sim)).sum(1).min().item())))
                    if r <= 0:
                        info = self._info(x, x, 0, 0, selected_scores=None,
                                          implementation="vectorized_bsm", num_accepted=0)
                        info["relevance_gate"] = gate_form
                        return x, token_ids, token_size, rep_for_orig, info
                elif gate_form == "add":
                    ra, rb = _ab(rel)
                    beta = float(getattr(self.config, "relevance_beta", 1.0))
                    rank_score = torch.where(torch.isfinite(best_sim),
                                             best_sim - beta * lam * (ra + rb), best_sim)
                else:
                    gate = (1.0 - lam * rel).clamp(0.0, 1.0).pow(pw)    # [B, N]
                    gate_a, gate_b_chosen = _ab(gate)
                    sim = (1.0 + best_sim) * 0.5 if gate_form == "signsafe" else best_sim
                    if gate_form == "source":
                        gated = sim * gate_a
                    else:
                        gated = sim * gate_a * gate_b_chosen
                    # keep axis-masked (-inf) rows at -inf (avoid -inf*0=nan promotion)
                    rank_score = torch.where(torch.isfinite(best_sim), gated, best_sim)

        # keep the r strongest A->B edges per sample (top-r by rank_score). r is a
        # single scalar across the batch -> exactly r sources dropped per sample
        # -> x_new stays a dense [B, N-r, D] (the invariant A/B/C rely on). Every
        # selected edge is finite because r <= #valid rows.
        cap_count = int(getattr(self.config, "bsm_cap_count", 0))
        cap_size = float(getattr(self.config, "bsm_cap_size", 0.0))
        cap_info = None
        if cap_count > 0 or cap_size > 0.0:
            if rank_score is not best_sim:
                raise NotImplementedError("bsm_cap_* is defined for the plain K-BSM order only")
            if a_idx.dim() == 2:
                size_a, size_b = token_size.gather(1, a_idx), token_size.gather(1, b_idx)
            else:
                size_a, size_b = token_size.index_select(1, a_idx), token_size.index_select(1, b_idx)
            cap_abs = cap_size * rep_for_orig.shape[1] / float(num_tokens - r) if cap_size > 0.0 else 0.0
            edge_a_local, edge_b_local, cap_info = self._capacity_select(
                scores, r, size_a.float(), size_b.float(), cap_count, cap_abs)
            edge_sim = scores.gather(2, edge_b_local.unsqueeze(-1)).squeeze(-1)
        else:
            _, edge_a_local = rank_score.topk(r, dim=1)             # [B, r]
            edge_b_local = best_b_local.gather(1, edge_a_local)     # [B, r]
            edge_sim = best_sim.gather(1, edge_a_local)             # [B, r] ACTUAL cosine

        # map A/B local indices back to ABSOLUTE token positions (per-sample or
        # shared, mirroring the a_idx/b_idx shape handled above).
        if a_idx.dim() == 2:
            source_pos = a_idx.gather(1, edge_a_local)              # [B, r]
            receiver_pos = b_idx.gather(1, edge_b_local)            # [B, r]
        else:
            source_pos = a_idx.unsqueeze(0).expand(batch_size, -1).gather(1, edge_a_local)   # [B, r]
            receiver_pos = b_idx.unsqueeze(0).expand(batch_size, -1).gather(1, edge_b_local) # [B, r]

        # NOTE on collisions (UNCHANGED): two A-tokens may pick the same B-token.
        # We scatter-add BOTH into that B (size-weighting stays exact). Sources are
        # unique (A disjoint, never overlaps B), so "drop sources, keep rest" stays
        # rectangular with exactly r removed per sample regardless of collisions.

        # -- Size-weighted merge (NUMERICALLY HARDENED) ----------------------
        # Bugfix (PiToMe NaN): the energy partition can funnel many redundant
        # sources into a FEW low-energy receivers, so a receiver token_size can
        # reach ~1e3 (vs ~4e2 for K-BSM even/odd). Under fp16 AMP (downstream
        # training) the weighted sum x*size then exceeds the fp16 max (65504) and
        # overflows to +/-inf -> inf/inf = nan, which poisons the encoder output
        # and the downstream loss (epoch1 non-finite). We therefore accumulate the
        # weighted average in FP32 and only cast back to x.dtype at the end. This
        # is a NO-OP for the fp32 encoder harness (x already fp32) -> K-BSM encoder
        # numbers are byte-identical; it only removes the fp16 overflow under AMP.
        x_f = x.float()
        size_f = token_size.float()
        source_ids = token_ids.gather(1, source_pos)            # [B, r]
        receiver_ids = token_ids.gather(1, receiver_pos)        # [B, r]
        source_weight = size_f.gather(1, source_pos)            # [B, r] fp32
        source_x = x_f.gather(1, source_pos.unsqueeze(-1).expand(-1, -1, dim))  # [B, r, D] fp32
        if rel_anchor > 0.0:
            # anchored averaging: relevant members dominate the merged feature;
            # token sizes (used by proportional attention) still count original tokens.
            feat_w = size_f * (1.0 + rel_anchor * rel_cur.float())
        else:
            feat_w = size_f
        src_feat_w = feat_w.gather(1, source_pos)
        weighted_src = source_x * src_feat_w.unsqueeze(-1)      # [B, r, D] fp32

        acc_x = torch.zeros_like(x_f)
        acc_w = torch.zeros_like(size_f)
        acc_fw = torch.zeros_like(size_f)
        acc_x.scatter_add_(1, receiver_pos.unsqueeze(-1).expand(-1, -1, dim), weighted_src)
        acc_w.scatter_add_(1, receiver_pos, source_weight)
        acc_fw.scatter_add_(1, receiver_pos, src_feat_w)
        recv_mask = torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        recv_mask.scatter_(1, receiver_pos, True)
        old_w = size_f                                           # [B, N] fp32
        new_w = old_w + acc_w
        new_fw = feat_w + acc_fw
        merged_x = (x_f * feat_w.unsqueeze(-1) + acc_x) / new_fw.clamp_min(1e-6).unsqueeze(-1)
        # guard against any residual non-finite (e.g. a non-finite incoming x row)
        merged_x = torch.nan_to_num(merged_x, nan=0.0, posinf=0.0, neginf=0.0)
        x_updated = torch.where(recv_mask.unsqueeze(-1), merged_x.to(x.dtype), x)
        token_size_updated = torch.where(recv_mask, new_w.to(token_size.dtype), token_size)

        # -- Drop sources, keep rest (rectangular, UNCHANGED) ----------------
        keep = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        keep.scatter_(1, source_pos, False)
        num_after = num_tokens - r
        x_new = x_updated[keep].reshape(batch_size, num_after, dim)
        ids_new = token_ids[keep].reshape(batch_size, num_after)
        size_new = token_size_updated[keep].reshape(batch_size, num_after)

        # -- Multi-layer rep_for_orig carry-forward (UNCHANGED) --------------
        num_original_tokens = rep_for_orig.shape[1]
        remap = (
            torch.arange(num_original_tokens, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        remap.scatter_(1, source_ids, receiver_ids)
        rep_new = remap.gather(1, rep_for_orig)

        info = self._info(
            x, x_new, r, r,
            selected_scores=edge_sim,
            implementation="vectorized_bsm",
            num_candidates=int(batch_size * na),
            num_candidate_cells=int(batch_size * na),
            num_accepted=r,
        )
        info["neg_sim_selected_frac"] = float((edge_sim < 0).float().mean().item())
        if cap_info is not None:
            info.update(cap_info)
        if rel_source_used is not None:
            info["relevance_gate"] = str(getattr(self.config, "relevance_gate", "mult"))
        info["bsm_match_metric"] = match_metric
        info["matching_metric"] = match_metric
        info["bsm_partition"] = partition_name
        info["merge_axis"] = axis
        if rel_source_used is not None:
            info["relevance_source"] = rel_source_used
            info["relevance_lambda"] = float(getattr(self.config, "relevance_lambda", 1.0))
        if energy_a is not None:
            e = energy_a.detach().float()
            info["pitome_energy_mean"] = float(e.mean().item())
            info["pitome_energy_min"] = float(e.min().item())
            info["pitome_energy_max"] = float(e.max().item())
            info["pitome_margin"] = float(getattr(self.config, "pitome_margin", 0.0))
        if fallback_reason is not None:
            info["fallback_reason"] = fallback_reason
        return x_new, ids_new, size_new, rep_new, info

    # ---------------------------------------------------------------------
    # NEW HELPER 1 — PiToMe energy partition.
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _bsm_energy_partition(self, metric, num_tokens):
        """Split tokens into A (high-energy sources) / B (low-energy receivers).

        metric: [B, N, d] L2-normalized matching vectors (unit rows).

        ENERGY (PiToMe, arXiv:2405.16148):
            energy_i = mean_j  relu( cos(i, j) - m )          (i != j)
        High energy <=> similar to many tokens <=> redundant => prefer SOURCE.
        Low energy  <=> isolated/unique             => protect  => RECEIVER pool.

        We rank tokens by energy and take the top-Na as A (sources pool) and the
        rest as B (dst pool). To keep K-BSM's invariants we use the SAME split
        sizes as the even/odd partition: Na = ceil(N/2), Nb = floor(N/2). This
        guarantees na >= nb >= 1 and r <= na exactly as before, and the existing
        r-cap floor((N-1)/2) <= na holds, so the rectangular [B, N-r, D] invariant
        is preserved.

        Energy is a per-sample [B, N] score; the A/B membership can therefore
        differ across the batch. We DON'T need a shared partition across the
        batch — only a shared scalar r — because the merge gathers by per-sample
        positions and drops exactly r sources per sample. (a_idx/b_idx below are
        per-sample [B, Na]/[B, Nb]; the caller's index_select/gather all accept
        per-sample indices via gather, see note.)

        COMPLEXITY: the full energy needs the N x N cosine gram = O(N^2 d) and
        O(N^2) memory. For V-JEPA2 ViT-L at 256px/64f, N starts at
        t_grid*h_grid*w_grid = 32*16*16 = 8192 and SHRINKS each merge layer, so
        N^2 is ~6.7e7 entries at the first merge layer — feasible but heavy. We
        therefore subsample anchor columns when N > pitome_energy_max_anchors:
        pick `A` uniformly-spaced anchor tokens and approximate
            energy_i ~= mean over anchors relu(cos(i, anchor) - m)
        which is O(N * A) time / O(N * A) memory and an unbiased estimate of the
        full mean (uniform anchors). Set pitome_energy_max_anchors<=0 to force
        full O(N^2). Default 2048 keeps the gram <= N*2048.

        Returns:
            a_idx_pos: [B, Na] LongTensor of ABSOLUTE positions (sources pool)
            b_idx_pos: [B, Nb] LongTensor of ABSOLUTE positions (dst pool)
            energy_a:  [B, Na] energy of the A tokens (for info/debug)
        NOTE — these are PER-SAMPLE [B, Na]/[B, Nb] indices (energy ranking
        differs per sample). The spliced `_forward_bsm` already dispatches on
        `a_idx.dim()`: for the 2-D (PiToMe) case it uses batched `gather` for
        a_metric/b_metric and for source_pos/receiver_pos; for the 1-D (K-BSM
        even/odd) case it uses `index_select`/expand. Nothing else differs.
        """
        batch_size, n, d = metric.shape
        margin = float(getattr(self.config, "pitome_margin", 0.0))
        max_anchors = int(getattr(self.config, "pitome_energy_max_anchors", 2048))

        if max_anchors > 0 and n > max_anchors:
            # uniformly-spaced anchor columns -> [B, A, d]
            anchor_idx = torch.linspace(
                0, n - 1, steps=max_anchors, device=metric.device
            ).round().long()
            anchor = metric.index_select(1, anchor_idx)               # [B, A, d]
            sim = torch.bmm(metric, anchor.transpose(1, 2))           # [B, N, A] cosine
            contrib = torch.relu(sim - margin)
            # subtract a token's self-contribution when it is itself an anchor so
            # the diagonal (cos=1) doesn't bias energy; tokens not in anchors are
            # unaffected. self_in_anchor[b,i] = relu(1 - m) if i is an anchor.
            self_mask = torch.zeros(batch_size, n, device=metric.device, dtype=metric.dtype)
            self_mask.index_fill_(1, anchor_idx, float(max(0.0, 1.0 - margin)))
            energy = (contrib.sum(dim=2) - self_mask) / float(max(1, max_anchors - 1))
        else:
            sim = torch.bmm(metric, metric.transpose(1, 2))           # [B, N, N] cosine
            contrib = torch.relu(sim - margin)
            # zero the diagonal (self cos=1) so it never inflates energy.
            diag = torch.arange(n, device=metric.device)
            contrib[:, diag, diag] = 0.0
            energy = contrib.sum(dim=2) / float(max(1, n - 1))        # [B, N]

        # SAME split sizes as even/odd so all downstream r-caps still hold.
        na = (n + 1) // 2
        # Sanitize before ranking: a non-finite incoming key/feature row would make
        # cos -> nan -> energy nan and torch.argsort with nan is UNDEFINED (garbage
        # permutation, could even drop/duplicate positions). Map non-finite energy
        # to a very low finite value so such tokens sort to the receiver (B) pool
        # and the partition stays a valid permutation. Finite energies are
        # untouched, so K-BSM is unaffected and normal PiToMe runs identically.
        energy = torch.nan_to_num(energy, nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4)
        # rank by DESCENDING energy: top-na = high energy = sources pool (A);
        # the remainder = low energy = receivers pool (B). stable sort keeps the
        # order deterministic for ties.
        order = torch.argsort(energy, dim=1, descending=True, stable=True)  # [B, N]
        a_idx_pos = order[:, :na].contiguous()                              # [B, Na]
        b_idx_pos = order[:, na:].contiguous()                              # [B, Nb]
        energy_a = energy.gather(1, a_idx_pos)                              # [B, Na]
        return a_idx_pos, b_idx_pos, energy_a

    # ---------------------------------------------------------------------
    # NEW HELPER 2 — axis-constraint mask.
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _bsm_axis_mask(self, scores, token_ids, a_idx, b_idx, h_grid, w_grid, axis):
        """Mask the [B, Na, Nb] cosine matrix by frame relationship.

        Frame of a token = ORIGINAL id // (h_grid*w_grid)  (VERIFIED mapping).
        token_ids: [B, N] original ids of the CURRENT tokens.
        a_idx/b_idx: either 1-D [Na]/[Nb] (positional/K-BSM) OR per-sample
            [B, Na]/[B, Nb] (PiToMe energy). Handled uniformly below.

        spatial  -> keep only SAME-frame edges  (mask where frame_a != frame_b)
        temporal -> keep only CROSS-frame edges  (mask where frame_a == frame_b)

        Returns (scores_masked, a_valid_row) where a_valid_row [B, Na] is True iff
        that A-row still has >=1 finite (allowed) edge. Rows that are all -inf are
        excluded from top-r by the caller (it caps r at the #valid rows).
        """
        batch_size, na, nb = scores.shape
        tokens_per_frame = int(h_grid * w_grid)

        # frame id per CURRENT token, then gather to A-side / B-side frames.
        frame_all = token_ids // tokens_per_frame                 # [B, N]
        if a_idx.dim() == 1:
            a_idx_b = a_idx.unsqueeze(0).expand(batch_size, -1)   # [B, Na]
            b_idx_b = b_idx.unsqueeze(0).expand(batch_size, -1)   # [B, Nb]
        else:
            a_idx_b = a_idx
            b_idx_b = b_idx
        frame_a = frame_all.gather(1, a_idx_b)                    # [B, Na]
        frame_b = frame_all.gather(1, b_idx_b)                    # [B, Nb]

        same = frame_a.unsqueeze(2) == frame_b.unsqueeze(1)       # [B, Na, Nb] bool
        if axis == "spatial":
            allowed = same          # keep same-frame
        else:                       # "temporal"
            allowed = ~same         # keep cross-frame

        neg_inf = torch.finfo(scores.dtype).min
        scores_masked = scores.masked_fill(~allowed, neg_inf)
        a_valid_row = allowed.any(dim=2)                          # [B, Na]
        return scores_masked, a_valid_row

    # ---------------------------------------------------------------------
    # NEW HELPER 3 — WAM per-token task relevance (motion saliency).
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _load_prior_relevance(self, source, batch_size, num_tokens, t_grid, h_grid, w_grid, device):
        """NEW(V2): load a precomputed per-token relevance PRIOR from config.relevance_path.

        Returns [batch_size, num_tokens] float on `device`, or None to signal
        "no prior -> fall back to the in-place motion signal (V1)". The prior is a
        CLIP-AGNOSTIC normalized importance map (e.g. offline predictor saliency or
        hand-joint density), shape [t*h*w] or [h*w] (tiled over time). Only consulted
        at the first merge layer. Robust: any missing/unreadable/mismatched file
        returns None (=> motion fallback), so this can NEVER crash a run.
        """
        # --- WAM_SUPPLEMENT_PATCH_20260729 -------------------------------------------
        # Score-family sources all route through the prior loader so the merge
        # maths below stays identical across arms (clean single-variable ablation).
        if source not in ("predictor_saliency", "handjoint", "flow", "egocomp",
                          "objmotion", "random", "antiflow", "prior"):
            return None
        path = str(getattr(self.config, "relevance_path", "") or "")
        if not path:
            return None
        # PER-CLIP store: `relevance_path` is a directory holding index.json.
        # Returns None on ANY problem, which falls back to the motion signal.
        if _os_.path.isdir(path):
            try:
                from src.models.utils import wam_prior_store as _wps
                store = _wps.get_store(path)
                if store is None:
                    return None
                arr = store.lookup(_wps.get_current_clip_keys(), int(num_tokens))
                if arr is None or arr.shape[0] != int(batch_size):
                    return None
                return torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            except Exception:
                return None
        import os as _os
        import numpy as _np
        cache = getattr(self, "_wam_prior_cache", None)
        if cache is None or cache.get("path") != path:
            if not _os.path.exists(path):
                return None
            try:
                arr = _np.load(path)
                rel = arr["rel"] if (hasattr(arr, "files") and "rel" in arr.files) else arr
                rel = _np.asarray(rel, dtype="float32").reshape(-1)
            except Exception:
                return None
            cache = {"path": path, "rel": rel}
            self._wam_prior_cache = cache
        rel = cache["rel"]
        n_thw, n_hw = int(t_grid * h_grid * w_grid), int(h_grid * w_grid)
        if rel.shape[0] == n_thw:
            vec = rel
        elif rel.shape[0] == n_hw:                      # spatial prior -> tile over time
            vec = _np.tile(rel, int(t_grid))
        else:
            return None                                 # shape mismatch -> motion fallback
        if int(vec.shape[0]) != int(num_tokens):
            return None
        try:                                            # (B) device-level guard: never raise
            t = torch.from_numpy(_np.ascontiguousarray(vec)).to(device=device, dtype=torch.float32)
            return t.unsqueeze(0).expand(int(batch_size), -1).contiguous()   # [B, t*h*w]
        except Exception:
            return None

    @torch.no_grad()
    def _wam_relevance(self, x, token_ids, rep_for_orig, t_grid, h_grid, w_grid):
        """Per-CURRENT-token task relevance in [0,1] (1 = protect hardest).

        Computed ONCE at the first merge layer (where N == num_original == t*h*w)
        as adjacent-frame feature change (motion); cached per ORIGINAL token id.
        At later (already-compressed) layers we GATHER the cached relevance by the
        current tokens' original ids (token_ids), since N != t*h*w there and the
        grid reshape in compute_importance would be invalid. Each layer is then
        per-sample min-max normalized to [0,1] (max -> 1) so the gate is consistent.

        relevance_source="motion" => compute_importance(..., "motion"); other
        sources fall through to the same motion default for now (V1).
        """
        batch_size, num_tokens, _ = x.shape
        num_original = int(rep_for_orig.shape[1])
        source = str(getattr(self.config, "relevance_source", "motion") or "motion")
        if source == "none":
            source = "motion"

        norm_mode = str(getattr(self.config, "relevance_norm", "max"))
        prop_mode = str(getattr(self.config, "relevance_prop", "self"))

        def _unit01(rel):
            if norm_mode == "rank":
                # per-sample rank in [0,1] over the CURRENT tokens (mean 0.5 at every
                # layer, so the total suppressed priority no longer depends on how
                # sparse the signal is); ties broken by position for determinism.
                n = rel.shape[1]
                order = rel.float().argsort(dim=1, stable=True)
                ranks = torch.empty_like(order)
                ranks.scatter_(1, order, torch.arange(n, device=rel.device).expand_as(order))
                return ranks.float() / max(n - 1, 1)
            rel = rel.float().clamp_min(0.0)
            rel = rel / rel.amax(dim=1, keepdim=True).clamp_min(1e-6)
            return rel

        def _shape_motion(rel):
            """M3 first-frame fill and E1 per-frame median compensation on the dense grid."""
            ff = str(getattr(self.config, "relevance_first_frame", "zero"))
            comp = str(getattr(self.config, "relevance_comp", "none"))
            if ff == "zero" and comp == "none":
                return rel
            g = rel.float().reshape(batch_size, int(t_grid), int(h_grid) * int(w_grid)).clone()
            if ff == "ffill" and int(t_grid) > 1:
                g[:, 0] = g[:, 1]
            if comp == "median":
                g = (g - g.median(dim=2, keepdim=True).values).clamp_min(0.0)
            return g.reshape(batch_size, -1)

        # First merge layer: dense grid -> compute & cache (token_ids == arange,
        # so the per-current-token vector IS indexed by original id).
        if num_tokens == num_original and num_tokens == int(t_grid * h_grid * w_grid):
            # NEW(V2): if a precomputed per-token prior is configured for this source
            # (predictor_saliency / handjoint + relevance_path), use it; otherwise
            # fall back to the in-place adjacent-frame motion signal (V1). Everything
            # downstream (cache, _unit01, gate, top-r, merge) is IDENTICAL either way,
            # so this is a pure signal-source swap (clean V1-vs-V2 ablation).
            rel = self._load_prior_relevance(source, batch_size, num_tokens,
                                             int(t_grid), int(h_grid), int(w_grid), x.device)
            self._wam_rel_actual_source = "prior" if rel is not None else "motion"
            if rel is None:
                # (C) a prior was REQUESTED but did not load -> warn ONCE so we never
                # silently run V1 while believing it is V2 (would corrupt the ablation).
                if (source in ("predictor_saliency", "handjoint", "flow", "egocomp",
                               "objmotion", "random", "antiflow", "prior")
                        and str(getattr(self.config, "relevance_path", "") or "")
                        and not getattr(self, "_wam_prior_warned", False)):
                    self._wam_prior_warned = True
                    print(f"[WAM][V2] WARNING: relevance_source={source!r} + relevance_path set, "
                          f"but prior did NOT load -> falling back to MOTION (V1). "
                          f"Check the .npz path/shape!", flush=True)
                rel = compute_importance(x, int(t_grid), int(h_grid), int(w_grid),
                                         "motion" if source not in ("norm", "norm_motion",
                                                                    "qk_global_hidden") else source)
            if source == "external":
                ext = EXTERNAL_RELEVANCE
                if ext is None or tuple(ext.shape) != (batch_size, num_tokens):
                    raise RuntimeError("relevance_source='external' but no matching "
                                       "EXTERNAL_RELEVANCE was set for this batch")
                rel = ext.to(device=x.device, dtype=torch.float32)
                self._wam_rel_actual_source = "external"
            if source == "scorer":
                if SCORER is None:
                    raise RuntimeError("relevance_source='scorer' but no scorer was set")
                with torch.autocast("cuda", enabled=False):
                    rel = torch.sigmoid(SCORER(x.float())).float()
                self._wam_rel_actual_source = "scorer"
            if source == "random_inplace":
                # control gate with the SAME marginal distribution under rank norm; seeded
                # per clip from its own content so it is independent of batching.
                rel = torch.empty(batch_size, num_tokens, device=x.device, dtype=torch.float32)
                for b in range(batch_size):
                    seed = int((x[b, :4, :4].float().abs().sum().item() * 1e4)) % (2 ** 31)
                    gen = torch.Generator(device=x.device).manual_seed(seed)
                    rel[b] = torch.rand(num_tokens, generator=gen, device=x.device)
                self._wam_rel_actual_source = "random_inplace"
            elif self._wam_rel_actual_source == "motion":
                rel = _shape_motion(rel)
            if rel is None:
                rel = torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.float32)
            self._wam_rel_orig = rel.detach().float()              # [B, num_original]
            return _unit01(rel)

        # Later layers: gather cached relevance by current tokens' original ids.
        cache = getattr(self, "_wam_rel_orig", None)
        if (cache is not None and cache.shape[0] == batch_size
                and cache.shape[1] == num_original):
            if prop_mode == "self":
                return _unit01(cache.gather(1, token_ids))
            # M4: aggregate over the original tokens each current token represents.
            # rep_for_orig[b, o] = original id of the current token representing o.
            agg = torch.zeros_like(cache)
            agg.scatter_reduce_(1, rep_for_orig, cache,
                                reduce="amax" if prop_mode == "max" else "mean",
                                include_self=False)
            return _unit01(agg.gather(1, token_ids))
        # Fallback: no cache (first merge layer wasn't dense) -> no protection.
        return torch.zeros(batch_size, num_tokens, device=x.device, dtype=torch.float32)
