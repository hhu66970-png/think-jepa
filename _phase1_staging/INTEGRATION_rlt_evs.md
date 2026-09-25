# Integrating RLT / EVS temporal pre-merge into `vision_transformer.py`

Status: code-only (no GPU here). This spec tells the GPU-side integrator exactly
where to insert the hook, how RoPE consumes the surviving positions, how it
co-exists with `merge_config`, and the relationship with `restore_dense`.

Files:
- New: `staging/temporal_premerge.py` — `rlt_premerge`, `evs_static_prune`,
  `pad_ragged_to_rectangular`.
- Contract source (read-only): `src/token_merge.py` — `init_token_merge_state`,
  `ids_to_coords`, `restore_dense_tokens`, `_apply_merge_from_positions`.

---

## 0. What "pre-merge" means here

These run **once**, right after `patch_embed` and **before transformer block 0**.
They are orthogonal to the in-block spatial K-BSM / local-2x2 merging in
`token_merge.py`. RLT/EVS shrink the temporal axis; K-BSM (run unchanged, at its
configured `merge_layers`) then continues on the already-reduced token set.

The dense grid is V-JEPA2 ViT-L @ 64 frames / 256x256 / patch16 / tubelet2:
`t_grid=32, h_grid=w_grid=16` → `N_orig = 32*16*16 = 8192` tokens.

---

## 1. Where to insert the hook in `vision_transformer.py`

`vision_transformer.py` is **not in this repo checkout**, so this is by
description. In V-JEPA2's `VisionTransformer.forward` the relevant region is:

```
x = self.patch_embed(x)                 # -> [B, N, D]  (N = 8192 here)
                                        # (some variants keep [B, T, H*W, D];
                                        #  flatten the middle two dims to [B,N,D])
pos / rope setup ...                    # RoPE is applied INSIDE each block's
                                        # attention from the token's (t,h,w),
                                        # NOT added here as an absolute PE.

# >>>>>>>>>>>>>>  INSERT PRE-MERGE HOOK HERE  <<<<<<<<<<<<<<
# (after patch_embed produces dense [B,N,D]; before the block loop;
#  and before init_token_merge_state, because the pre-merge DEFINES the
#  initial token_ids / token_size / rep_for_orig that K-BSM will carry.)

token_ids, token_size, rep_for_orig = init_token_merge_state(B, N, device, dtype)
for i, blk in enumerate(self.blocks):
    if self.merge_config.enabled and i in self.merge_config.merge_layers:
        x, token_ids, token_size, rep_for_orig, info = self.token_merger(
            x, token_ids, token_size, rep_for_orig, t_grid, h_grid, w_grid, attn_key)
    x = blk(x, ...,  rope_positions=(token_ids -> (t,h,w)))   # see §2
```

### Concretely, the hook

```python
from src.models.utils.temporal_premerge import rlt_premerge, evs_static_prune

pm = self.merge_config.pre_merge            # see §3 for the config schema
if pm is not None and pm.get("method") not in (None, "none"):
    if x.dim() == 4:                        # [B, T, H*W, D] -> [B, N, D]
        B_, T_, HW_, D_ = x.shape
        x = x.reshape(B_, T_ * HW_, D_)
    if pm["method"] == "rlt":
        x, token_ids, token_size, meta = rlt_premerge(
            x, t_grid, h_grid, w_grid,
            sim_threshold=pm.get("sim_threshold", 0.9),
            target_ratio=pm.get("target_ratio", None),   # rectangular if set
            rep_position=pm.get("rep_position", "start"))
    elif pm["method"] == "evs":
        x, token_ids, token_size, meta = evs_static_prune(
            x, t_grid, h_grid, w_grid,
            diff_threshold=pm.get("diff_threshold", 0.05),
            metric=pm.get("metric", "l2_rel"),
            target_ratio=pm.get("target_ratio", None))
    rep_for_orig = meta["rep_for_orig"]      # [B, N_orig] (rectangular mode)
    num_original_tokens = meta["num_tokens_before"]   # = 8192; stash for restore
    # NOTE: do NOT call init_token_merge_state afterwards — the pre-merge already
    # produced a CONSISTENT (token_ids, token_size, rep_for_orig) triple that
    # K-BSM composes onto.
else:
    token_ids, token_size, rep_for_orig = init_token_merge_state(B, N, device, dtype)
    num_original_tokens = N
```

After this, the block loop and `self.token_merger` (K-BSM) run **unchanged**:
K-BSM is grid-agnostic on the current token order (it partitions by even/odd
*position*, not by grid — see `_forward_bsm` in `token_merge_diagnostics.py`),
so a reduced, irregular token set at block 0 is already supported. Its
`rep_for_orig` carry-forward (`remap.scatter_(1, source_ids, receiver_ids);
rep_new = remap.gather(1, rep_for_orig)`) **composes** on top of the pre-merge's
`rep_for_orig` because both index by the same original-id space `0..N_orig-1`.

**Recommendation: wire `target_ratio` (rectangular) first.** The local-2x2
vectorized fast path in `token_merge.py` asserts equal per-sample lengths and
checks `token_ids == arange(N)` (`_can_vectorize_dense_grid`); after a pre-merge
that check fails and it would fall back to the slow Python path or error. K-BSM
(`bsm_ksim_gradual_vec`) has **no** such dense-grid requirement, so **pair the
pre-merge with K-BSM, not with the local-2x2 strategies.** If you must use
local-2x2 after a pre-merge, only the grid-agnostic BSM path is safe.

---

## 2. RoPE: how surviving tokens keep their (t, h, w)

V-JEPA2 ViT-L uses **RoPE** (rotary) attention, not an additive position
embedding. RoPE phase for a token is computed from its `(t, h, w)` grid index.
The whole point of "preserve position identity" is: **after pre-merge, the angle
for each surviving token must still be computed from its ORIGINAL `(t, h, w)`**,
not from its new compacted row index.

- **EVS**: trivially correct. EVS only *drops* tokens; every survivor keeps its
  exact original id. `token_ids_new` are the original flat ids; map them with
  `ids_to_coords(token_ids_new, h_grid, w_grid)` → `(t,h,w)` and feed those into
  the RoPE angle computation. No averaging → RoPE is **exact** for survivors.

- **RLT**: a run collapses several time steps at one `(h,w)` into one token. We
  report a single representative position per run via `rep_position`:
  - `"start"` (default): the run's **first** frame `(t_start, h, w)`. Matches the
    original RLT formulation; ids stay monotone within a location.
  - `"mid"`: the run's **middle** frame. Centers the rotary phase over the run's
    temporal extent; marginally better for long runs, loses "first-occurrence"
    semantics.
  Either way, `token_ids_new` already encodes that representative id, so the
  integrator does the **same** `ids_to_coords(token_ids_new)` → `(t,h,w)` → RoPE.
  No special-casing in the attention code; only the id source changed.

### Integration requirement on the attention/RoPE code

The block's attention must build its RoPE table **per-forward from the current
`token_ids`**, i.e. `rope_pos = ids_to_coords(token_ids, h_grid, w_grid)` and
index the cos/sin tables by `(t,h,w)`. If the existing code precomputes a single
static `[N_orig, ...]` RoPE table and indexes it by **row position**, it must be
changed to **gather by `token_ids`** (equivalently, gather rope angles with the
same `token_ids` used by K-BSM). V-JEPA2's RoPE attention already needs the
per-token grid index, so this is usually a 1-line change: index the rope table by
`token_ids`-derived `(t,h,w)` rather than by `arange(seq_len)`.

`token_size` (RLT run length; EVS all-ones) is the size-weighting used by
downstream K-BSM's size-weighted average and, if you enable proportional
attention, as the additive `log(size)` attention bias. EVS leaves it at 1, so it
is inert there; RLT sets it to the run length.

---

## 3. Co-existing with `merge_config`

`MergeConfig` (in `token_merge.py`) does **not** yet have a `pre_merge` field,
but the experiment runner (`run_token_merge_pca_experiment.py`) already threads a
flat `pre_merge_ratio` float through `apply_merge_config` / `normalize_merge_config`
and tags experiments `PRE_temporal` / `PRE_positional`. Recommended: replace the
ad-hoc float with a structured sub-config.

Add to `MergeConfig`:

```python
# None disables pre-merge. Otherwise a dict:
#   {"method": "rlt", "sim_threshold": 0.9, "target_ratio": 0.5,
#    "rep_position": "start"}
#   {"method": "evs", "diff_threshold": 0.05, "metric": "l2_rel",
#    "target_ratio": 0.5}
pre_merge: Optional[dict] = None
```

In `normalize_merge_config`, accept `config.get("pre_merge", None)`, validate
`method in {None,"none","rlt","evs"}`, coerce numeric fields, and (for back-compat)
if the legacy `pre_merge_ratio > 0` is present and `pre_merge` is None, synthesize
`{"method":"rlt","target_ratio": <ratio-as-keep-fraction>, ...}` — but note RLT's
`target_ratio` is the **kept fraction** (e.g. 0.5 keeps 50%), so translate
`pre_merge_ratio` (a *removal* ratio in the old runner) accordingly:
`target_ratio = 1 - pre_merge_ratio`. Double-check the old runner's sign
convention before flipping; the safe move is to add `pre_merge` and deprecate the
float.

`pre_merge` is consumed **only** by the hook in §1; `merge_layers`,
`merge_ratio`, `strategy` continue to drive the in-block K-BSM exactly as today.
Suggested experiment tags: `PRE_rlt__thr0.90__keep0.50`, `PRE_evs__l2_0.05`.

---

## 4. Relationship with `restore_dense`

`restore_dense_tokens(x, token_ids, rep_for_orig, num_original_tokens)` scatters a
compressed `[B, N', D]` back to the dense `[B, N_orig, D]` grid by, for each
original id, gathering the row whose `token_ids` equals that id's `rep_for_orig`.

Pre-merge makes tokens **fewer and irregular**, exactly like in-block merging, so
the same machinery applies — **with one critical wiring detail**:

- `restore_dense_tokens` needs `num_original_tokens = 8192` (the **pre-merge**
  input count), NOT the post-pre-merge count. Stash `meta["num_tokens_before"]`
  at hook time and pass it through (the model currently derives this from the
  dense grid; after a pre-merge you must use the stashed value).
- Pass the **final** `token_ids` and the **composed** `rep_for_orig` (pre-merge's
  `rep_for_orig`, then further remapped by every K-BSM layer). Because both layers
  index the same `0..8191` original-id space and K-BSM already does
  `rep_new = remap.gather(1, rep_for_orig)`, the composition is automatic — you do
  **not** write extra code, you just make sure the pre-merge's `rep_for_orig`
  (from `meta`) is the one seeded into the loop (see §1: do not overwrite it with
  `init_token_merge_state`).

**You must pick ONE of two downstream contracts (state it in the run config):**

1. **`restore_dense=True` (scatter back to 8192):** after the encoder, call
   `restore_dense_tokens` to re-expand to `[B, 8192, D]`. RLT-merged slots get the
   run mean broadcast to every original time step in the run; EVS-pruned slots get
   their temporal anchor's feature. This is the drop-in path if the **predictor /
   downstream head expects a dense 8192-token grid** (e.g. current PCA-fidelity
   harness compares against the dense baseline). Fidelity (`cos_vs_dense`) is then
   measured on the re-expanded tensor.

2. **`restore_dense=False` (keep compressed):** the predictor / head must accept a
   **variable-length, irregularly-positioned** token set and read positions from
   `token_ids`. This is the real-speedup path (fewer tokens through the
   predictor), but **the predictor must be made position-aware via `token_ids`**,
   mirroring the encoder RoPE change in §2. The default experiment runner uses
   `restore_dense=False` for the compressed timing path.

**Honest caveat on RLT + restore_dense (rectangular under-K samples):** in the
rare case a sample is so dynamic that its number of runs already exceeds the
requested `keep_k`, `rlt_premerge` clamps `keep_k` up to that sample's run count
and pads the *other* samples with size-0 inert slots (id set to 0). Those inert
slots carry id 0; if a real run in the same sample also has representative id 0
(i.e. it kept original token 0), `restore_dense_tokens`' scatter could route
original-id-0 to the inert row. Mitigation options for the integrator:
(a) use `target_ratio` large enough that no sample is under-K (typical EgoDex:
runs ≪ 8192, so keep≥0.3 is safe), or
(b) when `restore_dense=True`, build the scatter from `meta["rep_for_orig"]`
(which only references REAL runs) and ignore inert rows — already the case, since
`rep_for_orig` never points at an inert slot. The only residual risk is the id-0
collision above; the clean fix is to give inert slots an out-of-range sentinel id
and have `restore_dense_tokens` skip ids `>= num_original_tokens`. Flag this if
you ever see a sample with `meta["num_inert_slots"] > 0`.

EVS has no such issue: it never pads (ragged) or, in top-K, every kept token has a
unique original id and every original maps to a real kept anchor.

---

## 5. Minimal GPU verification (run on the GPU box, 1 EgoDex clip)

Use the existing harness (`run_token_merge_pca_experiment.py`,
`fidelity_vs_dense`) — it already returns `(cos, rel_l2)` vs the dense baseline.

1. Load **one** EgoDex clip → `[1, 3, 64, 256, 256]`.
2. Dense baseline: forward with `pre_merge=None`, record token count (8192) and
   features.
3. **RLT**: `pre_merge={"method":"rlt","sim_threshold":0.9,"target_ratio":0.5}`.
   Report:
   - `meta["num_runs_per_sample"]` and resulting token count → **how many tokens
     were cut** (`1 - N'/8192`).
   - `meta["merge_cos_mean"] / merge_cos_min` (similarity of accepted runs).
   - `cos_vs_dense` (with `restore_dense=True` to compare on the 8192 grid).
4. **EVS**: `pre_merge={"method":"evs","diff_threshold":0.05,"metric":"l2_rel"}`.
   Report `meta["kept_fraction"]`, tokens cut, `meta["pruned_change_max"]` (the
   most-dynamic token we dropped — sanity that we only dropped static ones), and
   `cos_vs_dense`.
5. Sweep `sim_threshold ∈ {0.95, 0.9, 0.85}` (RLT) and
   `diff_threshold ∈ {0.02, 0.05, 0.1}` (EVS); plot **tokens-cut vs cos_vs_dense**
   to find a working point (mirror the K-BSM `cos ≥ 0.90` workpoint search).
6. **Sanity asserts to run once:**
   - shapes: `tokens_new [B,K,D]`, `token_ids_new [B,K]`, `token_size_new [B,K]`,
     `meta["rep_for_orig"] [B,8192]`.
   - EVS: `token_size_new` all ones; all `token_ids_new` are unique per sample and
     in `[0,8191]`; frame-0 tokens (`id < 256`) all kept.
   - RLT: `token_size_new.sum(1) == 8192` per sample (every original frame is
     covered by exactly one run) — this is the strongest correctness check.
   - restore round-trip: `restore_dense_tokens(tokens_new, token_ids_new,
     meta["rep_for_orig"], 8192).shape == [B,8192,D]`.

---

## 6. HONEST expectations (no fabricated speedups)

RLT/EVS savings are **content-dependent** and unknown until measured on GPU.

- EgoDex is first-person with a **fixed camera + largely static background**, so
  static regions (background, table, torso) should form long RLT runs / be EVS-
  pruned heavily → real token reduction there.
- **Hand-motion / manipulated-object regions change every frame**, so those tokens
  will NOT merge (RLT) or be pruned (EVS) — and those are often the
  task-relevant tokens. Net reduction is therefore bounded by the static fraction
  of the clip, which varies per clip.
- Therefore **no "≈X× faster" number is claimed here.** The only way to know the
  real token-cut and the fidelity cost is the §5 sweep on actual EgoDex clips.
  Also note wall-clock speedup ≠ token-cut: irregular/short sequences can
  under-utilize the GPU, and the pre-merge itself costs one O(B·N·D) pass.
