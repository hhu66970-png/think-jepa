# Integration: PiToMe energy-score (`bsm_pitome_gradual_vec`) + axis constraint (`merge_axis`)

Two **additive** extensions to the diagnostic-only K-BSM path. The existing
`bsm_ksim_gradual_vec` behaviour is **byte-unchanged**: same even/odd partition,
same size-weighting, same multi-layer `rep_for_orig` carry-forward, same
rectangular `[B, N-r, D]` invariant.

All code lives in `staging/pitome_and_axis_constraint.py`. This file is the
splice map: it says *which file, which method, which lines*.

> NOTE — the `@torch.no_grad() if torch is not None else (lambda f: f)`
> decorators in the staging file are a **shim** so the staging file compiles on a
> torch-less laptop. When you splice the methods into `DiagnosticTokenMerger`,
> use a plain **`@torch.no_grad()`** decorator (the existing source style).

---

## 0. Files touched

| File | Methods / blocks |
|---|---|
| `src/models/utils/token_merge.py` | `MergeConfig` dataclass (+6 fields); `normalize_merge_config` (+6 reads, +1 allow-list entry); `_validate_merge_config` (+2 checks); `forward` bsm dispatch (widen 1 condition); `_can_vectorize_dense_grid` (widen 1 condition) |
| `src/models/utils/token_merge_diagnostics.py` | `VECTORIZED_STRATEGIES` / `NO_PYTHON_FALLBACK_STRATEGIES` (+1 each); `_method_name` (+1 branch); **replace** `_forward_bsm`; **add** `_bsm_energy_partition`, `_bsm_axis_mask` |
| `src/run_token_merge_pca_experiment.py` | add `--merge_axis`, `--pitome_margin`, `--pitome_energy_max_anchors` argparse + thread them into `apply_merge_config` (see §4) |

---

## 1. `token_merge.py` — `MergeConfig` dataclass

Insert **after** the existing `bsm_match_metric: str = "key"` line (~L41):

```python
    # Axis constraint for the BSM family (bsm_ksim_gradual_vec /
    # bsm_pitome_gradual_vec). free=no constraint (default, unchanged);
    # spatial=only same-frame merges; temporal=only cross-frame merges.
    # Frame = original_id // (h_grid*w_grid). Ignored by non-BSM strategies.
    merge_axis: str = "free"
    # PiToMe energy margin m: energy_i = mean_j relu(cos(i,j)-m). 0.0 => mean cos.
    pitome_margin: float = 0.0
    # PiToMe energy anchor budget. If N > this, energy uses N x anchors subsample
    # (O(N*anchors)); <=0 => always full O(N^2).
    pitome_energy_max_anchors: int = 2048
    # Pre-existing harness fields, currently MISSING from the dataclass+normalizer
    # (run_token_merge_pca_experiment.py passes them; today they are dropped).
    bsm_partition: str = "positional"
    pre_merge_ratio: float = 0.0
    bsm_protect_ratio: float = 0.0
```

> **Latent bug fix:** `run_token_merge_pca_experiment.py` (L110–112, L449–451)
> already passes `bsm_partition` / `pre_merge_ratio` / `bsm_protect_ratio` into
> `normalize_merge_config`, but the current dataclass + normalizer have no such
> fields, so those values are silently discarded. Adding them here makes the
> harness flags actually take effect. If you do **not** want to touch the
> pre-existing trio, you can omit them and the harness keeps silently dropping
> them as it does today — but `merge_axis`, `pitome_margin`,
> `pitome_energy_max_anchors` are **required** for these two features.

## 2. `token_merge.py` — `normalize_merge_config`

### 2a. allow multi-layer (tuple ~L69–71)
```python
    grid_agnostic_multilayer_strategies = (
        "bsm_ksim_gradual_vec",
        "bsm_pitome_gradual_vec",   # NEW
    )
```

### 2b. read the new fields (inside the `MergeConfig(...)` kwargs, ~L98–127)
```python
        merge_axis=str(config.get("merge_axis", "free")),
        pitome_margin=float(config.get("pitome_margin", 0.0)),
        pitome_energy_max_anchors=int(config.get("pitome_energy_max_anchors", 2048)),
        bsm_partition=str(config.get("bsm_partition", "positional")),
        pre_merge_ratio=float(config.get("pre_merge_ratio", 0.0)),
        bsm_protect_ratio=float(config.get("bsm_protect_ratio", 0.0)),
```

## 3. `token_merge.py` — `_validate_merge_config` (~L132)

Append:
```python
    if config.merge_axis not in ("free", "spatial", "temporal"):
        raise ValueError(
            f"merge_axis must be one of free|spatial|temporal, got {config.merge_axis!r}"
        )
    if config.strategy == "bsm_pitome_gradual_vec" and config.pitome_margin < 0.0:
        raise ValueError("pitome_margin must be >= 0.0")
```

## 4. `token_merge.py` — `forward` + `_can_vectorize_dense_grid`

`forward` bsm dispatch (~L299):
```python
        if self.config.strategy in ("bsm_ksim_gradual_vec", "bsm_pitome_gradual_vec"):
            return self._forward_bsm(
                x, token_ids, token_size, rep_for_orig,
                int(t_grid), int(h_grid), int(w_grid), attn_key,
            )
```

`_can_vectorize_dense_grid` defensive early-return (~L453):
```python
        if self.config.strategy in ("bsm_ksim_gradual_vec", "bsm_pitome_gradual_vec"):
            return True, None
```

> The new strategy is registered **only** in `DiagnosticTokenMerger`'s tuples, so
> the base `LocalTokenMerger.forward` still raises "Unsupported merge strategy"
> for it (line ~290–291) and the bsm branch is reached only via the subclass —
> exactly the existing K-BSM containment.

---

## 5. `token_merge_diagnostics.py`

### 5a. register strategy (tuples ~L36–45)
Add `"bsm_pitome_gradual_vec",` to **both** `VECTORIZED_STRATEGIES` and
`NO_PYTHON_FALLBACK_STRATEGIES`.

### 5b. `_method_name` (~L47–54)
Add, before `return super()._method_name()`:
```python
        if self.config.strategy == "bsm_pitome_gradual_vec":
            return "BSM_pitome_energy"
```

### 5c. replace `_forward_bsm` (L251–389)
Replace the whole method with the staging `_forward_bsm` (use plain
`@torch.no_grad()`). It is a superset of the current one:

* **unchanged**: metric/key selection + fallback flags, size-weighted
  scatter-add merge, multi-layer `rep_for_orig` remap, `r`-caps, rectangular
  invariant, and the K-BSM (positional even/odd) path.
* **new**: a partition step (energy split for PiToMe; even/odd preserved exactly
  for K-BSM), an axis mask applied to `scores` before the per-A argmax, and an
  `r_eff` cap that drops fully-masked A-rows out of top-r.

### 5d. add helpers
Copy `_bsm_energy_partition` and `_bsm_axis_mask` (plain `@torch.no_grad()`) as
new methods on `DiagnosticTokenMerger`.

### 5e. per-sample vs shared partition indices — already handled
The PiToMe partition returns **per-sample** `a_idx`/`b_idx` of shape `[B, Na]` /
`[B, Nb]` (energy ranking differs per sample); the K-BSM even/odd split is the
same 1-D `[Na]`/`[Nb]` for every sample. The staging `_forward_bsm` **already
dispatches on `a_idx.dim()`** at both touch points — `a_metric`/`b_metric`
gather and `source_pos`/`receiver_pos` mapping — so **no manual edit is required**;
copy the method as-is. `_bsm_axis_mask` also accepts both 1-D and `[B,*]`
indices. (Recorded here only so a reviewer knows the shape contract.)

---

## 6. PiToMe energy score — formula & complexity

Energy of token `i` over the L2-normalized matching metric (post-RoPE key, or
hidden-feature fallback — same metric K-BSM uses):

```
energy_i = (1 / (N-1)) * sum_{j != i} relu( cos(i, j) - m )       # m = pitome_margin
```

* **high energy** = similar to many tokens = **redundant** => put in **A** (sources, merged away).
* **low energy**  = isolated/unique                => put in **B** (receivers, protected).

Partition: rank by descending energy, take **top `Na = ceil(N/2)` as A**, the
rest as **B**. Using the *same* split sizes as even/odd guarantees `na >= nb >= 1`
and `r <= na` and the existing `r`-cap `floor((N-1)/2) <= na`, so the rectangular
`[B, N-r, D]` invariant and all `r`-caps are preserved with **zero** changes to
the merge/compaction code.

**Complexity & approximation.** Full energy = `N x N` cosine gram → `O(N^2 d)`
time, `O(N^2)` memory. For V-JEPA2 ViT-L @256px/64f the first merge layer has
`N = t_grid*h_grid*w_grid = 32*16*16 = 8192` (it shrinks every merge layer), so
`N^2 ≈ 6.7e7` — feasible but heavy. The helper subsamples **A uniformly-spaced
anchor columns** when `N > pitome_energy_max_anchors` (default 2048):

```
energy_i ≈ (1/(A-1)) * [ sum_a relu(cos(i, anchor_a) - m)  -  self_contribution_i ]
```

`O(N*A)` time/memory; an unbiased estimate of the full mean for uniform anchors;
the diagonal / self-term is removed so `cos(i,i)=1` never inflates energy. Set
`pitome_energy_max_anchors <= 0` to force exact `O(N^2)`.

---

## 7. Axis constraint — implementation & degeneracy

After `scores = bmm(a_metric, b_metric^T)` (`[B, Na, Nb]`) and **before**
`scores.max(dim=2)`:

* frame of a token = `original_id // (h_grid*w_grid)` (verified mapping — see §9).
  Derived from `token_ids` (original id), **not** the compacted position, and a
  merged token carries its **receiver's** original frame.
* build `same = frame_a[:,:,None] == frame_b[:,None,:]`.
  * `spatial`  → `allowed = same`   (keep only same-frame edges)
  * `temporal` → `allowed = ~same`  (keep only cross-frame edges)
  * `free`     → no mask (default; not even entered).
* `scores.masked_fill(~allowed, finfo.min)`.

**Degeneracy — fully-masked A-row.** If an A-token has no allowed B (e.g. a frame
with a single A-token under `spatial`, or all tokens in one frame under
`temporal`), its whole row is `-inf` and `best_sim = -inf`. The helper returns
`a_valid_row = allowed.any(dim=2)`; `_forward_bsm` then caps
`r_eff = min(r, min_over_batch(#valid A-rows))`. Because `-inf` rows sort to the
bottom, `topk(r_eff)` can only ever pick finite edges → **no `-inf` edge is ever
selected**, and `r_eff` is a single scalar so the output stays rectangular
(`[B, N - r_eff, D]`, exactly `r_eff` removed per sample). If `r_eff == 0` the
layer is a clean no-op (returns input with `num_accepted=0`).

> Edge note: using `torch.finfo(dtype).min` (not `-inf`) for the mask avoids
> `nan` if a later op multiplies a masked score by 0; `topk`/`max` ordering is
> identical to `-inf` for finite-vs-masked comparison.

---

## 8. Harness usage (`run_token_merge_pca_experiment.py`)

The harness already has `--bsm_partition` (positional|temporal) and
`--bsm_protect_ratio`. **Add** three flags and thread them through
`apply_merge_config` (which already forwards the dict to `normalize_merge_config`):

```python
    ap.add_argument("--merge_axis", default="free",
                    choices=["free", "spatial", "temporal"],
                    help="BSM axis constraint: free=any, spatial=same-frame only, "
                         "temporal=cross-frame only")
    ap.add_argument("--pitome_margin", type=float, default=0.0,
                    help="PiToMe energy margin m in mean_j relu(cos-m) "
                         "(bsm_pitome_gradual_vec only)")
    ap.add_argument("--pitome_energy_max_anchors", type=int, default=2048,
                    help="PiToMe energy anchor budget; N>this uses NxA subsample, "
                         "<=0 forces full O(N^2)")
```

Then in `apply_merge_config(...)` add params `merge_axis="free"`,
`pitome_margin=0.0`, `pitome_energy_max_anchors=2048`, put them in the `cfg`
dict, and pass them at the two call sites (the dense-baseline call can keep
defaults). Example dict additions:

```python
        "merge_axis": str(merge_axis),
        "pitome_margin": float(pitome_margin),
        "pitome_energy_max_anchors": int(pitome_energy_max_anchors),
```

And at the per-config call (~L445) add:
`merge_axis=args.merge_axis, pitome_margin=args.pitome_margin,
pitome_energy_max_anchors=args.pitome_energy_max_anchors`.

To also benchmark the new strategy, pass it via `--strategy` (it is **not**
auto-added to `build_config_matrix`, which currently hard-codes K-BSM for the
pre-merge case only; `--strategy bsm_pitome_gradual_vec` flows through the
generic `for strat in strategies` loop unchanged).

### Example invocations
```bash
# PiToMe energy-score, gradual multi-layer, key-metric, default free axis
python -m src.run_token_merge_pca_experiment \
  --strategy bsm_pitome_gradual_vec --merge_layers 4-12:2 --r_per_layer 0.15 \
  --bsm_match_metric key --out_dir runs/pitome_free

# PiToMe with a redundancy margin
python -m src.run_token_merge_pca_experiment \
  --strategy bsm_pitome_gradual_vec --pitome_margin 0.5 \
  --merge_layers 4-12:2 --r_per_layer 0.15 --out_dir runs/pitome_m05

# Axis ablation on the existing K-BSM (no PiToMe), same-frame only
python -m src.run_token_merge_pca_experiment \
  --strategy bsm_ksim_gradual_vec --merge_axis spatial \
  --merge_layers 4-12:2 --r_per_layer 0.15 --out_dir runs/ksim_spatial

# Cross-frame only (temporal redundancy)
python -m src.run_token_merge_pca_experiment \
  --strategy bsm_ksim_gradual_vec --merge_axis temporal \
  --merge_layers 4-12:2 --r_per_layer 0.15 --out_dir runs/ksim_temporal

# PiToMe + temporal axis combined
python -m src.run_token_merge_pca_experiment \
  --strategy bsm_pitome_gradual_vec --merge_axis temporal \
  --merge_layers 4-12:2 --r_per_layer 0.15 --out_dir runs/pitome_temporal
```

---

## 9. id ↔ (t,h,w) mapping — verification

Confirmed in `token_merge.py`:
* `init_token_merge_state` (L222–227): `token_ids = arange(num_tokens)` at the
  first merge layer ⇒ original id **==** flat grid position.
* `ids_to_coords` (L230–236): `tokens_per_frame = h_grid*w_grid`, `t = id // tpf`,
  `h = (id - t*tpf)//w_grid`, `w = rem - h*w_grid` ⇒ `id = t*(H*W)+h*W+w`.

So **`frame(id) = id // (h_grid*w_grid)`** — exactly what `_bsm_axis_mask` uses.
At layers > L_start `token_ids` is a survivor subset of original ids (the
position is compacted but the id is preserved), so deriving frame from the id is
correct across all merge layers. A merged token keeps the **receiver's** id and
hence the receiver's frame — a well-defined choice, consistent with how K-BSM
already treats merged-token identity.

Default grid for V-JEPA2 ViT-L @256px / 64f, patch 16, tubelet 2:
`t_grid = 64/2 = 32`, `h_grid = w_grid = 256/16 = 16`, `tokens_per_frame = 256`,
`N0 = 32*256 = 8192`. (These come from the harness `build_model`; the merger
receives them as `t_grid/h_grid/w_grid` and they are **not** hard-coded in the
new code — it reads `h_grid*w_grid` at runtime.)

---

## 10. Minimal GPU validation (once a GPU is available)

1. **Smoke, PiToMe, free axis** — one clip, single layer:
   ```bash
   python -m src.run_token_merge_pca_experiment \
     --strategy bsm_pitome_gradual_vec --merge_layers 8 --r_per_layer 0.15 \
     --repeats 1 --warmup 1 --out_dir /tmp/smoke_pitome
   ```
   Check the printed `tokens N>...` trajectory: each merge layer must remove
   **exactly** `r = floor(N_layer*ratio)` (capped at `floor((N-1)/2)`), so
   `tokens_final` matches `bsm_ksim_gradual_vec` at the same ratio. `cos_vs_dense`
   should be in a sane range (broadly comparable to K-BSM; PiToMe may differ
   slightly). In the JSON, per-layer info has `bsm_partition="energy"` and
   `pitome_energy_mean/min/max`.

2. **Axis: spatial** — `--strategy bsm_ksim_gradual_vec --merge_axis spatial`.
   Verify every merged pair is **same-frame**: `source_id//256 ==
   receiver_id//256`. (Use `dump_merge_decisions` or a one-off assert in
   `_forward_bsm` over `source_ids//tpf == receiver_ids//tpf`.) `merge_axis` is
   echoed in each layer's info dict.

3. **Axis: temporal** — `--merge_axis temporal`. Verify every merged pair is
   **cross-frame** (`source_id//256 != receiver_id//256`). On a heavily masked
   layer confirm `num_accepted` may be `< r` (degeneracy cap) yet `tokens_after`
   is still equal across the batch (rectangular).

4. **Degeneracy stress** — `--merge_axis temporal` with a tiny clip / few frames
   so some layers can mask many rows; confirm no crash, no `-inf`/`nan` in
   features (`assert torch.isfinite(featM).all()`), and `tokens_after` consistent
   per batch.

5. **Non-regression** — run `--strategy bsm_ksim_gradual_vec` with default
   `--merge_axis free` and confirm `tokens_final`, `cos_vs_dense`, and the
   per-layer trajectory are **identical** to a pre-change run (the K-BSM path is
   byte-unchanged; `merge_axis=free` skips the mask entirely).

6. **Anchor approximation sanity** — run PiToMe once with
   `--pitome_energy_max_anchors 0` (exact) and once with `2048` (default) on the
   same clip; `tokens_final` is identical and `cos_vs_dense` should be very
   close, confirming the subsample is a faithful energy proxy.
