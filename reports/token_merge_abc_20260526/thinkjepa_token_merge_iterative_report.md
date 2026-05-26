# ThinkJEPA Encoder-side Token Merge A/B/C 迭代实验报告

## 0. 一句话结论

本轮形成了一个完整闭环：先实现 A/B/C，再跑 full tiny-cache encoder / 512 encoder stress / downstream small compatibility，再由 critic 指出结论边界并追加两轮消融。最终结论是：**当前最保守可用候选仍是 A: `local_2x2_same_time_vec, r=0.05`；现有 B/C importance proxy 与 hybrid score 没有在 matched actual merge ratio 下稳定打败 A。**

这不是说 importance 永远没用，而是说当前这组 `norm/motion/norm_motion/qk_global_hidden + local_top1/hybrid score` 没有转化成可靠的 speed-fidelity gain。

## 1. Claim Boundary

- 数据范围：当前远端可用 tiny-cache，全 3 个 NPZ，完整 64 帧，online dense encoder。
- 这不是 paper-scale benchmark；downstream 只有 `2 train / 1 test`，只能叫 full-compatibility small test。
- 512x512 是 encoder-only stress test，不能解释 downstream accuracy。
- 最终 A/B/C 主结果和后续消融均要求 `any_fallback=False`；这表示比较没有混入会丢失 B/C 语义的 Python fallback 路径。
- `qk_global_hidden` 是 hidden-state global proxy，不是真实 attention map。
- `dynamic_ratio_mode`、`score_delta`、`debug_dump_scores` 当前为 metadata-only，不作为已实现能力宣传。

## 2. 代码实现闭环

改动点：

- `vjepa2/src/models/utils/token_merge.py`
  - 新增 `MergeConfig` 字段：importance/protection/threshold/hybrid score 参数。
  - 新增 `compute_importance()`：`norm`、`motion`、`norm_motion`、`qk_global_hidden`。
  - 新增 B: `local_2x2_importance_protected_vec`。
  - 新增 C: `local_2x2_hybrid_score_vec`。
  - B/C 禁止 fallback 到 Python similarity path，避免丢失 importance 语义。
  - 同时记录 `num_candidates` 和 `num_candidate_cells`，避免把 directed pair count 和 cell count 混淆。
- `tools/run_encoder_token_merge_full_pipeline.py`
  - 增加 A/B/C CLI 参数、`--merge_strategy` alias、r0 sanity、per-run JSONL。
  - 输出 `mean/median/p10/p1 cosine`、token error、importance/protection metadata。
- `cache_train/thinker_train.py` 和 `scripts/train.sh`
  - downstream 训练路径透传新增 dense JEPA merge 参数。
- `tools/analyze_token_merge_results.py`
  - 生成 `pareto_table.csv`、`best_configs.json`、`summary_by_method.md`。

## 3. 验证 Gate

已通过：

- 远端 `py_compile`：`token_merge.py`、`run_encoder_token_merge_full_pipeline.py`、`thinker_train.py`。
- `bash -n scripts/train.sh`。
- GPU synthetic：A/B/C `[1,64,8] -> [1,56,8] -> restore [1,64,8]`。
- B/C fallback block：非 dense grid 会显式报错，不会静默退化为 Python similarity。
- 256 full tiny-cache encoder：A/B/C，全 3 NPZ，完整 64 帧，online encoder。
- 512 encoder stress：A/B/C，全 3 NPZ，完整 64 帧，online encoder。
- downstream small compatibility：baseline + A/B/C r0.05，5 epochs。
- critic loop：至少三轮，包含实现审查、结果审查、消融审查。

## 4. 主结果表

下面是自动汇总表，主比较字段必须同时看 `speedup`、`tokens`、`cosine`、`p10`、`p1` 和 `fallback`。

| scope | config | speedup | tokens | cosine | p10 | p1 | memory delta MiB | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 256_A_same_time_vec | A_sim_same_time_vec_l8_r0.025_256 | 1.083x | 7988 | 0.9781 | 0.9666 | 0.5362 | -5.55 | False |
| 256_A_same_time_vec | A_sim_same_time_vec_l8_r0.05_256 | 1.093x | 7783 | 0.9579 | 0.8905 | 0.4349 | -8.41 | False |
| 256_A_same_time_vec | A_sim_same_time_vec_l8_r0.075_256 | 1.104x | 7578 | 0.9367 | 0.7915 | 0.3764 | -9.32 | False |
| 256_A_same_time_vec | A_sim_same_time_vec_l8_r0.1_256 | 1.116x | 7373 | 0.9161 | 0.7047 | 0.3509 | -15.86 | False |
| 256_A_same_time_vec | A_sim_same_time_vec_l8_r0.125_256 | 1.131x | 7168 | 0.8966 | 0.6407 | 0.3270 | -27.37 | False |
| 256_B_norm_motion | B_protect_norm_motion_local_top1_l8_r0.05_256 | 1.089x | 7783 | 0.9579 | 0.8900 | 0.4384 | -8.41 | False |
| 256_B_norm_motion | B_protect_norm_motion_local_top1_l8_r0.1_256 | 1.107x | 7373 | 0.9164 | 0.7049 | 0.3483 | -15.86 | False |
| 256_B_norm_motion | B_protect_norm_motion_local_top1_l8_r0.125_256 | 1.124x | 7168 | 0.8970 | 0.6441 | 0.3250 | -27.37 | False |
| 256_C_norm_motion | C_hybrid_norm_motion_local_top1_l8_r0.05_256 | 1.088x | 7783 | 0.9432 | 0.8199 | 0.3830 | -8.41 | False |
| 256_C_norm_motion | C_hybrid_norm_motion_local_top1_l8_r0.1_256 | 1.109x | 7373 | 0.8979 | 0.6401 | 0.3144 | -15.86 | False |
| 256_C_norm_motion | C_hybrid_norm_motion_local_top1_l8_r0.125_256 | 1.125x | 7168 | 0.8766 | 0.5845 | 0.3004 | -27.37 | False |
| 512_A_same_time_vec | A_sim_same_time_vec_l8_r0.05_512 | 1.067x | 31130 | 0.9678 | 0.9276 | 0.5207 | -38.45 | False |
| 512_A_same_time_vec | A_sim_same_time_vec_l8_r0.1_512 | 1.122x | 29492 | 0.9348 | 0.7919 | 0.4204 | -79.46 | False |
| 512_A_same_time_vec | A_sim_same_time_vec_l8_r0.125_512 | 1.144x | 28672 | 0.9186 | 0.7324 | 0.3905 | -97.48 | False |
| 512_B_norm_motion | B_protect_norm_motion_local_top1_l8_r0.05_512 | 1.068x | 31130 | 0.9680 | 0.9283 | 0.5213 | -38.45 | False |
| 512_B_norm_motion | B_protect_norm_motion_local_top1_l8_r0.1_512 | 1.124x | 29492 | 0.9349 | 0.7919 | 0.4205 | -79.46 | False |
| 512_B_norm_motion | B_protect_norm_motion_local_top1_l8_r0.125_512 | 1.147x | 28672 | 0.9189 | 0.7340 | 0.3918 | -97.48 | False |
| 512_C_norm_motion | C_hybrid_norm_motion_local_top1_l8_r0.05_512 | 1.071x | 31130 | 0.9568 | 0.8858 | 0.4186 | -38.45 | False |
| 512_C_norm_motion | C_hybrid_norm_motion_local_top1_l8_r0.1_512 | 1.125x | 29492 | 0.9177 | 0.7009 | 0.3485 | -79.46 | False |
| 512_C_norm_motion | C_hybrid_norm_motion_local_top1_l8_r0.125_512 | 1.148x | 28672 | 0.8993 | 0.6453 | 0.3300 | -97.48 | False |

## Downstream Small Compatibility

| config | best epoch | ADE | val loss | pred loss | latent cosine distance |
|---|---:|---:|---:|---:|---:|
| baseline | 5 | 1.1757 | 0.5354 | 9.6363 | 0.6413 |
| best_A_r005 | 5 | 1.2182 | 0.5724 | 9.6372 | 0.6409 |
| best_B_r005 | 5 | 1.2173 | 0.5731 | 9.6403 | 0.6409 |
| best_C_r005 | 5 | 1.2105 | 0.5641 | 9.6311 | 0.6403 |

## 5. Downstream Small Compatibility

该实验只证明 downstream 闭环可跑，不证明最终精度收益。当前 A/B/C r0.05 的 ADE 均差于 baseline。

| config | best epoch | ADE | val loss | pred loss | latent cosine distance |
|---|---:|---:|---:|---:|---:|
| baseline | 5 | 1.1757 | 0.5354 | 9.6363 | 0.6413 |
| best_A_r005 | 5 | 1.2182 | 0.5724 | 9.6372 | 0.6409 |
| best_B_r005 | 5 | 1.2173 | 0.5731 | 9.6403 | 0.6409 |
| best_C_r005 | 5 | 1.2105 | 0.5641 | 9.6311 | 0.6403 |

## 6. Iteration 2: Importance Source 消融

目标：回答 `importance_source` 本身有没有让 B 明确优于 A。

在 256 full tiny-cache、matched `actual_merge_ratio` 下：

| config | r | cosine | p10 | p1 | 结论 |
|---|---:|---:|---:|---:|---|
| A same-time vec | 0.05 | 0.9579 | 0.8905 | 0.4349 | 强 baseline |
| B norm | 0.05 | 0.9579 | 0.8905 | 0.4349 | 与 A 持平 |
| B motion | 0.05 | 0.9577 | 0.8883 | 0.4381 | 与 A 持平 |
| B norm_motion | 0.05 | 0.9579 | 0.8900 | 0.4384 | 与 A 持平 |
| B qk_global_hidden | 0.05 | 0.9577 | 0.8912 | 0.4321 | 与 A 持平 |

critic 结论：importance score 分布非退化，但当前 proxy 没有转化成比 A 更好的 merge 决策。

## 7. Iteration 3: Protection / Hybrid Score 分解

### B protection 分解

| config | r | cosine | p10 | p1 | 结论 |
|---|---:|---:|---:|---:|---|
| B norm_motion + local_top1 | 0.05 | 0.9579 | 0.8900 | 0.4384 | 与 A 持平 |
| B norm_motion + protect_none | 0.05 | 0.9579 | 0.8900 | 0.4384 | 与 local_top1 持平 |

结论：当前 `local_top1` protection 没观察到独立贡献。

### C hybrid score 分解

| config | r | cosine | p10 | p1 | 结论 |
|---|---:|---:|---:|---:|---|
| C beta=0.30 gamma=0.00 | 0.05 | 0.9464 | 0.8397 | 0.3900 | 低于 A |
| C beta=0.30 gamma=0.50 | 0.05 | 0.9432 | 0.8199 | 0.3830 | 低于 A |
| C beta=0.00 gamma=0.50 | 0.05 | 0.9514 | 0.8652 | 0.3978 | 低于 A |

结论：C 的 source penalty / receiver reward 都没有形成 Pareto 优势。

## 8. Critical Verdict

可以写：

- A/B/C 代码路径与实验链路已经接通。
- B/C 没有混入 fallback，比较口径比上一轮更干净。
- A r0.05 是当前最稳妥的保守候选。
- B 与 A 基本打平，尚无证据说明 importance protection 带来额外收益。
- C 当前不推荐作为主方法，因为 fidelity 明显低于 A/B。

不能写：

- importance 已证明有效。
- B 显著优于 A。
- C 调参后可用或最优。
- downstream 基本无损。
- 512 结果证明了下游泛化。

## 9. 下一步

如果继续做 importance 方向，应当换更强 signal 或不同机制，而不是继续在当前 proxy 上磨小数点。例如：

- 引入 predictor-side sensitivity / trajectory loss proxy。
- 用真实 attention diagnostic 只做离线分析，不进入 fast path。
- 设计更明确的 keep-then-merge，而不是在 2x2 cell 内只改 source/receiver。
- 扩大 cache 和 validation split 后，再做下游稳定性结论。

当前阶段可以收口为：**A 是可用 baseline；B/C 是 negative/neutral finding。**

## 10. 产物路径

- Encoder/full/stress 结果：`outputs/token_merge_abc_20260526_final/`
- Iteration 2 消融：`outputs/token_merge_abc_20260526_iter2/`
- Iteration 3 消融：`outputs/token_merge_abc_20260526_iter3/`
- 自动汇总：`reports/token_merge_abc_20260526/summary_by_method.md`
- Pareto 表：`reports/token_merge_abc_20260526/pareto_table.csv`
- Best configs JSON：`reports/token_merge_abc_20260526/best_configs.json`
