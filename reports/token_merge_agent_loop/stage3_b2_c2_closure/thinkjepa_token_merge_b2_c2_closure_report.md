# ThinkJEPA Encoder-side Token Merge B2/C2 闭环报告

## 结论先行

这轮实现和实验给出的结论是：

```text
B2/C2 没有形成优于 A 的新 Pareto 点。
A: local_2x2_same_time_vec 仍然是当前最稳 baseline。
B2-redundancy 会完全退化成 A 的 undirected pair selection。
B2-importance 确实改变了 merge 决策，但 selected similarity、tail fidelity 和 speed 都变差。
C2 similarity-gated importance 基本恢复到 A-like 行为，fidelity 接近 A，但没有超过 A。
```

因此本轮不进入 downstream 50 epoch。原因不是链路不通，而是 encoder-side full64 gate 没有通过：新策略没有在 matched ratio 下同时取得更低 latency 和不差的 fidelity。

## 本轮代码改动

修改位置：

```text
vjepa2/src/models/utils/token_merge.py
tools/run_encoder_token_merge_full_pipeline.py
tools/analyze_merge_pair_overlap.py
cache_train/thinker_train.py
scripts/train.sh
```

新增能力：

```text
1. decision_dump:
   在 encoder hot path 之外写出每个 merge decision 的 source/receiver/cell/similarity/importance。

2. B2 local_keep_then_merge_vec:
   支持 keep_source=redundancy / importance / importance_redundancy / random。

3. C2 local_2x2_similarity_gated_importance_vec:
   先用 similarity_gate_epsilon 限制候选，再用 importance 做 direction/tie-break。

4. pair overlap analyzer:
   读取 merge_decisions.jsonl，计算相对 A 的 cell overlap、undirected/directed pair overlap、similarity drop。

5. downstream 参数透传:
   thinker_train.py 和 scripts/train.sh 已支持 B2/C2 参数，但本轮 gate 未通过，所以不跑 downstream。
```

## 验证闭环

### Static / Schema

通过：

```text
python -m py_compile \
  vjepa2/src/models/utils/token_merge.py \
  tools/run_encoder_token_merge_full_pipeline.py \
  tools/analyze_merge_pair_overlap.py \
  cache_train/thinker_train.py

bash -n scripts/train.sh
```

### Synthetic correctness

远端 deterministic synthetic 通过：

```text
REMOTE_SYNTHETIC_V2_OK
```

覆盖内容：

```text
r=0 sanity
A same-time vectorized merge
B2 keep-then-merge
C2 similarity-gated importance
restore_dense shape
decision_dump shape
```

### Decision Dump / Pair Overlap

输入：

```text
tiny-cache part2，3 个 NPZ
512x512
完整 64 帧
merge layer = 12
ratios = 0.05, 0.125
```

产物：

```text
outputs/token_merge_agent_loop/stage2_decisions_A_B2_C2_l12_512.jsonl
reports/token_merge_agent_loop/stage2_pair_overlap/summary.md
reports/token_merge_agent_loop/stage2_pair_overlap_B2_importance/summary.md
```

关键结果：

| method | ratio | cell overlap vs A | undirected pair overlap vs A | directed pair overlap vs A | mean selected-sim drop |
|---|---:|---:|---:|---:|---:|
| B2 redundancy | 0.05 | 1.0000 | 1.0000 | 0.4980 | 0.000000 |
| B2 redundancy | 0.125 | 1.0000 | 1.0000 | 0.4942 | 0.000000 |
| C2 gated | 0.05 | 0.9835 | 0.9942 | 0.6400 | 0.000001 |
| C2 gated | 0.125 | 0.9931 | 0.9914 | 0.6425 | 0.000002 |
| B2 importance | 0.05 | 0.6345 | 0.7919 | 0.5263 | 0.000857 |
| B2 importance | 0.125 | 0.7565 | 0.6642 | 0.4397 | 0.002405 |

解释：

```text
B2-redundancy 本质上选择了和 A 一样的 undirected pair，所以它不是新方法，只是换了 source/receiver 方向。
C2 gated 几乎也选择 A 的 pair，它修复了旧 C 的 fidelity 崩坏风险，但代价是退回 A-like。
B2-importance 才真正改变决策，但它牺牲了 nearest-neighbor similarity。
```

## 512 Full64 Encoder Gate

输入：

```text
tiny-cache part2，3 个 NPZ
512x512
完整 64 帧
online dense encoder
merge layer = 12
ratios = 0.05, 0.10, 0.125
repeats = 3
warmup = 1
profile_segments = true
```

结果：

| method | ratio | speedup | latency ms | tokens after | mean cos | p10 | p1 | rel L2 | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0 | 1.0000 | 688.12 | 32768 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |  |
| A | 0.050 | 1.0442 | 658.96 | 31130 | 0.987576 | 0.981041 | 0.765688 | 0.161047 | False |
| A | 0.100 | 1.0803 | 636.98 | 29492 | 0.970596 | 0.907692 | 0.628527 | 0.246787 | False |
| A | 0.125 | 1.0951 | 628.36 | 28672 | 0.960979 | 0.871931 | 0.580908 | 0.283815 | False |
| B2 importance | 0.050 | 1.0392 | 662.16 | 31130 | 0.985741 | 0.978977 | 0.718412 | 0.172268 | False |
| B2 importance | 0.100 | 1.0765 | 639.24 | 29492 | 0.965566 | 0.884480 | 0.575162 | 0.266243 | False |
| B2 importance | 0.125 | 1.0895 | 631.59 | 28672 | 0.954171 | 0.839980 | 0.529484 | 0.306646 | False |
| C2 gated | 0.050 | 1.0396 | 661.89 | 31130 | 0.987544 | 0.980723 | 0.763927 | 0.161267 | False |
| C2 gated | 0.100 | 1.0773 | 638.77 | 29492 | 0.970577 | 0.907796 | 0.626646 | 0.246872 | False |
| C2 gated | 0.125 | 1.0908 | 630.84 | 28672 | 0.960924 | 0.872052 | 0.579315 | 0.283995 | False |

## Go / No-Go 判定

本轮判定为 No-Go，不进入 downstream。

原因：

```text
1. B2-importance 虽然改变了决策，但 fidelity 全面低于 A：
   mean_cos / p10 / p1 全部下降，relative L2 上升。

2. B2-importance 的 latency 也没有超过 A：
   r=0.05: A 1.0442x, B2 1.0392x
   r=0.10: A 1.0803x, B2 1.0765x
   r=0.125: A 1.0951x, B2 1.0895x

3. C2 gated 的 fidelity 接近 A，但 pair overlap 过高，属于 A-like neutral result：
   它不是证明 importance 有效，而是证明 similarity gate 能防止旧 C 破坏 nearest-neighbor fidelity。

4. 所有新策略 any_fallback=false，说明结果有效，不是 fallback 造成的假象。
```

## 研究解释

这一轮最重要的收获不是“又失败了”，而是把旧问题拆清楚了：

```text
如果保持 nearest-neighbor similarity，方法会自然退化到 A。
如果强行用 importance 改变 source/cell 选择，就会牺牲 feature-space 最近邻保真。
```

这说明当前 `norm_motion` importance proxy 还不能作为有效 decider signal。对于 ThinkJEPA dense encoder token merge，局部 feature similarity 仍然是一个非常强的 training-free criterion。

## 下一步建议

优先级最高的下一步不是继续调 B2/C2 小参数，而是做真正的 damage correlation：

```text
1. 采样少量 cell/pair；
2. 强制 merge 每个候选 pair；
3. 跑后续 encoder blocks；
4. 计算 true post-block damage；
5. 检验 norm/motion/qk proxy 是否真的能预测 damage。
```

如果 correlation 低，应收口为：

```text
A similarity-only + negative finding + diagnostic framework
```

如果 true attention 或某个 proxy 与 damage 有强相关，再设计下一轮 D 策略；否则继续把 importance 加进 merge policy 只会制造更复杂但不更好的 A-like 或 fidelity-negative 结果。
