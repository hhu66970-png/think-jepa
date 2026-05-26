# ThinkJEPA Encoder-side Token Merge Stage5/Stage6 闭环实验报告
## 结论先行
- Stage5 的 512x512 layer sweep 已补齐：A/B/C × layer 4/6/8/12 × ratio 0.05/0.10/0.125，全 3 个 tiny-cache NPZ、完整 64 帧、online dense encoder，`any_fallback=false`。
- 512 encoder stress 的最稳 Pareto 区域是 late merge：layer12。A/B 在 layer12 几乎重合，B 的 norm_motion/local_top1 protection 没有显示出稳定优于 A 的收益；C 的 feature fidelity 明显低于 A/B。
- Stage6 已补 256 ThinkJEPA downstream full tiny-cache：baseline、A_l12_r0.05/r0.10、B_l12_r0.05/r0.10、C_l12_r0.05，全部 50 epochs、固定 train/test manifest、online dense encoder。
- Downstream 50 epoch 的 ADE 差异非常小，A/B/C 没有显著破坏链路，但也不能证明真实精度提升；tiny-cache 只有 2 train + 1 test，不支持统计显著性或 full benchmark 结论。
- 当前最诚实的主结论：encoder dense patch tokens 存在可压缩冗余，late-layer similarity merge 是最稳 baseline；importance-protected B 没有证明优于 A，hybrid C 在 encoder fidelity 上更差。

## 实验范围
| 项目 | 设置 |
|---|---|
| 数据 | `/root/autodl-tmp/thinkjepa-work/tiny-cache/part2`，3 个 NPZ，`imgs` 完整 64 帧 |
| Stage5 | 512x512 encoder-only stress，`N=32768` dense patch tokens |
| Stage6 | 256 ThinkJEPA downstream path，train_tiny 2 samples，test_tiny 1 sample，50 epochs |
| checkpoint | `vjepa2/vitl.pt`，不换 checkpoint |
| merge位置 | encoder block 后，物理缩短 hidden patch token sequence，最后 `restore_dense=True` 接回 predictor |
| 方法 | A similarity-only；B norm_motion/local_top1 importance-protected；C hybrid score |

## Stage5: 512x512 Layer Sweep 核心结果
| scope | ratio | speedup | tokens_after | mean_cos | p10 | p1 | mem_delta | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_l12 | 0.050 | 1.044x | 31130 | 0.9876 | 0.9810 | 0.7657 | -37.65 | False |
| A_l12 | 0.100 | 1.082x | 29492 | 0.9706 | 0.9077 | 0.6285 | -80.26 | False |
| A_l12 | 0.125 | 1.097x | 28672 | 0.9610 | 0.8719 | 0.5809 | -102.48 | False |
| A_l4 | 0.050 | 1.091x | 31130 | 0.9486 | 0.8514 | 0.4142 | -35.46 | False |
| A_l4 | 0.100 | 1.164x | 29492 | 0.8992 | 0.6633 | 0.3348 | -79.06 | False |
| A_l4 | 0.125 | 1.193x | 28672 | 0.8771 | 0.6038 | 0.3114 | -97.48 | False |
| B_l12 | 0.050 | 1.044x | 31130 | 0.9876 | 0.9809 | 0.7645 | -37.65 | False |
| B_l12 | 0.100 | 1.082x | 29492 | 0.9706 | 0.9078 | 0.6272 | -80.26 | False |
| B_l12 | 0.125 | 1.097x | 28672 | 0.9610 | 0.8721 | 0.5797 | -102.48 | False |
| B_l4 | 0.050 | 1.093x | 31130 | 0.9488 | 0.8533 | 0.4129 | -35.46 | False |
| B_l4 | 0.100 | 1.168x | 29492 | 0.8998 | 0.6631 | 0.3331 | -79.06 | False |
| B_l4 | 0.125 | 1.200x | 28672 | 0.8778 | 0.6031 | 0.3098 | -97.48 | False |
| C_l12 | 0.050 | 1.041x | 31130 | 0.9772 | 0.9695 | 0.5603 | -37.65 | False |
| C_l12 | 0.100 | 1.080x | 29492 | 0.9538 | 0.8330 | 0.4736 | -80.26 | False |
| C_l12 | 0.125 | 1.095x | 28672 | 0.9419 | 0.7861 | 0.4481 | -102.48 | False |
| C_l4 | 0.050 | 1.093x | 31130 | 0.9356 | 0.7890 | 0.3511 | -35.46 | False |
| C_l4 | 0.100 | 1.169x | 29492 | 0.8851 | 0.6102 | 0.2950 | -79.06 | False |
| C_l4 | 0.125 | 1.200x | 28672 | 0.8609 | 0.5586 | 0.2794 | -97.48 | False |

解读：layer4 speedup 最大，但 p10/p1 cosine 明显低，属于更激进且更伤表征的配置；layer12 speedup 较小但 fidelity 最稳，更适合作为 downstream 候选。

## Stage5: Profile Breakdown 代表性结果
| scope | config | patch | pre_blocks | merge | post_blocks | restore | total_profiled |
|---|---|---:|---:|---:|---:|---:|---:|
| A_l12 | A_sim_same_time_vec_l12_r0.05_512 | 2.93 | 374.00 | 3.20 | 277.86 | 0.26 | 658.71 |
| A_l12 | A_sim_same_time_vec_l12_r0.125_512 | 2.92 | 374.79 | 3.24 | 245.62 | 0.23 | 627.18 |
| A_l12 | A_sim_same_time_vec_l12_r0.1_512 | 2.93 | 374.47 | 3.26 | 254.46 | 0.25 | 635.80 |
| A_l12 | baseline | 2.94 | 684.29 | 0.00 | 0.00 | 0.00 | 687.89 |
| B_l12 | B_protect_norm_motion_local_top1_l12_r0.05_512 | 2.93 | 374.75 | 3.79 | 277.98 | 0.24 | 660.10 |
| B_l12 | B_protect_norm_motion_local_top1_l12_r0.125_512 | 2.94 | 375.09 | 3.85 | 245.24 | 0.23 | 627.73 |
| B_l12 | B_protect_norm_motion_local_top1_l12_r0.1_512 | 2.93 | 374.84 | 3.83 | 254.16 | 0.23 | 636.39 |
| B_l12 | baseline | 2.94 | 685.26 | 0.00 | 0.00 | 0.00 | 688.87 |
| C_l12 | C_hybrid_norm_motion_local_top1_l12_r0.05_512 | 2.97 | 375.44 | 3.84 | 277.94 | 0.24 | 660.86 |
| C_l12 | C_hybrid_norm_motion_local_top1_l12_r0.125_512 | 2.96 | 375.97 | 3.85 | 245.31 | 0.22 | 628.70 |
| C_l12 | C_hybrid_norm_motion_local_top1_l12_r0.1_512 | 2.95 | 375.40 | 3.84 | 254.12 | 0.23 | 636.95 |
| C_l12 | baseline | 2.96 | 684.59 | 0.00 | 0.00 | 0.00 | 688.22 |

解读：late-layer merge 的收益来自 post-merge blocks 处理更短 sequence；merge module 和 restore_dense 在向量化版本下不是主要瓶颈。layer12 的加速幅度较小，是因为只压缩后半段较少 block。

## Stage6: 256 Downstream Full Tiny-cache 结果
| config | epochs | ADE | delta ADE | val_loss | pred_loss | latent_cos_dist | epoch50_time |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 50 | 0.117993 | +0.000000 | 0.005169 | 8.676202 | 0.554748 | 1.86s |
| A_l12_r005 | 50 | 0.116352 | -0.001641 | 0.005020 | 8.681848 | 0.554366 | 1.81s |
| A_l12_r010 | 50 | 0.117927 | -0.000066 | 0.005155 | 8.683505 | 0.553890 | 1.80s |
| B_l12_r005 | 50 | 0.116804 | -0.001189 | 0.005055 | 8.680966 | 0.554232 | 1.89s |
| B_l12_r010 | 50 | 0.117774 | -0.000219 | 0.005142 | 8.682595 | 0.553789 | 1.85s |
| C_l12_r005 | 50 | 0.116306 | -0.001687 | 0.005019 | 8.677083 | 0.554391 | 1.90s |

解读：所有候选都能完成 50 epochs，shape/restore/training 兼容性成立。ADE 差异只有约 0.001-0.002，在 1 个 test sample 上不能解释为真实精度提升；B 没有优于 A，C 虽然 tiny-cache ADE 略好，但 Stage5 encoder fidelity 明显更差，因此不应把 C 作为主方法。

## Critic 闭环结论
- Stage5 缺口已经补齐：不再只是 layer8，而是完整 layer 4/6/8/12 sweep。
- Stage6 已从 5 epoch compatibility 扩展到 50 epoch full tiny-cache downstream，但仍不是完整 EgoDex/ThinkJEPA benchmark。
- 继续调 B/C 小参数的优先级下降：B≈A，C encoder fidelity 更差。下一轮若继续，应换更强 importance signal 或改 predictor-side 接收 compressed tokens，而不是继续在同一 norm/motion proxy 上微调。
- Acceleration claim 应限定为 encoder-side stress / tiny-cache engineering evidence；不能写成 paper-scale downstream speedup 或 accuracy improvement。

## 产物路径
- Stage5 outputs: `outputs/token_merge_abc_20260526_stage5_layer_sweep_512_full64/`
- Stage5 analysis: `reports/token_merge_abc_20260526_stage5_layer_sweep/`
- Stage6 outputs: `outputs/token_merge_abc_20260526_stage6_downstream_l12_fulltiny/`
- Closure report: `reports/token_merge_abc_20260526_stage5_stage6_closure/thinkjepa_token_merge_stage5_stage6_closure_report.md`
- Summary CSVs: `reports/token_merge_abc_20260526_stage5_stage6_closure/stage5_layer_sweep_summary.csv`, `reports/token_merge_abc_20260526_stage5_stage6_closure/stage6_downstream_l12_fulltiny_summary.csv`, `reports/token_merge_abc_20260526_stage5_stage6_closure/stage5_profile_breakdown_l12.csv`
