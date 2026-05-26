# ThinkJEPA Token Merge A/B/C Result Summary

## Claim Boundary

- This is a full tiny-cache pipeline/stress evaluation, not a paper-scale benchmark.
- Stage 2 importance diagnostics are not proof that importance improves downstream quality.
- 512x512 results are encoder-only stress results, not downstream accuracy results.
- `qk_global_hidden` is a global-hidden proxy, not a true attention map.
- `dynamic_ratio_mode`, `score_delta`, and `debug_dump_scores` are metadata-only here.

## Encoder Summary

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

## Critical Read

- A and B are nearly tied at the same actual merge ratio; current `norm_motion` protection is not strong evidence that importance is useful.
- C is faster only because it merges the same amount, but its feature fidelity is worse than A/B at the same ratio.
- The best conservative candidate remains `A r=0.05`; `B r=0.05` is comparable but not clearly better.
- Downstream tiny-cache compatibility passes for all A/B/C r=0.05, but ADE is still worse than baseline on the one-sample validation split.
