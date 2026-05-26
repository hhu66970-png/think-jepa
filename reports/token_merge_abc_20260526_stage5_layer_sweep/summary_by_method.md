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
| A_l12 | A_sim_same_time_vec_l12_r0.05_512 | 1.044x | 31130 | 0.9876 | 0.9810 | 0.7657 | -37.65 | False |
| A_l12 | A_sim_same_time_vec_l12_r0.1_512 | 1.082x | 29492 | 0.9706 | 0.9077 | 0.6285 | -80.26 | False |
| A_l12 | A_sim_same_time_vec_l12_r0.125_512 | 1.097x | 28672 | 0.9610 | 0.8719 | 0.5809 | -102.48 | False |
| A_l4 | A_sim_same_time_vec_l4_r0.05_512 | 1.091x | 31130 | 0.9486 | 0.8514 | 0.4142 | -35.46 | False |
| A_l4 | A_sim_same_time_vec_l4_r0.1_512 | 1.164x | 29492 | 0.8992 | 0.6633 | 0.3348 | -79.06 | False |
| A_l4 | A_sim_same_time_vec_l4_r0.125_512 | 1.193x | 28672 | 0.8771 | 0.6038 | 0.3114 | -97.48 | False |
| A_l6 | A_sim_same_time_vec_l6_r0.05_512 | 1.078x | 31130 | 0.9618 | 0.9095 | 0.4739 | -38.65 | False |
| A_l6 | A_sim_same_time_vec_l6_r0.1_512 | 1.143x | 29492 | 0.9230 | 0.7455 | 0.3737 | -80.25 | False |
| A_l6 | A_sim_same_time_vec_l6_r0.125_512 | 1.169x | 28672 | 0.9048 | 0.6800 | 0.3515 | -101.48 | False |
| A_l8 | A_sim_same_time_vec_l8_r0.05_512 | 1.071x | 31130 | 0.9678 | 0.9276 | 0.5207 | -38.45 | False |
| A_l8 | A_sim_same_time_vec_l8_r0.1_512 | 1.128x | 29492 | 0.9348 | 0.7919 | 0.4204 | -79.46 | False |
| A_l8 | A_sim_same_time_vec_l8_r0.125_512 | 1.151x | 28672 | 0.9186 | 0.7324 | 0.3905 | -97.48 | False |
| B_l12 | B_protect_norm_motion_local_top1_l12_r0.05_512 | 1.044x | 31130 | 0.9876 | 0.9809 | 0.7645 | -37.65 | False |
| B_l12 | B_protect_norm_motion_local_top1_l12_r0.1_512 | 1.082x | 29492 | 0.9706 | 0.9078 | 0.6272 | -80.26 | False |
| B_l12 | B_protect_norm_motion_local_top1_l12_r0.125_512 | 1.097x | 28672 | 0.9610 | 0.8721 | 0.5797 | -102.48 | False |
| B_l4 | B_protect_norm_motion_local_top1_l4_r0.05_512 | 1.093x | 31130 | 0.9488 | 0.8533 | 0.4129 | -35.46 | False |
| B_l4 | B_protect_norm_motion_local_top1_l4_r0.1_512 | 1.168x | 29492 | 0.8998 | 0.6631 | 0.3331 | -79.06 | False |
| B_l4 | B_protect_norm_motion_local_top1_l4_r0.125_512 | 1.200x | 28672 | 0.8778 | 0.6031 | 0.3098 | -97.48 | False |
| B_l6 | B_protect_norm_motion_local_top1_l6_r0.05_512 | 1.080x | 31130 | 0.9620 | 0.9104 | 0.4722 | -38.65 | False |
| B_l6 | B_protect_norm_motion_local_top1_l6_r0.1_512 | 1.147x | 29492 | 0.9234 | 0.7454 | 0.3745 | -80.25 | False |
| B_l6 | B_protect_norm_motion_local_top1_l6_r0.125_512 | 1.174x | 28672 | 0.9052 | 0.6813 | 0.3499 | -101.48 | False |
| B_l8 | B_protect_norm_motion_local_top1_l8_r0.05_512 | 1.066x | 31130 | 0.9680 | 0.9283 | 0.5213 | -38.45 | False |
| B_l8 | B_protect_norm_motion_local_top1_l8_r0.1_512 | 1.121x | 29492 | 0.9349 | 0.7919 | 0.4205 | -79.46 | False |
| B_l8 | B_protect_norm_motion_local_top1_l8_r0.125_512 | 1.145x | 28672 | 0.9189 | 0.7340 | 0.3918 | -97.48 | False |
| C_l12 | C_hybrid_norm_motion_local_top1_l12_r0.05_512 | 1.041x | 31130 | 0.9772 | 0.9695 | 0.5603 | -37.65 | False |
| C_l12 | C_hybrid_norm_motion_local_top1_l12_r0.1_512 | 1.080x | 29492 | 0.9538 | 0.8330 | 0.4736 | -80.26 | False |
| C_l12 | C_hybrid_norm_motion_local_top1_l12_r0.125_512 | 1.095x | 28672 | 0.9419 | 0.7861 | 0.4481 | -102.48 | False |
| C_l4 | C_hybrid_norm_motion_local_top1_l4_r0.05_512 | 1.093x | 31130 | 0.9356 | 0.7890 | 0.3511 | -35.46 | False |
| C_l4 | C_hybrid_norm_motion_local_top1_l4_r0.1_512 | 1.169x | 29492 | 0.8851 | 0.6102 | 0.2950 | -79.06 | False |
| C_l4 | C_hybrid_norm_motion_local_top1_l4_r0.125_512 | 1.200x | 28672 | 0.8609 | 0.5586 | 0.2794 | -97.48 | False |
| C_l6 | C_hybrid_norm_motion_local_top1_l6_r0.05_512 | 1.080x | 31130 | 0.9495 | 0.8521 | 0.3832 | -38.65 | False |
| C_l6 | C_hybrid_norm_motion_local_top1_l6_r0.1_512 | 1.146x | 29492 | 0.9053 | 0.6581 | 0.3229 | -80.25 | False |
| C_l6 | C_hybrid_norm_motion_local_top1_l6_r0.125_512 | 1.173x | 28672 | 0.8853 | 0.6051 | 0.3034 | -101.48 | False |
| C_l8 | C_hybrid_norm_motion_local_top1_l8_r0.05_512 | 1.069x | 31130 | 0.9568 | 0.8858 | 0.4186 | -38.45 | False |
| C_l8 | C_hybrid_norm_motion_local_top1_l8_r0.1_512 | 1.125x | 29492 | 0.9177 | 0.7009 | 0.3485 | -79.46 | False |
| C_l8 | C_hybrid_norm_motion_local_top1_l8_r0.125_512 | 1.148x | 28672 | 0.8993 | 0.6453 | 0.3300 | -97.48 | False |

## Downstream Small Compatibility

| config | best epoch | ADE | val loss | pred loss | latent cosine distance |
|---|---:|---:|---:|---:|---:|

## Critical Read

- A and B are nearly tied at the same actual merge ratio; current `norm_motion` protection is not strong evidence that importance is useful.
- C is faster only because it merges the same amount, but its feature fidelity is worse than A/B at the same ratio.
- The best conservative candidate remains `A r=0.05`; `B r=0.05` is comparable but not clearly better.
- Downstream tiny-cache compatibility passes for all A/B/C r=0.05, but ADE is still worse than baseline on the one-sample validation split.
