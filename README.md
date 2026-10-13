# Token merging in a frozen V-JEPA 2 encoder

Experiment code for token merging inside a frozen V-JEPA 2 encoder. A deterministic attentive probe reads the
merged tokens and forecasts 3D hand joints (EgoDex) or classifies dives (Diving48). Every merger is compared with
two controls: encoding fewer input frames at matched cost, and the same relevance values at shuffled positions.

This branch holds the code only. Data, cached features, run outputs, and figure scripts are not included.

## Layout

- `vjepa2_models/`: the patched encoder and merge files (proportional attention, capped selection, key metric
  before RoPE, relevance variants). They derive from V-JEPA 2 (MIT License, Meta Platforms; `LICENSE` included).
- `patch/merge_and_pa_vs_baseline.patch`: all changes to the encoder and merge module relative to the base revision.
  Defaults reproduce the base revision bit for bit.
- `baseline_original/`: the encoder and merge files of the base revision, so that the bit exact tests run on their own.
- `wam_forecast/`: feature extraction (`extract.py`, `extract_videomae.py`, `pool_feats.py`), probe and classifier
  training (`train_probe.py`, `train_cls.py`), pairing and bootstrap statistics (`pairs.py`, `cvstats.py`,
  `offsplit.py`), dataset builders (`build_testset.py`, `build_xsplit.py`, `build_fps.py`, `build_diving.py`,
  `build_diving2.py`), relevance signals (`hand_masks.py`, `flow_maps.py`, `saliency.py`), timing (`bench.py`,
  `bench_fuse.py`, `profile_merge.py`), tests (`test_*.py`, `smoke*.py`), and the analysis script of every experiment
  (`*_analysis.py`, `rebuild_*.py`, `ade_fde_tables.py`).
- `drivers/`: the launch scripts of every experiment (feature extraction, job lists, GPU pool), named by an internal key.
- `requirements.txt`: Python 3.11, CUDA 12.4.

## Setup

1. Get the official V-JEPA 2 code and checkpoints (https://github.com/facebookresearch/vjepa2;
   `vitl.pt` SHA256 5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012,
   `vitg.pt` SHA256 67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e) and put the checkout at
   `$WAM_ROOT/repo/vjepa2`. Apply `patch/merge_and_pa_vs_baseline.patch`, or copy `vjepa2_models/` over `src/models/`.
2. Put `wam_forecast/` at `$WAM_ROOT/repo/wam_forecast`. `common.py` derives every path from `WAM_ROOT`
   (features via `WAM_FEAT_ROOT`, runs via `WAM_RUN_ROOT`); `extract.py` imports the encoder from
   `$WAM_ROOT/repo/vjepa2`. The VideoMAE scripts read `VIDEOMAE_PATH`, and the ViT-g scripts read `WAM_VITG`.
3. EgoDex and Diving48 must be obtained from their authors. `build_testset.py` builds the held out EgoDex episodes
   (seeds fixed in the file); `build_diving.py` and `build_diving2.py` prepare Diving48.
4. Cache features with `extract.py`, then run the probes with `make_jobs.py` and `gpu_pool.py`, as in `drivers/`.

## Tests

Run from `wam_forecast/`; tests that check results exit nonzero on failure.

```bash
BASELINE_DIR=../baseline_original python test_prop_attn.py   # proportional attention off: equal to the unpatched encoder bit for bit
BASELINE_DIR=../baseline_original python smoke_exact.py
python test_tome.py && python test_capacity.py && python test_priority_controls.py
```

`test_vidtldr.py`, `test_round_n.py`, `test_round3.py`, `test_videomae.py`, `test_current_pose.py`, and
`test_pitome_official.py` check the adapted baselines, merge schedules, the VideoMAE port, and the current pose input.
The unit tests accept `VJEPA2_ROOT` (default: `../vjepa2` next to `wam_forecast`).

## License

`vjepa2_models/` and `baseline_original/` derive from V-JEPA 2 and keep its MIT License (`LICENSE` in each folder).
