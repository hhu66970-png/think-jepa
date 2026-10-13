import datetime, sys
ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
open(sys.argv[1], "a").write(f"""- {ts}, ROUNDS S AND DV SPECIFICATION (seventh review; written before any feature or run of these rounds).
  ROUND S (does episode wise uniform sampling hide motion?). (a) Descriptive, existing runs: the motion and hand
  shuffle contrasts in tertiles of the sampling stride (video frames between consecutive sampled frames,
  clip_covariates.py stride_frames; part 2 median 1.74). (b) The 807 held out episodes of the 13 part 2 tasks with at
  least 64 frames (official test archive, build_fps.py), sampled two ways: fixed30 (64 consecutive frames at 30 fps,
  centred) and uniform (np.linspace over the episode). Per sampling, its own feature root, CV over the 727 non held out
  episodes (five folds, seeds 0-3), current pose: kbsmpa_L9, wampa_L9, wamshufpa_L9, flowpa_L9, flowshufpa_L9, pospa_L9,
  kbsmpa_L12, wampa_L12, wamshufpa_L12, pose only; video only: kbsmpa_L9, wampa_L9, wamshufpa_L9 (520 runs; no dense
  tokens, to save disk; vision headroom is read as copy last or pose only minus kbsmpa_L9). Primary S1, per sampling:
  wampa_L9 - wamshufpa_L9 and flowpa_L9 - flowshufpa_L9, current pose, 95% clip interval; below 0 under fixed30 but
  not under uniform would mean the sampling hid a motion signal. Secondary: the same at L12 and video only; each minus
  kbsmpa; pospa - kbsmpa. Analysis roundS_analysis.py, written before the runs.
  ROUND DV (second task, action recognition). Diving48 V2, the 1970 clips of its official test split (build_diving.py;
  mteb/diving48v2 copy of the official release): 32 frames evenly over the dive segment,
  frozen V-JEPA 2 ViT-L as everywhere, classified by an attentive probe of the same design (train_cls.py: 4 queries,
  3 blocks, restore readout, 60 epochs, lr 2e-4) in five fold CV over the 1970 clips, seeds 0-3. Configs: dense,
  kbsmpa_L9, kbsmpa_L12, pospa_L9, randmpa_L9, in16f, wampa_L9, wamshufpa_L9, wampa_L12, wamshufpa_L12, flowpa_L9,
  flowshufpa_L9, vidtldrpa_L9, vidtldrshufpa_L9, fmpa_L9, vidtldrnat_L9, fmnat_L9 (340 runs). Metric: per clip top 1
  accuracy (primary, in percentage points) and cross entropy (secondary). Validity: kbsmpa_L12 minus dense accuracy
  with its 95% clip interval below 0 (compression costs on this task). Primary DV1 (motion): wampa_L9 - wamshufpa_L9;
  DV2 (saliency): vidtldrpa_L9 - vidtldrshufpa_L9; DV3 (flow): flowpa_L9 - flowshufpa_L9; 95% clip intervals of the
  accuracy difference: above 0 means the signal beats its shuffle on this task. Secondary: L12 motion; each signal
  minus kbsmpa; pospa, randmpa, in16f minus kbsmpa_L9; saliency effect early minus late ((vidtldrnat - fmnat) -
  (vidtldrpa - fmpa)). Analysis roundDV_analysis.py, written before the runs.
""")
print(ts)
