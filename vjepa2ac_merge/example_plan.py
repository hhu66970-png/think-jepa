"""The method on any robot: one control step of goal image planning with merged tokens and fewer CEM samples.

  planner = MergedPlanner(pred_ckpt="ckpt/your_merge_aware_predictor.pt")      # method sim, n = 64, 102 samples
  planner.set_goal(goal_frame)
  action = planner.act(frame, state)                                           # send to the Cartesian controller

frame, goal_frame: uint8 [256, 256, 3] camera images, preprocessed exactly as in fine tuning (crop, resize).
state: [7] end effector pose in the V-JEPA 2-AC convention, robot base frame: x, y, z (m), roll, pitch, yaw (extrinsic
xyz Euler), gripper closedness 0 open .. 1 closed. action: [7] = xyz displacement (m, at most 0.05 per axis), zero
rotation, gripper delta.

Use a merge aware fine tuned predictor (s4_train.py, or ftlib.train on your own data). With the official or a plainly
fine tuned predictor, merged planning fails (H5 in PROTOCOL.md: 0.18 vs 0.92 success).

Smoke test with random frames (prints the action and the time per step for the method and for the plain baselines):
  python example_plan.py [--pred_ckpt ckpt/your_merge_aware_predictor.pt]
"""
import argparse
import time

import numpy as np
import torch

import acmerge as M


class MergedPlanner:
    def __init__(self, pred_ckpt="", ckpt=M.CKPT, method="sim", n=64, samples=102, cem_steps=10, rollout=2,
                 device="cuda", seed=0):
        """method: sim | pool | gate | rand | identity (no merging); n: groups per frame (64 or 16 as trained)."""
        self.dev = device
        self.enc, self.P = M.load_models(ckpt, device)
        if pred_ckpt:
            self.P.load_state_dict(torch.load(pred_ckpt, map_location="cpu"))
        self.method, self.n = method, n
        self.mpc = dict(samples=samples, cem_steps=cem_steps, rollout=rollout)
        self.gen = torch.Generator().manual_seed(seed)
        self.zg = None
        self.t = 0

    def set_goal(self, goal_frame):
        self.zg = M.encode(self.enc, goal_frame[None], self.dev)[0]                    # [256, 1408], full tokens

    @torch.no_grad()
    def act(self, frame, state, rel=None):
        """rel [256]: per patch relevance, needed only for method gate (e.g. a bump around the gripper)."""
        z = M.encode(self.enc, frame[None], self.dev)[0]                               # frozen encoder, once per step
        m = M.make_merge(self.method, z, self.n, rel=rel, seed=self.t)                 # 256 tokens -> n groups
        s = torch.as_tensor(np.asarray(state, np.float32), device=self.dev)
        self.t += 1
        return M.cem_plan(self.P, z, s, self.zg, m, mpc=self.mpc, gen=self.gen)       # restore readout, L1 to goal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_ckpt", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    frame, goal = (rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(2))
    state = np.array([0.5, 0.0, 0.26, -np.pi, 0.0, 0.0, 0.0], np.float32)
    pl = MergedPlanner(args.pred_ckpt)
    pl.set_goal(goal)
    for name, method, n, samples, iters in (("METHOD sim n=64, 102 x 10", "sim", 64, 102, 10),
                                            ("plain 256 tokens, 28 x 10", "identity", 256, 28, 10),
                                            ("plain 256 tokens, 102 x 10", "identity", 256, 102, 10)):
        pl.method, pl.n, pl.mpc = method, n, dict(samples=samples, cem_steps=iters, rollout=2)
        pl.act(frame, state)                                                            # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = pl.act(frame, state)
        torch.cuda.synchronize()
        print(f"{name:28s} {time.perf_counter() - t0:5.2f} s per step, "
              f"{M.predictor_tokens(n, 2) * samples / 1000:5.1f}k predictor tokens per CEM iteration, "
              f"action xyz {np.round(a[:3], 3)}")
