"""D1 (PROTOCOL.md): merge aware fine tuning on real DROID (droid_100, all 100 episodes), tested by CEM planning on
held out episodes.
  python d1_droid.py prep                         (decode, subsample by 4, crop as T2 -> $VJEPA2_CACHE frames + data/droid/d1_meta.npz)
  python d1_droid.py encode --part k/K            (frozen encoder tokens -> $VJEPA2_CACHE memmap, fp16)
  python -m torch.distributed.run --nproc_per_node 4 d1_droid.py train --kind plain|merge
  python d1_droid.py val                          (pick each kind's checkpoint by validation loss -> results/d1_val.json)
  python d1_droid.py eval --part k/K              (test transitions, every arm -> results/d1/eval_k.jsonl)
Split by episode index e (file order): e % 5 == 4 test, e % 10 == 3 validation, the rest train."""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "encode", "train", "val", "eval"])
ap.add_argument("--part", default="0/1")
ap.add_argument("--kind", default="merge")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--steps", type=int, default=1000)          # 2000 planned; 1000 for compute (PROTOCOL.md)
args = ap.parse_args()
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("VJEPA2_CACHE", os.path.join(ROOT, "cache"))      # as acmerge.CACHE (not imported in prep)
os.makedirs(CACHE, exist_ok=True)
FRAMES, TOK = os.path.join(CACHE, "d1_frames.npy"), os.path.join(CACHE, "d1_tok.npy")
META = os.path.join(ROOT, "data", "droid", "d1_meta.npz")
STRIDE, MOVE, CLIP = 4, 0.02, 3
SAVE_AT = (250, 500, 1000)


def frame(jpeg):                                                    # as T2: centre crop 240 x 180, resize to 256
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = im.size
    c = (w - 240) // 2
    return np.asarray(im.crop((c, 0, c + 240, h)).resize((256, 256), Image.BICUBIC))


def motion_rel(prev, cur):                                          # as T2: mean abs frame difference per patch
    d = np.abs(cur.astype(np.float32) - prev.astype(np.float32)).mean(-1)
    return d.reshape(16, 16, 16, 16).mean((1, 3)).flatten()


if args.mode == "prep":
    from tfrecord import example_pb2
    from tfrecord.reader import tfrecord_iterator
    shards = sorted(glob.glob(os.path.join(ROOT, "data", "droid", "r2d2-*.tfrecord")))
    assert len(shards) == 31, len(shards)
    fr, st, rel, off, names = [], [], [], [0], []
    for f in shards:
        for ei, rec in enumerate(tfrecord_iterator(f)):
            ex = example_pb2.Example()
            ex.ParseFromString(rec)
            Fe = ex.features.feature
            imgs = Fe["steps/observation/exterior_image_1_left"].bytes_list.value
            cart = np.array(Fe["steps/observation/cartesian_position"].float_list.value).reshape(-1, 6)
            grip = np.array(Fe["steps/observation/gripper_position"].float_list.value)[:, None]
            idx = np.arange(0, len(imgs), STRIDE)
            f_e = np.stack([frame(imgs[i]) for i in idx])
            fr.append(f_e)
            st.append(np.concatenate([cart, grip], 1)[idx].astype(np.float32))
            r = [motion_rel(f_e[max(t - 1, 0)], f_e[max(t, 1)]) for t in range(len(idx))]   # frame 0 uses 0 -> 1
            rel.append(np.stack(r).astype(np.float32))
            off.append(off[-1] + len(idx))
            names.append(f"{os.path.basename(f)}:{ei}")
        print(f, len(names), off[-1], flush=True)
    np.save(FRAMES, np.concatenate(fr))
    np.savez(META, states=np.concatenate(st), rel=np.concatenate(rel), off=np.array(off), names=np.array(names))
    print("episodes", len(names), "frames", off[-1])
    sys.exit()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import acmerge as M  # noqa: E402
import ftlib  # noqa: E402

meta = np.load(META)
ST, REL, OFF = meta["states"], meta["rel"], meta["off"]
E = len(OFF) - 1
EP = np.arange(E)
TEST, VAL = EP[EP % 5 == 4], EP[(EP % 10 == 3)]
TRAIN = np.setdiff1d(EP, np.concatenate([TEST, VAL]))
LEN = np.diff(OFF)


def ckpt_path(kind, step):
    return os.path.join(ROOT, "ckpt", f"d1_{kind}_{step}.pt")


def load_pred(path, dev):
    _, P = M.load_models(M.CKPT, dev)
    if path:
        P.load_state_dict(torch.load(path, map_location="cpu"))
    return P


if args.mode == "encode":
    k, K = map(int, args.part.split("/"))
    frames = np.load(FRAMES, mmap_mode="r")
    N = len(frames)
    if k == 0 and not os.path.exists(TOK):
        np.lib.format.open_memmap(TOK, mode="w+", dtype=np.float16, shape=(N, 256, 1408)).flush()
        open(TOK + ".ready", "w").write("ok")
    while not os.path.exists(TOK + ".ready"):
        time.sleep(5)
    mm = np.lib.format.open_memmap(TOK, mode="r+")
    enc, _ = M.load_models(M.CKPT, "cuda")
    for c, i in enumerate(range(0, N, 32)):
        if c % K == k:
            mm[i:i + 32] = M.encode(enc, frames[i:i + 32], "cuda").half().cpu().numpy()
    mm.flush()
    open(TOK + f".done{k}", "w").write("ok")
    print("encoded part", k, flush=True)

elif args.mode == "train":
    import torch.distributed as dist
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    P = load_pred("", f"cuda:{dist.get_rank()}")                 # the official predictor
    tok = np.load(TOK, mmap_mode="r")
    tr = TRAIN[LEN[TRAIN] >= CLIP]

    def sample(rng):
        e = int(rng.choice(tr))
        i = OFF[e] + int(rng.integers(0, LEN[e] - CLIP + 1))
        return tok[i:i + CLIP], ST[i:i + CLIP], REL[i]

    out = ckpt_path(args.kind, "x").replace("_x.pt", ".pt")
    ftlib.train(P, sample, out, os.path.join(ROOT, "results", f"d1_{args.kind}_log.json"), plain=args.kind == "plain",
                seed=args.seed, steps=args.steps, save_at=SAVE_AT, keep_all=True)

elif args.mode == "val":
    dev = "cuda"
    tok = np.load(TOK, mmap_mode="r")
    rng = np.random.default_rng(3)
    clips = [OFF[e] + t for e in VAL for t in range(0, LEN[e] - CLIP + 1, 2)]
    clips = sorted(rng.choice(clips, size=min(300, len(clips)), replace=False).tolist())
    res = {}
    for kind in ("plain", "merge"):
        res[kind] = {}
        for step in (0,) + SAVE_AT:
            P = load_pred("" if step == 0 else ckpt_path(kind, step), dev).eval()
            tot = []
            with torch.no_grad():
                for j, i in enumerate(clips):
                    z, a, s = ftlib.to_dev(tok[i:i + CLIP], ST[i:i + CLIP], dev)
                    if kind == "plain":
                        jl, sl = ftlib.plain_losses(P, z[None], a[None], s[None])
                    else:
                        m = M.make_merge(ftlib.METHODS[j % 5], z[0], ftlib.NS[(j // 5) % 2], rel=torch.from_numpy(REL[i]),
                                         seed=j)
                        jl, sl = ftlib.merged_losses(P, z, a, s, m)
                    tot.append(float(jl + sl))
            res[kind][step] = float(np.mean(tot))
            print(kind, step, res[kind][step], flush=True)
            del P
            torch.cuda.empty_cache()
    best = {k: min(v, key=v.get) for k, v in res.items()}
    json.dump(dict(loss=res, chosen=best, n_clips=len(clips)), open(os.path.join(ROOT, "results", "d1_val.json"), "w"),
              indent=1)
    print("chosen", best)

elif args.mode == "eval":
    k, K = map(int, args.part.split("/"))
    dev = "cuda"
    chosen = json.load(open(os.path.join(ROOT, "results", "d1_val.json")))["chosen"]
    preds = {"off": load_pred("", dev).eval()}
    for kind, tag in (("plain", "dp"), ("merge", "dm")):
        preds[tag] = load_pred("" if int(chosen[kind]) == 0 else ckpt_path(kind, chosen[kind]), dev).eval()
    ARMS = [("off_few28", "off", "identity", 256, 28), ("dp_few28", "dp", "identity", 256, 28),
            ("dm_sim64s102", "dm", "sim", 64, 102), ("dm_pool64s102", "dm", "pool", 64, 102),
            ("off_sim64s102", "off", "sim", 64, 102), ("dp_few102", "dp", "identity", 256, 102),
            ("dm_sim16", "dm", "sim", 16, 400)]
    tok = np.load(TOK, mmap_mode="r")
    cand = []
    for e in TEST:
        for t in range(LEN[e] - 1):
            i = OFF[e] + t
            if np.linalg.norm(ftlib.actions_of(ST[i:i + 2])[0, :3]) >= MOVE:
                cand.append(int(i))
    rng = np.random.default_rng(2)
    pick = sorted(rng.choice(cand, size=min(args.n_test, len(cand)), replace=False).tolist())
    out = os.path.join(ROOT, "results", "d1", f"eval_{k}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["i"] for l in open(out)} if os.path.exists(out) else set()
    print("test transitions", len(cand), "picked", len(pick), flush=True)
    for j, i in enumerate(pick):
        if j % K != k or i in done:
            continue
        z0, z1 = (F.layer_norm(torch.from_numpy(tok[x].astype(np.float32)).to(dev), (1408,)) for x in (i, i + 1))
        true = ftlib.actions_of(ST[i:i + 2])[0, :3]
        s0 = torch.from_numpy(ST[i]).to(dev)
        rec = dict(i=i, ep=int(np.searchsorted(OFF, i, side="right") - 1), true=true.tolist(), arms={})
        for name, pk, meth, n, smp in ARMS:
            m = M.make_merge(meth, z0, n, rel=torch.from_numpy(REL[i]), seed=i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            plan = M.cem_plan(preds[pk], z0, s0, z1, m, mpc=dict(samples=smp), gen=torch.Generator().manual_seed(i))
            torch.cuda.synchronize()
            rec["arms"][name] = dict(action=plan.tolist(), t=time.perf_counter() - t0,
                                     cos=float(np.dot(plan[:3], true) / (np.linalg.norm(plan[:3]) * np.linalg.norm(true) + 1e-9)))
        with open(out, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(i, {a: round(v["cos"], 2) for a, v in rec["arms"].items()}, flush=True)
