"""Round V: inference-time interventions on frozen forecasting probes, plus trajectories and attention for
visualisation. Exploratory (reused part-2 test split); no model is selected or retrained here.

  python intervene.py   -> runs/phaseA/roundV.json, runs/phaseA/roundV_viz.npz

Probes: runs/viz/<cfg>/<input>/s<seed>/model.pt, trained by train_probe.py --save_model (deterministic, so
their normal predictions reproduce the published runs; checked below).
Interventions on the DENSE probes (video only, and video plus current pose):
  swap     the video of every test clip replaced by another test clip's (fixed derangement), poses kept
  mean     every visual token replaced by the training mean token of its position
  <sel>@f  the tokens of a fraction f of the 4096 positions replaced by the training mean of that position,
           where sel = change (highest adjacent-frame feature change of the dense encoder output),
           random, hand (densest observed-hand cells), lowchange (lowest feature change)
"""
import json
import sys

import numpy as np
import torch

from aggregate import tcrit
from common import FEAT_ROOT, RUN_ROOT, W, clip_path, write_json
from train_probe import FeatureBank, ForecastProbe, errors, set_determinism

dev = torch.device("cuda")
set_determinism(0)
P = np.load(FEAT_ROOT / "poses.npz")
PAST = torch.from_numpy(P["past"]).to(dev)
FUT = torch.from_numpy(P["future"]).to(dev)
bank = FeatureBank("dense")
ok = P["avail"].copy() & np.isin(np.arange(len(P["avail"])), list(bank.where))
TEST = np.flatnonzero(P["is_test"] & ok)
TRAIN = np.flatnonzero(~P["is_test"] & ok)
HM = np.load(FEAT_ROOT / "hand_masks.npz")
VIZ = RUN_ROOT / "viz"
FRACS = (0.10, 0.25)


def load_probe(run):
    ck = torch.load(run / "model.pt", map_location="cpu")
    inp = ck["input"]
    m = ForecastProbe(inp in ("vis", "vispose", "zero"), inp in ("pose", "vispose"), d=ck["dim"], depth=ck["depth"],
                      D=ck["D"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["mu"].to(dev), ck["sd"].to(dev), inp


def past_k(inp_dir):
    if inp_dir.endswith("_k1"):                            # --pose_frames 1: only the current pose
        T0 = PAST.shape[1]
        return torch.cat([PAST[:, T0 - 1:T0].expand(-1, T0 - 1, -1, -1), PAST[:, T0 - 1:]], 1)
    return PAST


ATT = []


def hook_attention(model):
    """Record the cross-attention probabilities (mean over heads) of every block."""
    def wrap(blk):
        orig = blk._attn

        def f(q, k, v, bias=None):
            if k.shape[1] > 64 and getattr(blk, "_rec", False):   # cross-attention (visual keys)
                B, Nq, d = q.shape
                h, dh = blk.h, d // blk.h
                qq = q.view(B, Nq, h, dh).transpose(1, 2).float()
                kk = k.view(B, -1, h, dh).transpose(1, 2).float()
                a = (qq @ kk.transpose(-1, -2)) * dh ** -0.5
                if bias is not None:
                    a = a + bias[:, None, None, :].float()
                ATT.append(a.softmax(-1).mean(1).cpu())
            return orig(q, k, v, bias)
        blk._attn = f
    for blk in model.blocks:
        wrap(blk)


def predict(model, mu, sd, inp, past, rows, vis):
    pst = past[rows]
    base = pst[:, -1:] if inp in ("pose", "vispose") else mu.unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(vis, (pst - pst[:, -1:]) / sd, idx=None, compact=False)
    return base + y.float() * sd


# ------------------------------------------------------------------ training mean token per position
print("training mean token ...", flush=True)
acc = torch.zeros(4096, bank.D, dtype=torch.float64, device=dev)
for i in range(0, len(TRAIN), 32):
    acc += bank.batch(TRAIN[i:i + 32], dev).double().sum(0)
MEAN = (acc / len(TRAIN)).to(torch.float16)                                # [4096, D]


def change_score(tok):
    """tok [B,4096,D] dense encoder output (t,h,w order) -> adjacent-frame feature change [B,4096]."""
    g = tok.float().view(tok.shape[0], 16, 256, -1)
    c = torch.zeros(g.shape[:3], device=tok.device)
    c[:, 1:] = (g[:, 1:] - g[:, :-1]).norm(dim=-1)
    c[:, 0] = c[:, 1]
    return c.reshape(tok.shape[0], -1)


def select(kind, tok, rows, frac):
    n = int(round(frac * 4096))
    B = tok.shape[0]
    if kind == "change":
        return change_score(tok).topk(n, dim=1).indices
    if kind == "lowchange":
        return (-change_score(tok)).topk(n, dim=1).indices
    if kind == "hand":
        dens = torch.from_numpy(HM["density"][rows].astype(np.float32).reshape(B, -1)).to(dev)
        dens = dens + 1e-6 * torch.rand(dens.shape, generator=torch.Generator(device="cpu").manual_seed(1)).to(dev)
        return dens.topk(n, dim=1).indices
    gen = torch.Generator(device="cpu").manual_seed(int(rows[0]))
    return torch.stack([torch.randperm(4096, generator=gen)[:n] for _ in range(B)]).to(dev)


def intervene(kind, tok, rows):
    if kind == "normal":
        return tok
    if kind == "mean":
        return MEAN.unsqueeze(0).expand_as(tok).clone()
    sel, frac = kind.split("@")
    idx = select(sel, tok, rows, float(frac))
    out = tok.clone()
    out.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1]),
                 MEAN.unsqueeze(0).expand(tok.shape[0], -1, -1).gather(1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1])))
    return out


KINDS = ["normal", "swap", "mean"] + [f"{s}@{f}" for f in FRACS for s in ("change", "random", "hand", "lowchange")]
rng = np.random.RandomState(2026)
perm = rng.permutation(len(TEST))
DERANGE = TEST[np.roll(perm, 1)][np.argsort(perm)]                         # clip i gets another clip's video

res = {}
for inp_dir in ("vis", "vispose_k1"):
    past = past_k(inp_dir)
    for run in sorted((VIZ / "dense" / inp_dir).glob("s*")):
        model, mu, sd, inp = load_probe(run)
        seed = int(run.name[1:])
        pub = json.loads((run / "metrics.json").read_text())["ade_mm"]
        per = {}
        for kind in KINDS:
            e_all = []
            for i in range(0, len(TEST), 50):                         # same chunks as train_probe's eval
                rows = TEST[i:i + 50]
                src = DERANGE[i:i + 50] if kind == "swap" else rows
                tok = bank.batch(src, dev)
                tok = intervene("normal" if kind == "swap" else kind, tok, rows)
                e_all.append(errors(predict(model, mu, sd, inp, past, rows, tok), FUT[rows]).mean(dim=(1, 2)).cpu())
            per[kind] = torch.cat(e_all).numpy()
        assert abs(per["normal"].mean() - pub) < 1e-2, (run, per["normal"].mean(), pub)
        res.setdefault(inp_dir, {})[seed] = per
        print(inp_dir, seed, {k: round(float(v.mean()), 2) for k, v in per.items()}, flush=True)


def summ(per_seed, kind):
    s = sorted(per_seed)
    D = np.stack([per_seed[k][kind] - per_seed[k]["normal"] for k in s])
    d = D.mean(1)
    h = tcrit(len(s) - 1) * d.std(ddof=1) / np.sqrt(len(s))
    r = np.random.default_rng(0)
    bo = [D[r.integers(0, len(s), len(s))][:, r.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(4000)]
    return dict(d=float(d.mean()), seed_ci=[float(d.mean() - h), float(d.mean() + h)],
                clip_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                ade=float(np.mean([per_seed[k][kind].mean() for k in s])), n=len(s))


out = {inp: {k: summ(per, k) for k in KINDS} for inp, per in res.items()}


def contrast(per_seed, a, b):
    """paired a - b between two interventions (same clips, same seeds)"""
    ps = {k: {"normal": v[b], "x": v[a]} for k, v in per_seed.items()}
    return summ(ps, "x")


out["contrasts"] = {inp: {f"{s}@{f}-random@{f}": contrast(per, f"{s}@{f}", f"random@{f}")
                          for f in FRACS for s in ("change", "hand", "lowchange")} for inp, per in res.items()}
np.savez_compressed(W / "runs" / "phaseA" / "roundV_perclip.npz",
                    **{f"{inp}|s{sd}|{k}": v for inp, per in res.items() for sd, pk in per.items() for k, v in pk.items()})

# retrained references for the same inputs (published runs): pose only and dense
REF = {"vis": {}, "vispose_k1": {}}
for inp_dir, pose_dir in (("vispose_k1", "pose_k1"),):
    pr = [json.loads(f.read_text())["ade_mm"] for f in (RUN_ROOT / "test_main" / "none" / pose_dir).glob("s*/metrics.json")]
    REF[inp_dir]["pose_only_retrained"] = float(np.mean(pr)) if pr else None
out["retrained_reference"] = REF

# ------------------------------------------------------------------ trajectories and attention for figures
print("visualisation data ...", flush=True)
gain = res["vispose_k1"][0]["swap"] - res["vispose_k1"][0]["normal"]          # clips where vision matters most
pick = TEST[np.argsort(-gain)[:8]]
viz = dict(rows=pick, past=P["past"][pick], future=P["future"][pick], keys=P["keys"][pick])
frames, Ks = [], []
for r in pick:
    k = str(P["keys"][r])
    with np.load(clip_path((k.split("/")[0], k.split("/")[1]))) as z:
        frames.append(z["imgs"][[0, 15, 31]])
        Ks.append(z["cam_int"].reshape(3, 3))
viz["frames"], viz["cam_int"] = np.stack(frames), np.stack(Ks)
for cfg in ("dense", "kbsmpa_L9", "wampa_L9", "handmultpa_L9"):
    run = VIZ / cfg / "vispose_k1" / "s0"
    if not (run / "model.pt").exists():
        continue
    model, mu, sd, inp = load_probe(run)
    hook_attention(model)
    fb = bank if cfg == "dense" else FeatureBank(cfg)
    tok = fb.batch(pick, dev)
    for blk in model.blocks:
        blk._rec = True
    ATT.clear()
    pred = predict(model, mu, sd, inp, past_k("vispose_k1"), pick, tok)
    viz[f"pred_{cfg}"] = pred.cpu().numpy()
    viz[f"att_{cfg}"] = torch.stack(ATT).numpy().astype(np.float16)            # [blocks, B, 32, 4096+32]
    if cfg == "dense":
        viz["change"] = change_score(tok).cpu().numpy().astype(np.float16)
        viz["pred_dense_swap"] = predict(model, mu, sd, inp, past_k("vispose_k1"), pick,
                                         bank.batch(DERANGE[np.searchsorted(TEST, pick)], dev)).cpu().numpy()
    if fb is not bank:
        del fb
    torch.cuda.empty_cache()
# attention share on observed-hand cells, per probe (all test clips, seed 0, mean over blocks and queries)
shares = {}
for cfg in ("dense", "kbsmpa_L9", "wampa_L9", "handmultpa_L9"):
    run = VIZ / cfg / "vispose_k1" / "s0"
    if not (run / "model.pt").exists():
        continue
    model, mu, sd, inp = load_probe(run)
    hook_attention(model)
    for blk in model.blocks:
        blk._rec = True
    fb = bank if cfg == "dense" else FeatureBank(cfg)
    hs, vs, area = [], [], []
    for i in range(0, len(TEST), 25):
        rows = TEST[i:i + 25]
        ATT.clear()
        predict(model, mu, sd, inp, past_k("vispose_k1"), rows, fb.batch(rows, dev))
        a = torch.stack(ATT).mean(0).mean(1)                                   # [B, 4096+32]
        hand = torch.from_numpy(HM["mask"][rows].reshape(len(rows), -1))
        vis_mass = a[:, :4096].sum(1)
        hs.append(((a[:, :4096] * hand).sum(1) / vis_mass).numpy())
        vs.append(vis_mass.numpy())
        area.append(hand.float().mean(1).numpy())
    shares[cfg] = dict(hand_share_of_visual=float(np.concatenate(hs).mean()), visual_share=float(np.concatenate(vs).mean()),
                       hand_area=float(np.concatenate(area).mean()))
    if fb is not bank:
        del fb
out["attention"] = shares
write_json(out, W / "runs" / "phaseA" / "roundV.json")
np.savez_compressed(W / "runs" / "phaseA" / "roundV_viz.npz", **viz)
for inp, v in out.items():
    if inp in ("retrained_reference", "attention", "contrasts"):
        print(inp, v)
        continue
    for k, s in v.items():
        print(f"{inp:11s} {k:14s} ADE {s['ade']:.2f}  delta {s['d']:+.2f} seed[{s['seed_ci'][0]:+.2f},{s['seed_ci'][1]:+.2f}] "
              f"clip[{s['clip_ci'][0]:+.2f},{s['clip_ci'][1]:+.2f}]")
