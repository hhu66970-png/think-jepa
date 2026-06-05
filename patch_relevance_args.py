#!/usr/bin/env python
"""加 relevance_source/lambda/path 从训练命令到合并配置的透传(默认 none/空=V1,不影响现有跑)。"""
# --- thinker_train.py ---
p1 = "cache_train/thinker_train.py"
s = open(p1).read()
a1 = '        "lambda_motion": float(getattr(args, "dense_jepa_lambda_motion", 0.7)),'
add1 = a1 + (
    '\n        "relevance_source": str(getattr(args, "dense_jepa_relevance_source", "none")),'
    '\n        "relevance_lambda": float(getattr(args, "dense_jepa_relevance_lambda", 1.0)),'
    '\n        "relevance_path": str(getattr(args, "dense_jepa_relevance_path", "")),'
)
if '"relevance_source": str(getattr(args, "dense_jepa_relevance_source"' not in s:
    assert a1 in s, "dict anchor not found"
    s = s.replace(a1, add1, 1)
a2 = ('    parser.add_argument(\n'
      '        "--dense_jepa_importance_source",\n'
      '        type=str,\n'
      '        default="none",\n'
      '        help="importance score source: none, norm, motion, norm_motion, or qk_global_hidden",\n'
      '    )')
add2 = a2 + (
    '\n    parser.add_argument(\n'
    '        "--dense_jepa_relevance_source", type=str, default="none",\n'
    '        help="WAM relevance signal: none/motion (V1, in-place) OR predictor_saliency/handjoint (V2, +--dense_jepa_relevance_path)",\n'
    '    )\n'
    '    parser.add_argument(\n'
    '        "--dense_jepa_relevance_lambda", type=float, default=1.0,\n'
    '        help="WAM gate strength lambda in [0,1] (0=off==K-BSM)",\n'
    '    )\n'
    '    parser.add_argument(\n'
    '        "--dense_jepa_relevance_path", type=str, default="",\n'
    '        help="path to a precomputed per-token relevance prior .npz (V2)",\n'
    '    )'
)
if '--dense_jepa_relevance_source' not in s:
    assert a2 in s, "arg-def anchor not found"
    s = s.replace(a2, add2, 1)
open(p1, "w").write(s)

# --- scripts/train.sh ---
p2 = "scripts/train.sh"
t = open(p2).read()
ae = 'DENSE_JEPA_MERGE_STRATEGY="${DENSE_JEPA_MERGE_STRATEGY:-local_2x2_same_time}"'
add_e = ae + (
    '\nDENSE_JEPA_RELEVANCE_SOURCE="${DENSE_JEPA_RELEVANCE_SOURCE:-none}"'
    '\nDENSE_JEPA_RELEVANCE_LAMBDA="${DENSE_JEPA_RELEVANCE_LAMBDA:-1.0}"'
    '\nDENSE_JEPA_RELEVANCE_PATH="${DENSE_JEPA_RELEVANCE_PATH:-}"'
)
if 'DENSE_JEPA_RELEVANCE_SOURCE' not in t:
    assert ae in t, "train.sh env anchor not found"
    t = t.replace(ae, add_e, 1)
aa = '  --dense_jepa_importance_source "${DENSE_JEPA_IMPORTANCE_SOURCE}"'
add_a = aa + (
    '\n  --dense_jepa_relevance_source "${DENSE_JEPA_RELEVANCE_SOURCE}"'
    '\n  --dense_jepa_relevance_lambda "${DENSE_JEPA_RELEVANCE_LAMBDA}"'
    '\n  --dense_jepa_relevance_path "${DENSE_JEPA_RELEVANCE_PATH}"'
)
if '--dense_jepa_relevance_source' not in t:
    assert aa in t, "train.sh arg anchor not found"
    t = t.replace(aa, add_a, 1)
open(p2, "w").write(t)
print("patched thinker_train.py + train.sh (relevance passthrough; defaults none/'' = V1)")
