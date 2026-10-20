import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
import json, re, sys, os
import numpy as np
R = str(REPO)
sys.path.insert(0, R + "/scripts")
from parse_model_output import parse_model_output

def norm_str(v):
    if isinstance(v, list): v = v[0] if v else ""
    return re.sub(r"\s+", " ", str(v or "").strip().lower())
def norm_list(v):
    if isinstance(v, list): return {x.strip().lower() for x in v if isinstance(x,str) and x.strip()}
    if isinstance(v, str) and v.strip(): return {x.strip().lower() for x in re.split(r"[|;,]", v) if x.strip()}
    return set()
def macro_f1_categorical(golds, preds):   # copied from compute_field_metrics_latex_rescaled.py
    classes = set(golds) | set(preds); f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        if tp + fp == 0 or tp + fn == 0: continue
        pr = tp / (tp + fp); rc = tp / (tp + fn)
        if pr + rc == 0: continue
        f1s.append((2 * pr * rc / (pr + rc), tp + fn))
    if not f1s: return 0.0
    total = sum(s for _, s in f1s)
    return sum(f * s for f, s in f1s) / total if total else sum(f for f, _ in f1s) / len(f1s)
def weighted_f1_multilabel(gold_sets, pred_sets):  # same as generate_baseline_field_table_dev3-12_a2.py
    labels = set()
    for s in gold_sets: labels |= s
    for s in pred_sets: labels |= s
    if not labels: return 0.0
    num = den = 0.0
    for c in labels:
        tp = sum(1 for g, p in zip(gold_sets, pred_sets) if c in g and c in p)
        fp = sum(1 for g, p in zip(gold_sets, pred_sets) if c not in g and c in p)
        fn = sum(1 for g, p in zip(gold_sets, pred_sets) if c in g and c not in p)
        support = tp + fn
        if support == 0: continue
        f1 = 0.0
        if tp > 0:
            pr, rc = tp / (tp + fp), tp / (tp + fn); f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        num += f1 * support; den += support
    return num / den if den else 0.0

CAT = ["stanceTarget","knowledgeDomain","reasoningType","hasAssumption","primaryDomain","alternativeType",
       "inferenceStrength","logicalFallacy","targetExplanation","reasoningStructure","span1","span2","span3",
       "changeDecision","revisionType"]
WF1 = ["credibilityFactors","trustworthiness","biasDetection","heuristicDetection"]
ROWS = ["stanceTarget","knowledgeDomain","reasoningType","hasAssumption","primaryDomain","alternativeType",
        "inferenceStrength","credibilityFactors","logicalFallacy","trustworthiness","targetExplanation",
        "reasoningStructure","span1","span2","span3","biasDetection","heuristicDetection","changeDecision","revisionType"]

def score(field, gold_rows, pred_rows):
    if field in WF1:
        gs, ps = [], []
        for g, p in zip(gold_rows, pred_rows):
            if p == "UNMATCHED": continue
            gl = norm_list(g.get(field, ""))
            if not gl: continue
            gs.append(gl); ps.append(norm_list(p.get(field, "")))
        return (weighted_f1_multilabel(gs, ps) * 100, len(gs)) if gs else (None, 0)
    golds, preds = [], []
    for g, p in zip(gold_rows, pred_rows):
        if p == "UNMATCHED": continue
        gv = norm_str(g.get(field, ""))
        if gv in ("n/a", ""): continue
        golds.append(gv); preds.append(norm_str(p.get(field, "")))
    return (macro_f1_categorical(golds, preds) * 100, len(golds)) if golds else (None, 0)

A1 = json.load(open(R + "/data/a1_dev1-12_gold.json"))["annotations"]
A2 = json.load(open(R + "/data/a2_dev1-12_gold.json"))["annotations"]
a1 = {int(a["argumentIndex"]): a for a in A1}; a2 = {int(a["argumentIndex"]): a for a in A2}
idx = sorted(a1)
gold_rows = [a1[i] for i in idx]

mode = sys.argv[1] if len(sys.argv) > 1 else "human"
if mode == "validate":
    model = sys.argv[2]
    d = json.load(open(R + f"/data/baseline_complete_dev1-12/raw/{model}/raw_output.json"))
    by = {}
    for s in d["per_sample"]:
        by[int(s["argument_index"])] = s
    for f in ROWS:
        vals = []
        for seed in ("44", "45", "46"):
            preds = []
            for i in idx:
                txt = by.get(i, {}).get("per_seed", {}).get(seed, {}).get("generated_text", "")
                preds.append(parse_model_output(txt) if txt else "UNMATCHED")
            v, n = score(f, gold_rows, preds)
            if v is not None: vals.append(v)
        print(f"{f:22s} {np.mean(vals):5.1f} ±{np.std(vals):.1f}")
else:
    # Human ceiling, both directions: A2 scored against A1 gold, and A1 scored against A2 gold.
    out = {}
    for direction, g, p in (("A2vsA1", a1, a2), ("A1vsA2", a2, a1)):
        gr = [g[i] for i in idx]; pr = [p.get(i, "UNMATCHED") for i in idx]
        out[direction] = {}
        for f in ROWS:
            v, n = score(f, gr, pr)
            out[direction][f] = {"wtf1": v, "n": n}
            print(f"{direction} {f:22s} {'--' if v is None else f'{v:5.1f}'}  n={n}")
    json.dump(out, open(str(OUT / "human_wtf1_dev1-12.json"), "w"), indent=2)
    print("wrote", OUT / "human_wtf1_dev1-12.json")
