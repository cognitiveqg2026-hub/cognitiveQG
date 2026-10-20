import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
import json, os, math, sys
from fractions import Fraction as F
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = str(DATA)
DEV16 = os.path.join(ROOT, "causation/raw_judge_dev1-6")
DEV712 = os.path.join(ROOT, "causation/raw_judge_dev7-12")
BASE = os.path.join(ROOT, "baseline_complete_dev1-12")
MODELS = [("LLaMA-2","llama2","llama2","llama2"),("LLaMA-3","llama3","llama3","llama3"),
          ("Mistral","mistral","mistral","mistral"),("OLMo","olmo","olmo","olmo"),
          ("Qwen3","qwen3","qwen3","qwen3"),("Qwen2.5","qwen2","qwen","qwen"),
          ("GPT-4o","gpt4o","gpt-4o","gpt-4o"),("GPT-5.5","gpt5","gpt-5","gpt-5.5")]
CONDS = ["vanilla","baseline","noise","oracle"]
CONTRASTS = [("oracle_vs_noise","oracle","noise"),("oracle_vs_vanilla","oracle","vanilla"),
             ("noise_vs_vanilla","noise","vanilla"),("oracle_vs_baseline","oracle","baseline"),
             ("noise_vs_baseline","noise","baseline"),("vanilla_vs_baseline","vanilla","baseline")]

def load(path):
    with open(path, encoding="utf-8") as f: return json.load(f)["scored_pairs"]

def per_arg_raw(pairs):
    """{arg_index: exact Fraction mean of raw s in 1..5}"""
    by = defaultdict(list)
    for pr in pairs:
        if pr.get("model_missing") or pr.get("gold_missing"): continue
        j = pr.get("judge")
        if not isinstance(j, dict) or j.get("semantic_equivalence") is None: continue
        by[int(pr["arg_index"])].append(int(j["semantic_equivalence"]))
    return {k: F(sum(v), len(v)) for k, v in by.items() if v}

def cond_scores(d16, d712, bkey, cond):
    if cond == "baseline":
        return per_arg_raw(load(os.path.join(BASE, bkey, "judge_run2_judge.json")))
    if cond == "noise":
        p16 = os.path.join(DEV16, f"{d16}_noise_60args_judge.json")
        p712 = os.path.join(DEV712, f"{d712}_noise_v2_60args_judge.json")
    else:
        p16 = os.path.join(DEV16, f"{d16}_{cond}_60args_judge.json")
        p712 = os.path.join(DEV712, f"{d712}_{cond}_60args_judge.json")
    out = {}
    for k, v in per_arg_raw(load(p16)).items(): out[k - 1] = v
    for k, v in per_arg_raw(load(p712)).items(): out[k + 59] = v
    return out

def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, min((m - i) * ps[idx], 1.0)); adj[idx] = run
    return adj

def exact_wilcoxon(d_frac):
    """Wilcoxon on exact differences: zeros and ties detected exactly (Fractions),
    then passed to scipy as floats (equal Fractions -> identical floats)."""
    d = [x for x in d_frac]
    nz = [x for x in d if x != 0]
    # exact rank-biserial with exact tie handling
    absvals = sorted(set(abs(x) for x in nz))
    # average ranks for ties
    allabs = sorted(abs(x) for x in nz)
    rank_of = {}
    i = 0
    while i < len(allabs):
        j = i
        while j < len(allabs) and allabs[j] == allabs[i]: j += 1
        rank_of[allabs[i]] = F(i + 1 + j, 2)  # average of ranks i+1..j
        i = j
    wpos = sum(rank_of[abs(x)] for x in nz if x > 0)
    wneg = sum(rank_of[abs(x)] for x in nz if x < 0)
    rrb = float((wpos - wneg) / (wpos + wneg)) if nz else float("nan")
    darr = np.array([float(x) for x in d])
    stat, p = stats.wilcoxon(darr)  # defaults: zero_method='wilcox', correction=False, method='auto'
    return float(p), rrb, len(nz)

def to_scale(x, scale):
    return (x - 1) / 4 if scale == "s-1/4" else x / 5

def run(scale):
    data = {disp: {c: cond_scores(a, b, k, c) for c in CONDS} for disp, a, b, k in MODELS}
    means = []
    for disp, *_ in MODELS:
        for c in CONDS:
            v = np.array([float(to_scale(x, scale)) for x in data[disp][c].values()])
            means.append((disp, c, len(v), v.mean(), v.std(ddof=0) / math.sqrt(len(v))))
    rows = []
    for name, cx, cy in CONTRASTS:
        block = []
        for disp, *_ in MODELS:
            sx, sy = data[disp][cx], data[disp][cy]
            common = sorted(set(sx) & set(sy))
            dfr = [to_scale(sx[k], scale) - to_scale(sy[k], scale) for k in common]  # exact Fractions
            d = np.array([float(x) for x in dfr]); n = len(d)
            md = d.mean(); sd = d.std(ddof=1); se = sd / math.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            tp = float(stats.ttest_1samp(d, 0).pvalue)
            wp, rrb, nnz = exact_wilcoxon(dfr)
            block.append(dict(contrast=name, model=disp, n=n, mean_diff=md, ci_lo=md - tcrit * se,
                              ci_hi=md + tcrit * se, dz=md / sd, t_p=tp, w_p=wp, rrb=rrb, n_nonzero=nnz,
                              n_zero=n - nnz))
        for key in ("t_p", "w_p"):
            adj = holm([b[key] for b in block])
            for b, a in zip(block, adj): b[key + "_holm"] = a; b[key + "_sig"] = a <= 0.05
        rows += block
    return means, rows

if __name__ == "__main__":
    import csv
    outdir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else str(OUT / "causation_exact_s-1over4")
    os.makedirs(outdir, exist_ok=True)
    res = {s: run(s) for s in ("s-1/4", "s/5")}
    # scale-invariance check for rank stats
    a, b = res["s-1/4"][1], res["s/5"][1]
    bad = [(x["contrast"], x["model"]) for x, y in zip(a, b)
           if x["w_p"] != y["w_p"] or x["rrb"] != y["rrb"] or x["w_p_sig"] != y["w_p_sig"]]
    print("rank-stat differences between scales:", len(bad), bad[:5])
    means, rows = res["s-1/4"]
    with open(os.path.join(outdir, "condition_means_s-1over4.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","condition","n","mean","sem"]); w.writerows(means)
    with open(os.path.join(outdir, "contrasts_s-1over4.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("written", outdir)
