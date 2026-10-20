#!/usr/bin/env python3
"""
One-off analysis (not yet wired into the production script/tables): computes
every rank-based statistic (Wilcoxon p, Holm p, reject flag, r_rb) ONCE, on
the raw 1-5 judge scale -- no (s-1)/4 or s/5 normalization at all -- since
ranks are invariant to any positive affine rescaling of the input. Reuses
the real analyze_causation_judge_120_stats.py module for data loading,
index re-mapping, and the Holm/BH correction functions, so this is not a
reimplementation of that logic, only a substitution of the per-argument
scoring function.

Non-rank statistics (means, SEM, Delta, CI, t-test, d_z) are NOT recomputed
here -- those are read from the already-produced (s-1)/4 run
(results/out/causation_judge_120_stats/causation_judge_120_summary.json).
"""
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent.parent

spec = importlib.util.spec_from_file_location(
    "cstats", str(ROOT / "scripts/analyze_causation_judge_120_stats.py"))
cstats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cstats)


def per_arg_scores_raw(pairs):
    """Per-argument mean of the RAW judge score s in {1..5}, no normalization."""
    by = defaultdict(list)
    for pr in pairs or []:
        if pr.get("model_missing") or pr.get("gold_missing"):
            continue
        j = pr.get("judge")
        if not isinstance(j, dict):
            continue
        se = j.get("semantic_equivalence")
        if se is None:
            continue
        by[int(pr["arg_index"])].append(se)
    return {k: float(np.mean(v)) for k, v in by.items() if v}


def wilcoxon_raw(d_rounded):
    """Exactly mirrors wilcoxon_with_effect's zero handling / r_rb computation,
    and calls scipy.stats.wilcoxon with the same (default) zero_method,
    correction, alternative, method as the original script -- applied to the
    already-rounded, already-differenced d, matching the original's
    stats.wilcoxon(x, y) two-sample call exactly (wilcoxon(x,y) computes
    d=x-y internally and is equivalent to wilcoxon(d) for the same d)."""
    nonzero = d_rounded[d_rounded != 0]
    n_nonzero = len(nonzero)
    n_zero = int((d_rounded == 0).sum())
    if n_nonzero < 4:
        return {"n_nonzero": n_nonzero, "n_zero": n_zero, "W": float("nan"),
                "p": float("nan"), "rank_biserial_r": float("nan")}
    ranks = sps.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    r_rb = (w_pos - w_neg) / (w_pos + w_neg)
    try:
        stat, p = sps.wilcoxon(d_rounded)  # zero_method='wilcox', correction=False,
        stat, p = float(stat), float(p)    # alternative='two-sided', method='auto' (all defaults)
    except ValueError:
        stat, p = float("nan"), float("nan")
    return {"n_nonzero": n_nonzero, "n_zero": n_zero, "W": stat, "p": p,
            "rank_biserial_r": float(r_rb)}


def main():
    cstats.per_arg_scores = per_arg_scores_raw
    data = cstats.compute_all_condition_scores()

    results = {}
    for name, cond_x, cond_y in cstats.CONTRASTS:
        for disp, _, _, _ in cstats.MODELS:
            sx, sy = data[disp][cond_x], data[disp][cond_y]
            common = sorted(set(sx) & set(sy))
            n = len(common)
            key = (name, disp)
            if n < 4:
                results[key] = {"model": disp, "contrast": name, "n": n, "skipped": True}
                continue
            x = np.array([sx[k] for k in common])
            y = np.array([sy[k] for k in common])
            d = np.round(x - y, 10)
            wx = wilcoxon_raw(d)
            results[key] = {"model": disp, "contrast": name, "skipped": False,
                             "n": n, "mean_diff_raw_scale": float(d.mean()),
                             **wx}

    # Holm/BH correction within each contrast, across the 8 models -- identical
    # logic/function to the production script.
    for name, _, _ in cstats.CONTRASTS:
        keys = [(name, disp) for disp, _, _, _ in cstats.MODELS if not results[(name, disp)]["skipped"]]
        w_ps = [results[k]["p"] for k in keys]
        w_holm, w_holm_rej = cstats.holm_bonferroni(w_ps, 0.05)
        for i, k in enumerate(keys):
            results[k]["p_holm"] = float(w_holm[i])
            results[k]["p_holm_reject"] = bool(w_holm_rej[i])

    out_path = ROOT / "results/out/wilcoxon_raw_scale.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([results[(n, d)] for n, _, _ in cstats.CONTRASTS for d, *_ in cstats.MODELS], f, indent=2)
    print(f"Wrote {out_path}")

    print(f"\n{'contrast':20s} {'model':10s} {'n':>4s} {'n_zero':>6s} {'p':>12s} {'p_holm':>12s} {'reject':>7s} {'r_rb':>8s}")
    for name, _, _ in cstats.CONTRASTS:
        for disp, _, _, _ in cstats.MODELS:
            r = results[(name, disp)]
            print(f"{name:20s} {disp:10s} {r['n']:4d} {r.get('n_zero','-'):>6} "
                  f"{r['p']:12.6f} {r['p_holm']:12.6f} {str(r['p_holm_reject']):>7s} {r['rank_biserial_r']:8.4f}")

    print("\n=== zero-diff counts per contrast (summed across 8 models) ===")
    for name, _, _ in cstats.CONTRASTS:
        total_zero = sum(results[(name, disp)]["n_zero"] for disp, _, _, _ in cstats.MODELS)
        total_n = sum(results[(name, disp)]["n"] for disp, _, _, _ in cstats.MODELS)
        print(f"  {name:20s} {total_zero} zero / {total_n} total")

    print("\n=== headline counts ===")
    for cname, target in [("oracle_vs_noise", 6), ("noise_vs_vanilla", 5), ("oracle_vs_vanilla", 2)]:
        rows = [results[(cname, disp)] for disp, _, _, _ in cstats.MODELS]
        n_sig = sum(1 for r in rows if r["p_holm_reject"])
        pos = sum(1 for r in rows if r["mean_diff_raw_scale"] > 0)
        print(f"  {cname:20s} n_holm_sig={n_sig} (target {target})  positive_direction={pos}/8")


if __name__ == "__main__":
    main()
