#!/usr/bin/env python3
"""
Human-vs-LLM-judge agreement analysis for the dev1-12 semantic-equivalence study.

Inputs:
  --ratings-file   the Export JSON downloaded from the semeq_kit HTML tool
                    ({meta, annotations: {item_key: {rating, ts}}})
  --sample-file    the sampling manifest from sample_dev1-12_semantic_equivalence.py
                    ({meta, items: [...]}), which carries each item's existing
                    LLM-judge semantic_equivalence score

Joins on item_key (<model>_<arg_index>_<question_position>). Computes, over the
POOLED set of rated items (primary analysis, per the study design):
  - Spearman correlation (scipy.stats.spearmanr), bootstrap 95% CI, permutation p
    -- pattern/helpers mirrored from scripts/analyze_judge_vs_metrics.py
  - Weighted (quadratic) Cohen's kappa (sklearn.metrics.cohen_kappa_score)
  - ICC(3,1) -- two-way mixed-effects, single measures, consistency (hand-rolled;
    no existing repo implementation, no new dependency)

Per-model breakdowns (n=50 each, if the full stratified sample was drawn) are
reported descriptively as a ROBUSTNESS CHECK ONLY -- they are not independently
powered to a +-5% margin of error (that would need ~163-187 items per model; see
the sampling script).

Usage:
    python scripts/analyze_semeq_human_judge_agreement.py \
        --ratings-file path/to/semeq_..._ratings_<annotator>_<stamp>.json \
        --sample-file  data/annotation_sampling/dev1-12_semeq_sample_seed44.json \
        --output       results/annotation_sampling/dev1-12_semeq_agreement
"""
import argparse
import csv
import json
import os

import numpy as np

N_BOOTSTRAP = 1000
ALPHA = 0.05


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    from scipy.stats import spearmanr
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)


def _bootstrap_ci(x, y, corr_fn, n=N_BOOTSTRAP, alpha=ALPHA):
    mask = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n):
        idx = rng.integers(0, len(xm), size=len(xm))
        boot.append(corr_fn(xm[idx], ym[idx]))
    boot = np.array(boot)
    lo = float(np.nanpercentile(boot, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def _permutation_p(x, y, corr_fn, n=1000):
    mask = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return float("nan")
    observed = corr_fn(xm, ym)
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n):
        ym_perm = rng.permutation(ym)
        if abs(corr_fn(xm, ym_perm)) >= abs(observed):
            count += 1
    return count / n


def weighted_kappa(human: np.ndarray, judge: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    h = human.astype(int)
    j = judge.astype(int)
    return float(cohen_kappa_score(h, j, weights="quadratic"))


def icc_3_1(human: np.ndarray, judge: np.ndarray) -> float:
    """ICC(3,1): two-way mixed effects, single measures, consistency.
    Standard closed-form ANOVA calculation for n subjects x k=2 raters.
    """
    x = np.stack([human, judge], axis=1).astype(float)  # (n, k)
    n, k = x.shape
    if n < 2:
        return float("nan")
    grand_mean = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_total = ((x - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    if denom == 0:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


def load_pairs(ratings_file, sample_file):
    ratings = json.load(open(ratings_file, encoding="utf-8"))
    sample = json.load(open(sample_file, encoding="utf-8"))

    judge_by_key = {}
    model_by_key = {}
    for it in sample["items"]:
        judge_by_key[it["item_key"]] = it.get("judge_semantic_equivalence")
        model_by_key[it["item_key"]] = it["model"]

    rows = []
    for key, rec in ratings.get("annotations", {}).items():
        human_v = rec.get("rating")
        judge_v = judge_by_key.get(key)
        if human_v is None or judge_v is None:
            continue
        rows.append({"item_key": key, "model": model_by_key.get(key, "?"),
                     "human": float(human_v), "judge": float(judge_v)})
    return rows


def compute_scope(rows):
    human = np.array([r["human"] for r in rows], dtype=float)
    judge = np.array([r["judge"] for r in rows], dtype=float)
    n = len(rows)
    sp = _spearman(human, judge)
    sp_lo, sp_hi = _bootstrap_ci(human, judge, _spearman)
    p_val = _permutation_p(human, judge, _spearman)
    wk = weighted_kappa(human, judge) if n >= 2 else float("nan")
    icc = icc_3_1(human, judge) if n >= 2 else float("nan")
    return {
        "n": n,
        "mean_human": round(float(human.mean()), 4) if n else None,
        "mean_judge": round(float(judge.mean()), 4) if n else None,
        "spearman": round(sp, 4) if not np.isnan(sp) else None,
        "spearman_ci_lo": round(sp_lo, 4) if not np.isnan(sp_lo) else None,
        "spearman_ci_hi": round(sp_hi, 4) if not np.isnan(sp_hi) else None,
        "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
        "significant": bool(p_val < ALPHA) if not np.isnan(p_val) else None,
        "weighted_kappa": round(wk, 4) if not np.isnan(wk) else None,
        "icc_3_1": round(icc, 4) if not np.isnan(icc) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings-file", required=True)
    ap.add_argument("--sample-file", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_pairs(args.ratings_file, args.sample_file)
    if not rows:
        raise SystemExit("ERROR: no joined (human, judge) pairs found -- check item_key overlap")

    pooled = compute_scope(rows)
    print("=" * 70)
    print("POOLED (primary analysis)")
    print("=" * 70)
    for k, v in pooled.items():
        print(f"  {k:<16} {v}")

    models = sorted({r["model"] for r in rows})
    per_model = {}
    print()
    print("=" * 70)
    print("PER-MODEL (descriptive robustness check -- NOT independently powered")
    print("to +-5% margin of error; the stratified n=50/model does not meet that")
    print("bar on its own, only the pooled n does)")
    print("=" * 70)
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        per_model[m] = compute_scope(mrows)
        print(f"  {m:<9} n={per_model[m]['n']:<4} "
              f"mean_human={per_model[m]['mean_human']}  mean_judge={per_model[m]['mean_judge']}  "
              f"spearman={per_model[m]['spearman']}  kappa={per_model[m]['weighted_kappa']}  "
              f"icc={per_model[m]['icc_3_1']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    csv_path = f"{args.output}_agreement.csv"
    fieldnames = ["scope", "n", "mean_human", "mean_judge", "spearman",
                  "spearman_ci_lo", "spearman_ci_hi", "p_value", "significant",
                  "weighted_kappa", "icc_3_1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"scope": "POOLED", **pooled})
        for m in models:
            w.writerow({"scope": m, **per_model[m]})
    print(f"\nAgreement CSV saved: {csv_path}")
    print("(per-model rows are a robustness check only -- see note above)")


if __name__ == "__main__":
    main()
