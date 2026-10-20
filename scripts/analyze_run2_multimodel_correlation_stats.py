#!/usr/bin/env python3
"""
Statistical validity/robustness report for the run2 multimodel correlation figure
(docs/manuscript/figures/run2_multimodel_trace_vs_semanticeq_dev1-12_FIXED.pdf).

For each of the 8 models, and across them, computes: Pearson r (Fisher-z CI +
bootstrap CI), Spearman rho (bootstrap CI; rho's own p-value, which the figure
computes but never displays), Kendall's tau, effect sizes (Cohen 1988), Shapiro-
Wilk normality (x, y, OLS residuals), Fisher-z-based power analysis, Holm-
Bonferroni / Benjamini-Hochberg multiple-comparison correction across the 8
tests, and a Fisher-z heterogeneity test across the 8 per-model correlations.

DATA PROVENANCE -- read before changing --base:
  The obvious-looking canonical snapshot, results/_merged_corr_newoss_oldgpt/
  (built 2026-07-05 19:21 by scripts/jobs/_revert_gpt_correlation.py),
  reproduces the published figure exactly for 6/8 models but NOT for
  gpt-4o/gpt-5.5 -- for those two it instead reproduces a SUPERSEDED version
  of the same-named figure (confirmed: found byte-identical in a backup at
  docs/manuscript/_pre_update_backup_20260705/, whose gpt-4o/gpt-5.5 panels
  read r=0.115/p=0.272 and r=-0.074/p=0.421 -- exactly the "mismatched"
  numbers). gpt-4o/gpt-5.5's judge data had an arg_index-vs-gold-argumentIndex
  alignment bug; scripts/jobs/rerun_gpt_aligned.sh deleted and regenerated it
  directly in results/baseline_complete_dev1-12/ on 2026-07-05/06, finishing
  six minutes before the published figure was rendered (01:52/02:43 vs 02:49).
  That directory -- restricted to the 8 models below, since it later gained
  qwen3_14b/qwen3_32b which are NOT in the published figure -- is the
  verified-correct source: it reproduces all 8 published (r, p) pairs exactly
  (see PUBLISHED_REFERENCE and the validation section this script prints).
  Five other candidate directories and four aggregation variants (median vs
  mean, clipping on/off, model_missing filtering, arg-range restriction) were
  tried and ruled out before reaching this conclusion.

Usage:
  python scripts/analyze_run2_multimodel_correlation_stats.py
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

MODELS_8 = ["llama2", "llama3", "qwen", "mistral", "gpt-4o", "gpt-5.5", "olmo", "qwen3"]
MODEL_LABELS = {
    "llama2":  "LLaMA-2-7B",
    "llama3":  "LLaMA-3.1-8B",
    "qwen":    "Qwen2.5-7B",
    "mistral": "Mistral-7B-v0.3",
    "olmo":    "OLMo-2-7B",
    "gpt-4o":  "GPT-4o",
    "gpt-5.5": "GPT-5.5",
    "qwen3":   "Qwen3-8B",
}

DEFAULT_BASE = "results/baseline_complete_dev1-12"

# Published figure's own annotated values
# (docs/manuscript/figures/run2_multimodel_trace_vs_semanticeq_dev1-12_FIXED.pdf)
PUBLISHED_REFERENCE = {
    "llama2":  {"r": 0.297, "p": 0.001},
    "llama3":  {"r": 0.094, "p": 0.310},
    "qwen":    {"r": 0.029, "p": 0.757},
    "mistral": {"r": -0.056, "p": 0.540},
    "gpt-4o":  {"r": 0.145, "p": 0.115},
    "gpt-5.5": {"r": 0.110, "p": 0.233},
    "olmo":    {"r": 0.223, "p": 0.015},
    "qwen3":   {"r": 0.182, "p": 0.047},
}

POST_HOC_POWER_CAVEAT = (
    "Post-hoc/achieved power is computed by plugging the observed r back in as if it "
    "were the true effect size. Hoenig & Heisey (2001, The American Statistician) show "
    "this is a deterministic, monotonically-decreasing function of the p-value alone, "
    "so it carries no information beyond the p-value and must not be read as "
    "'non-significant = no real effect'. The min-n-for-80%-power and "
    "min-detectable-r-at-n statistics are the defensible retrospective framing."
)


# --------------------------------------------------------------------------- #
# Data loading -- ported from scripts/plot_run2_multimodel_correlation.py:48-63, 175-184
# --------------------------------------------------------------------------- #

def load_judge_se(model, base, judge_file="judge_run2_judge.json"):
    """Return {arg_index: avg_semantic_equivalence (1-5)} for non-skipped pairs."""
    path = os.path.join(base, model, judge_file)
    d = json.load(open(path, "r", encoding="utf-8"))
    by_arg = defaultdict(list)
    for p in d["scored_pairs"]:
        if p.get("model_missing", False):
            continue
        j = p.get("judge", p.get("llm_judge", {}))
        if j.get("_skipped"):
            continue
        se = j.get("semantic_equivalence")
        if se is not None:
            by_arg[int(p["arg_index"])].append(int(se))
    return {a: sum(v) / len(v) for a, v in by_arg.items()}


def load_trace_scores(base):
    return pd.read_csv(os.path.join(base, "reasoning_trace_scores.csv"))


def join_model_xy(model, trace_df, se_by_arg):
    """Join judge SE (1-5) against trace_jac on arg_index; normalize Y to 0-1."""
    model_trace = trace_df[trace_df["model"] == model].set_index("arg_index")
    xs, ys_raw, ids = [], [], []
    for arg_idx, se_val in sorted(se_by_arg.items()):
        if arg_idx in model_trace.index:
            xs.append(float(model_trace.loc[arg_idx, "trace_jac"]))
            ys_raw.append(se_val)
            ids.append(arg_idx)
    xs = np.array(xs, dtype=float)
    ys_raw = np.array(ys_raw, dtype=float)
    ys = np.clip((ys_raw - 1) / 4, 0.0, 1.0)
    return xs, ys, ys_raw, ids


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def fisher_ci_manual(r, n, alpha=0.05):
    """Hand-rolled Fisher-z CI, ported from scripts/plot_target_vs_reasoning_all8.py:57-63."""
    if n is None or n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return (math.tanh(z - zc * se), math.tanh(z + zc * se))


def bootstrap_ci(x, y, statistic_fn, n_boot=10000, alpha=0.05, seed=42):
    """Seeded percentile bootstrap CI for a paired bivariate statistic (house style)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_vals[i] = statistic_fn(x[idx], y[idx])
    lo, hi = np.nanpercentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def effect_size_label(value, thresholds=(0.1, 0.3, 0.5)):
    a = abs(value)
    if a < thresholds[0]:
        return "negligible"
    elif a < thresholds[1]:
        return "small"
    elif a < thresholds[2]:
        return "medium"
    return "large"


def significance_stars(p):
    """2-tier convention, matching the published figure's own annotation exactly."""
    if p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def shapiro_test(values, alpha=0.05):
    w, p = stats.shapiro(values)
    return {"W": float(w), "p": float(p), "normal_at_alpha": bool(p > alpha)}


def ols_residuals(x, y):
    slope, intercept, *_ = stats.linregress(x, y)
    return y - (slope * x + intercept)


def fisher_z_power(r, n, alpha=0.05):
    """Achieved/post-hoc two-tailed power for a correlation test (Fisher-z closed form)."""
    r = min(max(abs(r), 1e-6), 0.999999)
    z_r = np.arctanh(r)
    ncp = z_r * math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(ncp - z_crit) + stats.norm.cdf(-ncp - z_crit))


def required_n_for_power(r, power_target=0.80, alpha=0.05):
    """Minimum n for target power to detect |r| (Fisher-z closed form)."""
    r = min(max(abs(r), 1e-6), 0.999999)
    z_r = np.arctanh(r)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power_target)
    n = ((z_crit + z_beta) / z_r) ** 2 + 3
    return float(math.ceil(n))


def min_detectable_r(n, power_target=0.80, alpha=0.05):
    """Minimum |r| detectable at target power given actual n (Fisher-z closed form)."""
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power_target)
    z_r = (z_crit + z_beta) / math.sqrt(n - 3)
    return float(math.tanh(z_r))


def holm_bonferroni(pvals, alpha=0.05):
    """Step-down Holm correction. Returns (adjusted_p, reject) in ORIGINAL order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = min((m - i) * sorted_p[i], 1.0)
        running_max = max(running_max, val)
        adj_sorted[i] = running_max
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha


def benjamini_hochberg(pvals, alpha=0.05):
    """Step-up BH-FDR correction. Returns (adjusted_p, reject) in ORIGINAL order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj_sorted = np.empty(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        val = min((m / (i + 1)) * sorted_p[i], 1.0)
        running_min = min(running_min, val)
        adj_sorted[i] = running_min
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha


def fisher_z_heterogeneity_test(rs, ns):
    """Fisher-z chi-square homogeneity test across k independent correlations
    (inverse-variance weighted, analogous to Cochran's Q)."""
    rs = np.asarray(rs, dtype=float)
    ns = np.asarray(ns, dtype=float)
    zs = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    ws = ns - 3
    z_bar = np.sum(ws * zs) / np.sum(ws)
    q = float(np.sum(ws * (zs - z_bar) ** 2))
    df = len(rs) - 1
    p = float(stats.chi2.sf(q, df))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {
        "Q": q, "df": df, "p": p,
        "pooled_r_fisherz": float(math.tanh(z_bar)),
        "i2_like_pct": float(i2),
        "heterogeneous_at_alpha": bool(p < 0.05),
    }


def validate_against_published(model, r, p, tol_r=0.01, tol_p=0.02):
    ref = PUBLISHED_REFERENCE.get(model)
    if ref is None:
        return {"status": "NO_REFERENCE"}
    diff_r = r - ref["r"]
    diff_p = p - ref["p"]
    status = "PASS" if (abs(diff_r) <= tol_r and abs(diff_p) <= tol_p) else "FAIL"
    return {
        "published_r": ref["r"], "published_p": ref["p"],
        "diff_r": float(diff_r), "diff_p": float(diff_p),
        "status": status,
    }


# --------------------------------------------------------------------------- #
# Per-model orchestration
# --------------------------------------------------------------------------- #

def pearson_with_ci(x, y, alpha, n_boot, seed):
    res = stats.pearsonr(x, y)
    r, p = float(res.statistic), float(res.pvalue)
    ci = res.confidence_interval(confidence_level=1 - alpha)
    ci_boot = bootstrap_ci(x, y, lambda a, b: stats.pearsonr(a, b).statistic,
                            n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "r": r, "p": p,
        "ci95_fisherz": (float(ci.low), float(ci.high)),
        "ci95_fisherz_manual_check": fisher_ci_manual(r, len(x), alpha),
        "ci95_bootstrap": ci_boot,
        "r_squared": r ** 2,
        "effect_size": effect_size_label(r),
        "sig_stars": significance_stars(p),
    }


def spearman_with_ci(x, y, alpha, n_boot, seed):
    res = stats.spearmanr(x, y)
    rho, p = float(res.statistic), float(res.pvalue)
    ci_boot = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).statistic,
                            n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "rho": rho, "p": p,
        "ci95_bootstrap": ci_boot,
        "ci95_fisherz_approx": fisher_ci_manual(rho, len(x), alpha),
        "rho_squared": rho ** 2,
        "effect_size": effect_size_label(rho),
        "sig_stars": significance_stars(p),
    }


def kendall_tau_stat(x, y):
    res = stats.kendalltau(x, y)
    return {"tau": float(res.statistic), "p": float(res.pvalue)}


def compute_model_stats(model, xs, ys, alpha, power_target, n_boot, seed, tol_r, tol_p):
    n = len(xs)
    pearson = pearson_with_ci(xs, ys, alpha, n_boot, seed)
    spearman = spearman_with_ci(xs, ys, alpha, n_boot, seed)
    kendall = kendall_tau_stat(xs, ys)

    resid = ols_residuals(xs, ys)
    normality = {
        "x": shapiro_test(xs, alpha),
        "y": shapiro_test(ys, alpha),
        "residuals": shapiro_test(resid, alpha),
    }

    power = {
        "achieved_power": fisher_z_power(pearson["r"], n, alpha),
        "min_n_for_80pct_power": required_n_for_power(pearson["r"], power_target, alpha),
        "min_detectable_r_at_n": min_detectable_r(n, power_target, alpha),
        "caveat": POST_HOC_POWER_CAVEAT,
    }

    validation = validate_against_published(model, pearson["r"], pearson["p"], tol_r, tol_p)

    return {
        "model_label": MODEL_LABELS.get(model, model),
        "n": n,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "normality": normality,
        "power": power,
        "validation": validation,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def build_csv_rows(per_model, model_order):
    rows = []
    for model in model_order:
        d = per_model[model]
        p, s, k, nm, pw, v = (d["pearson"], d["spearman"], d["kendall"],
                              d["normality"], d["power"], d["validation"])
        rows.append({
            "model": model, "model_label": d["model_label"], "n": d["n"],
            "pearson_r": p["r"], "pearson_p": p["p"],
            "pearson_ci95_lo_fisherz": p["ci95_fisherz"][0], "pearson_ci95_hi_fisherz": p["ci95_fisherz"][1],
            "pearson_ci95_lo_bootstrap": p["ci95_bootstrap"][0], "pearson_ci95_hi_bootstrap": p["ci95_bootstrap"][1],
            "spearman_rho": s["rho"], "spearman_p": s["p"],
            "spearman_ci95_lo_bootstrap": s["ci95_bootstrap"][0], "spearman_ci95_hi_bootstrap": s["ci95_bootstrap"][1],
            "kendall_tau": k["tau"], "kendall_p": k["p"],
            "r_squared": p["r_squared"], "rho_squared": s["rho_squared"],
            "effect_size_pearson": p["effect_size"], "effect_size_spearman": s["effect_size"],
            "sig_stars_pearson": p["sig_stars"], "sig_stars_spearman": s["sig_stars"],
            "shapiro_x_W": nm["x"]["W"], "shapiro_x_p": nm["x"]["p"], "shapiro_x_normal_at_alpha": nm["x"]["normal_at_alpha"],
            "shapiro_y_W": nm["y"]["W"], "shapiro_y_p": nm["y"]["p"], "shapiro_y_normal_at_alpha": nm["y"]["normal_at_alpha"],
            "shapiro_resid_W": nm["residuals"]["W"], "shapiro_resid_p": nm["residuals"]["p"],
            "shapiro_resid_normal_at_alpha": nm["residuals"]["normal_at_alpha"],
            "achieved_power_pearson": pw["achieved_power"],
            "min_n_for_80pct_power_pearson": pw["min_n_for_80pct_power"],
            "min_detectable_r_at_n_80pct": pw["min_detectable_r_at_n"],
            "pearson_p_holm": d.get("pearson_p_holm"), "pearson_p_holm_reject": d.get("pearson_p_holm_reject"),
            "pearson_p_fdr_bh": d.get("pearson_p_fdr_bh"), "pearson_p_fdr_bh_reject": d.get("pearson_p_fdr_bh_reject"),
            "spearman_p_holm": d.get("spearman_p_holm"), "spearman_p_holm_reject": d.get("spearman_p_holm_reject"),
            "spearman_p_fdr_bh": d.get("spearman_p_fdr_bh"), "spearman_p_fdr_bh_reject": d.get("spearman_p_fdr_bh_reject"),
            "published_ref_r": v.get("published_r"), "published_ref_p": v.get("published_p"),
            "diff_r_vs_published": v.get("diff_r"), "diff_p_vs_published": v.get("diff_p"),
            "validation_status": v.get("status"),
        })
    return rows


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("Object of type %s is not JSON serializable" % type(o))


def write_outputs(summary, report_text, out_dir, stem, model_order):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "%s.csv" % stem)
    json_path = os.path.join(out_dir, "%s_summary.json" % stem)
    md_path = os.path.join(out_dir, "%s_report.md" % stem)

    rows = build_csv_rows(summary["per_model"], model_order)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {"csv": csv_path, "json": json_path, "md": md_path}


def format_report(summary):
    meta = summary["metadata"]
    per_model = summary["per_model"]
    cross = summary["cross_model"]
    models = meta["models"]

    lines = []
    lines.append("=" * 100)
    lines.append("RUN2 MULTIMODEL CORRELATION -- STATISTICAL VALIDITY REPORT")
    lines.append("Base: %s | Models: %d | alpha=%.2f | power_target=%.2f | bootstrap n=%d (seed=%d)" % (
        meta["base_dir"], len(models), meta["alpha"], meta["power_target"],
        meta["n_bootstrap"], meta["bootstrap_seed"]))
    lines.append("Source figure: %s" % meta["source_figure"])
    lines.append("=" * 100)
    lines.append("")

    lines.append("%-16s %4s  %9s %-3s %20s %9s   %9s %-3s %20s %9s" % (
        "Model", "n", "Pearson r", "", "95% CI(Fisher-z)", "p", "Spearman rho", "", "95% CI(boot)", "p"))
    lines.append("-" * 100)
    for m in models:
        d = per_model[m]
        p, s = d["pearson"], d["spearman"]
        ci_p = "[%.3f, %.3f]" % p["ci95_fisherz"]
        ci_s = "[%.3f, %.3f]" % s["ci95_bootstrap"]
        lines.append("%-16s %4d  %9.3f %-3s %20s %9.4f   %9.3f %-3s %20s %9.4f" % (
            d["model_label"], d["n"], p["r"], p["sig_stars"], ci_p, p["p"],
            s["rho"], s["sig_stars"], ci_s, s["p"]))
    lines.append("")

    lines.append("Kendall's tau (ties-robust) | Effect size (Cohen 1988) | Shapiro-Wilk normality p (x, y, OLS residuals)")
    lines.append("-" * 100)
    for m in models:
        d = per_model[m]
        k, nm = d["kendall"], d["normality"]
        lines.append("%-16s tau=%7.3f (p=%.4f)  effect[r]=%-10s effect[rho]=%-10s  shapiro p: x=%.3f y=%.3f resid=%.3f" % (
            d["model_label"], k["tau"], k["p"], d["pearson"]["effect_size"], d["spearman"]["effect_size"],
            nm["x"]["p"], nm["y"]["p"], nm["residuals"]["p"]))
    lines.append("")

    lines.append("Power analysis (Fisher-z closed form; post-hoc power is informational only -- see note below)")
    lines.append("-" * 100)
    for m in models:
        d = per_model[m]
        pw = d["power"]
        lines.append("%-16s achieved_power=%.3f   min_n_for_80%%_power=%7.0f   min_detectable_r@n=%.3f" % (
            d["model_label"], pw["achieved_power"], pw["min_n_for_80pct_power"], pw["min_detectable_r_at_n"]))
    lines.append("")

    lines.append("Multiple-comparison correction (%d Pearson p-values; separately, %d Spearman p-values)" % (
        len(models), len(models)))
    lines.append("-" * 100)
    for m in models:
        d = per_model[m]
        lines.append("%-16s Pearson: holm_p=%.4f(%s) bh_p=%.4f(%s)   Spearman: holm_p=%.4f(%s) bh_p=%.4f(%s)" % (
            d["model_label"],
            d["pearson_p_holm"], "reject" if d["pearson_p_holm_reject"] else "fail",
            d["pearson_p_fdr_bh"], "reject" if d["pearson_p_fdr_bh_reject"] else "fail",
            d["spearman_p_holm"], "reject" if d["spearman_p_holm_reject"] else "fail",
            d["spearman_p_fdr_bh"], "reject" if d["spearman_p_fdr_bh_reject"] else "fail"))
    lines.append("")

    hp, hs = cross["heterogeneity_pearson"], cross["heterogeneity_spearman"]
    lines.append("Heterogeneity across models (Fisher-z chi-square test, analogous to Cochran's Q)")
    lines.append("-" * 100)
    lines.append("  Pearson:              Q=%.2f  df=%d  p=%.4f  pooled_r=%.3f    I2-like=%.1f%%  -> %s" % (
        hp["Q"], hp["df"], hp["p"], hp["pooled_r_fisherz"], hp["i2_like_pct"],
        "heterogeneous at alpha=0.05" if hp["heterogeneous_at_alpha"] else "not distinguishable from homogeneous"))
    lines.append("  Spearman (approx.):   Q=%.2f  df=%d  p=%.4f  pooled_rho=%.3f  I2-like=%.1f%%  -> %s" % (
        hs["Q"], hs["df"], hs["p"], hs["pooled_r_fisherz"], hs["i2_like_pct"],
        "heterogeneous at alpha=0.05" if hs["heterogeneous_at_alpha"] else "not distinguishable from homogeneous"))
    lines.append("")

    lines.append("Validation vs published figure (%s)" % meta["source_figure"])
    lines.append("-" * 100)
    passing = [per_model[m]["model_label"] for m in models if per_model[m]["validation"].get("status") == "PASS"]
    failing = [per_model[m]["model_label"] for m in models if per_model[m]["validation"].get("status") == "FAIL"]
    lines.append("  PASS: %s (%d/%d)" % (", ".join(passing) or "none", len(passing), len(models)))
    if failing:
        lines.append("  FAIL: %s (%d/%d)" % (", ".join(failing), len(failing), len(models)))
    lines.append("")

    lines.append("NOTE on post-hoc power:")
    lines.append("  " + POST_HOC_POWER_CAVEAT)
    lines.append("")
    if len(meta["notes"]) > 1:
        lines.append("NOTE on data provenance:")
        lines.append("  " + meta["notes"][1])
        lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--judge-file", default="judge_run2_judge.json",
                     help="Per-model judge JSON filename to read (default: gpt-5.5's "
                          "judge_run2_judge.json). Pass judge_gpt6astra_judge.json / "
                          "judge_gpt56sol_judge.json to rescore with a different judge; "
                          "disables the PUBLISHED_REFERENCE validation, which is gpt-5.5-specific.")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power-target", type=float, default=0.80)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results/run2_multimodel_corr_stats")
    ap.add_argument("--output-stem", default="run2_multimodel_corr_stats")
    ap.add_argument("--validation-tol-r", type=float, default=0.01)
    ap.add_argument("--validation-tol-p", type=float, default=0.02)
    args = ap.parse_args()

    models = args.models if args.models else MODELS_8
    validate = args.models is None and args.judge_file == "judge_run2_judge.json"

    trace_df = load_trace_scores(args.base)

    per_model = {}
    skipped = []
    for model in models:
        try:
            se_by_arg = load_judge_se(model, args.base, args.judge_file)
        except FileNotFoundError as e:
            skipped.append([model, str(e)])
            continue
        xs, ys, ys_raw, ids = join_model_xy(model, trace_df, se_by_arg)
        if len(xs) < 3:
            skipped.append([model, "insufficient data (n<3)"])
            continue
        d = compute_model_stats(model, xs, ys, args.alpha, args.power_target,
                                 args.n_bootstrap, args.bootstrap_seed,
                                 args.validation_tol_r, args.validation_tol_p)
        if not validate:
            d["validation"] = {"status": "SKIPPED_CUSTOM_MODEL_SET"}
        per_model[model] = d

    model_order = [m for m in models if m in per_model]
    if not model_order:
        raise SystemExit("No models produced usable data -- check --base and judge_run2_judge.json availability.")

    pearson_ps = [per_model[m]["pearson"]["p"] for m in model_order]
    spearman_ps = [per_model[m]["spearman"]["p"] for m in model_order]

    p_holm, p_holm_rej = holm_bonferroni(pearson_ps, args.alpha)
    p_bh, p_bh_rej = benjamini_hochberg(pearson_ps, args.alpha)
    s_holm, s_holm_rej = holm_bonferroni(spearman_ps, args.alpha)
    s_bh, s_bh_rej = benjamini_hochberg(spearman_ps, args.alpha)

    for i, m in enumerate(model_order):
        per_model[m]["pearson_p_holm"] = float(p_holm[i])
        per_model[m]["pearson_p_holm_reject"] = bool(p_holm_rej[i])
        per_model[m]["pearson_p_fdr_bh"] = float(p_bh[i])
        per_model[m]["pearson_p_fdr_bh_reject"] = bool(p_bh_rej[i])
        per_model[m]["spearman_p_holm"] = float(s_holm[i])
        per_model[m]["spearman_p_holm_reject"] = bool(s_holm_rej[i])
        per_model[m]["spearman_p_fdr_bh"] = float(s_bh[i])
        per_model[m]["spearman_p_fdr_bh_reject"] = bool(s_bh_rej[i])

    rs = [per_model[m]["pearson"]["r"] for m in model_order]
    rhos = [per_model[m]["spearman"]["rho"] for m in model_order]
    ns = [per_model[m]["n"] for m in model_order]

    het_pearson = fisher_z_heterogeneity_test(rs, ns)
    het_spearman = fisher_z_heterogeneity_test(rhos, ns)
    het_spearman["note"] = ("Approximate: the Fisher-z sampling distribution is formally "
                             "derived for Pearson r; applied here to Spearman rho as a "
                             "supplementary check only.")

    cross_model = {
        "heterogeneity_pearson": het_pearson,
        "heterogeneity_spearman": het_spearman,
        "multiple_comparison_pearson": {
            "models": model_order, "raw_p": pearson_ps,
            "holm_adjusted_p": [float(v) for v in p_holm], "holm_reject": [bool(v) for v in p_holm_rej],
            "fdr_bh_adjusted_p": [float(v) for v in p_bh], "fdr_bh_reject": [bool(v) for v in p_bh_rej],
        },
        "multiple_comparison_spearman": {
            "models": model_order, "raw_p": spearman_ps,
            "holm_adjusted_p": [float(v) for v in s_holm], "holm_reject": [bool(v) for v in s_holm_rej],
            "fdr_bh_adjusted_p": [float(v) for v in s_bh], "fdr_bh_reject": [bool(v) for v in s_bh_rej],
        },
    }

    n_pass = sum(1 for m in model_order if per_model[m]["validation"].get("status") == "PASS")
    n_fail = sum(1 for m in model_order if per_model[m]["validation"].get("status") == "FAIL")
    validation_summary = {
        "n_pass": n_pass, "n_fail": n_fail,
        "failing_models": [m for m in model_order if per_model[m]["validation"].get("status") == "FAIL"],
        "overall_status": ("FULL_PASS" if (validate and n_fail == 0)
                            else ("PARTIAL_PASS" if n_pass else "SKIPPED")),
    }

    metadata = {
        "script": "analyze_run2_multimodel_correlation_stats.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_dir": args.base,
        "models": model_order,
        "alpha": args.alpha,
        "power_target": args.power_target,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "source_figure": "docs/manuscript/figures/run2_multimodel_trace_vs_semanticeq_dev1-12_FIXED.pdf",
        "notes": [
            POST_HOC_POWER_CAVEAT,
            "Data provenance: results/baseline_complete_dev1-12 is the verified-correct "
            "source for gpt-4o/gpt-5.5 (judge data realigned and regenerated 2026-07-05/06 "
            "by scripts/jobs/rerun_gpt_aligned.sh). The superficially-canonical "
            "results/_merged_corr_newoss_oldgpt reproduces a superseded figure version for "
            "those two models specifically. See this script's module docstring for the full "
            "evidence trail.",
        ],
        "skipped_models": skipped,
    }

    summary = {
        "metadata": metadata,
        "per_model": per_model,
        "cross_model": cross_model,
        "validation_summary": validation_summary,
    }

    report_text = format_report(summary)
    print(report_text)

    paths = write_outputs(summary, report_text, args.output_dir, args.output_stem, model_order)
    print("\nWritten -> %s" % ", ".join(paths.values()))


if __name__ == "__main__":
    main()
