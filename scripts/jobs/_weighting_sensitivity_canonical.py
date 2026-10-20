#!/usr/bin/env python3
"""
Weighting-sensitivity analysis, rebuilt as a thin layer on top of the EXACT
pipeline that produced results/baseline_complete_dev1-12/reasoning_trace_scores.csv.

Background:
  - That CSV's trace_jac is reproduced EXACTLY (max diff 1.11e-16) by
    compute_run2_trace_scores_complete.py (12 CAT + 4 ML + 23 FT = 39 fields),
    NOT by compute_run2_trace_scores.py (35 fields, max diff 0.046) despite
    run_baseline_complete_master.sh:84-85 literally invoking the latter --
    the CSV was evidently later regenerated/appended by the "_complete"
    script (it also contains qwen3_14b/qwen3_32b rows neither script run
    nor that shell job's ALL_MODELS produces).
  - The only code difference between the two sibling scripts is FT_FIELDS
    gaining deductiveTermX/Y, inductiveTermX/Y (confirmed by `diff`); these
    4 fields are genuinely non-empty in annotation_review.json (gold fill
    rate 13/120 and 80/120 respectively), not a vacuous inclusion.

This script therefore imports compute_run2_trace_scores_complete as `base`
(same as the ORIGINAL _weighting_sensitivity.py did), and hard-asserts, before
reporting anything else:
  (1) A reproduces trace_jac exactly,
  (2) LLaMA-2 Pearson r=0.297 / Holm p=0.0082 vs GPT-Sem (S1),
  (3) dev1-6 (arg_index<=59, 8 models) pooled mean=0.260 / median=0.254.

Adds, on top of the original script's A/A_eq/A_eq_noSR/A_abl(5-field)/A_eq_abl:
  - A_fs: field-share weights (12/39, 4/39, 23/39) instead of the fixed
    0.390/0.098/0.512 constants.
  - A_abl_a: 2-field exclusion (biasDetection, heuristicDetection only; 37
    fields) -- same exclusion as the hand-typed Table 5, on THIS (39-field)
    basis. (The original script's A_abl / this script's A_abl_b is the
    5-field exclusion, i.e. Table 34's definition, unchanged.)

Writes ONLY to results/weighting_sensitivity_canonical/ -- does not touch
results/weighting_sensitivity/, results/weighting_sensitivity_35/, or any
existing .tex file.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from iaa_core.metrics import jaccard_similarity  # noqa: E402
import compute_run2_trace_scores_complete as base  # noqa: E402  (39-field, matches the CSV exactly)
import analyze_run2_multimodel_correlation_stats as corr_base  # noqa: E402

REVIEW_PATH = ROOT / "results/baseline_complete_dev1-12/annotation_review.json"
TRACE_CSV = ROOT / "results/baseline_complete_dev1-12/reasoning_trace_scores.csv"
JUDGE_BASE = ROOT / "results/baseline_complete_dev1-12"
A2_GOLD_PATH = ROOT / "data/dev1-12_a2/k_dev1-12_gold.json"
RAW_PREDS_ROOT = ROOT / "results/baseline_complete_dev1-12/raw"

OUT_DIR = ROOT / "results/weighting_sensitivity_canonical"

MODELS = ["llama2", "llama3", "qwen", "qwen3", "mistral", "olmo", "gpt-4o", "gpt-5.5"]
MODEL_LABELS = {
    "llama2": "LLaMA-2-7B-Chat", "llama3": "LLaMA-3.1-8B-Instruct",
    "qwen": "Qwen2.5-7B-Instruct", "qwen3": "Qwen3-8B",
    "mistral": "Mistral-7B-Instruct", "olmo": "OLMo-2-7B-Instruct",
    "gpt-4o": "GPT-4o", "gpt-5.5": "GPT-5.5",
}
TABLE5_ABL_VALUES = {
    "gpt-4o": 0.322, "gpt-5.5": 0.268, "qwen": 0.279, "qwen3": 0.255,
    "olmo": 0.224, "llama3": 0.223, "mistral": 0.220, "llama2": 0.206,
}

SEEDS = ("44", "45", "46")
EXCLUDE_DEV1_2_INDICES = set(range(35, 55))
DEV16_MAX = 59

# ----------------------------------------------------------------------- #
# Field basis: exactly the sibling that reproduces the published CSV
# ----------------------------------------------------------------------- #
CAT_FIELDS = base.CAT_FIELDS          # 12
ML_FIELDS = base.ML_FIELDS            # 4
FT_FIELDS = base.FT_FIELDS            # 23 (the _complete variant -- matches the CSV)
W_CAT, W_ML, W_SPAN = base.W_CAT, base.W_ML, base.W_SPAN
norm_str, norm_list = base.norm_str, base.norm_list

FIELD_TYPE = {f: "CAT" for f in CAT_FIELDS}
FIELD_TYPE.update({f: "ML" for f in ML_FIELDS})
FIELD_TYPE.update({f: "FT" for f in FT_FIELDS})
assert len(FIELD_TYPE) == 39, f"expected 39 scored fields, got {len(FIELD_TYPE)}"

PHASE_FIELDS = {
    "Interpretation": ["initialUnderstandingTarget", "stanceTarget", "knowledgeDomain"],
    "Analysis": ["paraphrasingUnderstanding", "coreClaim", "minorClaim", "premise",
                 "reasoningType", "hasAssumption", "missingComponent"],
    "Inference": ["positiveConsequences", "negativeConsequences", "primaryDomain",
                  "alternativeType", "alternativeKeywords"],
    "Evaluation": ["inferenceStrength", "credibilityFactors", "logicalFallacy",
                   "trustworthiness", "trustworthinessExplanation"],
    "Explanation": ["targetExplanation", "reasoningStructure", "span1", "span2", "span3",
                     "goodEvidence", "badEvidence", "fallacySpan1", "fallacySpan2",
                     "noFallacyExplanation",
                     "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY"],
    "Self-Regulation": ["biasDetection", "heuristicDetection", "errorDetection",
                         "changeDecision", "revisionType"],
}
PHASES = list(PHASE_FIELDS.keys())
NON_SR_PHASES = [p for p in PHASES if p != "Self-Regulation"]

_all_phase_fields = [f for v in PHASE_FIELDS.values() for f in v]
assert len(_all_phase_fields) == 39 == len(set(_all_phase_fields))
assert sorted(_all_phase_fields) == sorted(FIELD_TYPE)

ABL_EXCLUDED_A = {"biasDetection", "heuristicDetection"}                       # 37 remain
ABL_EXCLUDED_B = ABL_EXCLUDED_A | {"span1", "span2", "span3"}                   # 34 remain -- Table 34's basis

W_CAT_FS = len(CAT_FIELDS) / len(FIELD_TYPE)   # 12/39
W_ML_FS = len(ML_FIELDS) / len(FIELD_TYPE)      # 4/39
W_SPAN_FS = len(FT_FIELDS) / len(FIELD_TYPE)    # 23/39

SCORE_VARIANTS = ["A", "A_eq", "A_eq_noSR", "A_fs",
                   "A_abl_a", "A_eq_abl_a", "A_abl_b", "A_eq_abl_b"]
ALPHA = 0.05
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


# ----------------------------------------------------------------------- #
# Per-field / per-argument scoring (identical primitives to both scripts)
# ----------------------------------------------------------------------- #

def field_unit_score(fields, model, field):
    ftype = FIELD_TYPE[field]
    fd = fields.get(field, {})
    pred_raw = fd.get(model, "")
    if pred_raw == "UNMATCHED":
        return None
    if ftype == "CAT":
        g = norm_str(fd.get("gold", ""))
        if g in ("n/a", ""):
            return None
        return 1.0 if norm_str(pred_raw) == g else 0.0
    if ftype == "ML":
        g = norm_list(fd.get("gold", ""))
        if not g:
            return None
        p = norm_list(pred_raw)
        union = g | p
        return len(g & p) / len(union) if union else 0.0
    g = norm_str(fd.get("gold", ""))
    if g in ("n/a", ""):
        return None
    return jaccard_similarity(g, norm_str(pred_raw))


def all_field_unit_scores(fields, model):
    return {f: field_unit_score(fields, model, f) for f in FIELD_TYPE}


def grouped_mean(unit_scores, field_subset):
    vals = [unit_scores[f] for f in field_subset if unit_scores[f] is not None]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def phase_mean_with_skip(phase_scores, phase_ns, phases_to_use):
    valid = [phase_scores[p] for p in phases_to_use if phase_ns[p] > 0]
    skipped = len(phases_to_use) - len(valid)
    if not valid:
        return None, skipped
    return sum(valid) / len(valid), skipped


def _ablated_variant(unit, excluded):
    cat = [f for f in CAT_FIELDS if f not in excluded]
    ml = [f for f in ML_FIELDS if f not in excluded]
    ft = [f for f in FT_FIELDS if f not in excluded]
    cat_s, _ = grouped_mean(unit, cat)
    ml_s, _ = grouped_mean(unit, ml)
    ft_s, _ = grouped_mean(unit, ft)
    a_abl = W_CAT * cat_s + W_ML * ml_s + W_SPAN * ft_s
    phase_scores, phase_ns = {}, {}
    for phase in PHASES:
        subset = [f for f in PHASE_FIELDS[phase] if f not in excluded]
        phase_scores[phase], phase_ns[phase] = grouped_mean(unit, subset)
    a_eq_abl, skip = phase_mean_with_skip(phase_scores, phase_ns, PHASES)
    return a_abl, a_eq_abl, skip


def per_arg_row(entry, model, arg_index=None):
    fields = entry["fields"]
    unit = all_field_unit_scores(fields, model)

    cat_score, cat_n = grouped_mean(unit, CAT_FIELDS)
    multi_jac, ml_n = grouped_mean(unit, ML_FIELDS)
    span_jac, ft_n = grouped_mean(unit, FT_FIELDS)
    A = W_CAT * cat_score + W_ML * multi_jac + W_SPAN * span_jac
    A_fs = W_CAT_FS * cat_score + W_ML_FS * multi_jac + W_SPAN_FS * span_jac

    phase_scores, phase_ns = {}, {}
    for phase in PHASES:
        phase_scores[phase], phase_ns[phase] = grouped_mean(unit, PHASE_FIELDS[phase])
    A_eq, skip_eq = phase_mean_with_skip(phase_scores, phase_ns, PHASES)
    A_eq_noSR, skip_nosr = phase_mean_with_skip(phase_scores, phase_ns, NON_SR_PHASES)

    A_abl_a, A_eq_abl_a, skip_abl_a = _ablated_variant(unit, ABL_EXCLUDED_A)
    A_abl_b, A_eq_abl_b, skip_abl_b = _ablated_variant(unit, ABL_EXCLUDED_B)

    assert cat_n + ml_n + ft_n == sum(phase_ns.values())
    idx = arg_index if arg_index is not None else int(entry["arg_index"])
    return {
        "arg_index": idx, "model": model,
        "cat_score": cat_score, "multi_jac": multi_jac, "span_jac": span_jac,
        "A": A, "A_eq": A_eq, "A_eq_noSR": A_eq_noSR, "A_fs": A_fs,
        "A_abl_a": A_abl_a, "A_eq_abl_a": A_eq_abl_a,
        "A_abl_b": A_abl_b, "A_eq_abl_b": A_eq_abl_b,
    }


def build_per_arg_df(review, models, expected_n=None):
    rows = []
    for model in models:
        for entry in review:
            rows.append(per_arg_row(entry, model))
    df = pd.DataFrame(rows)
    if expected_n is not None:
        counts = df.groupby("model").size()
        for model in models:
            if counts.get(model, 0) != expected_n:
                print(f"WARNING: model {model} has {counts.get(model, 0)} rows, expected {expected_n}")
    return df


# ----------------------------------------------------------------------- #
# HARD VALIDATION GATES (requested explicitly -- must pass before anything
# else is reported)
# ----------------------------------------------------------------------- #

def gate1_reproduces_csv(s1_df):
    existing = pd.read_csv(TRACE_CSV)
    existing = existing[existing["model"].isin(MODELS)]
    m = existing.merge(s1_df, on=["arg_index", "model"], suffixes=("_csv", "_new"))
    assert len(m) == len(existing), f"row count mismatch: {len(m)} vs {len(existing)}"
    diff = (m["trace_jac"] - m["A"]).abs()
    max_diff = diff.max()
    assert max_diff < 1e-9, (
        f"GATE 1 FAILED: A does not reproduce trace_jac exactly, max diff={max_diff:.2e}")
    print(f"[GATE 1 PASS] A reproduces reasoning_trace_scores.csv's trace_jac exactly "
          f"(max diff={max_diff:.2e}, {len(m)} rows).")


def gate2_llama2_pearson(s1_df):
    gpt_sem = corr_base.load_judge_se("llama2", str(JUDGE_BASE))
    sub = s1_df[s1_df["model"] == "llama2"]
    common = sorted(set(sub["arg_index"]) & set(gpt_sem.keys()))
    x = sub.set_index("arg_index").loc[common, "A"].values.astype(float)
    y = np.array([gpt_sem[k] for k in common], dtype=float)
    r, p = scipy_stats.pearsonr(x, y)
    # Holm correction across the 8 models, matching the published analysis exactly.
    pvals = []
    model_order = []
    for model in MODELS:
        se = corr_base.load_judge_se(model, str(JUDGE_BASE))
        subm = s1_df[s1_df["model"] == model]
        commonm = sorted(set(subm["arg_index"]) & set(se.keys()))
        xm = subm.set_index("arg_index").loc[commonm, "A"].values.astype(float)
        ym = np.array([se[k] for k in commonm], dtype=float)
        _, pm = scipy_stats.pearsonr(xm, ym)
        pvals.append(pm)
        model_order.append(model)
    holm_p, _ = corr_base.holm_bonferroni(pvals, ALPHA)
    llama2_holm = holm_p[model_order.index("llama2")]
    assert round(r, 3) == 0.297, f"GATE 2 FAILED: LLaMA-2 r={r:.4f}, expected 0.297"
    assert round(llama2_holm, 4) == 0.0082, (
        f"GATE 2 FAILED: LLaMA-2 Holm p={llama2_holm:.4f}, expected 0.0082")
    print(f"[GATE 2 PASS] LLaMA-2 r={r:.4f} (matches 0.297), Holm p={llama2_holm:.4f} (matches 0.0082).")
    return model_order, pvals, holm_p


def gate3_pooled_mean_median(s1_df):
    dev16 = s1_df[s1_df["arg_index"] <= DEV16_MAX]
    mean_ = dev16["A"].mean()
    median_ = dev16["A"].median()
    assert round(mean_, 3) == 0.260, f"GATE 3 FAILED: pooled mean={mean_:.4f}, expected 0.260"
    assert round(median_, 3) == 0.254, f"GATE 3 FAILED: pooled median={median_:.4f}, expected 0.254"
    print(f"[GATE 3 PASS] dev1-6 pooled mean={mean_:.4f} (matches 0.260), "
          f"median={median_:.4f} (matches 0.254), n={len(dev16)}.")


# ----------------------------------------------------------------------- #
# S2/S3, ranks, rank agreement (structurally identical to the other two scripts)
# ----------------------------------------------------------------------- #

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_a2_dev3_12_review():
    a2mod = _load_module("a2_dev312_reuse_canon",
                           SCRIPTS / "generate_baseline_field_table_dev3-12_a2.py")
    g7 = a2mod.g7
    combined_gold = a2mod.load_anns(str(A2_GOLD_PATH))
    preds = a2mod.load_pooled_preds(str(RAW_PREDS_ROOT), len(combined_gold))
    combined_gold = a2mod.filter_dev3_12(combined_gold, preds, expected_total=len(combined_gold))
    kept_indices = sorted(int(a["argumentIndex"]) for a in combined_gold)
    assert len(kept_indices) == 100

    per_seed_reviews = {}
    for s in SEEDS:
        raw_review = g7.build_review(combined_gold, preds, s)
        for entry, ann in zip(raw_review, combined_gold):
            entry["_idx"] = int(ann["argumentIndex"])
            entry["arg_index"] = int(ann["argumentIndex"])
        for entry, ann in zip(raw_review, combined_gold):
            idx = entry["_idx"]
            for slot in ("X", "Y"):
                for kind in ("deductive", "inductive"):
                    field = f"{kind}Term{slot}"
                    gval = str(ann.get(field, "") or "").strip()
                    fe = {"gold": gval}
                    for m in MODELS:
                        pred = preds[m][s].get(idx)
                        fe[m] = str(pred.get(field, "") or "").strip() if pred else "UNMATCHED"
                    entry["fields"][field] = fe
        per_seed_reviews[s] = raw_review
    return per_seed_reviews, kept_indices


def build_a2_per_arg_df():
    per_seed_reviews, kept_indices = load_a2_dev3_12_review()
    per_seed_dfs = []
    for s in SEEDS:
        df_s = build_per_arg_df(per_seed_reviews[s], MODELS, expected_n=100)
        df_s["seed"] = s
        per_seed_dfs.append(df_s)
    all_seeds_df = pd.concat(per_seed_dfs, ignore_index=True)
    averaged = all_seeds_df.groupby(["arg_index", "model"])[SCORE_VARIANTS].mean().reset_index()
    return averaged, kept_indices


def build_s3_per_arg_df(s1_per_arg_df):
    return s1_per_arg_df[~s1_per_arg_df["arg_index"].isin(EXCLUDE_DEV1_2_INDICES)].reset_index(drop=True)


def build_model_summary(per_arg_df, models):
    summary = per_arg_df.groupby("model")[SCORE_VARIANTS].mean()
    summary = summary.reindex(models)
    summary["n_args"] = per_arg_df.groupby("model").size().reindex(models)
    summary["model_label"] = [MODEL_LABELS[m] for m in summary.index]
    for v in SCORE_VARIANTS:
        summary[f"{v}_rank"] = summary[v].rank(ascending=False, method="min").astype(int)
    return summary


def rank_agreement_table(summary, setting):
    rows = []
    for alt in SCORE_VARIANTS:
        if alt == "A":
            continue
        rho, rho_p = scipy_stats.spearmanr(summary["A"], summary[alt])
        tau, tau_p = scipy_stats.kendalltau(summary["A"], summary[alt])
        rows.append({
            "setting": setting, "score_a": "A", "score_b": alt,
            "spearman_rho": rho, "spearman_p": rho_p,
            "kendall_tau": tau, "kendall_p": tau_p,
            "top1_unchanged": bool(summary["A"].idxmax() == summary[alt].idxmax()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- #
# S1 correlation sensitivity, all score variants
# ----------------------------------------------------------------------- #

def correlation_sensitivity(per_arg_df, gpt_sem_by_model, models, setting, score_vars):
    rows = []
    per_score_pvals = {v: [] for v in score_vars}
    per_score_models = {v: [] for v in score_vars}
    for score_var in score_vars:
        for model in models:
            se = gpt_sem_by_model.get(model, {})
            sub = per_arg_df[per_arg_df["model"] == model]
            common = sorted(set(sub["arg_index"]) & set(se.keys()))
            if len(common) < 4:
                continue
            x = sub.set_index("arg_index").loc[common, score_var].values.astype(float)
            y = np.array([se[k] for k in common], dtype=float)
            pear = corr_base.pearson_with_ci(x, y, ALPHA, 10000, BOOTSTRAP_SEED)
            rows.append({
                "setting": setting, "score_variant": score_var, "model": model, "n": len(common),
                "pearson_r": pear["r"], "pearson_p": pear["p"],
                "pearson_ci95_lo": pear["ci95_fisherz"][0], "pearson_ci95_hi": pear["ci95_fisherz"][1],
            })
            per_score_pvals[score_var].append(pear["p"])
            per_score_models[score_var].append(model)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for score_var in score_vars:
        ms = per_score_models[score_var]
        if not ms:
            continue
        holm_p, holm_rej = corr_base.holm_bonferroni(per_score_pvals[score_var], ALPHA)
        for i, m in enumerate(ms):
            mask = (df["setting"] == setting) & (df["score_variant"] == score_var) & (df["model"] == m)
            df.loc[mask, "pearson_p_holm"] = holm_p[i]
            df.loc[mask, "pearson_p_holm_reject"] = bool(holm_rej[i])
    return df


def load_gpt_sem_by_model(models):
    out = {}
    for model in models:
        try:
            out[model] = corr_base.load_judge_se(model, str(JUDGE_BASE))
        except FileNotFoundError:
            out[model] = {}
    return out


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("CANONICAL WEIGHTING-SENSITIVITY -- 39-FIELD BASIS (compute_run2_trace_scores_complete.py)")
    print("=" * 90)
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    s1_df = build_per_arg_df(review, MODELS, expected_n=120)
    assert len(s1_df) == 120 * len(MODELS)

    print("\n--- Hard validation gates (must pass before anything else) ---")
    gate1_reproduces_csv(s1_df)
    gate2_llama2_pearson(s1_df)
    gate3_pooled_mean_median(s1_df)

    s1_df.to_csv(OUT_DIR / "per_argument_scores_S1.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'per_argument_scores_S1.csv'}")

    s1_summary = build_model_summary(s1_df, MODELS)
    print("\nS1 model summary:")
    for m in s1_summary.index:
        r = s1_summary.loc[m]
        print(f"  {m:10s} A={r['A']:.4f}(#{r['A_rank']}) A_eq={r['A_eq']:.4f}(#{r['A_eq_rank']}) "
              f"A_fs={r['A_fs']:.4f}(#{r['A_fs_rank']}) "
              f"A_abl_a={r['A_abl_a']:.4f}(#{r['A_abl_a_rank']}) "
              f"A_abl_b={r['A_abl_b']:.4f}(#{r['A_abl_b_rank']})")

    print("\n--- Table 5 reproduction check (A_abl_a = 2-field excl., 37 fields) ---")
    abl_rows = []
    for m in MODELS:
        recomputed_a = s1_summary.loc[m, "A_abl_a"]
        recomputed_b = s1_summary.loc[m, "A_abl_b"]
        published = TABLE5_ABL_VALUES.get(m)
        match_a = published is not None and round(recomputed_a, 3) == published
        match_b = published is not None and round(recomputed_b, 3) == published
        abl_rows.append({"model": m, "A_abl_a_37field": recomputed_a, "A_abl_b_34field": recomputed_b,
                         "table5_published": published, "A_abl_a_matches": match_a, "A_abl_b_matches": match_b})
        print(f"  {m:10s} A_abl_a={recomputed_a:.4f} (match={match_a})  "
              f"A_abl_b={recomputed_b:.4f} (match={match_b})  Table5={published}")
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(OUT_DIR / "table5_reproduction_check.csv", index=False)
    print(f"Wrote {OUT_DIR / 'table5_reproduction_check.csv'}")

    print("\n" + "=" * 90)
    print("S2 (dev3-12, A2 gold, n=100)")
    print("=" * 90)
    s2_df, _ = build_a2_per_arg_df()
    s2_df.to_csv(OUT_DIR / "per_argument_scores_S2.csv", index=False)
    s2_summary = build_model_summary(s2_df, MODELS)
    for m in s2_summary.index:
        r = s2_summary.loc[m]
        print(f"  {m:10s} A={r['A']:.4f}(#{r['A_rank']}) A_eq={r['A_eq']:.4f}(#{r['A_eq_rank']})")

    print("\n" + "=" * 90)
    print("S3 (dev3-12, A1 gold, n=100)")
    print("=" * 90)
    s3_df = build_s3_per_arg_df(s1_df)
    assert len(s3_df) == 100 * len(MODELS)
    s3_df.to_csv(OUT_DIR / "per_argument_scores_S3.csv", index=False)
    s3_summary = build_model_summary(s3_df, MODELS)
    for m in s3_summary.index:
        r = s3_summary.loc[m]
        print(f"  {m:10s} A={r['A']:.4f}(#{r['A_rank']}) A_eq={r['A_eq']:.4f}(#{r['A_eq_rank']})")

    rank_rows = []
    for setting, summ in [("S1", s1_summary), ("S2", s2_summary), ("S3", s3_summary)]:
        for m in summ.index:
            row = {"setting": setting, "model": m, "model_label": MODEL_LABELS[m]}
            for v in SCORE_VARIANTS:
                row[v] = summ.loc[m, v]
                row[f"{v}_rank"] = summ.loc[m, f"{v}_rank"]
            rank_rows.append(row)
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / "model_scores_rankings.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'model_scores_rankings.csv'}")

    rank_agree_df = pd.concat([
        rank_agreement_table(s1_summary, "S1"),
        rank_agreement_table(s2_summary, "S2"),
        rank_agreement_table(s3_summary, "S3"),
    ], ignore_index=True)
    rank_agree_df.to_csv(OUT_DIR / "rank_agreement.csv", index=False)
    print(f"Wrote {OUT_DIR / 'rank_agreement.csv'}")
    print(f"Rank agreement (Spearman rho) range: "
          f"[{rank_agree_df['spearman_rho'].min():.3f}, {rank_agree_df['spearman_rho'].max():.3f}]")

    print("\n" + "=" * 90)
    print("S1 trace-question correlation, all score variants (Pearson, Fisher-z CI, Holm)")
    print("=" * 90)
    gpt_sem_s1 = load_gpt_sem_by_model(MODELS)
    corr_s1 = correlation_sensitivity(s1_df, gpt_sem_s1, MODELS, "S1",
                                       score_vars=("A", "A_eq", "A_fs", "A_abl_a", "A_abl_b"))
    corr_s1.to_csv(OUT_DIR / "correlation_sensitivity.csv", index=False)
    print(f"Wrote {OUT_DIR / 'correlation_sensitivity.csv'}")
    for v in ("A", "A_eq", "A_fs", "A_abl_a", "A_abl_b"):
        sub = corr_s1[corr_s1["score_variant"] == v].set_index("model")
        print(f"\nUnder {v}:")
        for m in MODELS:
            if m in sub.index:
                row = sub.loc[m]
                print(f"  {m:10s} r={row['pearson_r']:+.4f}  p={row['pearson_p']:.4f}  "
                      f"Holm_p={row['pearson_p_holm']:.4f}")

    print("\n" + "=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)
    print("[PASS] A reproduces reasoning_trace_scores.csv's trace_jac exactly")
    print("[PASS] LLaMA-2 r=0.297, Holm p=0.0082 (S1, under A)")
    print("[PASS] dev1-6 pooled mean=0.260, median=0.254 (under A)")

    best_abl = None
    for _, row in abl_df.iterrows():
        if row["A_abl_a_matches"]:
            best_abl = "A_abl_a (2-field: bias+heuristic, 37 fields)" if best_abl is None else best_abl + " [partial]"
    n_match_a = abl_df["A_abl_a_matches"].sum()
    n_match_b = abl_df["A_abl_b_matches"].sum()
    print(f"\nTable 5 reproduction: A_abl_a (2-field, 37 fields) matches {n_match_a}/8 models exactly to 3dp; "
          f"A_abl_b (5-field, 34 fields) matches {n_match_b}/8 models exactly to 3dp. "
          f"Neither reproduces Table 5 for ALL 8 models." if (n_match_a < 8 and n_match_b < 8) else
          "One definition reproduces Table 5 for all 8 models -- see table5_reproduction_check.csv.")

    return {"s1_summary": s1_summary, "s2_summary": s2_summary, "s3_summary": s3_summary,
            "rank_agree_df": rank_agree_df, "corr_s1": corr_s1}


if __name__ == "__main__":
    main()
