#!/usr/bin/env python3
"""
Compute per-field metrics (vs gold) for all 6 models, grouped by cognitive phase.
Outputs a LaTeX booktabs table suitable for inclusion in a manuscript.

Metric per field type:
  CAT  (single-select categorical)  — weighted macro-F1 over all classes
  ML   (multi-label list)           — instance-averaged micro-F1 (set intersection)
  FT   (free-text / span)           — BERTScore F1 (roberta-large, batched)

Usage:
  python scripts/compute_field_metrics_latex.py \
      --review  results/baseline/comparison_run2_20260525/annotation_review.json \
      --run-tag Run2P4structured \
      --output  results/baseline/comparison_run2_20260525/field_metrics_run2.tex
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import torch
from bert_score import score as bert_score_fn

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
MODEL_LABELS = {
    "llama2": "LLaMA-2",
    "llama3": "LLaMA-3",
    "qwen": "Qwen",
    "mistral": "Mistral",
    "olmo": "OLMo",
    "qwen3": "Qwen3",
    "gpt-4o": "GPT-4o",
    "gpt-5.5": "GPT-5.5",
}

# Field ordering and phase groupings match cognitiveqg-iaa-sem-60.tex exactly
PHASE_FIELDS = {
    "Interpretation": [
        ("initialUnderstandingTarget", "Initial Understanding",     "FT"),
        ("stanceTarget",               "Stance Target",             "CAT"),
        ("knowledgeDomain",            "Knowledge Domain",          "CAT"),
    ],
    "Analysis": [
        ("paraphrasingUnderstanding",  "Paraphrase of Core Claim",  "FT"),
        ("coreClaim",                  "Core Claim",                "FT"),
        ("minorClaim",                 "Minor Claim",               "FT"),
        ("premise",                    "Premise",                   "FT"),
        ("reasoningType",              "Reasoning Type",            "CAT"),
        ("hasAssumption",              "Has Assumption",            "CAT"),
        ("missingComponent",           "Missing Component",         "FT"),
    ],
    "Inference": [
        ("positiveConsequences",       "Positive Consequences",     "FT"),
        ("negativeConsequences",       "Negative Consequences",     "FT"),
        ("primaryDomain",              "Primary Domain Affected",   "CAT"),
        ("alternativeType",            "Alternative Type",          "CAT"),
        ("alternativeKeywords",        "Alternative Keywords",      "FT"),
    ],
    "Evaluation": [
        ("inferenceStrength",          "Inference Score",           "CAT"),
        ("credibilityFactors",         "Credibility Factors",       "ML"),
        ("logicalFallacy",             "Logical Fallacy",           "CAT"),
        ("trustworthiness",            "Trustworthiness",           "ML"),
        ("trustworthinessExplanation", "Trust Explanation",         "FT"),
    ],
    "Explanation": [
        ("targetExplanation",          "Target Explanation",        "CAT"),
        ("reasoningStructure",         "Reasoning Structure",       "CAT"),
        ("span1",                      "Span Assignment (span1)",   "CAT"),
        ("span2",                      "Span Assignment (span2)",   "CAT"),
        ("span3",                      "Span Assignment (span3)",   "CAT"),
        ("goodEvidence",               "Good Evidence",             "FT"),
        ("badEvidence",                "Bad Evidence",              "FT"),
        ("fallacySpan1",               "Fallacy Span 1",            "FT"),
        ("fallacySpan2",               "Fallacy Span 2",            "FT"),
        ("noFallacyExplanation",       "No Fallacy Explanation",    "FT"),
    ],
    "Self-Regulation": [
        ("biasDetection",              "Bias Detection",            "ML"),
        ("heuristicDetection",         "Heuristic Detection",       "ML"),
        ("errorDetection",             "Error Detection",           "FT"),
        ("changeDecision",             "Change Decision",           "CAT"),
        ("revisionType",               "Revision Type",             "CAT"),
    ],
    "Socratic Question Generation": [
        ("socraticQuestion1",          "Socratic Question 1",       "FT"),
        ("socraticQuestion2",          "Socratic Question 2",       "FT"),
        ("socraticQuestion3",          "Socratic Question 3",       "FT"),
    ],
}

METRIC_LABEL = {
    "CAT": "Wt.\\ F1",
    "ML":  "Micro-F1",
    "FT":  "BERTScore",
}

PHASE_COLORS = {
    "Interpretation":      r"\rowcolor[gray]{0.93}",
    "Analysis":            "",
    "Inference":           r"\rowcolor[gray]{0.93}",
    "Evaluation":          "",
    "Explanation":         r"\rowcolor[gray]{0.93}",
    "Self-Regulation":     "",
    "Question Generation": r"\rowcolor[gray]{0.93}",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def norm_str(v) -> str:
    if isinstance(v, list):
        v = v[0] if v else ""
    return re.sub(r"\s+", " ", str(v or "").strip().lower())


def norm_list(v) -> set:
    if isinstance(v, list):
        return {x.strip().lower() for x in v if x.strip()}
    if isinstance(v, str) and v.strip():
        return {x.strip().lower() for x in re.split(r"[|;,]", v) if x.strip()}
    return set()


def set_micro_f1(gold_set: set, pred_set: set) -> float:
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    common = len(gold_set & pred_set)
    if common == 0:
        return 0.0
    prec = common / len(pred_set)
    rec  = common / len(gold_set)
    return 2 * prec * rec / (prec + rec)


def macro_f1_categorical(golds: list, preds: list) -> float:
    classes = set(golds) | set(preds)
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        if tp + fp == 0 or tp + fn == 0:
            continue
        pr = tp / (tp + fp)
        rc = tp / (tp + fn)
        if pr + rc == 0:
            continue
        f1s.append((2 * pr * rc / (pr + rc), tp + fn))
    if not f1s:
        return 0.0
    total = sum(s for _, s in f1s)
    if total == 0:
        return sum(f for f, _ in f1s) / len(f1s)
    return sum(f * s for f, s in f1s) / total


# ── BERTScore batched precomputation ─────────────────────────────────────────

def precompute_bert_scores(review: list) -> dict:
    """
    Collect all (field, model) FT pairs, run BERTScore in one batch,
    return dict[(field, model)] -> list of per-instance F1 (or None if skipped).
    """
    ft_fields = {field for phase_list in PHASE_FIELDS.values()
                 for field, _, mtype in phase_list if mtype == "FT"}

    # Build index: (field, model) -> list of (arg_idx, gold, pred)
    index = defaultdict(list)
    for arg_idx, arg in enumerate(review):
        for field in ft_fields:
            entry = arg["fields"].get(field, {})
            gold_raw = entry.get("gold", "")
            for model in MODELS:
                pred_raw = entry.get(model, "")
                g = norm_str(gold_raw)
                p = norm_str(pred_raw)
                index[(field, model)].append((arg_idx, g, p))

    # Flatten into one big batch, keeping track of positions
    # Empty pred → score 0 directly; missing gold → skip (None)
    batch_refs, batch_hyps, batch_keys = [], [], []
    zero_keys = []   # (key, arg_idx) pairs that get score=0 (non-empty gold, empty pred)
    for key, triples in index.items():
        for arg_idx, g, p in triples:
            if g in ("n/a", "") or p == "unmatched":
                batch_keys.append((key, arg_idx, None))      # no gold → skip
            elif p == "":
                zero_keys.append((key, arg_idx))             # empty pred → 0
                batch_keys.append((key, arg_idx, None))
            else:
                batch_refs.append(g)
                batch_hyps.append(p)
                batch_keys.append((key, arg_idx, len(batch_refs) - 1))

    print(f"  Running BERTScore on {len(batch_refs)} pairs (roberta-large)...")
    if batch_refs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, F = bert_score_fn(
            batch_hyps, batch_refs,
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
            device=device,
        )
        f_scores = F.tolist()
    else:
        f_scores = []

    # Reconstruct per-(field, model) lists
    results: dict[tuple, list] = defaultdict(lambda: [None] * len(review))
    for key, arg_idx, batch_idx in batch_keys:
        if batch_idx is not None:
            results[key][arg_idx] = f_scores[batch_idx]
    for key, arg_idx in zero_keys:
        results[key][arg_idx] = 0.0

    return dict(results)


# ── Per-field score computation ───────────────────────────────────────────────

def compute_cat_score(review, field, model) -> float | None:
    golds, preds = [], []
    for arg in review:
        entry = arg["fields"].get(field, {})
        g = norm_str(entry.get("gold", ""))
        p = norm_str(entry.get(model, ""))
        if p == "unmatched" or g in ("n/a", ""):
            continue
        golds.append(g)
        preds.append(p)
    if not golds:
        return None
    return macro_f1_categorical(golds, preds)


def compute_ml_score(review, field, model) -> float | None:
    scores = []
    for arg in review:
        entry = arg["fields"].get(field, {})
        g = norm_list(entry.get("gold", ""))
        p = norm_list(entry.get(model, ""))
        if entry.get(model) == "UNMATCHED" or not g:
            continue
        scores.append(set_micro_f1(g, p))
    return sum(scores) / len(scores) if scores else None


def compute_ft_score(bert_cache, field, model) -> float | None:
    vals = [v for v in bert_cache.get((field, model), []) if v is not None]
    return sum(vals) / len(vals) if vals else None


# ── LaTeX builder ─────────────────────────────────────────────────────────────

def fmt(val) -> str:
    if val is None:
        return "--"
    return f"{val * 100:.1f}"


def build_latex(review: list, bert_cache: dict, run_tag: str) -> str:
    # 10 columns: Field + Metric + 8 models
    total_cols = 10
    model_headers = " & ".join(r"\textbf{" + MODEL_LABELS[m] + r"}" for m in MODELS)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{3.2cm} l r r r r r r r r}",
        r"\hline",
        rf"\textbf{{Annotation Field}} & \textbf{{Metric}} & {model_headers} \\",
        r"\hline",
    ]

    for phase, fields in PHASE_FIELDS.items():
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")

        for field, display, metric in fields:
            if metric == "CAT":
                scores = [compute_cat_score(review, field, m) for m in MODELS]
            elif metric == "ML":
                scores = [compute_ml_score(review, field, m) for m in MODELS]
            else:
                scores = [compute_ft_score(bert_cache, field, m) for m in MODELS]

            score_cols = " & ".join(fmt(s) for s in scores)
            met_label  = METRIC_LABEL[metric]

            lines.append(f"{display} & {met_label} & {score_cols} \\\\")

        lines.append(r"\hline")

    label = re.sub(r"[^a-zA-Z0-9_]", "", run_tag)
    lines += [
        r"\end{tabular}",
        rf"\caption{{Per-field scores by cognitive phase ({run_tag}). "
        r"Wt.\,F1 = categorical weighted macro-F1; "
        r"Micro-F1 = multi-label set micro-F1; "
        r"BERTScore = BERTScore F1 (roberta-large). "
        r"Field ordering matches Table~\ref{tab:iaa-human}.}",
        rf"\label{{tab:model-f1-{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review",  default="results/baseline_dev1-12/annotation_review.json")
    ap.add_argument("--run-tag", default="Run2P4structured")
    ap.add_argument("--output",  default="results/baseline_dev1-12/field_metrics_run2.tex")
    args = ap.parse_args()

    review = json.loads(Path(args.review).read_text(encoding="utf-8"))

    print("Pre-computing BERTScores...")
    bert_cache = precompute_bert_scores(review)

    print("Building table...")
    latex = build_latex(review, bert_cache, args.run_tag)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX table → {args.output}")

    # Plain-text summary
    print(f"\n{'Phase':<18} {'Field':<28} {'Type':<4} " +
          "  ".join(f"{m[:6]:>7}" for m in MODELS))
    print("-" * 108)
    for phase, fields in PHASE_FIELDS.items():
        for field, display, metric in fields:
            if metric == "CAT":
                scores = [compute_cat_score(review, field, m) for m in MODELS]
            elif metric == "ML":
                scores = [compute_ml_score(review, field, m) for m in MODELS]
            else:
                scores = [compute_ft_score(bert_cache, field, m) for m in MODELS]
            s_str = "  ".join(f"{s*100:7.1f}" if s is not None else "      -" for s in scores)
            print(f"{phase:<18} {display:<28} {metric:<4} {s_str}")
        print()


if __name__ == "__main__":
    main()
