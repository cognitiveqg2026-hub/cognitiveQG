#!/usr/bin/env python3
"""
Compute per-field metrics (model vs gold) using IAA-aligned metrics:
  CAT  — Gwet's AC          (iaa_core/metrics.py:gwet_ac_score)
  ML   — Jaccard (set)      (|A∩B| / |A∪B|, per instance then averaged)
  FT   — Jaccard (token)    (iaa_core/metrics.py:jaccard_similarity)

Field ordering matches cognitiveqg-iaa-sem-60.tex exactly.
Display format: 3-decimal floats (0.341 style) matching the IAA table.

Usage:
  python scripts/compute_field_metrics_jac_gwet.py \
      --review  results/baseline/comparison_run2_20260525/annotation_review.json \
      --run-tag Run2P4structured \
      --output  docs/manuscript/model-baselinescore-run2-jac-gwet.tex
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from iaa_core.metrics import gwet_ac_score, jaccard_similarity

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "deepseek_chat"]
MODEL_LABELS = {
    "llama2": "LLaMA-2",
    "llama3": "LLaMA-3",
    "qwen": "Qwen",
    "mistral": "Mistral",
    "olmo": "OLMo",
    "deepseek_chat": "DeepSeek",
}

METRIC_LABEL = {
    "CAT": "Gwet's AC",
    "ML":  "Jaccard",
    "FT":  "Jaccard",
}

# Field ordering matches cognitiveqg-iaa-sem-60.tex exactly
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
        ("span1",                      "Span Assignment (span1)",   "FT"),
        ("span2",                      "Span Assignment (span2)",   "FT"),
        ("span3",                      "Span Assignment (span3)",   "FT"),
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


# ── Normalisation helpers ─────────────────────────────────────────────────────

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


def fmt(val) -> str:
    if val is None:
        return "--"
    return f"{val:.3f}"


# ── Per-field score functions ─────────────────────────────────────────────────

def compute_cat_gwet(review, field, model) -> float | None:
    golds, preds = [], []
    for arg in review:
        entry = arg["fields"].get(field, {})
        g = norm_str(entry.get("gold", ""))
        p = norm_str(entry.get(model, ""))
        if p == "unmatched" or g in ("n/a", ""):
            continue
        golds.append(g)
        preds.append(p)
    if len(golds) < 2:
        return None
    return float(gwet_ac_score(golds, preds))


def compute_jac_ft(review, field, model) -> float | None:
    scores = []
    for arg in review:
        entry = arg["fields"].get(field, {})
        g = norm_str(entry.get("gold", ""))
        p = norm_str(entry.get(model, ""))
        if entry.get(model) == "UNMATCHED" or g in ("n/a", ""):
            continue
        scores.append(jaccard_similarity(g, p))
    return sum(scores) / len(scores) if scores else None


def compute_jac_ml(review, field, model) -> float | None:
    scores = []
    for arg in review:
        entry = arg["fields"].get(field, {})
        if entry.get(model) == "UNMATCHED":
            continue
        g = norm_list(entry.get("gold", ""))
        p = norm_list(entry.get(model, ""))
        if not g:
            continue
        union = g | p
        scores.append(len(g & p) / len(union) if union else 0.0)
    return sum(scores) / len(scores) if scores else None


def field_score(review, field, metric, model) -> float | None:
    if metric == "CAT":
        return compute_cat_gwet(review, field, model)
    elif metric == "ML":
        return compute_jac_ml(review, field, model)
    else:
        return compute_jac_ft(review, field, model)


# ── LaTeX builder ─────────────────────────────────────────────────────────────

def build_latex(review, run_tag) -> str:
    total_cols = 9   # Field + Metric + 6 models + Avg
    model_headers = " & ".join(r"\textbf{" + MODEL_LABELS[m] + r"}" for m in MODELS)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{3.2cm} l r r r r r r r}",
        r"\hline",
        rf"\textbf{{Annotation Field}} & \textbf{{Metric}} & {model_headers} & \textbf{{Avg}} \\",
        r"\hline",
    ]

    for phase, fields in PHASE_FIELDS.items():
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")

        for field, display, metric in fields:
            scores = [field_score(review, field, metric, m) for m in MODELS]
            valid  = [s for s in scores if s is not None]
            avg    = sum(valid) / len(valid) if valid else None
            score_cols = " & ".join(fmt(s) for s in scores)
            lines.append(
                f"{display} & {METRIC_LABEL[metric]} & {score_cols} & {fmt(avg)} \\\\"
            )

        lines.append(r"\hline")

    label = re.sub(r"[^a-zA-Z0-9_]", "", run_tag)
    lines += [
        r"\end{tabular}",
        rf"\caption{{Per-field model-vs-gold scores by cognitive phase ({run_tag}). "
        r"Gwet's AC for categorical fields; Jaccard similarity for span/text and "
        r"multi-label fields. Metrics match the human IAA study "
        r"(Table~\ref{tab:iaa-human}) to enable direct human--model comparison. "
        r"Jaccard for span fields equals the span component of the reasoning trace "
        r"alignment score (§\ref{sec:trace}).}",
        rf"\label{{tab:model-baseline-jac-gwet-{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review",  default="results/baseline_dev1-12/annotation_review.json")
    ap.add_argument("--run-tag", default="Run2P4structured")
    ap.add_argument("--output",  default="docs/manuscript/model-baselinescore-run2-jac-gwet.tex")
    args = ap.parse_args()

    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    latex  = build_latex(review, args.run_tag)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(latex, encoding="utf-8")
    print(f"Wrote → {args.output}")

    # Plain-text summary
    print(f"\n{'Phase':<20} {'Field':<28} {'Type':<4} " +
          "  ".join(f"{m[:6]:>7}" for m in MODELS) + "    Avg")
    print("-" * 115)
    for phase, fields in PHASE_FIELDS.items():
        for field, display, metric in fields:
            scores = [field_score(review, field, metric, m) for m in MODELS]
            valid  = [s for s in scores if s is not None]
            avg    = sum(valid) / len(valid) if valid else None
            s_str  = "  ".join(f"{s:7.3f}" if s is not None else "      -" for s in scores)
            a_str  = f"{avg:7.3f}" if avg is not None else "      -"
            print(f"{phase:<20} {display:<28} {metric:<4} {s_str}  {a_str}")
        print()


if __name__ == "__main__":
    main()
