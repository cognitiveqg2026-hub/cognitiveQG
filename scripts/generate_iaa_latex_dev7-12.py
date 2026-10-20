#!/usr/bin/env python3
"""
Generate the dev7-12 A2-vs-A1 IAA LaTeX table (Overleaf-ready).

Mirrors the existing manuscript table docs/manuscript/cognitiveqg-iaa-sem-60.tex
(\\label{tab:iaa-human}, n=20) as a drop-in companion for n=55, reading values
from the computed IAA results JSON. Categorical rows report Cohen's kappa,
PABAK, observed agreement (OA) and Gwet's AC; text/span rows report Jaccard,
ROUGE-L and BERT. The GPT-Sem (LLM-judge) column is reserved as an empty
placeholder, since that pass is deferred.

Usage:
    python scripts/generate_iaa_latex_dev7-12.py \
        [--json outputs/results/iaa/iaa_dev7-12_a2_vs_a1.json] \
        [--output docs/manuscript/cognitiveqg-iaa-dev7-12.tex] \
        [--n 55]
"""

import argparse
import json
import math
import os

# Each row mirrors a row of cognitiveqg-iaa-sem-60.tex.
#   (display_name, kind, field_or_fields)
#   kind: 'cat' -> categorical (shaded);  'text' -> text/span field
#   field_or_fields: a JSON field name, or a list of candidates for a merged row
#                    (the candidate with the most valid pairs is used).
PHASES = [
    ("Interpretation", [
        ("Initial Understanding",     "text", "initialUnderstandingTarget"),
        ("Stance Target",             "cat",  "stanceTarget"),
        ("Knowledge Domain",          "cat",  "knowledgeDomain"),
    ]),
    ("Analysis", [
        ("Paraphrase of Core Claim",  "text", "paraphrasingUnderstanding"),
        ("Core Claim",                "text", "coreClaim"),
        ("Minor Claim",               "text", "minorClaim"),
        ("Premise",                   "text", "premise"),
        ("Reasoning Type",            "cat",  "reasoningType"),
        ("Has Assumption",            "cat",  "hasAssumption"),
        ("Missing Component",         "text", "missingComponent"),
    ]),
    ("Inference", [
        ("Positive Consequences",     "text", "positiveConsequences"),
        ("Negative Consequences",     "text", "negativeConsequences"),
        ("Primary Domain Affected",   "cat",  "primaryDomain"),
        ("Alternative Type",          "cat",  "alternativeType"),
        ("Alternative Keywords",      "text", "alternativeKeywords"),
    ]),
    ("Evaluation", [
        ("Inference Score",           "cat",  "inferenceStrength"),
        ("Credibility Factors",       "cat",  "credibilityFactors"),
        ("Logical Fallacy",           "cat",  "logicalFallacy"),
        ("Trustworthiness",           "cat",  "trustworthiness"),
        ("Trust Explanation",         "text", "trustworthinessExplanation"),
    ]),
    ("Explanation", [
        ("Target Explanation",        "cat",  "targetExplanation"),
        ("Reasoning Structure",       "cat",  "reasoningStructure"),
        ("Span Assignment (span1)",   "cat",  "span1"),
        ("Span Assignment (span2)",   "cat",  "span2"),
        ("Span Assignment (span3)",   "cat",  "span3"),
        (r"Inductive/Ded.\ Term X",   "text", ["inductiveTermX", "deductiveTermX"]),
        (r"Inductive/Ded.\ Term Y",   "text", ["inductiveTermY", "deductiveTermY"]),
        ("Good Evidence",             "text", "goodEvidence"),
        ("Bad Evidence",              "text", "badEvidence"),
        ("Fallacy Span",              "text", ["fallacySpan1", "fallacySpan2"]),
        ("No Fallacy Explanation",    "text", "noFallacyExplanation"),
    ]),
    ("Self-Regulation", [
        ("Bias Detection",            "cat",  "biasDetection"),
        ("Heuristic Detection",       "cat",  "heuristicDetection"),
        ("Error Detection",           "text", "errorDetection"),
        ("Change Decision",           "cat",  "changeDecision"),
        ("Revision Phases",           "cat",  "revisionPhases"),
        ("Revision Reason",           "text", "revisionReason"),
        ("Revision Type",             "cat",  "revisionType"),
    ]),
    ("Socratic Question Generation", [
        ("Socratic Question 1",       "text", "socraticQuestion1"),
        ("Socratic Question 2",       "text", "socraticQuestion2"),
        ("Socratic Question 3",       "text", "socraticQuestion3"),
    ]),
]


def fmt(x):
    """Format a metric value to 3 decimals, or '--' when missing/NaN."""
    if x is None:
        return "--"
    if isinstance(x, float) and math.isnan(x):
        return "--"
    return f"{x:.3f}"


def valid_pairs(entry):
    if not entry:
        return 0
    return (entry.get("coverage") or {}).get("valid_pairs", 0)


def pick_field(data, field_or_fields):
    """Resolve a row's source entry; for a merged row pick the best-covered field."""
    if isinstance(field_or_fields, str):
        return data.get(field_or_fields)
    best, best_cov = None, -1
    for name in field_or_fields:
        entry = data.get(name)
        cov = valid_pairs(entry)
        if entry is not None and cov > best_cov:
            best, best_cov = entry, cov
    return best


def cat_row(display, entry):
    """Shaded categorical row: C'Kappa | PABAK | OA | Gwet's AC | (GPT-Sem reserved)."""
    e = entry or {}
    k = fmt(e.get("cohens_kappa"))
    p = fmt(e.get("pabak"))
    oa = fmt(e.get("observed_agreement"))
    ac = fmt(e.get("gwet_ac"))
    return (f"\\rowcolor[gray]{{0.93}}{display:<24} "
            f"& C'Kappa & {k} & PABAK & {p} & OA & {oa} & Gwet's AC & {ac} & -- & -- \\\\")


def text_row(display, entry):
    """Text/span row: Jaccard | ROUGE-L | (reserved) | BERT | GPT-Sem (reserved)."""
    e = entry or {}
    if valid_pairs(e) == 0:
        j = r = b = "--"
    else:
        j = fmt((e.get("jaccard") or {}).get("mean"))
        r = fmt((e.get("rouge_l") or {}).get("mean"))
        b = fmt((e.get("bert") or {}).get("mean"))
    return (f"{display:<24} "
            f"& Jaccard & {j} & ROUGE-L & {r} & -- & -- & BERT & {b} & GPT-Sem & -- \\\\")


def build_table(data, n):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        "",
        r"\begin{tabular}{p{3cm} l r l r l r l r l r}",
        r"\hline",
        r"\textbf{Annotation Field} &",
        r"\textbf{Metric} & \textbf{Score} &",
        r"\textbf{Metric} & \textbf{Score} &",
        r"\textbf{Metric} & \textbf{Score} &",
        r"\textbf{Metric} & \textbf{Score} &",
        r"\textbf{Metric} & \textbf{Score} \\",
        r"\hline",
        "",
    ]
    for phase, rows in PHASES:
        lines.append(r"\multicolumn{11}{l}{\textit{%s}} \\" % phase)
        for display, kind, field in rows:
            entry = pick_field(data, field)
            lines.append(cat_row(display, entry) if kind == "cat" else text_row(display, entry))
        lines.append(r"\hline")
        lines.append("")

    lines += [
        r"\end{tabular}",
        "",
        r"\caption{Inter-annotator agreement between the two human annotators"
        r" (A2 vs.\ A1) on the dev7--12 batch ($n = %d$ arguments, positional"
        r" matching). Categorical fields (shaded rows,"
        r" \colorbox[gray]{0.93}{\phantom{x}}) report Cohen's $\kappa$, PABAK,"
        r" observed agreement (OA) and Gwet's AC; text/span fields (white rows)"
        r" report Jaccard similarity, ROUGE-L and BERT cosine similarity"
        r" (sentence-transformers, all-MiniLM-L6-v2). The GPT-Sem column is"
        r" reserved: the LLM-as-judge semantic-equivalence pass has not yet been"
        r" run for this batch. All categorical fields use the same four agreement"
        r" coefficients (no per-field Exact-Match/Jaccard variants). Merged rows"
        r" (Inductive/Ded.\ Term X/Y, Fallacy Span) report the better-covered"
        r" sub-field; ``--'' marks fields with no valid pairs.}" % n,
        r"\label{tab:iaa-human-dev7-12}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(root, "outputs/results/iaa/iaa_dev7-12_a2_vs_a1.json"))
    ap.add_argument("--output", default=os.path.join(root, "docs/manuscript/cognitiveqg-iaa-dev7-12.tex"))
    ap.add_argument("--n", type=int, default=55)
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    tex = build_table(data, args.n)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(tex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
