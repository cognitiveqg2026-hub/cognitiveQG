#!/usr/bin/env python3
"""
Generate the dev7-12 (n=55) per-field baseline table — the "model-baselinescore"
table (Open-Source models, Wt.F1 / Micro-F1 / BERTScore, mean±std over seeds,
bold best-in-row, shaded Phase-avg rows), matching
docs/manuscript/model-baselinescore-run2-phavg-20260602.tex but for the 6
open-source models on the dev7-12 A1 gold.

Scores are computed with the *canonical* metric functions from
compute_field_metrics_latex.py (support-weighted macro-F1 for CAT, set micro-F1
for ML, roberta-large BERTScore for FT/spans), evaluated per inference seed and
aggregated to mean±std. Predictions are parsed from each model's raw_output.json
per seed via parse_model_output().

Run on the GPU host (roberta-large BERTScore):
    python scripts/generate_baseline_field_table_dev7-12.py \
        --raw-root results/baseline_dev7-12/raw_run2_20260525 \
        --gold     data/dev7-12_A1/p_dev7-12_gold.json \
        --output   docs/manuscript/model-baselinescore-dev7-12.tex
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "utils"))

import compute_field_metrics_latex_rescaled as cfm  # noqa: E402  (RESCALED BERTScore metric funcs)
from parse_model_output import parse_model_output   # noqa: E402

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
MODEL_LABELS = {"llama2": "LLaMA-2", "llama3": "LLaMA-3", "qwen": "Qwen",
                "mistral": "Mistral", "olmo": "OLMo", "qwen3": "Qwen3",
                "gpt-4o": "GPT-4o", "gpt-5.5": "GPT-5.5"}
SEEDS = ["44", "45", "46"]

METRIC_LABEL = {"CAT": r"Wt.\ F1", "ML": "Micro-F1", "FT": "BERTScore"}

# Row spec mirrors model-baselinescore-run2-phavg (spans = BERTScore "Span Eval.").
# (display, metric, gold_field)
PHASES = [
    ("Interpretation", [
        ("Initial Understanding",  "FT",  "initialUnderstandingTarget"),
        ("Stance Target",          "CAT", "stanceTarget"),
        ("Knowledge Domain",       "CAT", "knowledgeDomain"),
    ]),
    ("Analysis", [
        ("Paraphrase of Core Claim", "FT", "paraphrasingUnderstanding"),
        ("Core Claim",  "FT", "coreClaim"),
        ("Minor Claim", "FT", "minorClaim"),
        ("Premise",     "FT", "premise"),
        ("Reasoning Type", "CAT", "reasoningType"),
        ("Has Assumption", "CAT", "hasAssumption"),
        ("Missing Component", "FT", "missingComponent"),
    ]),
    ("Inference", [
        ("Positive Consequences", "FT", "positiveConsequences"),
        ("Negative Consequences", "FT", "negativeConsequences"),
        ("Primary Domain Affected", "CAT", "primaryDomain"),
        ("Alternative Type", "CAT", "alternativeType"),
        ("Alternative Keywords", "FT", "alternativeKeywords"),
    ]),
    ("Evaluation", [
        ("Inference Score", "CAT", "inferenceStrength"),
        ("Credibility Factors", "ML", "credibilityFactors"),
        ("Logical Fallacy", "CAT", "logicalFallacy"),
        ("Trustworthiness", "ML", "trustworthiness"),
        ("Trust Explanation", "FT", "trustworthinessExplanation"),
    ]),
    ("Explanation", [
        ("Target Explanation", "CAT", "targetExplanation"),
        ("Reasoning Structure", "CAT", "reasoningStructure"),
        ("Span Eval.\\ (span1)", "FT", "span1"),
        ("Span Eval.\\ (span2)", "FT", "span2"),
        ("Span Eval.\\ (span3)", "FT", "span3"),
        ("Good Evidence", "FT", "goodEvidence"),
        ("Bad Evidence", "FT", "badEvidence"),
        ("Fallacy Span 1", "FT", "fallacySpan1"),
        ("Fallacy Span 2", "FT", "fallacySpan2"),
        ("No Fallacy Explanation", "FT", "noFallacyExplanation"),
    ]),
    ("Self-Regulation", [
        ("Bias Detection", "ML", "biasDetection"),
        ("Heuristic Detection", "ML", "heuristicDetection"),
        ("Error Detection", "FT", "errorDetection"),
        ("Change Decision", "CAT", "changeDecision"),
        ("Revision Type", "CAT", "revisionType"),
    ]),
    ("Socratic Question Generation", [
        ("Socratic Question 1", "FT", "socraticQuestion1"),
        ("Socratic Question 2", "FT", "socraticQuestion2"),
        ("Socratic Question 3", "FT", "socraticQuestion3"),
    ]),
]
# Explanation phase-avg uses only the 4 cross-model-comparable fields (dagger).
EXPLANATION_AVG_FIELDS = {"targetExplanation", "reasoningStructure",
                          "goodEvidence", "badEvidence"}

# Gold field -> prediction key (only this one differs).
PRED_FIELD_MAP = {"trustworthinessExplanation": "trustExplanation"}


def load_predictions(raw_root):
    """preds[model][seed][arg_index] = parsed field dict."""
    preds = {}
    for model in MODELS:
        path = os.path.join(raw_root, model, "raw_output.json")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}; {model} -> empty")
            preds[model] = {s: {} for s in SEEDS}
            continue
        raw = json.load(open(path))
        per = {s: {} for s in SEEDS}
        for sample in raw["per_sample"]:
            idx = int(sample["argument_index"])
            for s in SEEDS:
                txt = sample.get("per_seed", {}).get(s, {}).get("generated_text", "")
                per[s][idx] = parse_model_output(txt) if txt else {}
        preds[model] = per
    return preds


def build_review(gold, preds, seed):
    """A cfm-style review for one seed: [{fields:{f:{gold, <model>:pred}}}]."""
    review = []
    for ann in gold:
        idx = int(ann["argumentIndex"])
        entry = {"fields": {}}
        for _, _, field in [r for _, rows in PHASES for r in rows]:
            fe = {"gold": ann.get(field, "")}
            pcol = PRED_FIELD_MAP.get(field, field)
            for m in MODELS:
                pred = preds[m][seed].get(idx)
                fe[m] = pred.get(pcol, "") if pred else "UNMATCHED"
            entry["fields"][field] = fe
        review.append(entry)
    return review


def score_cell(field, metric, model, reviews, bert_caches):
    """mean,std (in %) over seeds for (field, model); (None, None) if all missing."""
    vals = []
    for s in SEEDS:
        if metric == "CAT":
            v = cfm.compute_cat_score(reviews[s], field, model)
        elif metric == "ML":
            v = cfm.compute_ml_score(reviews[s], field, model)
        else:
            v = cfm.compute_ft_score(bert_caches[s], field, model)
        if v is not None:
            vals.append(v * 100.0)
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def cell_tex(mean, std, is_best):
    if mean is None:
        return "--"
    body = f"\\textbf{{{mean:.1f}}}" if is_best else f"{mean:.1f}"
    return f"{body}{{\\tiny$\\pm${std:.1f}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="results/baseline_dev7-12/raw_run2_20260525")
    ap.add_argument("--gold", default="data/dev7-12_A1/p_dev7-12_gold.json")
    ap.add_argument("--output", default="docs/manuscript/model-baselinescore-dev7-12.tex")
    ap.add_argument("--span-json", default=None,
                    help="Optional span_bertscore JSON to fill span1/2/3 rows")
    ap.add_argument("--n", type=int, default=55)
    args = ap.parse_args()

    span_data = json.load(open(args.span_json)) if args.span_json else None
    gold = json.load(open(args.gold))["annotations"]
    preds = load_predictions(args.raw_root)

    # Make cfm use our model list and field/metric mapping (so its BERTScore
    # precompute treats spans as FT).
    cfm.MODELS = MODELS
    cfm.PHASE_FIELDS = {ph: [(f, disp, met) for (disp, met, f) in rows]
                        for ph, rows in PHASES}

    reviews, bert_caches = {}, {}
    for s in SEEDS:
        print(f"Scoring seed {s} ...")
        reviews[s] = build_review(gold, preds, s)
        bert_caches[s] = cfm.precompute_bert_scores(reviews[s])

    # Compute every cell: results[(field)] -> {model: (mean,std)} ; track metric/display.
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{3.2cm} l c c c c c c c c}",
        r"\hline",
        r"\textbf{Annotation Field} & \textbf{Metric} & "
        r"\multicolumn{6}{c}{\textbf{Open-Source}} & "
        r"\multicolumn{2}{c}{\textbf{Closed-Source}} \\",
        "& & " + " & ".join(rf"\textbf{{{MODEL_LABELS[m]}}}" for m in MODELS) + r" \\",
        r"\hline",
    ]
    total_cols = 2 + len(MODELS)
    for phase, rows in PHASES:
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")
        # accumulate per-model field means for the phase average
        phase_means = {m: [] for m in MODELS}
        for display, metric, field in rows:
            if span_data is not None and field in ("span1", "span2", "span3"):
                # Dedicated span pipeline (label->field text + BERTScore, no seed std).
                cells = {}
                for m in MODELS:
                    sd = span_data.get(m, {}).get(field)
                    cells[m] = (sd["F1"], 0.0) if sd and sd.get("n", 0) > 0 else (None, None)
            else:
                cells = {m: score_cell(field, metric, m, reviews, bert_caches) for m in MODELS}
            means = [cells[m][0] for m in MODELS]
            best = max([v for v in means if v is not None], default=None)
            # Don't bold a row where nothing scored above ~0 (all-zero ties).
            bold_ok = best is not None and best > 0.05
            row = [display, METRIC_LABEL[metric]]
            for m in MODELS:
                mean, std = cells[m]
                row.append(cell_tex(mean, std, bold_ok and mean is not None
                                    and abs(mean - best) < 1e-9))
                include = (phase != "Explanation") or (field in EXPLANATION_AVG_FIELDS)
                if include and mean is not None:
                    phase_means[m].append(mean)
            lines.append(" & ".join(row) + r" \\")
        # Phase avg row
        avg = {m: (np.mean(phase_means[m]) if phase_means[m] else None) for m in MODELS}
        best_avg = max([v for v in avg.values() if v is not None], default=None)
        bold_avg_ok = best_avg is not None and best_avg > 0.05
        label = r"\textit{Phase avg}"
        if phase == "Explanation":
            label += r"$^\dagger$"
        cols = []
        for m in MODELS:
            v = avg[m]
            if v is None:
                cols.append("--")
            elif bold_avg_ok and abs(v - best_avg) < 1e-9:
                cols.append(rf"\textbf{{{v:.1f}}}")
            else:
                cols.append(f"{v:.1f}")
        lines.append(rf"\rowcolor[gray]{{0.92}} {label} & & " + " & ".join(cols) + r" \\")
        lines.append(r"\hline")

    lines += [
        r"\end{tabular}",
        rf"\caption{{Per-field model-vs-gold scores by cognitive phase on the "
        rf"dev7--12 batch ($n={args.n}$, A1 gold; P4 prompt, mean\,$\pm$\,std over "
        r"3 inference seeds: 44, 45, 46), open-source models only. "
        r"Wt.\,F1 = categorical weighted macro-F1; Micro-F1 = multi-label set "
        r"micro-F1; BERTScore = BERTScore F1 (roberta-large). Span Eval.\ (span1--3) "
        r"use text-span BERTScore. \textbf{Bold} = best per row. Shaded rows = "
        r"phase-level averages over fields with valid predictions. "
        r"$^\dagger$Explanation phase avg over 4 universal fields (Target "
        r"Explanation, Reasoning Structure, Good/Bad Evidence). "
        r"{--} = no valid predictions. Field ordering matches "
        r"Table~\ref{tab:iaa-human}.}",
        r"\label{tab:model-f1-dev7-12}",
        r"\end{table*}",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    open(args.output, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
