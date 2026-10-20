#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, DATA, OUT, TABLE_DIR
"""
Builds the three new LSAT tables (mean-over-references main + A1/A2 Human
appendix), matching the dev1-12 Table-3 pattern (phase-avg rows, Wt.F1 for
all categorical/multi-label fields, Human column sourced from lsat-iaa.tex
for BERTScore rows), using the already-computed scoring results:
  - lsat_meanref_result.json   (CAT + FT fields, span-role F1; A1/A2 gold;
                                 transformers==5.9.0, GPU host)
  - lsat_wtf1_multilabel_result.json  (the 4 fields re-scored with dev1-12's
                                 weighted_f1_multilabel instead of plain
                                 Micro-F1, per the approved addendum)
  - lsat_socratic_robust_result.json  (socraticQuestion1/2/3 model cells only;
                                 re-scored with extract_questions_robust()
                                 instead of the strict parser, to match the
                                 dev1-12 pipeline's own handling of these
                                 three fields -- see rescore_lsat_field_metrics_3seed.py)
Human BERTScore-row values are looked up from table/lsat-iaa.tex directly
(not recomputed), same convention as the dev1-12 appendix tables.
"""
import json
import os

ROOT = str(OUT)
OUT_TABLE_DIR = str(TABLE_DIR)

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
MODEL_LABELS = {"llama2": "LLaMA-2", "llama3": "LLaMA-3", "qwen": "Qwen", "mistral": "Mistral",
                "olmo": "OLMo", "qwen3": "Qwen3", "gpt-4o": "GPT-4o", "gpt-5.5": "GPT-5.5"}

# (display, metric_type, field_key). metric_type in {CAT, WTF1, FT, SPANROLE}
PHASES = [
    ("Interpretation", [
        ("Initial Understanding", "FT", "initialUnderstandingTarget"),
        ("Stance Target", "CAT", "stanceTarget"),
        ("Knowledge Domain", "CAT", "knowledgeDomain"),
    ]),
    ("Analysis", [
        ("Paraphrase of Core Claim", "FT", "paraphrasingUnderstanding"),
        ("Core Claim", "FT", "coreClaim"),
        ("Minor Claim", "FT", "minorClaim"),
        ("Premise", "FT", "premise"),
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
        ("Credibility Factors", "WTF1", "credibilityFactors"),
        ("Logical Fallacy", "CAT", "logicalFallacy"),
        ("Trustworthiness", "WTF1", "trustworthiness"),
        ("Trust Explanation", "FT", "trustworthinessExplanation"),
    ]),
    ("Explanation", [
        ("Target Explanation", "CAT", "targetExplanation"),
        ("Reasoning Structure", "CAT", "reasoningStructure"),
        (r"Span Eval.\ (span1)", "SPANBERT", "span1"),
        (r"Span Eval.\ (span2)", "SPANBERT", "span2"),
        (r"Span Eval.\ (span3)", "SPANBERT", "span3"),
        (r"Inductive/Ded.\ Term X", "TERM", "termX"),
        (r"Inductive/Ded.\ Term Y", "TERM", "termY"),
        ("Good Evidence", "FT", "goodEvidence"),
        ("Bad Evidence", "FT", "badEvidence"),
        ("Fallacy Span 1", "FT", "fallacySpan1"),
        ("Fallacy Span 2", "FT", "fallacySpan2"),
        ("No Fallacy Explanation", "FT", "noFallacyExplanation"),
    ]),
    ("Self-Regulation", [
        ("Bias Detection", "WTF1", "biasDetection"),
        ("Heuristic Detection", "WTF1", "heuristicDetection"),
        ("Error Detection", "FT", "errorDetection"),
        ("Change Decision", "CAT", "changeDecision"),
        ("Revision Type", "CAT", "revisionType"),
    ]),
    ("Socratic Question Generation", [
        ("Socratic Question 1", "SOCRATIC", "socraticQuestion1"),
        ("Socratic Question 2", "SOCRATIC", "socraticQuestion2"),
        ("Socratic Question 3", "SOCRATIC", "socraticQuestion3"),
    ]),
]
METRIC_LABEL = {"SPANBERT": "BERTScore", "CAT": r"Wt.\ F1", "WTF1": r"Wt.\ F1", "FT": "BERTScore", "SPANROLE": "F1", "TERM": "BERTScore",
                "SOCRATIC": "BERTScore"}
# Explanation phase-avg: same 4-universal-field convention as dev1-12 (excludes span-role/fallacy-extra rows).
EXPLANATION_AVG_FIELDS = {"targetExplanation", "reasoningStructure", "goodEvidence", "badEvidence"}

# ---- Human BERTScore-row source: table/lsat-iaa.tex BERT column, x100 ----
# (taken from lsat-iaa.tex). Fallacy Span 2 has
# no separate IAA row (lsat-iaa.tex merges span1/span2 into one "Fallacy
# Span" row) -- matching the dev1-12 appendix convention, that merged value
# maps to Fallacy Span 1 only; Fallacy Span 2 Human = "--".
IAA_BERT_X100 = {
    "initialUnderstandingTarget": 49.8, "paraphrasingUnderstanding": 62.3, "coreClaim": 92.8,
    "minorClaim": 97.5, "premise": 89.4, "missingComponent": 51.3,
    "positiveConsequences": 34.1, "negativeConsequences": 37.9, "alternativeKeywords": 32.1,
    "trustworthinessExplanation": 37.0,
    "goodEvidence": 34.7, "badEvidence": 36.5, "fallacySpan1": 77.9, "fallacySpan2": None,
    "noFallacyExplanation": 57.1, "errorDetection": None,
    "socraticQuestion1": 41.7, "socraticQuestion2": 36.7, "socraticQuestion3": 38.5,
}

d = json.load(open(os.path.join(ROOT, "lsat_meanref_result.json")))
wtf1 = json.load(open(os.path.join(ROOT, "lsat_wtf1_multilabel_result.json")))
termxy = json.load(open(os.path.join(ROOT, "lsat_termxy_result.json")))
socratic = json.load(open(os.path.join(ROOT, "lsat_socratic_robust_result.json")))

# Term X/Y Human cells: per the task, use the PUBLISHED lsat-iaa.tex values
# as-is (same convention as every other BERTScore Human cell), not the
# gold-vs-gold HUMAN_A2vsA1/A1vsA2 values already sitting in
# lsat_termxy_result.json (those exist but are intentionally unused here).
TERM_HUMAN_X100 = {"termX": 68.1, "termY": 82.4}


SPAN_EVAL = json.load(open(os.path.join(str(OUT), "span_eval_bertscore.json")))


def get_model_cell(tag, metric_type, field, model):
    """Returns (mean_x100, std_x100) or None."""
    if metric_type == "CAT":
        v = d[tag]["fields"].get(f"{field}|{model}")
        if v is None or v.get("mean") is None:
            return None
        return v["mean"] * 100, v["std"] * 100
    if metric_type == "WTF1":
        v = wtf1[tag][field].get(model)
        if v is None:
            return None
        return v["mean"] * 100, v["std"] * 100
    if metric_type == "FT":
        v = d[tag]["fields"].get(f"{field}|{model}")
        if v is None or v.get("mean") is None:
            return None
        return v["mean"] * 100, v["std"] * 100
    if metric_type == "TERM":
        v = termxy[tag][field].get(model)
        if v is None:
            return None
        return v["mean"] * 100, v["std"] * 100
    if metric_type == "SOCRATIC":
        v = socratic[tag][field].get(model)
        if v is None:
            return None
        return v["mean"] * 100, v["std"] * 100
    if metric_type == "SPANROLE":
        r = d[tag]["span_roles"][model][field]
        return r["mean"], r["std"]
    if metric_type == "SPANBERT":   # results/revision/span_eval_bertscore.json (already x100)
        r = SPAN_EVAL["lsat"][tag][f"{field}|{model}"]
        return None if r["mean"] is None else (r["mean"], r["std"])
    raise ValueError(metric_type)


def get_human_cell(human_tag, metric_type, field):
    """Returns mean_x100 (no std) or None."""
    if metric_type == "CAT":
        v = d[human_tag]["fields"].get(field)
        return v * 100 if v is not None else None
    if metric_type == "WTF1":
        v = wtf1[human_tag].get(field)
        return v * 100 if v is not None else None
    if metric_type == "FT":
        return IAA_BERT_X100.get(field)
    if metric_type == "TERM":
        return TERM_HUMAN_X100.get(field)
    if metric_type == "SOCRATIC":
        return IAA_BERT_X100.get(field)
    if metric_type == "SPANROLE":
        r = d[human_tag]["span_roles"][field]
        return r[0]
    if metric_type == "SPANBERT":
        return SPAN_EVAL["lsat"][human_tag][field]
    raise ValueError(metric_type)


def fmt_mean_std(cell):
    if cell is None:
        return "--"
    m, s = cell
    return f"{m:.1f}" + r"{\tiny$\pm$" + f"{s:.1f}" + "}"


def fmt_mean_only(v):
    if v is None:
        return "--"
    return f"{v:.1f}"


def best_bold(cells_by_model, get_mean):
    """cells_by_model: {model: cell_or_value}. Returns set of models tied for
    best DISPLAYED (1dp) mean, ties included."""
    displayed = {}
    for m, c in cells_by_model.items():
        v = get_mean(c)
        if v is not None:
            displayed[m] = round(v, 1)
    if not displayed:
        return set()
    best = max(displayed.values())
    return {m for m, v in displayed.items() if v == best}


def build_rows(tag, human_tag, with_std):
    """Returns (data_lines, phase_avg_values_for_bold_check) as LaTeX row
    strings, bold applied to best model per row (human excluded)."""
    lines = []
    for phase, items in PHASES:
        total_cols = 2 + len(MODELS) + 1  # field+metric+8 models+human
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")
        phase_means = {m: [] for m in MODELS}
        human_phase_vals = []
        for display, mtype, field in items:
            cells = {m: get_model_cell(tag, mtype, field, m) for m in MODELS}
            bold_models = best_bold(cells, lambda c: c[0] if c else None)
            row_cells = []
            for m in MODELS:
                c = cells[m]
                txt = fmt_mean_std(c) if with_std else fmt_mean_only(c[0] if c else None)
                if m in bold_models and c is not None:
                    # wrap only the numeric mean in \textbf, keep the ±std suffix outside
                    if with_std:
                        mean_txt = f"{c[0]:.1f}"
                        rest = txt[len(mean_txt):]
                        txt = r"\textbf{" + mean_txt + "}" + rest
                    else:
                        txt = r"\textbf{" + txt + "}"
                row_cells.append(txt)
                if c is not None and (phase != "Explanation" or field in EXPLANATION_AVG_FIELDS):
                    phase_means[m].append(c[0])
            human_v = get_human_cell(human_tag, mtype, field)
            human_txt = fmt_mean_only(human_v)
            if human_v is not None and (phase != "Explanation" or field in EXPLANATION_AVG_FIELDS):
                human_phase_vals.append(human_v)
            lines.append(f"{display} & {METRIC_LABEL[mtype]} & " + " & ".join(row_cells) + f" & {human_txt}" + r" \\")
        # phase avg row
        avg = {m: (sum(v) / len(v) if v else None) for m, v in phase_means.items()}
        bold_avg = best_bold(avg, lambda v: v)
        avg_cells = []
        for m in MODELS:
            v = avg[m]
            txt = fmt_mean_only(v)
            if m in bold_avg and v is not None:
                txt = r"\textbf{" + txt + "}"
            avg_cells.append(txt)
        label = r"\textit{Phase avg}"
        human_avg_txt = fmt_mean_only(sum(human_phase_vals) / len(human_phase_vals) if human_phase_vals else None)
        lines.append(rf"\rowcolor[gray]{{0.92}} {label} & & " + " & ".join(avg_cells) + f" & {human_avg_txt}" + r" \\")
        lines.append(r"\hline")
    return lines


HEADER = r"""\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
%s
\begin{tabular}{p{3.2cm} l c c c c c c c c | c}
\hline
\textbf{Annotation Field} & \textbf{Metric} & \multicolumn{6}{c}{\textbf{Open-Source}} & \multicolumn{2}{c}{\textbf{Closed-Source}} & \textbf{Human} \\
& & \textbf{LLaMA-2} & \textbf{LLaMA-3} & \textbf{Qwen} & \textbf{Mistral} & \textbf{OLMo} & \textbf{Qwen3} & \textbf{GPT-4o} & \textbf{GPT-5.5} & \textbf{%s} \\
\hline"""

FOOTER = r"""\end{tabular}
\end{table*}
"""

SPAN_ROLE_NOTE = (r"Span role rows: one-vs-rest F1 for each argumentative role, pooled over the "
                   r"three span slots of the reasoning structure.")

# ---------------------------------------------------------------------------
# Table 1: mean-over-references main
# ---------------------------------------------------------------------------
caption_main = (
    r"\caption{Per-field model scores on the LSAT batch ($n=20$; P1 prompt, mean over 3 "
    r"inference seeds), averaged over the two expert references. Metrics, the Human column, "
    r"the rule for a missing reference, bold and ``--'' follow "
    r"Table~\ref{tab:model-baseline-jac-gwet-Run2P4structured}; Human values of the BERTScore rows other "
    r"than Span Eval come from Table~\ref{tab:iaa-lsat}. Scores against A1 and against A2, with seed-level standard "
    r"deviations, are in Tables~\ref{tab:lsat-baseline-a1-human} and~\ref{tab:lsat-baseline-a2-human}.}"
)
main_lines = [
    r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
    caption_main,
    r"\label{tab:model-f1-LSATbaseline}",
    r"\begin{tabular}{p{3.2cm} l c c c c c c c c | c}",
    r"\hline",
    r"\textbf{Annotation Field} & \textbf{Metric} & \multicolumn{6}{c}{\textbf{Open-Source}} & \multicolumn{2}{c}{\textbf{Closed-Source}} & \textbf{Human} \\",
    r"& & \textbf{LLaMA-2} & \textbf{LLaMA-3} & \textbf{Qwen} & \textbf{Mistral} & \textbf{OLMo} & \textbf{Qwen3} & \textbf{GPT-4o} & \textbf{GPT-5.5} & \textbf{A1$\leftrightarrow$A2} \\",
    r"\hline",
]


def meanref_cell(mtype, field, model):
    a1 = get_model_cell("A1", mtype, field, model)
    a2 = get_model_cell("A2", mtype, field, model)
    if a1 is None and a2 is None:
        return None
    if a1 is None:
        return a2[0]
    if a2 is None:
        return a1[0]
    return (a1[0] + a2[0]) / 2


def meanref_human(mtype, field):
    h1 = get_human_cell("HUMAN_A2vsA1", mtype, field)
    h2 = get_human_cell("HUMAN_A1vsA2", mtype, field)
    if h1 is None and h2 is None:
        return None
    if h1 is None:
        return h2
    if h2 is None:
        return h1
    return (h1 + h2) / 2


for phase, items in PHASES:
    total_cols = 2 + len(MODELS) + 1
    main_lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{phase}}}}} \\")
    phase_means = {m: [] for m in MODELS}
    human_phase_vals = []
    for display, mtype, field in items:
        vals = {m: meanref_cell(mtype, field, m) for m in MODELS}
        bold_models = best_bold(vals, lambda v: v)
        row_cells = []
        for m in MODELS:
            v = vals[m]
            txt = fmt_mean_only(v)
            if m in bold_models and v is not None:
                txt = r"\textbf{" + txt + "}"
            row_cells.append(txt)
            if v is not None and (phase != "Explanation" or field in EXPLANATION_AVG_FIELDS):
                phase_means[m].append(v)
        human_v = meanref_human(mtype, field)
        if human_v is not None and (phase != "Explanation" or field in EXPLANATION_AVG_FIELDS):
            human_phase_vals.append(human_v)
        main_lines.append(f"{display} & {METRIC_LABEL[mtype]} & " + " & ".join(row_cells) + f" & {fmt_mean_only(human_v)}" + r" \\")
    avg = {m: (sum(v) / len(v) if v else None) for m, v in phase_means.items()}
    bold_avg = best_bold(avg, lambda v: v)
    avg_cells = []
    for m in MODELS:
        v = avg[m]
        txt = fmt_mean_only(v)
        if m in bold_avg and v is not None:
            txt = r"\textbf{" + txt + "}"
        avg_cells.append(txt)
    label = r"\textit{Phase avg}"
    human_avg_txt = fmt_mean_only(sum(human_phase_vals) / len(human_phase_vals) if human_phase_vals else None)
    main_lines.append(rf"\rowcolor[gray]{{0.92}} {label} & & " + " & ".join(avg_cells) + f" & {human_avg_txt}" + r" \\")
    main_lines.append(r"\hline")
main_lines += [r"\end{tabular}", r"\end{table*}"]

with open(os.path.join(OUT_TABLE_DIR, "lsat-baseline_meanref_main.tex"), "w") as f:
    f.write("\n".join(main_lines) + "\n")
print("wrote lsat-baseline_meanref_main.tex")

# ---------------------------------------------------------------------------
# Tables 2/3: A1-human and A2-human appendix
# ---------------------------------------------------------------------------
for tag, human_tag, human_col, label, caption_extra in [
    ("A1", "HUMAN_A2vsA1", "A2 vs A1", "tab:lsat-baseline-a1-human",
     r"Human = annotator A2 scored against the A1 gold as a reference ceiling "
     r"(single annotation, no seeds; not counted for bold): Wt.\,F1 and Span Eval "
     r"rows use the same scoring code as the models, and "
     r"the other BERTScore rows are the rescaled BERTScore values from the "
     r"inter-annotator agreement analysis (Table~\ref{tab:iaa-lsat}). "
     r"``--'' = no human reference available (A2 left the field empty, or "
     r"no BERTScore agreement was computed for that field)."),
    ("A2", "HUMAN_A1vsA2", "A1 vs A2", "tab:lsat-baseline-a2-human",
     r"Human = annotator A1 scored against the A2 gold (single annotation, "
     r"no seeds; not counted for bold): Wt.\,F1 and Span Eval rows use the "
     r"same scoring code as the models, and the other BERTScore rows are the "
     r"rescaled BERTScore values from the inter-annotator agreement "
     r"analysis (Table~\ref{tab:iaa-lsat}). ``--'' = no reference available "
     r"(A2 left the field empty, or no BERTScore agreement was computed "
     r"for that field)."),
]:
    gold_name = "A1 gold" if tag == "A1" else "A2 gold"
    caption = (
        rf"\caption{{Per-field model-vs-gold scores by cognitive phase on the LSAT "
        rf"batch ($n=20$, {gold_name}; P1 prompt, mean\,$\pm$\,std over 3 "
        rf"inference seeds: 44, 45, 46). Wt.\,F1 = support-weighted F1 "
        rf"(weighted macro-F1 over classes; multi-label fields scored over "
        rf"the label-set indicator); BERTScore = BERTScore F1 (roberta-large). "
        rf"\textbf{{Bold}} = best displayed mean per row, including ties. "
        rf"Shaded rows = phase-level averages over fields with valid "
        rf"predictions. {caption_extra}" + "}"
    )
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
        caption,
        rf"\label{{{label}}}",
        r"\begin{tabular}{p{3.2cm} l c c c c c c c c | c}",
        r"\hline",
        r"\textbf{Annotation Field} & \textbf{Metric} & \multicolumn{6}{c}{\textbf{Open-Source}} & \multicolumn{2}{c}{\textbf{Closed-Source}} & \textbf{Human} \\",
        rf"& & \textbf{{LLaMA-2}} & \textbf{{LLaMA-3}} & \textbf{{Qwen}} & \textbf{{Mistral}} & \textbf{{OLMo}} & \textbf{{Qwen3}} & \textbf{{GPT-4o}} & \textbf{{GPT-5.5}} & \textbf{{{human_col}}} \\",
        r"\hline",
    ]
    lines += build_rows(tag, human_tag, with_std=True)
    lines += [r"\end{tabular}", r"\end{table*}"]
    fname = f"lsat-baseline_{tag}_human.tex"
    with open(os.path.join(OUT_TABLE_DIR, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {fname}")
