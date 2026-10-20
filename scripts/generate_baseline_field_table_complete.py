#!/usr/bin/env python3
"""
Unified dev1-12 (n=120) per-field baseline table — pools predictions from:
  * dev1-6  (60): results/baseline/comparison_run2_20260525/annotation_review.json
                  (already-parsed field preds per model; single pred, reused across seeds)
  * dev7-12 (55): results/baseline_dev7-12/raw_run2_20260525/<model>/raw_output.json
  * dev12b  (5) : results/baseline_dev12b/raw/<model>/raw_output.json  (optional)

Same layout/metrics as the dev7-12 generator (Wt.F1 / Micro-F1 / BERTScore,
mean±std over 3 seeds, bold best-in-row, shaded Phase-avg rows). If dev12b raw is
absent the table is the interim n=115; with it, n=120.

Run on the GPU host (roberta-large BERTScore):
    python scripts/generate_baseline_field_table_dev1-12.py \
        --output docs/manuscript/model-baselinescore-dev1-12.tex
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.abspath(os.path.dirname(__file__))

# Load the dev7-12 generator module (hyphenated filename -> importlib).
_spec = importlib.util.spec_from_file_location(
    "g712", os.path.join(SCRIPTS, "generate_baseline_field_table_dev7-12_rescaled.py"))
g7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g7)

MODELS, SEEDS, PHASES = g7.MODELS, g7.SEEDS, g7.PHASES
MODEL_LABELS, METRIC_LABEL = g7.MODEL_LABELS, g7.METRIC_LABEL
EXPLANATION_AVG_FIELDS, PRED_FIELD_MAP = g7.EXPLANATION_AVG_FIELDS, g7.PRED_FIELD_MAP
cfm = g7.cfm


def load_anns(path):
    d = json.load(open(path))
    return d.get("annotations", d.get("completedAnnotations", []))


def load_dev16_review(path):
    """dev1-6 annotation_review.json -> cfm-style entries (already parsed, all seeds same)."""
    review = json.load(open(path))
    return [{"fields": e["fields"]} for e in review]


def load_pooled_preds(dev712_root, dev12b_root, offset):
    """preds[model][seed][arg_index]: dev7-12 at 0.., dev12b at offset.."""
    from parse_model_output import parse_model_output
    from extract_socratic_robust import extract_questions_robust  # robust question parse
    preds = {m: {s: {} for s in SEEDS} for m in MODELS}
    for model in MODELS:
        for root, base in ((dev712_root, 0), (dev12b_root, offset)):
            path = os.path.join(root, model, "raw_output.json")
            if not os.path.exists(path):
                if base == 0:
                    print(f"  WARNING missing {path}")
                continue
            for sample in json.load(open(path))["per_sample"]:
                idx = base + int(sample["argument_index"])
                for s in SEEDS:
                    txt = sample.get("per_seed", {}).get(s, {}).get("generated_text", "")
                    p = parse_model_output(txt) if txt else {}
                    if txt:  # override socratic Q with the robust extractor (template parser misses variants)
                        qs = extract_questions_robust(txt)
                        p["socraticQuestion1"] = qs[0] if len(qs) > 0 else ""
                        p["socraticQuestion2"] = qs[1] if len(qs) > 1 else ""
                        p["socraticQuestion3"] = qs[2] if len(qs) > 2 else ""
                    preds[model][s][idx] = p
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev16-review", default="results/baseline/comparison_run2_20260525/annotation_review.json")
    ap.add_argument("--dev712-root", default="results/baseline_complete_dev1-12/raw")
    ap.add_argument("--dev712-gold", default="data/dev1-12_A1/p_dev1-12_gold.json")
    ap.add_argument("--dev12b-root", default="results/__none__")
    ap.add_argument("--dev12b-gold", default="data/__none__")
    ap.add_argument("--output", default="docs/manuscript/model-baselinescore-dev1-12_RESCALED.tex")
    ap.add_argument("--span-json", default=None, help="pooled span_bertscore JSON to fill span1/2/3")
    args = ap.parse_args()
    span_data = json.load(open(args.span_json)) if args.span_json else None

    # UNIFIED complete run: all 120 args come from one raw dir on the FIXED gold.
    # (No separate dev1-6 review, no dev12b; predictions loaded per-seed from raw.)
    dev16 = []
    combined_gold = load_anns(args.dev712_gold)           # 120 args, argumentIndex 0..119
    print(f"unified complete-run gold: {len(combined_gold)} args")

    preds = load_pooled_preds(args.dev712_root, args.dev12b_root, len(combined_gold))

    cfm.MODELS = MODELS
    cfm.PHASE_FIELDS = {ph: [(f, disp, met) for (disp, met, f) in rows] for ph, rows in PHASES}

    reviews, bert_caches = {}, {}
    for s in SEEDS:
        print(f"Scoring seed {s} ...")
        reviews[s] = dev16 + g7.build_review(combined_gold, preds, s)
        bert_caches[s] = cfm.precompute_bert_scores(reviews[s])
    n = len(reviews[SEEDS[0]])
    print(f"pooled n = {n}")

    # ── Build the table (same layout as the dev7-12 generator) ────────────────
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
        phase_means = {m: [] for m in MODELS}
        for display, metric, field in rows:
            if span_data is not None and field in ("span1", "span2", "span3"):
                cells = {}
                for m in MODELS:
                    sd = span_data.get(m, {}).get(field)
                    cells[m] = (sd["F1"], 0.0) if sd and sd.get("n", 0) > 0 else (None, None)
            else:
                cells = {m: g7.score_cell(field, metric, m, reviews, bert_caches) for m in MODELS}
            means = [cells[m][0] for m in MODELS]
            best = max([v for v in means if v is not None], default=None)
            bold_ok = best is not None and best > 0.05
            row = [display, METRIC_LABEL[metric]]
            for m in MODELS:
                mean, std = cells[m]
                row.append(g7.cell_tex(mean, std, bold_ok and mean is not None and abs(mean - best) < 1e-9))
                if ((phase != "Explanation") or (field in EXPLANATION_AVG_FIELDS)) and mean is not None:
                    phase_means[m].append(mean)
            lines.append(" & ".join(row) + r" \\")
        avg = {m: (np.mean(phase_means[m]) if phase_means[m] else None) for m in MODELS}
        best_avg = max([v for v in avg.values() if v is not None], default=None)
        bold_avg_ok = best_avg is not None and best_avg > 0.05
        label = r"\textit{Phase avg}" + (r"$^\dagger$" if phase == "Explanation" else "")
        cols = []
        for m in MODELS:
            v = avg[m]
            cols.append("--" if v is None else
                        (rf"\textbf{{{v:.1f}}}" if bold_avg_ok and abs(v - best_avg) < 1e-9 else f"{v:.1f}"))
        lines.append(rf"\rowcolor[gray]{{0.92}} {label} & & " + " & ".join(cols) + r" \\")
        lines.append(r"\hline")

    lines += [
        r"\end{tabular}",
        rf"\caption{{Per-field model-vs-gold scores by cognitive phase on the "
        rf"dev1--12 batch ($n={n}$, A1 gold; P4 prompt, mean\,$\pm$\,std over 3 "
        r"inference seeds: 44, 45, 46). Wt.\,F1 = "
        r"categorical weighted macro-F1; Micro-F1 = multi-label set micro-F1; "
        r"BERTScore = BERTScore F1 (roberta-large). \textbf{Bold} = best per row. "
        r"Shaded rows = phase-level averages over fields with valid predictions. "
        r"$^\dagger$Explanation phase avg over 4 universal fields. {--} = no valid "
        r"predictions. dev1-6 predictions are single-seed (reused across seeds).}",
        rf"\label{{tab:model-baseline-jac-gwet-Run2P4structured}}",
        r"\end{table*}",
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    open(args.output, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {args.output} (n={n})")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(SCRIPTS, "utils"))
    main()
