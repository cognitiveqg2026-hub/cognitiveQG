#!/usr/bin/env python3
"""
dev3-12 (n=100) per-field baseline table for A2 gold. Copy of
generate_baseline_field_table_complete_v2_a2.py with ONE additional change:

  (D) Exclude the dev1-2 "annotator calibration round" (per
      docs/manuscript/access_final.tex: "initial calibration was conducted on
      dev1--dev2") from the dev1-12 pool before scoring.

      Same exclusion set as the A1 variant (generate_baseline_field_table_dev3-12.py),
      verified to apply identically to both annotators -- see that script's
      docstring for the full derivation. Kept set: {0..34} u {55..119} =
      dev3-6 (40) + dev7-12 main block (55) + separately appended dev12b (5)
      = 100 args.

      No new raw inference needed; --dev712-root stays the SAME shared,
      annotator-independent raw pool (results/baseline_complete_dev1-12/raw)
      used by the dev1-12 A2 table.

(A)/(B) (Wt.F1 metric patch, Inductive/Ded. Term X/Y rows) are unchanged from
generate_baseline_field_table_complete_v2_a2.py. This variant deliberately
does NOT include the Qwen3-14B/32B scale-series patch that the A1 script
has -- that asymmetry already exists between the two dev1-12 scripts and is
preserved here rather than normalized.

Run on the GPU host (roberta-large BERTScore), in the cognitiveQG venv:
    python scripts/generate_baseline_field_table_dev3-12_a2.py \
        --output docs/manuscript/model-baselinescore-dev3-12_kim.tex
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.abspath(os.path.dirname(__file__))

_spec = importlib.util.spec_from_file_location(
    "g712", os.path.join(SCRIPTS, "generate_baseline_field_table_dev7-12_rescaled.py"))
g7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g7)

MODELS, SEEDS = g7.MODELS, g7.SEEDS
MODEL_LABELS, METRIC_LABEL = g7.MODEL_LABELS, g7.METRIC_LABEL
EXPLANATION_AVG_FIELDS = g7.EXPLANATION_AVG_FIELDS
cfm = g7.cfm

# ── (A)+(B) patch the field registry in memory ───────────────────────────────
ML_TO_WF1 = {"credibilityFactors", "trustworthiness", "biasDetection", "heuristicDetection"}
METRIC_LABEL["WF1"] = r"Wt.\ F1"


def patch_phases(phases):
    out = []
    for phase, rows in phases:
        new_rows = []
        for display, metric, field in rows:
            if field in ML_TO_WF1:            # (A) Micro-F1 -> Wt. F1
                metric = "WF1"
            new_rows.append((display, metric, field))
            if field == "span3":              # (B) insert term rows after the span block
                new_rows.append((r"Inductive/Ded.\ Term X", "FT", "termX"))
                new_rows.append((r"Inductive/Ded.\ Term Y", "FT", "termY"))
        out.append((phase, new_rows))
    return out


PHASES = patch_phases(g7.PHASES)
g7.PHASES = PHASES
cfm.MODELS = MODELS
cfm.PHASE_FIELDS = {ph: [(f, disp, met) for (disp, met, f) in rows] for ph, rows in PHASES}


# ── (A) proper multi-label support-weighted F1 ───────────────────────────────
def weighted_f1_multilabel(gold_sets, pred_sets):
    labels = set()
    for s in gold_sets:
        labels |= s
    for s in pred_sets:
        labels |= s
    if not labels:
        return 0.0
    num = den = 0.0
    for c in labels:
        tp = sum(1 for g, p in zip(gold_sets, pred_sets) if c in g and c in p)
        fp = sum(1 for g, p in zip(gold_sets, pred_sets) if c not in g and c in p)
        fn = sum(1 for g, p in zip(gold_sets, pred_sets) if c in g and c not in p)
        support = tp + fn
        if support == 0:
            continue
        f1 = 0.0
        if tp > 0:
            pr, rc = tp / (tp + fp), tp / (tp + fn)
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        num += f1 * support
        den += support
    return num / den if den else 0.0


def compute_wf1_score(review, field, model):
    gs, ps = [], []
    for arg in review:
        entry = arg["fields"].get(field, {})
        g = cfm.norm_list(entry.get("gold", ""))
        p = cfm.norm_list(entry.get(model, ""))
        if entry.get(model) == "UNMATCHED" or not g:
            continue
        gs.append(g)
        ps.append(p)
    if not gs:
        return None
    return weighted_f1_multilabel(gs, ps)


def score_cell(field, metric, model, reviews, bert_caches):
    if metric == "WF1":
        vals = [compute_wf1_score(reviews[s], field, model) for s in SEEDS]
        vals = [v * 100.0 for v in vals if v is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
    return g7.score_cell(field, metric, model, reviews, bert_caches)


# ── (B) merged inductive/deductive term injection ────────────────────────────
def _term(d, slot):
    """(value, type) using whichever of inductive/deductive term<slot> is populated."""
    iv = str(d.get("inductiveTerm" + slot, "") or "").strip()
    if iv:
        return iv, "ind"
    dv = str(d.get("deductiveTerm" + slot, "") or "").strip()
    if dv:
        return dv, "ded"
    return "", None


def inject_terms(review, gold, preds, seed):
    for i, ann in enumerate(gold):
        idx = int(ann["argumentIndex"])
        for slot in ("X", "Y"):
            gval, gtype = _term(ann, slot)
            fe = {"gold": gval}
            for m in MODELS:
                pred = preds[m][seed].get(idx)
                if not pred:
                    fe[m] = "UNMATCHED"
                elif gtype == "ind":
                    fe[m] = str(pred.get("inductiveTerm" + slot, "") or "").strip()
                elif gtype == "ded":
                    fe[m] = str(pred.get("deductiveTerm" + slot, "") or "").strip()
                else:
                    fe[m] = str(pred.get("inductiveTerm" + slot, "")
                                or pred.get("deductiveTerm" + slot, "") or "").strip()
            review[i]["fields"]["term" + slot] = fe


def load_anns(path):
    d = json.load(open(path))
    return d.get("annotations", d.get("completedAnnotations", []))


def load_pooled_preds(dev712_root, offset):
    from parse_model_output import parse_model_output
    from extract_socratic_robust import extract_questions_robust
    preds = {m: {s: {} for s in SEEDS} for m in MODELS}
    for model in MODELS:
        path = os.path.join(dev712_root, model, "raw_output.json")
        if not os.path.exists(path):
            print(f"  WARNING missing {path}")
            continue
        for sample in json.load(open(path))["per_sample"]:
            idx = int(sample["argument_index"])
            for s in SEEDS:
                txt = sample.get("per_seed", {}).get(s, {}).get("generated_text", "")
                p = parse_model_output(txt) if txt else {}
                if txt:
                    qs = extract_questions_robust(txt)
                    p["socraticQuestion1"] = qs[0] if len(qs) > 0 else ""
                    p["socraticQuestion2"] = qs[1] if len(qs) > 1 else ""
                    p["socraticQuestion3"] = qs[2] if len(qs) > 2 else ""
                preds[model][s][idx] = p
    return preds


# ── (D) dev1-2 calibration-round exclusion (identical to the A1 variant) ───
EXCLUDE_DEV1_2_INDICES = set(range(35, 55))  # dev1b:35-39, dev1a:40-44, dev2:45-54
EXPECTED_DEV3_12_N = 100


def filter_dev3_12(combined_gold, preds, expected_total):
    kept = [a for a in combined_gold if int(a["argumentIndex"]) not in EXCLUDE_DEV1_2_INDICES]
    kept_idx = {int(a["argumentIndex"]) for a in kept}
    expected_idx = set(range(expected_total)) - EXCLUDE_DEV1_2_INDICES

    print(f"Excluded indices: 35-54 ({len(EXCLUDE_DEV1_2_INDICES)})")
    print(f"Kept index count: {len(kept)}")
    print("Kept index range: 0-34,55-119")

    if len(kept) != EXPECTED_DEV3_12_N:
        raise SystemExit(
            f"DEV3-12 FILTER ABORT (no scoring run): expected {EXPECTED_DEV3_12_N} args "
            f"after excluding dev1-2 calibration indices {sorted(EXCLUDE_DEV1_2_INDICES)}, "
            f"got {len(kept)} from an input of {len(combined_gold)}. Re-verify the index "
            f"split before trusting any partial result."
        )
    if kept_idx != expected_idx:
        raise SystemExit(
            "DEV3-12 FILTER ABORT (no scoring run): kept argumentIndex set does not equal "
            f"the expected complement of the exclusion set. Missing={sorted(expected_idx - kept_idx)} "
            f"Unexpected={sorted(kept_idx - expected_idx)}"
        )

    all_ok = True
    for m in MODELS:
        for s in SEEDS:
            have = set(preds[m][s].keys())
            missing = kept_idx - have
            if missing:
                all_ok = False
                print(f"Prediction index alignment: FAIL ({m}, seed {s}) "
                      f"missing={sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        if all(not (kept_idx - set(preds[m][s].keys())) for s in SEEDS):
            print(f"Prediction index alignment: PASS ({m}, {len(kept_idx)}/{len(kept_idx)})")
    if not all_ok:
        raise SystemExit(
            "DEV3-12 FILTER ABORT (no scoring run): one or more models/seeds are missing "
            "predictions for kept dev3-12 indices. See per-model FAIL lines above."
        )

    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev712-root", default="results/baseline_complete_dev1-12/raw")
    ap.add_argument("--dev712-gold", default="data/dev1-12_a2/k_dev1-12_gold.json")
    ap.add_argument("--output", default="docs/manuscript/model-baselinescore-dev3-12_kim.tex")
    ap.add_argument("--span-json", default=None)
    args = ap.parse_args()
    span_data = json.load(open(args.span_json)) if args.span_json else None

    combined_gold = load_anns(args.dev712_gold)
    print(f"unified complete-run gold (A2): {len(combined_gold)} args")
    preds = load_pooled_preds(args.dev712_root, len(combined_gold))

    combined_gold = filter_dev3_12(combined_gold, preds, expected_total=len(combined_gold))

    reviews, bert_caches = {}, {}
    for s in SEEDS:
        print(f"Scoring seed {s} ...")
        reviews[s] = g7.build_review(combined_gold, preds, s)
        inject_terms(reviews[s], combined_gold, preds, s)      # (B)
        bert_caches[s] = cfm.precompute_bert_scores(reviews[s])
    n = len(reviews[SEEDS[0]])
    print(f"pooled n = {n}")

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
                cells = {m: score_cell(field, metric, m, reviews, bert_caches) for m in MODELS}
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
        rf"dev3--12 batch (dev1--2 calibration round excluded; $n={n}$, A2 gold; "
        r"P4 prompt, mean\,$\pm$\,std over 3 "
        r"inference seeds: 44, 45, 46). Wt.\,F1 = support-weighted F1 (weighted "
        r"macro-F1 over classes; multi-label fields scored over the label-set "
        r"indicator); BERTScore = BERTScore F1 (roberta-large). Span Eval.\ and "
        r"Inductive/Ded.\ Term X/Y use BERTScore; each term row scores the model's "
        r"term of whichever slot (inductive or deductive) the gold populates. "
        r"\textbf{Bold} = best per row. Shaded rows = phase-level "
        r"averages over fields with valid predictions. $^\dagger$Explanation phase "
        r"avg over 4 universal fields. {--} = no valid predictions. dev3-6 "
        r"predictions are single-seed (reused across seeds).}",
        rf"\label{{tab:model-baseline-jac-gwet-dev3-12-kim}}",
        r"\end{table*}",
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    open(args.output, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {args.output} (n={n})")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(SCRIPTS, "utils"))
    main()
