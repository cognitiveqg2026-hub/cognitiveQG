#!/usr/bin/env python3
"""
COMPLETE variant (duplicate of compute_run2_trace_scores.py).
Difference: the FT (token-Jaccard) field list additionally includes the
inductive/deductive Term X/Y fields, so the Combined Reasoning Trace Score
(X-axis of the correlation figure) reflects them. Per argument only the
filled pair is scored (empty-gold fields are skipped), matching how gold
stores exactly one of {deductive,inductive} term pair.

Compute per-argument reasoning trace scores for specified models and
append/update rows in reasoning_trace_scores.csv.

Columns:
  arg_index, model, cat_score, multi_jac, span_jac, span_bert,
  trace_jac, trace_bert

Formulas:
  cat_score  = fraction of CAT fields where norm(model) == norm(gold)
  multi_jac  = avg set-Jaccard for ML fields
  span_jac   = avg token-Jaccard for FT fields (iaa_core jaccard_similarity)
  trace_jac  = 0.390*cat_score + 0.098*multi_jac + 0.512*span_jac
  span_bert / trace_bert = 0.0 (BERT not recomputed here)

Usage:
  python scripts/compute_run2_trace_scores.py \
      --models gpt-4o gpt-5.5 \
      --append
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iaa_core.metrics import jaccard_similarity

REVIEW_PATH = Path("results/baseline_dev1-12/annotation_review.json")
CSV_PATH    = Path("results/baseline_dev1-12/reasoning_trace_scores.csv")

W_CAT  = 0.390
W_ML   = 0.098
W_SPAN = 0.512

CAT_FIELDS = [
    "stanceTarget", "knowledgeDomain", "reasoningType", "hasAssumption",
    "primaryDomain", "alternativeType", "inferenceStrength", "logicalFallacy",
    "changeDecision", "revisionType", "targetExplanation", "reasoningStructure",
]
ML_FIELDS = [
    "credibilityFactors", "trustworthiness", "biasDetection", "heuristicDetection",
]
FT_FIELDS = [
    "initialUnderstandingTarget", "coreClaim", "minorClaim", "premise",
    "paraphrasingUnderstanding", "missingComponent", "positiveConsequences",
    "negativeConsequences", "alternativeKeywords", "trustworthinessExplanation",
    "errorDetection", "goodEvidence", "badEvidence", "fallacySpan1", "fallacySpan2",
    "span1", "span2", "span3", "noFallacyExplanation",
    # COMPLETE-variant additions: inductive/deductive Term X/Y (only the filled
    # pair per arg contributes; empty-gold fields are skipped by the n/a guard).
    "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY",
]


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


def per_arg_scores(review: list, model: str) -> list[dict]:
    rows = []
    for entry in review:
        arg_idx = int(entry["arg_index"])
        fields  = entry["fields"]

        # CAT score: exact match fraction
        cat_hits, cat_total = 0, 0
        for f in CAT_FIELDS:
            fd = fields.get(f, {})
            g  = norm_str(fd.get("gold", ""))
            p  = fd.get(model, "")
            if p == "UNMATCHED" or g in ("n/a", ""):
                continue
            p = norm_str(p)
            cat_total += 1
            if g == p:
                cat_hits += 1
        cat_score = cat_hits / cat_total if cat_total else 0.0

        # ML score: avg set-Jaccard
        ml_scores = []
        for f in ML_FIELDS:
            fd = fields.get(f, {})
            if fd.get(model) == "UNMATCHED":
                continue
            g = norm_list(fd.get("gold", ""))
            p = norm_list(fd.get(model, ""))
            if not g:
                continue
            union = g | p
            ml_scores.append(len(g & p) / len(union) if union else 0.0)
        multi_jac = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0

        # FT score: avg token-Jaccard
        ft_scores = []
        for f in FT_FIELDS:
            fd = fields.get(f, {})
            if fd.get(model) == "UNMATCHED":
                continue
            g = norm_str(fd.get("gold", ""))
            p = norm_str(fd.get(model, ""))
            if g in ("n/a", ""):
                continue
            ft_scores.append(jaccard_similarity(g, p))
        span_jac = sum(ft_scores) / len(ft_scores) if ft_scores else 0.0

        trace_jac = W_CAT * cat_score + W_ML * multi_jac + W_SPAN * span_jac

        rows.append({
            "arg_index":  arg_idx,
            "model":      model,
            "cat_score":  cat_score,
            "multi_jac":  multi_jac,
            "span_jac":   span_jac,
            "span_bert":  0.0,
            "trace_jac":  trace_jac,
            "trace_bert": 0.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review",  default=str(REVIEW_PATH))
    ap.add_argument("--output",  default=str(CSV_PATH))
    ap.add_argument("--models",  nargs="+", required=True)
    ap.add_argument("--append",  action="store_true",
                    help="Load existing CSV and replace rows for specified models")
    args = ap.parse_args()

    review = json.loads(Path(args.review).read_text(encoding="utf-8"))

    all_rows = []
    for model in args.models:
        rows = per_arg_scores(review, model)
        all_rows.extend(rows)
        avg_trace = sum(r["trace_jac"] for r in rows) / len(rows)
        avg_span  = sum(r["span_jac"]  for r in rows) / len(rows)
        avg_cat   = sum(r["cat_score"] for r in rows) / len(rows)
        print("%-12s  n=%d  cat=%.3f  multi_jac=%.3f  span_jac=%.3f  trace_jac=%.3f" % (
            model, len(rows), avg_cat,
            sum(r["multi_jac"] for r in rows)/len(rows),
            avg_span, avg_trace))

    new_df = pd.DataFrame(all_rows)

    out_path = Path(args.output)
    if args.append and out_path.exists():
        existing = pd.read_csv(out_path)
        # Drop old rows for these models, then append
        existing = existing[~existing["model"].isin(args.models)]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.sort_values(["arg_index", "model"], inplace=True)
        combined.to_csv(out_path, index=False)
        print("Appended → %s  (%d rows total)" % (out_path, len(combined)))
    else:
        new_df.sort_values(["arg_index", "model"], inplace=True)
        new_df.to_csv(out_path, index=False)
        print("Wrote → %s  (%d rows)" % (out_path, len(new_df)))


if __name__ == "__main__":
    main()
