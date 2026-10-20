#!/usr/bin/env python3
"""
Standalone Combined Reasoning Trace Score (A) computation -- Eq. (2) of
the manuscript, exactly. Weights are proportional to group size, so
every field contributes equally when all fields are filled; empty
expert fields are skipped within their group.

A_i = 16/41*EM_i + 4/41*J_multi_i + 21/41*J_span_i

  EM_i      = mean exact match over 16 single-label fields (revisionPhases
              compared as a label SET: exact match = identical sets)
  J_multi_i = mean set Jaccard over 4 multi-label fields
  J_span_i  = mean token-level Jaccard over 21 span/text fields
  (41 fields scored; field lists, normalization, tokenization and n/a
  handling below are an independent reimplementation matching
  scripts/compute_run2_trace_scores_v41.py in the original repo -- this
  file imports nothing from that repo.)

Field accounting: 44 annotation fields = 41 trace fields + 3 Socratic
question fields (excluded, dependent variable). All 41 trace fields are
now scored (previously, revisionPhases and revisionReason had no scoring
rule and were skipped -- that's the only field-coverage difference from
the package's earlier 39-field version).

Input: a single JSON file ("annotation_review.json" format) -- a list of
per-argument records, each:
    {"arg_index": <int>, "fields": {<field_name>: {"gold": <value>,
     "<model>": <value>, ...}, ...}}
This is the exact merged gold+prediction structure the original pipeline
scores from; this script does not parse raw model text.

Output: one CSV row per (arg_index, model) with EM, J_multi, J_span, A,
and the variants A_eq, A_eq_noSR, A_abl, A_eq_abl, A_old (the previously
published score, 0.390/0.098/0.512 over 12/4/23 fields, kept only for
side-by-side comparison).

Usage:
    python score_trace.py --review data/annotation_review.json \
        --models llama2 llama3 qwen qwen3 mistral olmo gpt-4o gpt-5.5 \
        --output out/per_argument_scores.csv
"""
import argparse
import json
import re
from pathlib import Path

# --------------------------------------------------------------------- #
# Field lists -- 16 EM + 4 ML + 21 FT = 41, matching
# compute_run2_trace_scores_v41.py exactly (Eq. 2 of the manuscript).
# --------------------------------------------------------------------- #
EM_FIELDS = [
    "stanceTarget", "knowledgeDomain", "reasoningType", "hasAssumption",
    "primaryDomain", "alternativeType", "inferenceStrength", "logicalFallacy",
    "changeDecision", "revisionType", "targetExplanation", "reasoningStructure",
    "span1", "span2", "span3", "revisionPhases",
]
ML_FIELDS = [
    "credibilityFactors", "trustworthiness", "biasDetection", "heuristicDetection",
]
FT_FIELDS = [
    "initialUnderstandingTarget", "coreClaim", "minorClaim", "premise",
    "paraphrasingUnderstanding", "missingComponent", "positiveConsequences",
    "negativeConsequences", "alternativeKeywords", "trustworthinessExplanation",
    "errorDetection", "goodEvidence", "badEvidence", "fallacySpan1", "fallacySpan2",
    "noFallacyExplanation",
    "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY",
    "revisionReason",
]
assert len(EM_FIELDS) == 16 and len(ML_FIELDS) == 4 and len(FT_FIELDS) == 21

# revisionPhases' gold is list-valued; compare as a label SET rather than
# a plain string (exact match = identical sets, not string equality).
EM_SET_FIELDS = {"revisionPhases"}

FIELD_TYPE = {f: "EM" for f in EM_FIELDS}
FIELD_TYPE.update({f: "ML" for f in ML_FIELDS})
FIELD_TYPE.update({f: "FT" for f in FT_FIELDS})
assert len(FIELD_TYPE) == 41

# Six Facione phases -> field membership (for A_eq / A_eq_noSR / A_eq_abl).
# Self-Regulation now includes revisionPhases and revisionReason (they
# were unscored before, so absent from every phase in the 39-field
# version of this file).
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
                         "changeDecision", "revisionType", "revisionPhases", "revisionReason"],
}
PHASES = list(PHASE_FIELDS.keys())
NON_SR_PHASES = [p for p in PHASES if p != "Self-Regulation"]
_all_phase_fields = [f for v in PHASE_FIELDS.values() for f in v]
assert len(_all_phase_fields) == 41 == len(set(_all_phase_fields))
assert sorted(_all_phase_fields) == sorted(FIELD_TYPE)

W_EM, W_ML, W_FT = 16 / 41, 4 / 41, 21 / 41

# A_abl: drop span1/2/3 (EM) and biasDetection/heuristicDetection (ML) ->
# 13 + 2 + 21 = 36 fields, reweighted by group size.
ABL_EXCLUDED = {"span1", "span2", "span3", "biasDetection", "heuristicDetection"}
EM_ABL = [f for f in EM_FIELDS if f not in ABL_EXCLUDED]
ML_ABL = [f for f in ML_FIELDS if f not in ABL_EXCLUDED]
FT_ABL = [f for f in FT_FIELDS if f not in ABL_EXCLUDED]
assert len(EM_ABL) == 13 and len(ML_ABL) == 2 and len(FT_ABL) == 21
W_EM_ABL, W_ML_ABL, W_FT_ABL = 13 / 36, 2 / 36, 21 / 36

# Previous (published) grouping, kept only as "A_old" for side-by-side
# comparison -- not part of Eq. (2).
OLD_CAT_FIELDS = [
    "stanceTarget", "knowledgeDomain", "reasoningType", "hasAssumption",
    "primaryDomain", "alternativeType", "inferenceStrength", "logicalFallacy",
    "changeDecision", "revisionType", "targetExplanation", "reasoningStructure",
]
OLD_ML_FIELDS = ML_FIELDS
OLD_FT_FIELDS = [
    "initialUnderstandingTarget", "coreClaim", "minorClaim", "premise",
    "paraphrasingUnderstanding", "missingComponent", "positiveConsequences",
    "negativeConsequences", "alternativeKeywords", "trustworthinessExplanation",
    "errorDetection", "goodEvidence", "badEvidence", "fallacySpan1", "fallacySpan2",
    "span1", "span2", "span3", "noFallacyExplanation",
    "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY",
]
assert len(OLD_CAT_FIELDS) == 12 and len(OLD_FT_FIELDS) == 23
OLD_FIELD_TYPE = {f: "CAT" for f in OLD_CAT_FIELDS}
OLD_FIELD_TYPE.update({f: "ML" for f in OLD_ML_FIELDS})
OLD_FIELD_TYPE.update({f: "FT" for f in OLD_FT_FIELDS})
W_OLD_CAT, W_OLD_ML, W_OLD_FT = 0.390, 0.098, 0.512


# --------------------------------------------------------------------- #
# Normalization, Jaccard -- independent reimplementations
# --------------------------------------------------------------------- #

def norm_str(v) -> str:
    if isinstance(v, list):
        v = v[0] if v else ""
    return re.sub(r"\s+", " ", str(v or "").strip().lower())


def norm_list(v) -> set:
    if isinstance(v, list):
        return {x.strip().lower() for x in v if str(x).strip()}
    if isinstance(v, str) and v.strip():
        return {x.strip().lower() for x in re.split(r"[|;,]", v) if x.strip()}
    return set()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Token-level (word-split) Jaccard, matching iaa_core.metrics.jaccard_similarity."""
    if text1 is None or text2 is None or text1 == "" or text2 == "":
        return 0.0
    tokens1 = set(str(text1).lower().split())
    tokens2 = set(str(text2).lower().split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    union = tokens1 | tokens2
    return len(tokens1 & tokens2) / len(union) if union else 0.0


# --------------------------------------------------------------------- #
# Per-field unit score (1 value per field per argument per model)
# --------------------------------------------------------------------- #

def field_unit_score(fields: dict, model: str, field: str):
    """Returns a float in [0,1], or None if this field should be skipped
    for this argument (gold is n/a/empty, or model is UNMATCHED)."""
    ftype = FIELD_TYPE[field]
    fd = fields.get(field, {})
    pred_raw = fd.get(model, "")
    if pred_raw == "UNMATCHED":
        return None
    if ftype == "EM":
        if field in EM_SET_FIELDS:
            g = norm_list(fd.get("gold", ""))
            if not g:
                return None
            return 1.0 if g == norm_list(pred_raw) else 0.0
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
    # FT
    g = norm_str(fd.get("gold", ""))
    if g in ("n/a", ""):
        return None
    return jaccard_similarity(g, norm_str(pred_raw))


def field_unit_score_old(fields: dict, model: str, field: str):
    """Previous (published) scoring, for the A_old comparison column --
    byte-for-byte as the package's earlier 0.390/0.098/0.512 version."""
    ftype = OLD_FIELD_TYPE[field]
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


def all_field_unit_scores(fields: dict, model: str) -> dict:
    return {f: field_unit_score(fields, model, f) for f in FIELD_TYPE}


def all_field_unit_scores_old(fields: dict, model: str) -> dict:
    return {f: field_unit_score_old(fields, model, f) for f in OLD_FIELD_TYPE}


def grouped_mean(unit_scores: dict, field_subset):
    vals = [unit_scores[f] for f in field_subset if unit_scores[f] is not None]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def phase_mean_with_skip(phase_scores: dict, phase_ns: dict, phases_to_use):
    """Average phase scores for one argument, skipping any phase with zero
    valid fields for that argument."""
    valid = [phase_scores[p] for p in phases_to_use if phase_ns[p] > 0]
    skipped = len(phases_to_use) - len(valid)
    if not valid:
        return None, skipped
    return sum(valid) / len(valid), skipped


def _ablated_variant(unit: dict):
    em_abl, _ = grouped_mean(unit, EM_ABL)
    ml_abl, _ = grouped_mean(unit, ML_ABL)
    ft_abl, _ = grouped_mean(unit, FT_ABL)
    a_abl = W_EM_ABL * em_abl + W_ML_ABL * ml_abl + W_FT_ABL * ft_abl
    phase_scores, phase_ns = {}, {}
    for phase in PHASES:
        subset = [f for f in PHASE_FIELDS[phase] if f not in ABL_EXCLUDED]
        phase_scores[phase], phase_ns[phase] = grouped_mean(unit, subset)
    a_eq_abl, _ = phase_mean_with_skip(phase_scores, phase_ns, PHASES)
    return a_abl, a_eq_abl


def per_arg_row(entry: dict, model: str) -> dict:
    fields = entry["fields"]
    unit = all_field_unit_scores(fields, model)
    old_unit = all_field_unit_scores_old(fields, model)

    em, _ = grouped_mean(unit, EM_FIELDS)
    j_multi, _ = grouped_mean(unit, ML_FIELDS)
    j_span, _ = grouped_mean(unit, FT_FIELDS)
    A = W_EM * em + W_ML * j_multi + W_FT * j_span

    phase_scores, phase_ns = {}, {}
    for phase in PHASES:
        phase_scores[phase], phase_ns[phase] = grouped_mean(unit, PHASE_FIELDS[phase])
    A_eq, _ = phase_mean_with_skip(phase_scores, phase_ns, PHASES)
    A_eq_noSR, _ = phase_mean_with_skip(phase_scores, phase_ns, NON_SR_PHASES)

    A_abl, A_eq_abl = _ablated_variant(unit)

    old_cat, _ = grouped_mean(old_unit, OLD_CAT_FIELDS)
    old_ml, _ = grouped_mean(old_unit, OLD_ML_FIELDS)
    old_ft, _ = grouped_mean(old_unit, OLD_FT_FIELDS)
    A_old = W_OLD_CAT * old_cat + W_OLD_ML * old_ml + W_OLD_FT * old_ft

    return {
        "arg_index": int(entry["arg_index"]), "model": model,
        "EM": em, "J_multi": j_multi, "J_span": j_span,
        "A": A, "A_eq": A_eq, "A_eq_noSR": A_eq_noSR,
        "A_abl": A_abl, "A_eq_abl": A_eq_abl, "A_old": A_old,
    }


def score_review(review: list, models) -> list:
    rows = []
    for model in models:
        for entry in review:
            rows.append(per_arg_row(entry, model))
    return rows


def write_csv(rows, out_path: Path):
    cols = ["arg_index", "model", "EM", "J_multi", "J_span",
            "A", "A_eq", "A_eq_noSR", "A_abl", "A_eq_abl", "A_old"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(repr(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", required=True, help="Path to annotation_review.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    rows = score_review(review, args.models)
    write_csv(rows, Path(args.output))

    for model in args.models:
        model_rows = [r for r in rows if r["model"] == model]
        mean_a = sum(r["A"] for r in model_rows) / len(model_rows)
        print(f"{model:10s} n={len(model_rows):3d}  A_mean={mean_a:.6f}")
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
