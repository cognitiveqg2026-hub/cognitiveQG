#!/usr/bin/env python3
"""
Parameterized copy of the project's compute_iaa.py (annotated_data/compute_iaa.py)
— the methodology behind the manuscript IAA table. Reads two aligned annotation
JSONs (A2, A1), computes per-field cat/multi/span IAA metrics, and dumps a
structured JSON for LaTeX-table generation.

Field spec, phase grouping, and metric definitions are copied verbatim from
compute_iaa.py; only the inputs/outputs are parameterized.

Usage:
    python scripts/compute_iaa_generic.py --a2 A2.json --a1 A1.json --out results/iaa/iaa_X.json
"""

import argparse
import json

from sklearn.metrics import cohen_kappa_score
from rouge_score import rouge_scorer as rouge_lib
from bert_score import score as bert_score_fn

ap = argparse.ArgumentParser()
ap.add_argument("--a1", required=True)
ap.add_argument("--a2", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

with open(args.a1) as f:
    a1_anns = json.load(f)["annotations"]
with open(args.a2) as f:
    a2_anns = json.load(f)["annotations"]

N = len(a1_anns)


# ── Helper functions (verbatim) ───────────────────────────────────────────────
def jaccard(a, b):
    a, b = set(str(a).lower().split()), set(str(b).lower().split())
    if not a and not b: return None
    if not a or not b: return None
    return len(a & b) / len(a | b)


def rouge_l(a, b):
    if not a or not b: return None
    scorer = rouge_lib.RougeScorer(['rougeL'], use_stemmer=False)
    return scorer.score(a, b)['rougeL'].fmeasure


def pabak(oa): return 2 * oa - 1


def gwet_ac(cats, a_labels, b_labels):
    n = len(a_labels)
    if n == 0: return None
    oa = sum(a == b for a, b in zip(a_labels, b_labels)) / n
    all_labels = list(cats)
    pa = {c: (a_labels.count(c) + b_labels.count(c)) / (2 * n) for c in all_labels}
    pe = sum(p * (1 - p) for p in pa.values()) / (len(all_labels) - 1) if len(all_labels) > 1 else 0
    if pe >= 1: return None
    return (oa - pe) / (1 - pe)


def safe_kappa(a, b):
    if len(set(a + b)) < 2: return 0.0
    try:    return cohen_kappa_score(a, b)
    except: return None


def obs_agree(a, b):
    n = len(a)
    return sum(x == y for x, y in zip(a, b)) / n if n else None


def list_to_str(v):
    if isinstance(v, list): return ' '.join(sorted(str(x) for x in v))
    return str(v) if v else ''


def pairs(field, empty_vals=('', None, [], 'none', 'None')):
    out = []
    for p, k in zip(a1_anns, a2_anns):
        out.append((p.get(field), k.get(field)))
    return out


def nonempty_pairs(field):
    return [(p, k) for p, k in pairs(field)
            if p not in ('', None) and k not in ('', None)
            and str(p).strip() and str(k).strip()]


def bert_avg(field):
    ps = nonempty_pairs(field)
    if not ps: return None
    refs = [str(p) for p, _ in ps]
    hyps = [str(k) for _, k in ps]
    _, _, F1 = bert_score_fn(hyps, refs, lang='en', model_type="roberta-large",
                             rescale_with_baseline=True, verbose=False)
    return float(F1.mean())


def cat_metrics(field, all_cats=None):
    ps = [(str(p), str(k)) for p, k in pairs(field)
          if p not in ('', None, []) and k not in ('', None, [])]
    if len(ps) < 2: return (None, None, None, None)
    a_l = [x[0] for x in ps]; b_l = [x[1] for x in ps]
    cats = all_cats or sorted(set(a_l + b_l))
    oa = obs_agree(a_l, b_l)
    kap = safe_kappa(a_l, b_l)
    pb = pabak(oa)
    gw = gwet_ac(cats, a_l, b_l)
    return kap, pb, oa, gw


def multicheck_metrics(field, all_cats=None):
    ps = pairs(field)
    exact = sum(sorted(str(p) if isinstance(p, list) else [p]) ==
                sorted(str(k) if isinstance(k, list) else [k])
                for p, k in ps) / N
    jacs = [x for x in (jaccard(list_to_str(p), list_to_str(k)) for p, k in ps) if x is not None]
    jac_avg = sum(jacs) / len(jacs) if jacs else None
    a_l = [list_to_str(p) for p, _ in ps]
    b_l = [list_to_str(k) for _, k in ps]
    cats = sorted(set(a_l + b_l))
    oa = obs_agree(a_l, b_l)
    kap = safe_kappa(a_l, b_l)
    gw = gwet_ac(cats, a_l, b_l)
    return kap, exact, jac_avg, gw


def span_metrics(field):
    ps = nonempty_pairs(field)
    if not ps: return (None, None, None)
    jac = sum(x for x in (jaccard(p, k) for p, k in ps) if x is not None) / len(ps)
    rl = sum(x for x in (rouge_l(str(p), str(k)) for p, k in ps) if x is not None) / len(ps)
    bs = bert_avg(field)
    return jac, rl, bs


print(f"Computing IAA metrics on {N} pairs (BERT may take a minute)...\n")
results = {}

results['Initial Understanding']    = ('span', span_metrics('initialUnderstandingTarget'))
results['Stance Target']            = ('cat',  cat_metrics('stanceTarget', ['favor', 'against', 'neutral']))
results['Knowledge Domain']         = ('cat',  cat_metrics('knowledgeDomain',
                                       ['technical-scientific', 'legal-policy', 'cultural-social',
                                        'economic-business', 'government-law', 'no-specialized']))
results['Paraphrase of Core Claim'] = ('span', span_metrics('paraphrasingUnderstanding'))
results['Core Claim']               = ('span', span_metrics('coreClaim'))
results['Minor Claim']              = ('span', span_metrics('minorClaim'))
results['Premise']                  = ('span', span_metrics('premise'))
results['Reasoning Type']           = ('cat',  cat_metrics('reasoningType', ['inductive', 'deductive', 'none']))
results['Has Assumption']           = ('cat',  cat_metrics('hasAssumption', ['yes', 'no']))
results['Missing Component']        = ('span', span_metrics('missingComponent'))
results['Positive Consequences']    = ('span', span_metrics('positiveConsequences'))
results['Negative Consequences']    = ('span', span_metrics('negativeConsequences'))
results['Primary Domain Affected']  = ('cat',  cat_metrics('primaryDomain',
                                       ['personal-physical', 'interpersonal', 'social',
                                        'organisational', 'government-policy-law', 'economic']))
results['Alternative Type']         = ('cat',  cat_metrics('alternativeType',
                                       ['applies-when', 'other-factors', 'misunderstands']))
results['Alternative Keywords']     = ('span', span_metrics('alternativeKeywords'))
results['Inference Score']          = ('cat',  cat_metrics('inferenceStrength', ['1', '2', '3']))
results['Credibility Factors']      = ('multi', multicheck_metrics('credibilityFactors'))
results['Logical Fallacy']          = ('cat',  cat_metrics('logicalFallacy',
                                       ['none', 'false-dilemma', 'hasty-generalization', 'ad-hominem',
                                        'strawman', 'circular-reasoning', 'false-cause']))
results['Trustworthiness']          = ('multi', multicheck_metrics('trustworthiness'))
results['Trust Explanation']        = ('span', span_metrics('trustworthinessExplanation'))
results['Target Explanation']       = ('cat',  cat_metrics('targetExplanation',
                                       ['explicitly-stated', 'repeated-emphasis', 'core-causal-claim',
                                        'evaluative-language']))
results['Reasoning Structure']      = ('cat',  cat_metrics('reasoningStructure',
                                       ['specific-cases', 'consequence-based', 'analogical',
                                        'sign-indicator', 'expert-authority', 'none']))
results['Span Eval. (span1)']       = ('cat',  cat_metrics('span1',
                                       ['major-claim', 'minor-claim', 'premise', 'not-available']))
results['Span Eval. (span2)']       = ('cat',  cat_metrics('span2',
                                       ['major-claim', 'minor-claim', 'premise', 'not-available']))
results['Span Eval. (span3)']       = ('cat',  cat_metrics('span3',
                                       ['major-claim', 'minor-claim', 'premise', 'not-available']))
results['Inductive/Ded. Term X']    = ('span', span_metrics('inductiveTermX'))
results['Inductive/Ded. Term Y']    = ('span', span_metrics('inductiveTermY'))
results['Good Evidence']            = ('span', span_metrics('goodEvidence'))
results['Bad Evidence']             = ('span', span_metrics('badEvidence'))
results['Fallacy Span']             = ('span', span_metrics('fallacySpan1'))
results['No Fallacy Explanation']   = ('span', span_metrics('noFallacyExplanation'))
results['Bias Detection']           = ('multi', multicheck_metrics('biasDetection'))
results['Heuristic Detection']      = ('multi', multicheck_metrics('heuristicDetection'))
results['Error Detection']          = ('span', span_metrics('errorDetection'))
results['Change Decision']          = ('cat',  cat_metrics('changeDecision', ['yes', 'no']))
results['Revision Phases']          = ('multi', multicheck_metrics('revisionPhases'))
results['Revision Reason']          = ('span', span_metrics('revisionReason'))
results['Revision Type']            = ('cat',  cat_metrics('revisionType',
                                       ['minor-adjustment', 'major-change', 'complete-revision',
                                        'no-change', '']))
results['Socratic Question 1']      = ('span', span_metrics('socraticQuestion1'))
results['Socratic Question 2']      = ('span', span_metrics('socraticQuestion2'))
results['Socratic Question 3']      = ('span', span_metrics('socraticQuestion3'))

sections = {
    'INTERPRETATION': ['Initial Understanding', 'Stance Target', 'Knowledge Domain'],
    'ANALYSIS': ['Paraphrase of Core Claim', 'Core Claim', 'Minor Claim', 'Premise', 'Reasoning Type', 'Has Assumption', 'Missing Component'],
    'INFERENCE': ['Positive Consequences', 'Negative Consequences', 'Primary Domain Affected', 'Alternative Type', 'Alternative Keywords'],
    'EVALUATION': ['Inference Score', 'Credibility Factors', 'Logical Fallacy', 'Trustworthiness', 'Trust Explanation'],
    'EXPLANATION': ['Target Explanation', 'Reasoning Structure',
                    'Span Eval. (span1)', 'Span Eval. (span2)', 'Span Eval. (span3)',
                    'Inductive/Ded. Term X', 'Inductive/Ded. Term Y', 'Good Evidence', 'Bad Evidence', 'Fallacy Span', 'No Fallacy Explanation'],
    'SELF-REGULATION': ['Bias Detection', 'Heuristic Detection', 'Error Detection', 'Change Decision',
                        'Revision Phases', 'Revision Reason', 'Revision Type'],
    'SOCRATIC QUESTIONS': ['Socratic Question 1', 'Socratic Question 2', 'Socratic Question 3'],
}

METRIC_KEYS = {
    'cat':   ['kappa', 'pabak', 'oa', 'gwet'],
    'multi': ['kappa', 'exact', 'jaccard', 'gwet'],
    'span':  ['jaccard', 'rouge_l', 'bert'],
}


def fmt(v): return '  -   ' if v is None else f"{v:+.3f}"


for section, fields in sections.items():
    print(f"\n== {section} ==")
    for f in fields:
        typ, vals = results[f]
        labels = {'span': ['Jaccard', 'ROUGE-L', 'BERT'],
                  'cat':  ['Kappa', 'PABAK', 'OA', 'Gwet'],
                  'multi': ['Kappa', 'Exact', 'Jaccard', 'Gwet']}[typ]
        cells = "  ".join(f"{lab}={fmt(v)}" for lab, v in zip(labels, vals))
        print(f"  {f:<33} {typ:<5} {cells}")

# Structured JSON for the LaTeX generator.
out = {"n": N, "sections": sections, "results": {}}
for field, (typ, vals) in results.items():
    out["results"][field] = {"type": typ,
                             "metrics": dict(zip(METRIC_KEYS[typ], vals))}

import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(f"\nSaved structured results -> {args.out}")
