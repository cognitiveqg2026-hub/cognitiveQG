#!/usr/bin/env python3
"""
parse_model_output.py

Parses raw model-generated text into a normalized dict matching the gold
annotation schema (p_combined_all_instances_with_ids.json).

Handles all five model output styles:
  - Tab-asterisk:   \t* fieldName: value          (llama2, qwen, olmo)
  - Dash-space:     - fieldName: value             (llama3)
  - Bold-numbered:  1) **FIELD**\n  fieldName: v   (deepseek)
  - Plain numbered: 1. FIELD\n  fieldName: v       (any)
  - Inline:         fieldName: value               (fallback)

Also strips DeepSeek-R1 <think>...</think> chain-of-thought blocks.

Usage (standalone):
    python scripts/parse_model_output.py \
        --input  results/deepseek_pipeline_60args_2026-05-09.json \
        --output results/deepseek_normalized_60args_2026-05-09.json

    # or parse a single raw text string for testing:
    python scripts/parse_model_output.py --text "stanceTarget: against\n..."
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher


# ── Value normalization maps ──────────────────────────────────────────────────
# Model output value  →  gold schema value

_STANCE = {
    "favor": "favor", "against": "against", "neutral": "neutral",
}

_KNOWLEDGE_DOMAIN = {
    "technical/scientific": "technical-scientific",
    "legal/policy": "legal-policy",
    "cultural/social": "cultural-social",
    "economic/business": "economic-business",
    "government/law": "government-law",
    "no specialized knowledge": "no-specialized",
    "none": "no-specialized",
}

_REASONING_TYPE = {
    "inductive": "inductive",
    "deductive": "deductive",
    "none of the above": "none-of-the-above",
    "none": "none-of-the-above",
}

_HAS_ASSUMPTION = {
    "yes=complete": "yes",
    "yes": "yes",
    "no=missing components": "no",
    "no=missing component": "no",
    "missing component": "no",
    "missing components": "no",
    "no": "no",
}

_PRIMARY_DOMAIN = {
    "personal/physical behaviour": "personal",
    "personal/physical behavior": "personal",
    "interpersonal relationships": "interpersonal",
    "social/cultural norms": "social",
    "organisational practices": "organisational",
    "organizational practices": "organisational",
    "government/policy/law": "government",
    "economic systems": "economic",
}

_ALTERNATIVE_TYPE = {
    "applies when": "applies-when",
    "other factors": "other-factors",
    "misunderstands": "misunderstands",
}

_INFERENCE_STRENGTH = {
    "1=weak": 1, "1": 1, "weak": 1,
    "2=moderate": 2, "2": 2, "moderate": 2,
    "3=strong": 3, "3": 3, "strong": 3,
}

_CREDIBILITY = {
    "personal bias": "personal-bias",
    "hearsay/secondhand": "hearsay",
    "hearsay": "hearsay",
    "emotional reasoning": "emotional-reasoning",
    "limited expertise": "limited-expertise",
    "none identified": "none",
    "none": "none",
}

_LOGICAL_FALLACY = {
    "no logical fallacy": "no-fallacy",
    "none": "no-fallacy",
    "false dilemma": "false-dilemma",
    "hasty generalisation": "hasty-generalization",
    "hasty generalization": "hasty-generalization",
    "ad hominem": "ad-hominem",
    "strawman": "strawman",
    "circular reasoning": "circular-reasoning",
    "false cause": "false-cause",
}

_TRUSTWORTHINESS = {
    "evidence-based": "trusted",
    "logical consistency": "logical",
    "common knowledge": "common-knowledge",
    "cannot be trusted": "untrusted",
    "untrusted": "untrusted",
}

_BIAS = {
    "fear/anxiety": "fear-anxiety",
    "first impression": "first-impression",
    "emotional reaction": "emotional-reaction",
    "pattern matching": "pattern-matching",
    "insufficient self-checking": "insufficient-self-checking",
    "none detected": "none-detected",
    "none": "none-detected",
}

_HEURISTIC = {
    "went with gut feeling": "gut-feeling",
    "used familiar pattern": "familiar-pattern",
    "followed surface cues": "surface-cues",
    "none detected": "none-detected",
    "none": "none-detected",
}

_CHANGE_DECISION = {
    "yes": "yes", "no": "no",
}

_REVISION_TYPE = {
    "minor-adjustment": "minor-adjustment",
    "minor adjustment": "minor-adjustment",
    "major-change": "major-change",
    "major change": "major-change",
    "complete-revision": "complete-revision",
    "complete revision": "complete-revision",
    "no-change": "no-change",
    "no change": "no-change",
    "": "",
}

# targetExplanationReason: prose sentence → gold short code
_TARGET_EXPLANATION_PROSE = {
    "It is explicitly stated as the conclusion (e.g., \"therefore/thus/so\").": "explicit-conclusion",
    "It is repeated or foregrounded through emphasis and evidence selection.": "repeated-emphasis",
    "It is the core causal claim (X causes Y) that drives the argument's implication.": "causal-claim",
    "It is stated early and framed with clear evaluative language (good/bad/should).": "early-evaluative",
}

# reasoningStructure: prose sentence → gold short code
_REASONING_STRUCTURE_PROSE = {
    "Uses specific cases, stories, or instances to support claims": "specific-cases",
    "Justifies action by their positive or negative outcomes": "outcome-justification",
    "Compare current citation to similar known situation": "analogical-comparison",
    "Use indicator to suggest underlying condition": "indicator-condition",
    "Appeal to expert knowledge or credible source": "expert-appeal",
    "None of the others": "none-of-the-others",
}


def _fuzzy_match(value: str, mapping: dict) -> str | None:
    """Return the mapped gold value for `value` using exact then fuzzy matching."""
    key = value.strip().lower()
    if key in mapping:
        return mapping[key]
    # Try substring containment
    for prose, code in mapping.items():
        if prose.lower() in key or key in prose.lower():
            return code
    # Fuzzy similarity fallback (threshold 0.55)
    best_score, best_code = 0.0, None
    for prose, code in mapping.items():
        score = SequenceMatcher(None, key, prose.lower()).ratio()
        if score > best_score:
            best_score, best_code = score, code
    return best_code if best_score >= 0.55 else None


def _normalize_single(value: str, mapping: dict) -> str:
    result = _fuzzy_match(value, mapping)
    return result if result is not None else value.strip()


def _normalize_list(raw: str, mapping: dict) -> list[str]:
    """Split on '; ' or ',' and normalize each element."""
    parts = [p.strip() for p in re.split(r";|,", raw) if p.strip()]
    result = []
    for p in parts:
        normalized = _fuzzy_match(p, mapping)
        result.append(normalized if normalized is not None else p.lower())
    return result or ["none-detected"]


# ── Think-block stripping ─────────────────────────────────────────────────────

def strip_think_block(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> chain-of-thought."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Raw text → flat key-value dict ───────────────────────────────────────────

# Lookup for normalising spaced field names (e.g. DeepSeek: "stance Target" → "stanceTarget")
_KNOWN_FIELDS_LOWER = {
    f.lower(): f for f in [
        "initialUnderstandingTarget", "stanceTarget", "knowledgeDomain",
        "coreClaim", "minorClaim", "premise", "paraphrasingUnderstanding",
        "reasoningType", "hasAssumption", "missingComponent",
        "positiveConsequences", "negativeConsequences", "primaryDomain",
        "alternativeType", "alternativeKeywords",
        "inferenceStrength", "credibilityFactors", "logicalFallacy", "trustworthiness",
        "targetExplanationReason", "reasoningStructure",
        "goodEvidence", "badEvidence", "trustExplanation",
        "fallacySpan1", "fallacySpan2", "noFallacyExplanation",
        "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY",
        "span1", "span2", "span3",
        "biasDetection", "heuristicDetection", "errorDetection",
        "changeDecision", "revisionPhases", "revisionReason", "revisionType",
        "targetExplanation",
        "question", "target", "sourceField",
    ]
}


def normalize_field_names(text: str) -> str:
    """Collapse spaced field names at line-start produced by DeepSeek.

    E.g. 'stance Target:' → 'stanceTarget:', 'positive Consequences:' → 'positiveConsequences:'.
    Only collapses tokens that, after space removal, match a known field name exactly.
    """
    def _collapse(m):
        raw = m.group(1)
        key = raw.replace(" ", "").lower()
        if key in _KNOWN_FIELDS_LOWER:
            return _KNOWN_FIELDS_LOWER[key] + ":"
        return m.group(0)

    return re.sub(r"^[ \t]*([A-Za-z][A-Za-z ]+):", _collapse, text, flags=re.MULTILINE)


# All known field names in the output
_ALL_FIELDS = {
    "initialUnderstandingTarget", "targetExplanation", "stanceTarget",
    "knowledgeDomain", "coreClaim", "minorClaim", "premise",
    "paraphrasingUnderstanding", "reasoningType", "hasAssumption",
    "missingComponent", "positiveConsequences", "negativeConsequences",
    "primaryDomain", "alternativeType", "alternativeKeywords",
    "inferenceStrength", "credibilityFactors", "logicalFallacy",
    "trustworthiness", "trustExplanation", "targetExplanationReason",
    "reasoningStructure", "biasDetection", "heuristicDetection",
    "errorDetection", "changeDecision", "revisionType",
    "goodEvidence", "badEvidence", "fallacySpan1", "fallacySpan2",
    "span1", "span2", "span3",
    "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY",
    "noFallacyExplanation",
    "question", "target", "sourceField",
}

# Regex: capture "fieldName: rest of line" regardless of leading bullet/number
_FIELD_RE = re.compile(
    r"^\s*(?:\*\s*|-\s*|\d+[.)]\s*|\*{1,2})?"  # optional bullet / number
    r"(initialUnderstandingTarget|targetExplanation|stanceTarget|knowledgeDomain"
    r"|coreClaim|minorClaim|premise|paraphrasingUnderstanding|reasoningType"
    r"|hasAssumption|missingComponent|positiveConsequences|negativeConsequences"
    r"|primaryDomain|alternativeType|alternativeKeywords|inferenceStrength"
    r"|credibilityFactors|logicalFallacy|trustworthiness|trustExplanation"
    r"|targetExplanationReason|reasoningStructure|biasDetection|heuristicDetection"
    r"|errorDetection|changeDecision|revisionType"
    r"|goodEvidence|badEvidence|fallacySpan1|fallacySpan2"
    r"|span1|span2|span3"
    r"|deductiveTermX|deductiveTermY|inductiveTermX|inductiveTermY"
    r"|noFallacyExplanation"
    r"|question|target|sourceField)"
    r"[ \t]*:[ \t]*(.*)$",
    re.MULTILINE,
)


def extract_raw_fields(text: str) -> list[tuple[str, str]]:
    """Return ordered list of (field_name, raw_value) from generated text."""
    text = strip_think_block(text)
    text = normalize_field_names(text)
    matches = _FIELD_RE.findall(text)
    # Clean up values: strip markdown bold/italic and "Not provided"
    result = []
    for field, value in matches:
        value = re.sub(r"\*+", "", value).strip()
        if value.lower() in ("not provided", "n/a", "none provided"):
            value = ""
        result.append((field, value))

    # Fallback: if initialUnderstandingTarget was not labeled, extract the first
    # non-empty line before the first matched field. Models often write the topic
    # as a free-text preamble sentence instead of an explicit labeled key.
    if not any(f == "initialUnderstandingTarget" for f, _ in result) and matches:
        first_field_name = matches[0][0]
        m = re.search(rf"^(.*?){re.escape(first_field_name)}\s*:", text, re.DOTALL)
        if m:
            preamble = m.group(1)
            for line in preamble.splitlines():
                line = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*|#+\s*)", "", line).strip()
                line = re.sub(r"(?i)^phase\s*\d+.*$", "", line).strip()
                if line and len(line) > 3:
                    result.insert(0, ("initialUnderstandingTarget", line))
                    break

    return result


def extract_socratic_questions(raw_fields: list[tuple[str, str]]) -> list[dict]:
    """Group consecutive question/target/sourceField triplets."""
    questions = []
    i = 0
    while i < len(raw_fields):
        field, value = raw_fields[i]
        if field == "question" and value:
            q = {"question": value, "target": "", "source_field": ""}
            j = i + 1
            while j < len(raw_fields) and raw_fields[j][0] in ("target", "sourceField"):
                if raw_fields[j][0] == "target":
                    q["target"] = raw_fields[j][1]
                else:
                    q["source_field"] = raw_fields[j][1]
                j += 1
            questions.append(q)
            i = j
        else:
            i += 1
    return questions[:3]  # max 3


# ── Main normalization function ───────────────────────────────────────────────

def parse_model_output(generated_text: str) -> dict:
    """
    Parse raw model output and return a normalized dict matching the gold schema.

    Fields that the model doesn't output are set to "" or [] as appropriate.
    Socratic questions are stored as socraticQuestion1/2/3 (flat, like gold).
    """
    raw_fields = extract_raw_fields(generated_text)
    raw = {}
    for field, value in raw_fields:
        if field not in ("question", "target", "sourceField"):
            # Last occurrence wins (some models repeat fields across sections)
            raw[field] = value

    questions = extract_socratic_questions(raw_fields)

    def get(field):
        return raw.get(field, "")

    # Extract span1/2/3 from [X] → [Y] → [Z] bracket pattern in reasoningStructure
    # when models embed spans inside the structure string (run1 / P4_full_guideline)
    _rs_raw = get("reasoningStructure")
    _brackets = [
        s.strip() for s in re.findall(r'\[([^\]]+)\]', _rs_raw)
        if s.strip() and s.strip().lower() not in ('...', '…', 'span1', 'span2', 'span3',
                                                    'term x', 'term y', 'term z')
    ]
    _span1 = get("span1") or (_brackets[0] if len(_brackets) > 0 else "")
    _span2 = get("span2") or (_brackets[1] if len(_brackets) > 1 else "")
    _span3 = get("span3") or (_brackets[2] if len(_brackets) > 2 else "")

    normalized = {
        # Free-text fields — pass through as-is
        "initialUnderstandingTarget": get("initialUnderstandingTarget"),
        "coreClaim":                  get("coreClaim"),
        "minorClaim":                 get("minorClaim"),
        "premise":                    get("premise"),
        "paraphrasingUnderstanding":  get("paraphrasingUnderstanding"),
        "missingComponent":           get("missingComponent"),
        "positiveConsequences":       get("positiveConsequences"),
        "negativeConsequences":       get("negativeConsequences"),
        "alternativeKeywords":        get("alternativeKeywords"),
        "trustExplanation":           get("trustExplanation"),
        "errorDetection":             get("errorDetection"),
        "goodEvidence":               get("goodEvidence"),
        "badEvidence":                get("badEvidence"),
        "fallacySpan1":               get("fallacySpan1"),
        "fallacySpan2":               get("fallacySpan2"),
        "span1":                      _span1,
        "span2":                      _span2,
        "span3":                      _span3,
        "deductiveTermX":             get("deductiveTermX"),
        "deductiveTermY":             get("deductiveTermY"),
        "inductiveTermX":             get("inductiveTermX"),
        "inductiveTermY":             get("inductiveTermY"),
        "noFallacyExplanation":       get("noFallacyExplanation"),

        # Normalized single-select fields
        "stanceTarget":     _normalize_single(get("stanceTarget"),    _STANCE),
        "knowledgeDomain":  _normalize_single(get("knowledgeDomain"), _KNOWLEDGE_DOMAIN),
        "reasoningType":    _normalize_single(get("reasoningType"),   _REASONING_TYPE),
        "hasAssumption":    _normalize_single(get("hasAssumption"),   _HAS_ASSUMPTION),
        "primaryDomain":    _normalize_single(get("primaryDomain"),   _PRIMARY_DOMAIN),
        "alternativeType":  _normalize_single(get("alternativeType"), _ALTERNATIVE_TYPE),
        "logicalFallacy":   _normalize_single(get("logicalFallacy"),  _LOGICAL_FALLACY),
        "changeDecision":   _normalize_single(get("changeDecision"),  _CHANGE_DECISION),
        "revisionType":     _normalize_single(get("revisionType"),    _REVISION_TYPE),

        # Prose → short-code fields
        "targetExplanation":       _normalize_single(get("targetExplanationReason"),
                                                      _TARGET_EXPLANATION_PROSE),
        "reasoningStructure":      _normalize_single(get("reasoningStructure"),
                                                      _REASONING_STRUCTURE_PROSE),

        # Integer field
        "inferenceStrength": _INFERENCE_STRENGTH.get(
            get("inferenceStrength").lower().strip(), get("inferenceStrength")
        ),

        # Multi-select → list fields
        "credibilityFactors": _normalize_list(get("credibilityFactors"), _CREDIBILITY),
        "trustworthiness":    _normalize_list(get("trustworthiness"),    _TRUSTWORTHINESS),
        "biasDetection":      _normalize_list(get("biasDetection"),      _BIAS),
        "heuristicDetection": _normalize_list(get("heuristicDetection"), _HEURISTIC),

        # Socratic questions (flat, matching gold schema)
        "socraticQuestion1": questions[0]["question"] if len(questions) > 0 else "",
        "socraticQuestion2": questions[1]["question"] if len(questions) > 1 else "",
        "socraticQuestion3": questions[2]["question"] if len(questions) > 2 else "",

        # Metadata for traceability
        "_socraticDetails": questions,  # includes target + sourceField per question
    }

    return normalized


# ── Pipeline JSON → normalized JSON ──────────────────────────────────────────

def normalize_pipeline_json(pipeline_path: str, output_path: str,
                             seed_strategy: str = "preferred") -> dict:
    """
    Read a pipeline output JSON (from inference.py) and produce a normalized
    JSON file where each sample has per-seed parsed annotations plus an
    'aggregated' field using the preferred/best seed.

    seed_strategy:
        "preferred"  — use the seed with the highest field_rate
        "seed_44"    — always use seed 44
        "all"        — include all seeds
    """
    with open(pipeline_path, encoding="utf-8") as f:
        pipeline = json.load(f)

    results = []
    per_sample = pipeline.get("per_sample", [])

    for sample in per_sample:
        argument      = sample.get("argument", "")
        argument_id   = sample.get("argument_id")
        master_id     = sample.get("master_argument_id")
        per_seed      = sample.get("per_seed", {})

        seed_results = {}
        best_seed = None
        best_field_rate = -1.0

        for seed_key, seed_data in per_seed.items():
            generated_text = seed_data.get("generated_text", "")
            parsed = parse_model_output(generated_text)
            field_rate = seed_data.get("metrics", {}).get("field_rate", 0.0)
            seed_results[seed_key] = {
                "parsed": parsed,
                "field_rate": field_rate,
                "num_tokens": seed_data.get("metrics", {}).get("num_tokens_generated", 0),
            }
            if field_rate > best_field_rate:
                best_field_rate = field_rate
                best_seed = seed_key

        # Aggregated: use best seed or seed_44
        if seed_strategy == "seed_44":
            agg_key = "seed_44"
        elif seed_strategy == "preferred":
            agg_key = best_seed
        else:
            agg_key = best_seed

        aggregated = seed_results[agg_key]["parsed"] if agg_key and agg_key in seed_results else {}

        results.append({
            "argument":           argument,
            "argument_id":        argument_id,
            "master_argument_id": master_id,
            "best_seed":          agg_key,
            "best_field_rate":    best_field_rate,
            "aggregated":         aggregated,
            "per_seed":           seed_results,
        })

    output = {
        "metadata": {
            "source": pipeline_path,
            "model":  pipeline.get("model", "unknown"),
            "date":   pipeline.get("date_tag", ""),
            "total_samples": len(results),
            "seed_strategy": seed_strategy,
        },
        "samples": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Normalized {len(results)} samples → {output_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse model output into gold-schema normalized JSON"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  help="Pipeline JSON from inference.py")
    group.add_argument("--text",   help="Raw generated text string (for testing)")

    parser.add_argument("--output", help="Output JSON path (required with --input)")
    parser.add_argument("--seed-strategy", default="preferred",
                        choices=["preferred", "seed_44", "all"],
                        help="Which seed to use for the aggregated result")
    args = parser.parse_args()

    if args.text:
        result = parse_model_output(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.output:
        print("ERROR: --output is required with --input", file=sys.stderr)
        sys.exit(1)

    normalize_pipeline_json(args.input, args.output, args.seed_strategy)


if __name__ == "__main__":
    main()
