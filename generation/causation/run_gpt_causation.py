#!/usr/bin/env python3
"""
run_gpt_causation.py

GPT-4 / GPT-5 baseline causation experiment for CognitiveQG.

Four experimental conditions (mirrors the open-source causation experiment):
  vanilla      -- zero-shot direct: argument → 3 Socratic questions  (matches vanilla_v3)
  trace        -- full 6-phase Facione pipeline self-generated       (matches vanilla_v2)
  human_trace  -- human cognitive trace injected → questions         (matches gold)
  intervention -- expert annotation of the next argument injected (Noise),
                  argument i receives the annotation of argument (i+1) mod N

Usage:
    # Single condition
    python causation/run_gpt_causation.py \
        --model gpt-4o --condition vanilla \
        --gold-json path/to/dev1-12_gold.json \
        --seeds 44 45 46 --output-dir results/causation

    # All 4 conditions in one invocation
    python causation/run_gpt_causation.py \
        --model gpt-4o --run-all \
        --gold-json path/to/dev1-12_gold.json \
        --seeds 44 45 46 --output-dir results/causation

    # Dry-run: print prompts without calling the API
    python causation/run_gpt_causation.py \
        --model gpt-4o --condition trace --dry-run --num-samples 1

Set API key:
    export OPENAI_API_KEY=sk-...

Output:
    results/causation/gpt_{model_slug}_{condition}_20args_{YYYY-MM-DD}.json
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not found. Install: pip install openai")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]  # package root
RESULTS_DIR  = BASE_DIR / "results" / "causation"
TODAY        = datetime.now().strftime("%Y-%m-%d")
DEFAULT_DATA = BASE_DIR / "data/gold_annotations.json"

SUPPORTED_MODELS = ["gpt-4o", "gpt-5", "gpt-5.5"]
CONDITIONS       = ["vanilla", "trace", "human_trace", "intervention"]

# Per-model temperature overrides (default applies to all others)
MODEL_TEMP = {
    "gpt-4o": 0.3,
}
TEMP_DEFAULT = 0.2   # gpt-4.5-preview, gpt-5, gpt-5.5, etc.

# Models that use max_completion_tokens instead of max_tokens
COMPLETION_TOKENS_MODELS = {"gpt-5", "gpt-5.5", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"}

# Models that do not support custom temperature (only default=1 allowed)
FIXED_TEMP_MODELS = {"gpt-5", "gpt-5.5", "o1", "o1-mini", "o3", "o3-mini"}
MAX_TOKENS_FULL        = 1200   # trace / human_trace / intervention (gpt-4o etc.)
MAX_TOKENS_FULL_LARGE  = 8000   # all conditions on reasoning models (gpt-5, gpt-5.5)
MAX_TOKENS_DIRECT      = 400    # vanilla (non-reasoning models)

# Models that need a larger output budget — reasoning tokens consume the budget internally
LARGE_OUTPUT_MODELS = {"gpt-5", "gpt-5.5"}
MAX_RETRIES       = 3
RETRY_DELAY       = 5.0

# ── Normalization tables (gold annotation values → prompt-readable labels) ─────
# Sourced from run_oracle_causation.py

_NORM_SCALAR = {
    "knowledgeDomain": {
        "government-law":       "Government/Law",
        "no-specialized":       "No specialized knowledge",
        "technical-scientific": "Technical/Scientific",
        "legal-policy":         "Legal/Policy",
        "cultural-social":      "Cultural/Social",
        "economic-business":    "Economic/Business",
    },
    "reasoningType": {
        "inductive":  "Inductive",
        "deductive":  "Deductive",
        "none":       "None of the above",
    },
    "hasAssumption": {
        "yes": "Yes=Complete",
        "no":  "No=Missing Components",
    },
    "primaryDomain": {
        "personal-physical": "Personal/Physical behaviour",
        "interpersonal":     "Interpersonal relationships",
        "social":            "Social/Cultural norms",
        "organizational":    "Organisational practices",
        "organisational":    "Organisational practices",
        "government":        "Government/Policy/Law",
        "economic":          "Economic systems",
    },
    "alternativeType": {
        "applies-when":   "applies when",
        "other-factors":  "other factors",
        "misunderstands": "misunderstands",
    },
    "inferenceStrength": {
        1: "1=Weak", 2: "2=Moderate", 3: "3=Strong",
        "1": "1=Weak", "2": "2=Moderate", "3": "3=Strong",
    },
    "logicalFallacy": {
        "none":                 "No logical fallacy",
        "hasty-generalization": "Hasty generalisation",
        "hasty-generalisation": "Hasty generalisation",
        "circular":             "Circular reasoning",
        "false-dilemma":        "False dilemma",
        "ad-hominem":           "Ad hominem",
        "strawman":             "Strawman",
        "false-cause":          "False cause",
    },
    "changeDecision": {
        "yes": "Yes",
        "no":  "No",
    },
    "stanceTarget": {
        "favor":   "favor",
        "against": "against",
        "neutral": "neutral",
    },
}

_NORM_LIST = {
    "credibilityFactors": {
        "bias":      "Personal bias",
        "hearsay":   "Hearsay/secondhand",
        "emotional": "Emotional reasoning",
        "expertise": "Limited expertise",
        "none":      "None identified",
    },
    "trustworthiness": {
        "logical":   "Logical consistency",
        "evidence":  "Evidence-based",
        "common":    "Common knowledge",
        "untrusted": "Cannot be trusted",
    },
    "biasDetection": {
        "none-detected":          "None detected",
        "fear-anxiety":           "Fear/anxiety",
        "first-impression":       "First impression",
        "emotional-reaction":     "Emotional reaction",
        "pattern-matching":       "Pattern matching",
        "insufficient-checking":  "Insufficient self-checking",
    },
    "heuristicDetection": {
        "none-detected":    "None detected",
        "gut-feeling":      "Went with gut feeling",
        "familiar-pattern": "Used familiar pattern",
        "surface-cues":     "Followed surface cues",
    },
}


def _norm(field: str, value) -> str:
    if field in _NORM_SCALAR:
        table = _NORM_SCALAR[field]
        if isinstance(value, (int, float)):
            return table.get(value, table.get(str(int(value)), str(value)))
        return table.get(str(value).lower().strip(), str(value))
    if field in _NORM_LIST:
        table = _NORM_LIST[field]
        if isinstance(value, list):
            normalized = [table.get(str(v).lower().strip(), str(v)) for v in value]
            return "; ".join(normalized)
        return table.get(str(value).lower().strip(), str(value))
    return str(value) if value is not None else ""


# ── System Prompts ─────────────────────────────────────────────────────────────

# Condition 1: vanilla — no reasoning, direct questions
VANILLA_SYSTEM = """You are a critical thinking assistant. Given an argument, generate exactly 3 Socratic questions that probe its assumptions, missing evidence, or logical weaknesses.

Output ONLY valid JSON with no markdown fences and no commentary:
{
  "socraticQuestions": [
    {"question": "", "target": "", "sourceField": ""}
  ]
}

- "question": the Socratic question text
- "target": the short phrase or weakness the question probes
- "sourceField": one of: coreClaim | premise | minorClaim | hasAssumption | missingComponent | logicalFallacy | inferenceStrength | credibilityFactors | trustworthiness | reasoningType | positiveConsequences | negativeConsequences | alternativeType | stanceTarget | knowledgeDomain"""

# Condition 2: trace — full 6-phase Facione pipeline (reused from generate_gpt_socratic.py)
TRACE_SYSTEM = """You are a critical thinking annotator trained on the CognitiveQG annotation scheme
based on Facione's Critical Thinking Framework. Analyze the argument below through
each of the six cognitive stages using the exact guidelines below, then generate
3 Socratic questions.

------------------------------------------------------------
STAGE 1: INTERPRETATION
------------------------------------------------------------
Initial Understanding of Target:
  - Restate the core target of the argument (Topic, Person, or Idea).
  - Complete: "The argument talks about ___."

Stance Toward the Core Target:
  - Select: [favor | against | neutral]

Knowledge Domain:
  - Select one: [Technical/Scientific | Legal/Policy | Cultural/Social |
    Economic/Business | Government/Law | No specialized knowledge]

------------------------------------------------------------
STAGE 2: ANALYSIS
------------------------------------------------------------
Argument Structure Diagnosis (identify spans):
  - Core Claim: The main assertion.
  - Minor Claim: A subsidiary assertion that serves the core claim.
  - Premise: The evidence or reason given to support the claim.

Paraphrasing Initial Understanding:
  - Paraphrase the core claim into the simplest sentence possible.

Inductive vs. Deductive Reasoning:
  - Select: [Inductive | Deductive | None of the above]

Implicit Assumption:
  - Select: [Yes=Complete | No=Missing Components]
  - If missing, write what is needed as short keywords.

------------------------------------------------------------
STAGE 3: INFERENCE
------------------------------------------------------------
Consequence Prediction:
  - Positive consequences (max 2 keywords/phrases).
  - Negative consequences (max 2 keywords/phrases).

Primary Domain Affected:
  - Select one: [Personal/Physical behavior | Interpersonal relationships |
    Social/Cultural norms | Organizational practices | Government/Policy/Law |
    Economic systems]

Alternative Possibility Recognition (choose one + up to 3 keywords):
  - "...this only applies when ___, ___, ___"
  - "...other factors are more important like ___, ___, ___"
  - "...this misunderstands ___, ___, ___"

------------------------------------------------------------
STAGE 4: EVALUATION
------------------------------------------------------------
Inference Strength (rate 1-3):
  - 3 (Strong): Premises directly and logically lead to the core claim.
  - 2 (Moderate): Premises provide some support but require one significant assumption.
  - 1 (Weak): Premises provide minimal support; major logical gaps.

Credibility Factors (select all that apply):
  [Personal bias | Hearsay/secondhand | Emotional reasoning | Limited expertise | None identified]

Logical Fallacy (select one):
  [No logical fallacy | False dilemma | Hasty generalization | Ad hominem |
   Strawman | Circular reasoning | False cause]

Trustworthiness (select all that apply):
  [Evidence-based | Logical consistency | Common knowledge | Cannot be trusted]

------------------------------------------------------------
STAGE 5: EXPLANATION
------------------------------------------------------------
Target Explanation:
  Explain why you identified the target as primary.

Reasoning Structure (select one):
  [Specific cases | Outcomes | Comparison | Indicators | Expert appeal | None]

Inductive/Deductive Template:
  - Deductive: "General principle [Term X] -> Specific conclusion [Term Y]"
  - Inductive: "Specific instance [Term X] -> General principle [Term Y]"

Evidential Justification:
  - Good evidence: evidence from the text supporting your inference.
  - Bad evidence: weaknesses in the evidence.

Trustworthiness and Fallacy Explanation.

------------------------------------------------------------
STAGE 6: SELF-REGULATION
------------------------------------------------------------
Bias Detection (select up to 2):
  [Fear/anxiety | First impression | Emotional reaction | Pattern matching |
   Insufficient self-checking | None detected]

Heuristic Shortcut Detection (select up to 2):
  [Went with gut feeling | Used familiar pattern | Followed surface cues | None detected]

Error Detection: describe any errors or write "None detected."

Change Decision: [Yes | No]

------------------------------------------------------------
OUTPUT FORMAT — return valid JSON only, no markdown fences, no commentary.
------------------------------------------------------------
{
  "interpretation": {
    "initialUnderstandingTarget": "",
    "stanceTarget": "",
    "knowledgeDomain": "",
    "interpretationJustification": ""
  },
  "analysis": {
    "coreClaim": "",
    "minorClaim": "",
    "premise": "",
    "paraphrasingUnderstanding": "",
    "reasoningType": "",
    "hasAssumption": "",
    "missingComponent": ""
  },
  "inference": {
    "positiveConsequences": [],
    "negativeConsequences": [],
    "primaryDomain": "",
    "alternativeType": "",
    "alternativeKeywords": []
  },
  "evaluation": {
    "inferenceStrength": "",
    "credibilityFactors": [],
    "logicalFallacy": "",
    "trustworthiness": [],
    "trustworthinessExplanation": ""
  },
  "explanation": {
    "targetExplanation": "",
    "targetExplanationText": "",
    "reasoningStructure": "",
    "reasoningReconstruction": "",
    "inductiveDeductiveMapping": "",
    "goodEvidence": "",
    "badEvidence": "",
    "fallacyJustification": ""
  },
  "selfRegulation": {
    "biasDetection": [],
    "heuristicDetection": [],
    "errorDetection": "",
    "changeDecision": "",
    "revisionType": "",
    "revisionPhases": [],
    "revisionReason": ""
  },
  "socraticQuestions": [
    {"question": "", "target": "", "sourceField": ""}
  ]
}"""

# Condition 3 & 4: human_trace / intervention — receives pre-built trace
HUMAN_TRACE_SYSTEM = """You are a Socratic question generator. You are given a cognitive reasoning analysis of an argument prepared by a human annotator following the Facione Critical Thinking Framework. Use the provided analysis to generate exactly 3 Socratic questions that probe the argument's weaknesses.

Output ONLY valid JSON with no markdown fences and no commentary:
{
  "socraticQuestions": [
    {"question": "", "target": "", "sourceField": ""}
  ]
}

- "question": the Socratic question text
- "target": the short phrase or weakness the question probes
- "sourceField": one of: coreClaim | premise | minorClaim | hasAssumption | missingComponent | logicalFallacy | inferenceStrength | credibilityFactors | trustworthiness | reasoningType | positiveConsequences | negativeConsequences | alternativeType | stanceTarget | knowledgeDomain"""

HUMAN_TRACE_USER_TEMPLATE = """Argument:
{argument}

Human Cognitive Analysis:
- Stance: {stanceTarget}
- Knowledge Domain: {knowledgeDomain}
- Core Claim: {coreClaim}
- Reasoning Type: {reasoningType}
- Has Assumption: {hasAssumption}
- Primary Domain: {primaryDomain}
- Inference Strength: {inferenceStrength}
- Logical Fallacy: {logicalFallacy}
- Change Decision: {changeDecision}"""


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_vanilla_messages(argument: str) -> list[dict]:
    return [
        {"role": "system", "content": VANILLA_SYSTEM},
        {"role": "user",   "content": f"Argument:\n{argument}"},
    ]


def build_trace_messages(argument: str) -> list[dict]:
    return [
        {"role": "system", "content": TRACE_SYSTEM},
        {"role": "user",   "content": f"Argument:\n{argument}"},
    ]


def build_human_trace_messages(argument: str, annotation: dict) -> list[dict]:
    user_content = HUMAN_TRACE_USER_TEMPLATE.format(
        argument        = argument,
        stanceTarget    = _norm("stanceTarget",     annotation.get("stanceTarget", "")),
        knowledgeDomain = _norm("knowledgeDomain",  annotation.get("knowledgeDomain", "")),
        coreClaim       = annotation.get("coreClaim", ""),
        reasoningType   = _norm("reasoningType",    annotation.get("reasoningType", "")),
        hasAssumption   = _norm("hasAssumption",    annotation.get("hasAssumption", "")),
        primaryDomain   = _norm("primaryDomain",    annotation.get("primaryDomain", "")),
        inferenceStrength = _norm("inferenceStrength", annotation.get("inferenceStrength", "")),
        logicalFallacy  = _norm("logicalFallacy",   annotation.get("logicalFallacy", "")),
        changeDecision  = _norm("changeDecision",   annotation.get("changeDecision", "")),
    )
    return [
        {"role": "system", "content": HUMAN_TRACE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def build_messages(condition: str, argument: str, annotation: dict,
                   model: str = "") -> tuple[list[dict], int]:
    """Return (messages, max_tokens) for the given condition."""
    full_budget = MAX_TOKENS_FULL_LARGE if model in LARGE_OUTPUT_MODELS else MAX_TOKENS_FULL
    vanilla_budget = MAX_TOKENS_FULL_LARGE if model in LARGE_OUTPUT_MODELS else MAX_TOKENS_DIRECT
    if condition == "vanilla":
        return build_vanilla_messages(argument), vanilla_budget
    elif condition == "trace":
        return build_trace_messages(argument), full_budget
    elif condition in ("human_trace", "intervention"):
        return build_human_trace_messages(argument, annotation), full_budget
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


# ── API call ───────────────────────────────────────────────────────────────────

def get_temperature(model: str) -> float:
    return MODEL_TEMP.get(model, TEMP_DEFAULT)


def call_api(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    seed: int,
) -> str:
    tokens_key = "max_completion_tokens" if model in COMPLETION_TOKENS_MODELS else "max_tokens"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                **{tokens_key: max_tokens},
            )
            if model not in FIXED_TEMP_MODELS:
                kwargs["temperature"] = get_temperature(model)
            # seed is supported on most GPT models
            kwargs["seed"] = seed
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] API error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return ""


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_response(raw: str, condition: str) -> tuple[dict, list[str]]:
    """Return (pipeline_output, questions) from raw LLM response."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    pipeline: dict = {}
    questions: list[str] = []

    try:
        parsed = json.loads(cleaned)

        if condition == "vanilla" or condition in ("human_trace", "intervention"):
            sq = parsed.get("socraticQuestions", [])
        else:
            # trace: full pipeline output
            pipeline = {
                "interpretation":  parsed.get("interpretation", {}),
                "analysis":        parsed.get("analysis", {}),
                "inference":       parsed.get("inference", {}),
                "evaluation":      parsed.get("evaluation", {}),
                "explanation":     parsed.get("explanation", {}),
                "self_regulation": parsed.get("selfRegulation", {}),
            }
            sq = parsed.get("socraticQuestions", [])

        if isinstance(sq, list):
            for item in sq:
                if isinstance(item, dict):
                    q = item.get("question", "").strip()
                elif isinstance(item, str):
                    q = item.strip()
                else:
                    continue
                if q:
                    questions.append(q)

    except json.JSONDecodeError:
        # Fallback: extract numbered questions from raw text
        matches = re.findall(r'\d+[.)]\s*(.+?)(?=\n\d+[.)]|$)', raw, re.DOTALL)
        questions = [m.strip() for m in matches if m.strip()]

    return pipeline, questions[:3]


# ── Single condition run ───────────────────────────────────────────────────────

def run_condition(
    client: OpenAI,
    model: str,
    annotations: list[dict],
    condition: str,
    seeds: list[int],
    num_samples: int,
    output_path: str,
    dry_run: bool = False,
):
    subset = annotations[:num_samples]
    n = len(subset)

    print(f"\n{'#'*72}")
    print(f"  Model: {model} | Condition: {condition} | n={n} | seeds={seeds}")
    print(f"{'#'*72}\n")

    per_sample = []

    for pos, ann in enumerate(subset):
        idx      = ann["argumentIndex"]
        argument = ann["argument"]

        # For intervention (Noise): inject the expert annotation of the next argument.
        if condition == "intervention":
            inj_ann = annotations[(pos + 1) % len(annotations)]
        else:
            inj_ann = ann

        print(f"[{idx+1:2}/{n}] {argument[:80]}...")

        per_seed: dict = {}
        for seed in seeds:
            messages, max_tokens = build_messages(condition, argument, inj_ann, model=model)

            if dry_run:
                print(f"  [dry-run] seed={seed}")
                print(f"  SYSTEM: {messages[0]['content'][:200]}...")
                print(f"  USER:   {messages[1]['content'][:200]}...")
                per_seed[str(seed)] = {
                    "generated_text": "[DRY RUN]",
                    "pipeline_output": {},
                    "questions": [],
                    "metrics": {"num_socratic_questions": 0, "generation_time_sec": 0.0},
                }
                continue

            t0  = time.time()
            raw = call_api(client, model, messages, max_tokens, seed)
            elapsed = time.time() - t0

            pipeline, questions = parse_response(raw, condition)

            print(f"  seed={seed} -> {len(questions)}q ({elapsed:.1f}s)")
            for i, q in enumerate(questions, 1):
                print(f"    {i}) {q[:90]}")

            per_seed[str(seed)] = {
                "generated_text":  raw,
                "pipeline_output": pipeline,
                "questions":       questions,
                "metrics": {
                    "num_socratic_questions": len(questions),
                    "generation_time_sec":    round(elapsed, 3),
                },
            }

        # Aggregate across seeds
        nq_list = [v["metrics"]["num_socratic_questions"] for v in per_seed.values()]
        t_list  = [v["metrics"]["generation_time_sec"]    for v in per_seed.values()]
        seed_agg = {
            "num_socratic_questions_avg": round(mean(nq_list), 3) if nq_list else 0,
            "generation_time_sec_avg":   round(mean(t_list), 3)  if t_list  else 0,
        }

        sample = {
            "argument_index": idx,
            "argument_id":    idx,
            "argument":       argument,
            "per_seed":       per_seed,
            "seed_aggregate": seed_agg,
        }
        if condition == "intervention":
            sample["noise_source"] = {
                "injected_argumentIndex": inj_ann.get("argumentIndex"),
            }
        per_sample.append(sample)

    # Global aggregate
    all_nq = [s["seed_aggregate"]["num_socratic_questions_avg"] for s in per_sample]
    all_t  = [s["seed_aggregate"]["generation_time_sec_avg"]   for s in per_sample]
    aggregate = {
        "num_socratic_questions_avg": round(mean(all_nq), 3) if all_nq else 0,
        "num_socratic_questions_std": round(stdev(all_nq), 3) if len(all_nq) > 1 else 0,
        "generation_time_sec_avg":   round(mean(all_t),  3)  if all_t  else 0,
        "generation_time_sec_std":   round(stdev(all_t), 3)  if len(all_t) > 1 else 0,
    }

    output = {
        "config": {
            "model":         model,
            "condition":     condition,
            "prompt_version": "gpt_v1",
            "seeds":         seeds,
            "num_samples":   n,
            "temperature":   get_temperature(model),
            "max_tokens":    MAX_TOKENS_FULL if condition != "vanilla" else MAX_TOKENS_DIRECT,
            "generated_at":  datetime.now().isoformat(),
        },
        "per_sample": per_sample,
        "aggregate":  aggregate,
    }

    if not dry_run:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved -> {output_path}")
        print(f"Overall avg questions: {aggregate['num_socratic_questions_avg']:.2f}")


# ── run-all ────────────────────────────────────────────────────────────────────

def run_all_conditions(client, model, annotations, seeds, num_samples, output_dir, dry_run):
    for condition in CONDITIONS:
        model_slug = model.replace("/", "-").replace(".", "-")
        out_path = os.path.join(
            output_dir, f"gpt_{model_slug}_{condition}_{num_samples}args_{TODAY}.json"
        )
        run_condition(client, model, annotations, condition, seeds, num_samples, out_path, dry_run)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPT-4/GPT-5 baseline causation experiment for CognitiveQG"
    )
    parser.add_argument("--model", required=True,
                        help="OpenAI model: " + " | ".join(SUPPORTED_MODELS))
    parser.add_argument("--condition", choices=CONDITIONS,
                        help="Single condition to run (ignored if --run-all)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 4 conditions sequentially")
    parser.add_argument("--gold-json", default=str(DEFAULT_DATA),
                        help="Path to gold annotation JSON")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of arguments to process (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46],
                        help="Random seeds for generation (paper: 44 45 46)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR),
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling the API")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to wait between API calls (default: 0.5)")
    args = parser.parse_args()

    if not args.run_all and not args.condition:
        parser.error("Provide --condition or --run-all")

    with open(args.gold_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data["annotations"]
    print(f"Loaded {len(annotations)} annotations from {args.gold_json}")


    client = None
    if not args.dry_run:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)

    model_slug = args.model.replace("/", "-").replace(".", "-")

    if args.run_all:
        run_all_conditions(
            client, args.model, annotations,
            args.seeds, args.num_samples, args.output_dir, args.dry_run,
        )
    else:
        out_path = os.path.join(
            args.output_dir,
            f"gpt_{model_slug}_{args.condition}_{args.num_samples}args_{TODAY}.json",
        )
        run_condition(
            client, args.model, annotations,
            args.condition, args.seeds, args.num_samples, out_path, args.dry_run,
        )


if __name__ == "__main__":
    main()
