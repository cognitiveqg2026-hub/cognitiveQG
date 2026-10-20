#!/usr/bin/env python3
"""
run_oracle_causation.py

Oracle causation experiment: inject gold (or noise) cognitive reasoning
phases 1-6 and let LLaMA-2-7B-Chat (off-the-shelf, 4-bit NF4) generate
the Socratic questions.

Conditions:
  gold    -- inject the matching human annotation for each argument
  noise   -- inject the expert annotation of the next argument, (i+1) mod N
             (same format as gold, content mismatched to the argument)
  vanilla -- skip all 6 phases entirely; generate questions directly from argument

Usage:
    python run_oracle_causation.py \
        --gold-json data/Dev_combined/p_dev_combined_2026-02-26.json \
        --condition vanilla \
        --num-samples 20 \
        --seeds 44 \
        --max-new-tokens 256 \
        --output results/oracle_vanilla_20args_2026-05-02.json

    # Dry-run: print prompt for arg 0 without loading the model
    python scripts/run_oracle_causation.py \
        --gold-json data/Dev_combined/p_dev_combined_2026-02-26.json \
        --condition vanilla --dry-run
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

# model_configs.py lives at the package root, one level above this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── System prompt (identical to inference.py) ─────────────────────────────────
SYSTEM_PROMPT = """You are a structured cognitive analyst trained under Facione's Critical Thinking Framework.
You MUST analyze the argument through six phases in order:

1) Interpretation
2) Analysis
3) Inference
4) Evaluation
5) Explanation
6) Self-Regulation

Then generate up to 3 Socratic questions grounded in your analysis.

CRITICAL RULES
- Follow the phase order strictly.
- When a field requests an "exact span", copy a short phrase exactly from the argument.
- If multiple spans could work, choose the FIRST best short span.
- Do NOT invent facts not present in the argument.
- If uncertain, leave the field blank as "" (empty string).
- Use ONLY the allowed dropdown/checkbox labels EXACTLY as listed below.
- For fields that allow multiple selections, join them with "; " (semicolon + space).
- For keyword-limited fields, respect the limit (max 2 or max 3).
- Output must match the required template exactly (headings and field names).
- No extra commentary outside the template.

ALLOWED LABEL SETS (MUST MATCH EXACTLY)

[INTERPRETATION]
- stanceTarget (choose 1):
  favor | against | neutral

- knowledgeDomain (choose 1):
  Technical/Scientific | Legal/Policy | Cultural/Social | Economic/Business | Government/Law | No specialized knowledge

[ANALYSIS]
- reasoningType (choose 1):
  Inductive | Deductive | None of the above

- hasAssumption (choose 1):
  Yes=Complete | No=Missing Components

[INFERENCE]
- primaryDomain (choose 1):
  Personal/Physical behaviour | Interpersonal relationships | Social/Cultural norms | Organisational practices | Government/Policy/Law | Economic systems

- alternativeType (choose 1):
  applies when | other factors | misunderstands

- positiveConsequences: up to 2 keywords/phrases
- negativeConsequences: up to 2 keywords/phrases
- alternativeKeywords: up to 3 keywords/phrases

[EVALUATION]
- inferenceStrength (choose 1):
  1=Weak | 2=Moderate | 3=Strong

- credibilityFactors (choose 1+; use "; " if multiple):
  Personal bias | Hearsay/secondhand | Emotional reasoning | Limited expertise | None identified

- logicalFallacy (choose 1):
  No logical fallacy | False dilemma | Hasty generalisation | Ad hominem | Strawman | Circular reasoning | False cause

- trustworthiness (choose 1+; use "; " if multiple):
  Evidence-based | Logical consistency | Common knowledge | Cannot be trusted

[EXPLANATION]
- targetExplanationReason (choose 1):
  It is explicitly stated as the conclusion (e.g., "therefore/thus/so"). |
  It is repeated or foregrounded through emphasis and evidence selection. |
  It is the core causal claim (X causes Y) that drives the argument's implication. |
  It is stated early and framed with clear evaluative language (good/bad/should).

- reasoningStructure (choose 1):
  Uses specific cases, stories, or instances to support claims |
  Justifies action by their positive or negative outcomes |
  Compare current citation to similar known situation |
  Use indicator to suggest underlying condition |
  Appeal to expert knowledge or credible source |
  None of the others

[SELF-REGULATION]
- biasDetection (choose up to 2; use "; " if multiple):
  Fear/anxiety | First impression | Emotional reaction | Pattern matching | Insufficient self-checking | None detected

- heuristicDetection (choose up to 2; use "; " if multiple):
  Went with gut feeling | Used familiar pattern | Followed surface cues | None detected

- changeDecision (choose 1):
  Yes | No

- revisionType (choose 1):
  minor-adjustment | major-change | complete-revision | no-change

SOCRATIC QUESTIONS (MAX 3)
For each question:
- question: the Socratic question
- target: what weakness it probes (short phrase)
- sourceField: MUST be one of these exact field names:
  initialUnderstandingTarget | stanceTarget | knowledgeDomain |
  coreClaim | minorClaim | premise | paraphrasingUnderstanding | reasoningType |
  hasAssumption | missingComponent |
  positiveConsequences | negativeConsequences | primaryDomain |
  alternativeType | alternativeKeywords |
  inferenceStrength | credibilityFactors | logicalFallacy |
  trustworthiness | trustExplanation |
  targetExplanationReason | reasoningStructure |
  biasDetection | heuristicDetection | errorDetection"""


# ── Normalization tables (gold annotation values → prompt allowed labels) ──────

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
        "applies-when":  "applies when",
        "other-factors": "other factors",
        "misunderstands": "misunderstands",
    },
    "inferenceStrength": {
        1: "1=Weak",  2: "2=Moderate",  3: "3=Strong",
        "1": "1=Weak", "2": "2=Moderate", "3": "3=Strong",
    },
    "logicalFallacy": {
        "none":               "No logical fallacy",
        "hasty-generalization": "Hasty generalisation",
        "hasty-generalisation": "Hasty generalisation",
        "circular":           "Circular reasoning",
        "false-dilemma":      "False dilemma",
        "ad-hominem":         "Ad hominem",
        "strawman":           "Strawman",
        "false-cause":        "False cause",
    },
    "changeDecision": {
        "yes": "Yes",
        "no":  "No",
    },
    "targetExplanation": {
        "core-causal-claim":
            "It is the core causal claim (X causes Y) that drives the argument's implication.",
        "repeated-emphasis":
            "It is repeated or foregrounded through emphasis and evidence selection.",
        "evaluative-language":
            "It is stated early and framed with clear evaluative language (good/bad/should).",
        "explicit-conclusion":
            'It is explicitly stated as the conclusion (e.g., "therefore/thus/so").',
    },
    "reasoningStructure": {
        "specific-cases": "Uses specific cases, stories, or instances to support claims",
        "outcomes":       "Justifies action by their positive or negative outcomes",
        "comparison":     "Compare current citation to similar known situation",
        "indicators":     "Use indicator to suggest underlying condition",
        "expert":         "Appeal to expert knowledge or credible source",
        "none":           "None of the others",
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
        "none-detected":            "None detected",
        "fear-anxiety":             "Fear/anxiety",
        "first-impression":         "First impression",
        "emotional-reaction":       "Emotional reaction",
        "pattern-matching":         "Pattern matching",
        "insufficient-self-checking": "Insufficient self-checking",
    },
    "heuristicDetection": {
        "none-detected":     "None detected",
        "gut-feeling":       "Went with gut feeling",
        "familiar-pattern":  "Used familiar pattern",
        "surface-cues":      "Followed surface cues",
    },
}

def _norm_scalar(field: str, value) -> str:
    """Normalise a single categorical value. Returns original if not in map."""
    if value is None:
        return ""
    s = str(value).strip()
    mapping = _NORM_SCALAR.get(field, {})
    return mapping.get(s, mapping.get(s.lower(), s))


def _norm_list(field: str, values) -> str:
    """Normalise a list field and join with '; '."""
    if not values:
        return ""
    if isinstance(values, str):
        values = [values]
    mapping = _NORM_LIST.get(field, {})
    parts = []
    for v in values:
        s = str(v).strip()
        parts.append(mapping.get(s, mapping.get(s.lower(), s)))
    return "; ".join(p for p in parts if p)


def _get(ann: dict, key: str, default: str = "") -> str:
    """Return annotation field as a clean string."""
    v = ann.get(key, default)
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(str(x).strip() for x in v if x)
    return str(v).strip().rstrip("\n")


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_oracle_prompt_v2(argument: str, annotation: dict) -> str:
    """
    v2 oracle prompt: inject the annotation as "pre-analysis notes",
    then let the model generate all 7 sections.

    Unlike v1 (which pre-fills phases 1-6 and asks for section 7 only),
    v2 asks for the full output, so the format matches the baseline run.
    """
    a = annotation

    hints = []
    for field, label in [
        ("stanceTarget",      _norm_scalar("stanceTarget",      a.get("stanceTarget", ""))),
        ("knowledgeDomain",   _norm_scalar("knowledgeDomain",   a.get("knowledgeDomain", ""))),
        ("coreClaim",         _get(a, "coreClaim")),
        ("reasoningType",     _norm_scalar("reasoningType",     a.get("reasoningType", ""))),
        ("hasAssumption",     _norm_scalar("hasAssumption",     a.get("hasAssumption", ""))),
        ("primaryDomain",     _norm_scalar("primaryDomain",     a.get("primaryDomain", ""))),
        ("inferenceStrength", _norm_scalar("inferenceStrength", a.get("inferenceStrength", ""))),
        ("logicalFallacy",    _norm_scalar("logicalFallacy",    a.get("logicalFallacy", ""))),
        ("changeDecision",    _norm_scalar("changeDecision",    a.get("changeDecision", ""))),
    ]:
        if label:
            hints.append(f"  - {field}: {label}")

    hint_block = "\n".join(hints) if hints else "  (none)"

    user_msg = (
        f"The following argument has been pre-analyzed. Use these notes as your "
        f"starting point and generate all six Facione phases plus Socratic questions.\n\n"
        f"Pre-analysis notes:\n{hint_block}\n\n"
        f"Argument:\n{argument}"
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] "
        "Here is my analysis of the argument using Facione's Critical Thinking Framework:\n\n"
        "1. INTERPRETATION\n"
    )


def build_vanilla_prompt_v2(argument: str) -> str:
    """
    v2 vanilla prompt: no annotation hint, full 7-section generation.
    """
    user_msg = f"Argument:\n{argument}"
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] "
        "Here is my analysis of the argument using Facione's Critical Thinking Framework:\n\n"
        "1. INTERPRETATION\n"
    )


def build_vanilla_prompt_v3(argument: str) -> str:
    """
    v3 vanilla prompt: no schema, no Facione phases, no system prompt.
    Minimal format hint so the model knows the expected output structure.
    """
    return (
        f"<s>[INST] Read the following argument and generate up to 3 Socratic questions "
        f"that probe its weaknesses. For each question use this exact format:\n"
        f"* question: <the Socratic question>\n"
        f"* target: <short phrase describing what weakness it probes>\n"
        f"* sourceField: <which aspect of the argument it targets>\n\n"
        f"Argument:\n{argument} [/INST] "
        "7. SOCRATIC QUESTIONS (MAX 3)\n"
    )


def build_oracle_prompt(argument: str, annotation: dict) -> str:
    """
    v1 oracle prompt: pre-fills all 6 Facione phases from annotation and
    primes the model to generate only section 7 (Socratic questions).

    Legacy prompt, kept for reference; build_oracle_prompt_v2 is the default.
    """
    a = annotation  # shorthand

    phases = (
        "1. INTERPRETATION\n"
        f"\t* initialUnderstandingTarget: {_get(a, 'initialUnderstandingTarget')}\n"
        f"\t* stanceTarget: {_norm_scalar('stanceTarget', a.get('stanceTarget', ''))}\n"
        f"\t* knowledgeDomain: {_norm_scalar('knowledgeDomain', a.get('knowledgeDomain', ''))}\n"
        "\n"
        "2. ANALYSIS\n"
        f"\t* coreClaim: {_get(a, 'coreClaim')}\n"
        f"\t* minorClaim: {_get(a, 'minorClaim')}\n"
        f"\t* premise: {_get(a, 'premise')}\n"
        f"\t* paraphrasingUnderstanding: {_get(a, 'paraphrasingUnderstanding')}\n"
        f"\t* reasoningType: {_norm_scalar('reasoningType', a.get('reasoningType', ''))}\n"
        f"\t* hasAssumption: {_norm_scalar('hasAssumption', a.get('hasAssumption', ''))}\n"
        f"\t* missingComponent: {_get(a, 'missingComponent')}\n"
        f"\t* deductiveTermX: {_get(a, 'deductiveTermX')}\n"
        f"\t* deductiveTermY: {_get(a, 'deductiveTermY')}\n"
        f"\t* inductiveTermX: {_get(a, 'inductiveTermX')}\n"
        f"\t* inductiveTermY: {_get(a, 'inductiveTermY')}\n"
        f"\t* goodEvidence: {_get(a, 'goodEvidence')}\n"
        f"\t* badEvidence: {_get(a, 'badEvidence')}\n"
        "\n"
        "3. INFERENCE\n"
        f"\t* positiveConsequences: {_get(a, 'positiveConsequences')}\n"
        f"\t* negativeConsequences: {_get(a, 'negativeConsequences')}\n"
        f"\t* primaryDomain: {_norm_scalar('primaryDomain', a.get('primaryDomain', ''))}\n"
        f"\t* alternativeType: {_norm_scalar('alternativeType', a.get('alternativeType', ''))}\n"
        f"\t* alternativeKeywords: {_get(a, 'alternativeKeywords')}\n"
        "\n"
        "4. EVALUATION\n"
        f"\t* inferenceStrength: {_norm_scalar('inferenceStrength', a.get('inferenceStrength', ''))}\n"
        f"\t* credibilityFactors: {_norm_list('credibilityFactors', a.get('credibilityFactors', []))}\n"
        f"\t* logicalFallacy: {_norm_scalar('logicalFallacy', a.get('logicalFallacy', ''))}\n"
        f"\t* trustworthiness: {_norm_list('trustworthiness', a.get('trustworthiness', []))}\n"
        f"\t* trustworthinessExplanation: {_get(a, 'trustworthinessExplanation')}\n"
        f"\t* noFallacyExplanation: {_get(a, 'noFallacyExplanation')}\n"
        f"\t* fallacySpan1: {_get(a, 'fallacySpan1')}\n"
        f"\t* fallacySpan2: {_get(a, 'fallacySpan2')}\n"
        "\n"
        "5. EXPLANATION\n"
        f"\t* targetExplanationReason: {_norm_scalar('targetExplanation', a.get('targetExplanation', ''))}\n"
        f"\t* reasoningStructure: {_norm_scalar('reasoningStructure', a.get('reasoningStructure', ''))}\n"
        "\n"
        "6. SELF-REGULATION\n"
        f"\t* biasDetection: {_norm_list('biasDetection', a.get('biasDetection', []))}\n"
        f"\t* heuristicDetection: {_norm_list('heuristicDetection', a.get('heuristicDetection', []))}\n"
        f"\t* errorDetection: {_get(a, 'errorDetection')}\n"
        f"\t* changeDecision: {_norm_scalar('changeDecision', a.get('changeDecision', ''))}\n"
        f"\t* revisionType: {_get(a, 'revisionType')}\n"
        f"\t* revisionReason: {_get(a, 'revisionReason')}\n"
        "\n"
        "7. SOCRATIC QUESTIONS (MAX 3)\n"
    )

    user_msg = (
        f"Argument:\n{argument}\n\n"
        "The six-phase cognitive analysis has already been completed. "
        "Generate ONLY the SOCRATIC QUESTIONS section (max 3 questions)."
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] "
        "Here is my analysis of the argument using Facione's Critical Thinking Framework:\n\n"
        f"{phases}"
    )


# ── Vanilla prompt (no phases — argument only, full generation) ───────────────

def build_vanilla_prompt(argument: str) -> str:
    """
    v1 vanilla prompt: skips all 6 phases, jumps directly to section 7.
    Legacy prompt, kept for reference.
    """
    user_msg = (
        f"Argument:\n{argument}\n\n"
        "Do NOT perform any of the six reasoning phases. "
        "Generate ONLY the SOCRATIC QUESTIONS section (max 3 questions)."
    )
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] "
        "7. SOCRATIC QUESTIONS (MAX 3)\n"
    )


# ── Output validation (questions-only) ───────────────────────────────────────

def validate_oracle_output(text: str) -> dict:
    """Validate the generated questions section only."""
    q_pattern = re.compile(
        r'(?:^\s*\*\s*|\d+[.)]\s*\*?\s*)question:\s*(.+?)$', re.MULTILINE
    )
    questions = q_pattern.findall(text)
    n_q = len(questions)
    return {
        "has_socratic_questions": n_q > 0,
        "num_socratic_questions": n_q,
        # Placeholder rates so output JSON is compatible with baseline format
        "sections_found": 1, "sections_total": 7,
        "fields_found": 0,   "fields_total": 23,
        "label_correct": 0,  "label_total": 0,
        "section_rate":  1 / 7,
        "field_rate":    0.0,
        "label_accuracy": 0.0,
    }


# ── Single inference call ─────────────────────────────────────────────────────

def run_oracle_single(model, tokenizer, argument: str, annotation,
                      seed: int, max_new_tokens: int, device: str,
                      condition: str = "gold", prompt_version: str = "v2") -> dict:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if prompt_version == "v2":
        if condition == "vanilla":
            prompt = build_vanilla_prompt_v2(argument)
        else:
            prompt = build_oracle_prompt_v2(argument, annotation)
    elif prompt_version == "v3":
        if condition == "vanilla":
            prompt = build_vanilla_prompt_v3(argument)
        else:
            prompt = build_oracle_prompt_v2(argument, annotation)
    else:
        if condition == "vanilla":
            prompt = build_vanilla_prompt(argument)
        else:
            prompt = build_oracle_prompt(argument, annotation)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metrics = validate_oracle_output(generated_text)
    metrics["generation_time_sec"] = elapsed
    metrics["num_tokens_generated"] = len(generated_ids)

    return {"generated_text": generated_text, "metrics": metrics}


def aggregate_metrics(all_metrics: list) -> dict:
    import numpy as np
    keys = ["num_socratic_questions", "generation_time_sec", "num_tokens_generated"]
    agg = {}
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            agg[f"{key}_avg"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Oracle causation experiment: inject gold or noise annotations"
    )
    parser.add_argument(
        "--gold-json", required=True,
        help="Path to p_dev_combined JSON (gold annotations)"
    )
    parser.add_argument(
        "--condition", choices=["gold", "noise", "vanilla"], default="gold",
        help="gold = inject the matching expert annotation; noise = inject the expert "
             "annotation of the next argument (i+1); vanilla = no trace, generate questions directly"
    )
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of arguments (default: all in --gold-json)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--max-new-tokens", type=int, default=1500,
                        help="Token budget for generation (paper: 1500)")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v2",
                        help="v2 (default): full 7-section generation seeded with annotation hint; "
                             "v1: pre-fill phases 1-6, generate section 7 only (legacy)")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompt for the first argument and exit (no model loading)"
    )
    args = parser.parse_args()

    today = __import__("datetime").date.today().isoformat()
    output_path = args.output or (
        f"results/oracle_{args.condition}_{args.num_samples}args_{today}.json"
    )

    # ── Load gold annotations ─────────────────────────────────────────────────
    with open(args.gold_json, encoding="utf-8") as f:
        gold_data = json.load(f)
    annotations = gold_data["annotations"][: args.num_samples]
    n = len(annotations)
    print(f"Loaded {n} gold annotations from {args.gold_json}")

    # ── Build annotation list for this condition ──────────────────────────────
    if args.condition == "gold":
        injected = annotations  # annotation[i] → argument[i]
    elif args.condition == "noise":
        # Rotate by +1: argument i gets the expert annotation of argument (i+1) % n
        injected = [annotations[(i + 1) % n] for i in range(n)]
        print(f"Noise condition: argument i receives the annotation of argument (i+1) % {n}")
    else:
        # Vanilla: no annotation injected — placeholder list so the per-argument code is the same for all conditions
        injected = [None] * n
        print("Vanilla condition: all 6 phases skipped, questions generated directly from argument text")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 70)
        print(f"DRY RUN — condition: {args.condition}")
        print("=" * 70)
        ann = injected[0]
        arg_text = annotations[0]["argument"]
        if args.prompt_version == "v2":
            prompt = build_vanilla_prompt_v2(arg_text) if args.condition == "vanilla" \
                     else build_oracle_prompt_v2(arg_text, ann)
        elif args.prompt_version == "v3":
            prompt = build_vanilla_prompt_v3(arg_text) if args.condition == "vanilla" \
                     else build_oracle_prompt_v2(arg_text, ann)
        else:
            prompt = build_vanilla_prompt(arg_text) if args.condition == "vanilla" \
                     else build_oracle_prompt(arg_text, ann)
        print(f"\nArgument index: {annotations[0]['argumentIndex']}")
        if args.condition == "noise":
            print(f"Injected annotation from: argumentIndex {ann.get('argumentIndex')}")
        print("\n--- FULL PROMPT ---\n")
        print(prompt)
        print("\n--- END PROMPT ---")
        print(f"\nPrompt length: {len(prompt)} chars")

        # Normalization sanity check: warn only if raw value has NO entry in map
        print("\n--- Normalization sanity check (all annotations) ---")
        errors = []
        for i, a in enumerate(injected):
            if a is None:
                continue
            for field in ["knowledgeDomain", "reasoningType", "hasAssumption",
                          "primaryDomain", "alternativeType", "logicalFallacy",
                          "changeDecision", "targetExplanation", "reasoningStructure"]:
                raw = a.get(field, "")
                if raw == "" or raw is None:
                    continue
                s = str(raw).strip()
                mapping = _NORM_SCALAR.get(field, {})
                if s not in mapping and s.lower() not in mapping:
                    errors.append(
                        f"  [WARN] arg {i} {field}={repr(raw)} -> not in normalization map"
                    )
        if errors:
            for e in errors:
                print(e)
        else:
            print("  All categorical fields normalised successfully.")
        return

    # ── Load model (off-the-shelf weights, 4-bit NF4) ─────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from model_configs import get_model_config

    config = get_model_config("llama2")
    model_id = config["model_id"]
    token = (os.getenv("HUGGING_FACE_HUB_TOKEN")
             or os.getenv("HF_TOKEN")
             or os.getenv("HF_HUB_TOKEN"))

    load_kwargs = {
        "device_map": "auto",
        "token": token,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
    }
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Run inference ─────────────────────────────────────────────────────────
    results = {
        "config": {
            "model":          "llama2_oracle",
            "condition":      args.condition,
            "noise":          "rotation_i+1" if args.condition == "noise" else None,
            "prompt_version": args.prompt_version,
            "model_id":       model_id,
            "seeds":          args.seeds,
            "num_samples":    n,
            "max_new_tokens": args.max_new_tokens,
            "gold_json":      args.gold_json,
        },
        "per_sample": [],
        "aggregate":  {},
    }

    all_metrics = []

    for i, (ann_arg, ann_inject) in enumerate(zip(annotations, injected)):
        argument = ann_arg["argument"]
        arg_idx  = ann_arg.get("argumentIndex")  # Keep for backward compat
        arg_id   = ann_arg.get("argument_id", ann_arg.get("id", i))  # Use argument_id if available

        print(f"\n{'='*60}")
        print(f"Argument {arg_id} [{i+1}/{n}]: {argument[:80]}...")
        if args.condition == "noise" and ann_inject is not None:
            print(f"  Injecting annotation from argumentIndex {ann_inject.get('argumentIndex')}")
        print(f"{'='*60}")

        sample_result = {
            "argument_index": arg_idx,
            "argument_id":    arg_id,
            "argument":       argument,
            "per_seed":       {},
            "seed_aggregate": {},
        }

        if args.condition == "noise":
            sample_result["noise_source"] = {
                "injected_argumentIndex": ann_inject.get("argumentIndex"),
            }
        seed_metrics = []

        for seed in args.seeds:
            print(f"  Seed {seed}:")
            result = run_oracle_single(
                model, tokenizer, argument, ann_inject,
                seed=seed, max_new_tokens=args.max_new_tokens, device=device,
                condition=args.condition, prompt_version=args.prompt_version,
            )
            m = result["metrics"]
            print(f"    Socratic Qs: {m['num_socratic_questions']}")
            print(f"    Time:        {m['generation_time_sec']:.1f}s "
                  f"({m['num_tokens_generated']} tokens)")

            sample_result["per_seed"][str(seed)] = {
                "generated_text": result["generated_text"],
                "metrics":        m,
            }
            seed_metrics.append(m)
            all_metrics.append(m)

        sample_result["seed_aggregate"] = aggregate_metrics(seed_metrics)
        results["per_sample"].append(sample_result)

    results["aggregate"] = aggregate_metrics(all_metrics)

    avg_q = results["aggregate"].get("num_socratic_questions_avg", 0)
    print(f"\n{'='*60}")
    print(f"DONE — condition: {args.condition}")
    print(f"  Avg Socratic questions/arg: {avg_q:.2f}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
