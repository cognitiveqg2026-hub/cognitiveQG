#!/usr/bin/env python3
"""
inference.py

Run baseline inference with an off-the-shelf open-source model,
loaded in 4-bit NF4 quantization, using multiple seeds (44, 45, 46).
Reports per-seed results and avg+std statistics.

Usage:
    python inference.py \
        --model llama3 \
        --output results/inference/llama3_results.json

    python inference.py \
        --model llama2 \
        --prompt-file prompts/P4_complete_20260704.txt \
        --test-csv data/Dev_combined/p_dev_combined_2026-02-26.csv \
        --num-samples 3 \
        --seeds 44 45 46 \
        --output outputs/results/inference/inference_results.json
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# model_configs.py lives at the package root, one level above this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_configs import get_model_config, list_models, _PRIME


# ── System prompt (default; normally overridden via --prompt-file) ───────────
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


# ── Validation ────────────────────────────────────────────────────────────────

EXPECTED_SECTIONS = [
    "1. INTERPRETATION",
    "2. ANALYSIS",
    "3. INFERENCE",
    "4. EVALUATION",
    "5. EXPLANATION",
    "6. SELF-REGULATION",
    "7. SOCRATIC QUESTIONS",
]

EXPECTED_FIELDS = [
    "initialUnderstandingTarget", "stanceTarget", "knowledgeDomain",
    "coreClaim", "minorClaim", "premise", "reasoningType", "hasAssumption",
    "positiveConsequences", "negativeConsequences", "primaryDomain",
    "alternativeType", "alternativeKeywords",
    "inferenceStrength", "credibilityFactors", "logicalFallacy", "trustworthiness",
    "targetExplanationReason", "reasoningStructure",
    "biasDetection", "heuristicDetection", "changeDecision", "revisionType",
]

ALLOWED_VALUES = {
    "stanceTarget": {"favor", "against", "neutral"},
    "knowledgeDomain": {"Technical/Scientific", "Legal/Policy", "Cultural/Social",
                        "Economic/Business", "Government/Law", "No specialized knowledge"},
    "reasoningType": {"Inductive", "Deductive", "None of the above"},
    "hasAssumption": {"Yes=Complete", "No=Missing Components"},
    "inferenceStrength": {"1=Weak", "2=Moderate", "3=Strong"},
    "logicalFallacy": {"No logical fallacy", "False dilemma", "Hasty generalisation",
                       "Ad hominem", "Strawman", "Circular reasoning", "False cause"},
    "changeDecision": {"Yes", "No"},
    "revisionType": {"minor-adjustment", "major-change", "complete-revision", "no-change"},
}


def strip_think_block(text: str) -> str:
    """Remove DeepSeek-R1 chain-of-thought <think>...</think> block from output."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip()


# Section name keywords for flexible matching (order-insensitive, format-insensitive)
_SECTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}INTERPRETATION\*{0,2}", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}ANALYSIS\*{0,2}", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}INFER[A-Z]*\*{0,2}", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}EVALUATION\*{0,2}", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}EXPLANATION\*{0,2}", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}SELF.REGULATION\*{0,2}", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\d+[.)]\s*\*{0,2}SOCRATIC[A-Z ]*\*{0,2}", re.IGNORECASE),
]


def validate_output(text: str) -> dict:
    """Validate generated output structure and labels. Returns metrics dict."""
    # Strip chain-of-thought block (DeepSeek-R1 and similar reasoning models)
    clean_text = strip_think_block(text)

    metrics = {
        "sections_found": 0,
        "sections_total": len(EXPECTED_SECTIONS),
        "fields_found": 0,
        "fields_total": len(EXPECTED_FIELDS),
        "label_correct": 0,
        "label_total": 0,
        "has_socratic_questions": False,
        "num_socratic_questions": 0,
    }

    # Check sections — try exact match first, then flexible pattern
    for i, section in enumerate(EXPECTED_SECTIONS):
        if section in clean_text or _SECTION_PATTERNS[i].search(clean_text):
            metrics["sections_found"] += 1

    # Check fields
    for field in EXPECTED_FIELDS:
        if f"{field}:" in clean_text:
            metrics["fields_found"] += 1

    # Check label correctness
    for field, allowed in ALLOWED_VALUES.items():
        pattern = rf"{field}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, clean_text)
        if match:
            metrics["label_total"] += 1
            value = match.group(1).strip()
            if value in allowed or value == "":
                metrics["label_correct"] += 1

    # Check Socratic questions — accept both "1) question:" and "- question:" formats
    q_pattern = r'(?:\d+[.)]\s*question:|-\s*question:)'
    questions = re.findall(q_pattern, clean_text, re.IGNORECASE)
    metrics["num_socratic_questions"] = len(questions)
    metrics["has_socratic_questions"] = len(questions) > 0

    # Compute rates
    metrics["section_rate"] = metrics["sections_found"] / metrics["sections_total"]
    metrics["field_rate"] = metrics["fields_found"] / metrics["fields_total"]
    metrics["label_accuracy"] = (
        metrics["label_correct"] / metrics["label_total"]
        if metrics["label_total"] > 0 else 0.0
    )

    return metrics


# ── Inference ─────────────────────────────────────────────────────────────────

def build_prompt(argument: str, config: dict, tokenizer=None) -> str:
    """Build the inference prompt using model-specific template."""
    user_msg = f"Argument:\n{argument}"
    prompt_fn = config.get("prompt_builder")

    if prompt_fn is None:
        # Fallback to LLaMA 2 style — prime the response to prevent base chat behavior
        return (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{user_msg} [/INST] Here is my analysis of the argument using Facione's Critical Thinking Framework:\n\n1. INTERPRETATION\n"
        )

    if config.get("needs_tokenizer_for_template", False):
        return prompt_fn(SYSTEM_PROMPT, user_msg, tokenizer=tokenizer)
    else:
        return prompt_fn(SYSTEM_PROMPT, user_msg)


def run_inference_single(model, tokenizer, argument: str, seed: int,
                         config: dict,
                         max_new_tokens: int = 1500, device: str = "cuda") -> dict:
    """Run inference for a single argument with a specific seed."""
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = build_prompt(argument, config, tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Per-model overrides (e.g. deepseek_chat's pad_token_id fix) take
    # precedence over these defaults, so merge rather than pass both.
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    generate_kwargs.update(config.get("generation_kwargs", {}))
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    elapsed = time.time() - start_time

    # Decode only the generated tokens (skip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    # Fix residual BPE space/newline markers (deepseek-llm and similar tokenizers)
    # Use explicit Unicode escapes to avoid any source-encoding ambiguity
    generated_text = generated_text.replace("Ġ", " ").replace("Ċ", "\n")
    # Prepend priming so parse_generated_text() can extract initialUnderstandingTarget
    generated_text = _PRIME + generated_text

    # Validate
    metrics = validate_output(generated_text)
    metrics["generation_time_sec"] = elapsed
    metrics["num_tokens_generated"] = len(generated_ids)

    return {
        "generated_text": generated_text,
        "metrics": metrics,
    }


def aggregate_metrics(all_metrics: list) -> dict:
    """Compute avg and std across seed runs."""
    keys = [
        "section_rate", "field_rate", "label_accuracy",
        "num_socratic_questions", "generation_time_sec", "num_tokens_generated",
    ]
    agg = {}
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            agg[f"{key}_avg"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg


def main():
    parser = argparse.ArgumentParser(description="Multi-seed baseline inference for CognitiveQG")
    parser.add_argument("--model", default="llama2", choices=list_models(),
                        help="Model family (default: llama2)")
    parser.add_argument(
        "--test-csv",
        default="data/Dev_combined/p_dev_combined_2026-02-26.csv",
        help="CSV with test arguments",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of arguments (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46], help="Seeds to use")
    parser.add_argument("--max-new-tokens", type=int, default=1500, help="Max tokens to generate (paper: 1500)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate environment, paths, and HF token without loading the model")
    parser.add_argument("--prompt-file", default=None,
                        help="Path to a .txt prompt file (e.g. prompts/P4_complete_20260704.txt). Overrides hardcoded SYSTEM_PROMPT.")
    args = parser.parse_args()

    global SYSTEM_PROMPT
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as _pf:
            SYSTEM_PROMPT = _pf.read().strip()
        print(f"[INFO] Loaded system prompt from: {args.prompt_file}")

    config = get_model_config(args.model)
    model_id = config["model_id"]
    token = (os.getenv("HUGGING_FACE_HUB_TOKEN")
             or os.getenv("HF_TOKEN")
             or os.getenv("HF_HUB_TOKEN"))
    output_path = args.output or f"outputs/results/inference/inference_{args.model}_results.json"

    print(f"Model: {config['display_name']} ({args.model})")

    # ── Dry-run: validate everything without loading the model ─────────────────
    if args.dry_run:
        import sys as _sys
        ok = True

        print("\n[dry-run] Checking environment...")

        # HF token
        if token:
            print("  [OK] HF token found")
        else:
            print("  [FAIL] No HF token found. Set HUGGING_FACE_HUB_TOKEN or run: huggingface-cli login")
            ok = False

        # HF repo accessibility
        try:
            from huggingface_hub import model_info
            model_info(model_id, token=token)
            print(f"  [OK] HF repo accessible: {model_id}")
        except Exception as e:
            print(f"  [FAIL] Cannot access HF repo {model_id}: {e}")
            ok = False

        # Test CSV
        if os.path.isfile(args.test_csv):
            df_check = pd.read_csv(args.test_csv)
            print(f"  [OK] Test CSV exists: {args.test_csv} ({len(df_check)} rows)")
        else:
            print(f"  [FAIL] Test CSV not found: {args.test_csv}")
            ok = False

        # CUDA
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  [WARN] CUDA not available — will run on CPU (slow)")

        # Output dir writable
        out_dir = os.path.dirname(output_path) or "."
        if os.path.isdir(out_dir) or not os.path.exists(out_dir):
            print(f"  [OK] Output path: {output_path}")
        else:
            print(f"  [FAIL] Output directory is not writable: {out_dir}")
            ok = False

        print()
        if ok:
            print("[dry-run] All checks passed. Ready to submit.")
        else:
            print("[dry-run] One or more checks FAILED. Fix the issues above before submitting.")
            _sys.exit(1)
        return

    # ── Load model (off-the-shelf weights, 4-bit NF4) ──────────────────────────
    trust_remote = config.get("trust_remote_code", False)
    quant_config = config.get("quantization")

    load_kwargs = {
        "device_map": "auto",
        "token": token,
        "trust_remote_code": trust_remote,
    }
    max_memory = config.get("max_memory")
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    if quant_config:
        print("Loading base model with 4-bit quantization...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_config)
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    tokenizer_kwargs = config.get("tokenizer_kwargs", {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote,
        token=token,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load test data ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.test_csv)
    test_samples = df if args.num_samples is None else df.head(args.num_samples)
    print(f"Running inference on {len(test_samples)} samples with seeds {args.seeds}")

    # ── Run inference ──────────────────────────────────────────────────────────
    results = {
        "config": {
            "model": args.model,
            "model_id": model_id,
            "seeds": args.seeds,
            "num_samples": len(test_samples),
            "max_new_tokens": args.max_new_tokens,
        },
        "per_sample": [],
        "aggregate": {},
    }

    all_metrics_across_all = []

    for idx, row in test_samples.iterrows():
        argument = str(row["argument"])
        print(f"\n{'='*60}")
        print(f"Sample {idx}: {argument[:80]}...")
        print(f"{'='*60}")

        sample_result = {
            "argument_index": int(row["argumentIndex"]),
            "argument": argument,
            "per_seed": {},
            "seed_aggregate": {},
        }

        seed_metrics = []

        for seed in args.seeds:
            print(f"\n  Seed {seed}:")
            result = run_inference_single(
                model, tokenizer, argument, seed,
                config=config,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )

            m = result["metrics"]
            print(f"    Sections: {m['sections_found']}/{m['sections_total']} "
                  f"({m['section_rate']:.0%})")
            print(f"    Fields:   {m['fields_found']}/{m['fields_total']} "
                  f"({m['field_rate']:.0%})")
            print(f"    Labels:   {m['label_correct']}/{m['label_total']} "
                  f"({m['label_accuracy']:.0%})")
            print(f"    Socratic: {m['num_socratic_questions']} questions")
            print(f"    Time:     {m['generation_time_sec']:.1f}s "
                  f"({m['num_tokens_generated']} tokens)")

            sample_result["per_seed"][str(seed)] = {
                "generated_text": result["generated_text"],
                "metrics": m,
            }
            seed_metrics.append(m)
            all_metrics_across_all.append(m)

        # Aggregate per sample across seeds
        sample_result["seed_aggregate"] = aggregate_metrics(seed_metrics)
        results["per_sample"].append(sample_result)

        # Print per-sample aggregate
        agg = sample_result["seed_aggregate"]
        print(f"\n  Aggregate (avg +/- std) for sample {idx}:")
        print(f"    Section rate:  {agg.get('section_rate_avg', 0):.2%} "
              f"+/- {agg.get('section_rate_std', 0):.2%}")
        print(f"    Field rate:    {agg.get('field_rate_avg', 0):.2%} "
              f"+/- {agg.get('field_rate_std', 0):.2%}")
        print(f"    Label acc:     {agg.get('label_accuracy_avg', 0):.2%} "
              f"+/- {agg.get('label_accuracy_std', 0):.2%}")

    # ── Global aggregate ───────────────────────────────────────────────────────
    results["aggregate"] = aggregate_metrics(all_metrics_across_all)

    print(f"\n{'='*60}")
    print("OVERALL RESULTS (all samples, all seeds)")
    print(f"{'='*60}")
    agg = results["aggregate"]
    print(f"  Section rate:     {agg.get('section_rate_avg', 0):.2%} "
          f"+/- {agg.get('section_rate_std', 0):.2%}")
    print(f"  Field rate:       {agg.get('field_rate_avg', 0):.2%} "
          f"+/- {agg.get('field_rate_std', 0):.2%}")
    print(f"  Label accuracy:   {agg.get('label_accuracy_avg', 0):.2%} "
          f"+/- {agg.get('label_accuracy_std', 0):.2%}")
    print(f"  Socratic Qs:      {agg.get('num_socratic_questions_avg', 0):.1f} "
          f"+/- {agg.get('num_socratic_questions_std', 0):.1f}")
    print(f"  Gen time:         {agg.get('generation_time_sec_avg', 0):.1f}s "
          f"+/- {agg.get('generation_time_sec_std', 0):.1f}s")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
