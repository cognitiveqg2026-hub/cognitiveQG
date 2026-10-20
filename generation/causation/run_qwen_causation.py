#!/usr/bin/env python3
"""
run_qwen_causation.py

Qwen version of the CognitiveQG causation experiment.

This script mirrors scripts/inference/run_oracle_causation.py and preserves:
  - the same gold/noise/vanilla intervention logic,
  - the same annotation normalization helpers,
  - the same prompt semantics and experiment conditions,
  - the same per_sample / per_seed / aggregate JSON output format.

Qwen-specific changes:
  - model loading comes from scripts/inference/model_configs.py (--model qwen,
    qwen_14b, qwen_1_5b, etc.),
  - prompts are rendered through tokenizer.apply_chat_template() because Qwen
    models use ChatML-style role templates rather than LLaMA [INST] markers,
  - tokenizer is loaded with the registry's Qwen tokenizer settings,
  - Qwen 2.5 models are normally loaded with trust_remote_code=True and 4-bit
    NF4 quantization, as defined in the registry.

Dependencies:
  pip install torch transformers accelerate bitsandbytes numpy

Environment:
  source ~/envs/cognitiveQG/bin/activate
  cd ~/cognitiveQG
  export HF_HUB_TOKEN=<token-if-needed>

Single-condition examples:
  python scripts/inference/run_qwen_causation.py \\
      --model qwen \\
      --gold-json data/Dev_combined/p_dev_combined_2026-02-26.json \\
      --condition gold \\
      --num-samples 20 \\
      --seeds 44 \\
      --prompt-version v2 \\
      --output results/causation/qwen_oracle_20args_2026-05-18.json

  python scripts/inference/run_qwen_causation.py \\
      --model qwen_14b \\
      --gold-json data/Dev_combined/p_dev_combined_2026-02-26.json \\
      --condition vanilla \\
      --prompt-version v3 \\
      --output results/causation/qwen14b_v3_vanilla_20args_2026-05-18.json

Full four-condition experiment, matching the LLaMA 2 semantic-causation setup:
  python scripts/inference/run_qwen_causation.py \\
      --model qwen \\
      --gold-json data/Dev_combined/p_dev_combined_2026-02-26.json \\
      --run-all \\
      --num-samples 20 \\
      --seeds 44 \\
      --output-dir results/causation
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import date

import numpy as np

# Local imports: reuse the LLaMA 2 experiment's intervention/evaluation logic.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
# model_configs.py lives at the package root, one level above this file.
sys.path.insert(0, os.path.dirname(THIS_DIR))

from model_configs import get_model_config, list_models
from run_oracle_causation import (  # noqa: E402
    SYSTEM_PROMPT,
    _NORM_SCALAR,
    aggregate_metrics,
    build_oracle_prompt_v2,
    build_vanilla_prompt_v2,
    validate_oracle_output,
)


ASSISTANT_PRIMER = (
    "Here is my analysis of the argument using Facione's Critical Thinking "
    "Framework:\n\n1. INTERPRETATION\n"
)

QUESTIONS_ONLY_PRIMER = "7. SOCRATIC QUESTIONS (MAX 3)\n"

# Minimal system prompt for vanilla v3 — avoids conflicting with the Facione
# framework (SYSTEM_PROMPT) which causes degenerate outputs when paired with
# a phase-7-only primer.
VANILLA_V3_SYSTEM = (
    "You are a critical thinking assistant. "
    "Generate Socratic questions that probe the weaknesses of arguments."
)


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Set best-effort reproducibility controls for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except Exception:
        # Dry-run can execute without torch fully available.
        pass


def extract_qwen_user_and_assistant(raw_prompt: str) -> tuple[str, str]:
    """
    Convert the LLaMA-reference prompt into Qwen chat-template components.

    The reference implementation constructs one string that contains the system
    prompt, user content, and assistant primer. To preserve the exact experiment
    wording while using Qwen's chat template, we extract only the user content
    from that reference prompt and supply the same assistant primer separately.
    """
    if "[/INST]" not in raw_prompt:
        return raw_prompt, ASSISTANT_PRIMER

    before_inst, after_inst = raw_prompt.split("[/INST]", 1)
    user_part = before_inst
    if "<</SYS>>" in user_part:
        user_part = user_part.split("<</SYS>>", 1)[1]
    user_part = user_part.strip()

    assistant_part = after_inst.strip()
    if not assistant_part:
        assistant_part = ASSISTANT_PRIMER
    return user_part, assistant_part


def build_qwen_vanilla_v3_prompt(tokenizer, user_msg: str) -> str:
    """
    Clean prompt for vanilla v3: minimal system message, no phase primer,
    add_generation_prompt=True so the model generates freely.

    The full SYSTEM_PROMPT (Facione 6-phase) must NOT be used here — it
    contradicts the v3 zero-shot task and causes newline loops, ePub
    repetition, and placeholder outputs in Qwen2.5.
    """
    messages = [
        {"role": "system",  "content": VANILLA_V3_SYSTEM},
        {"role": "user",    "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def build_qwen_chat_prompt(tokenizer, user_msg: str, assistant_primer: str,
                           enable_thinking: bool = True) -> str:
    """
    Render prompt with the tokenizer's native chat template.

    The model's generation begins inside the assistant turn. We include a
    partial assistant message as the final chat message and set
    add_generation_prompt=False so the model continues from that primer.
    For Qwen3, enable_thinking=False suppresses <think> blocks that would
    corrupt structured output parsing.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_primer},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_qwen_prompt(tokenizer, argument: str, annotation, condition: str,
                      prompt_version: str, enable_thinking: bool = True) -> str:
    """Build the chat-template prompt corresponding to the LLaMA reference condition."""
    if prompt_version == "v3" and condition == "vanilla":
        user_msg = (
            "Read the following argument and generate up to 3 Socratic questions "
            "that probe its weaknesses. For each question use this exact format:\n"
            "* question: <the Socratic question>\n"
            "* target: <short phrase describing what weakness it probes>\n"
            "* sourceField: <which aspect of the argument it targets>\n\n"
            f"Argument:\n{argument}"
        )
        return build_qwen_vanilla_v3_prompt(tokenizer, user_msg)

    if condition == "vanilla":
        raw_prompt = build_vanilla_prompt_v2(argument)
    else:
        raw_prompt = build_oracle_prompt_v2(argument, annotation)

    user_msg, assistant_primer = extract_qwen_user_and_assistant(raw_prompt)
    return build_qwen_chat_prompt(tokenizer, user_msg, assistant_primer,
                                  enable_thinking=enable_thinking)


def run_qwen_single(model, tokenizer, argument: str, annotation, seed: int,
                    max_new_tokens: int, device: str, condition: str,
                    prompt_version: str, deterministic: bool = False,
                    temperature: float = 0.1, top_p: float = 0.9,
                    repetition_penalty: float = 1.15,
                    enable_thinking: bool = True) -> dict:
    """Run one generation call and return LLaMA-compatible metrics."""
    import torch

    set_reproducibility(seed, deterministic=deterministic)
    prompt = build_qwen_prompt(tokenizer, argument, annotation, condition, prompt_version,
                               enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metrics = validate_oracle_output(generated_text)
    metrics["generation_time_sec"] = elapsed
    metrics["num_tokens_generated"] = int(len(generated_ids))
    return {"generated_text": generated_text, "metrics": metrics}


def load_gold_annotations(path: str, num_samples: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        gold_data = json.load(f)
    return gold_data["annotations"][:num_samples]


def build_injected_annotations(annotations: list[dict], condition: str) -> list:
    """Return the annotation injected for each argument (None for vanilla).

    gold:  the expert annotation of the same argument.
    noise: the expert annotation of the next argument, (i+1) mod n.
    """
    n = len(annotations)
    if condition == "gold":
        return annotations
    if condition == "noise":
        return [annotations[(i + 1) % n] for i in range(n)]
    return [None] * n


def load_qwen_model_and_tokenizer(model_name: str,
                                  load_in_4bit: bool = True):
    """Load an off-the-shelf model in 4-bit NF4 and its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = get_model_config(model_name)
    model_id = config["model_id"]
    token = (
        os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )
    trust_remote = config.get("trust_remote_code", True)

    load_kwargs = {
        "device_map": "auto",
        "token": token,
        "trust_remote_code": trust_remote,
    }

    if load_in_4bit:
        quant_config = config.get("quantization") or {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        }
        load_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_config)
    else:
        load_kwargs["torch_dtype"] = torch.float16

    print(f"Loading Qwen base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    tokenizer_kwargs = config.get("tokenizer_kwargs", {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=trust_remote,
        **tokenizer_kwargs,
    )
    tokenizer.padding_side = config.get("padding_side", "right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, tokenizer, device, model_id


def dry_run(args, condition: str, prompt_version: str) -> None:
    """Print the first generated Qwen prompt without loading model weights."""
    from transformers import AutoTokenizer

    config = get_model_config(args.model)
    token = (
        os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        token=token,
        trust_remote_code=config.get("trust_remote_code", True),
        **config.get("tokenizer_kwargs", {}),
    )
    annotations = load_gold_annotations(args.gold_json, args.num_samples)
    injected = build_injected_annotations(annotations, condition)
    enable_thinking = not args.model.startswith("qwen3")
    prompt = build_qwen_prompt(
        tokenizer,
        annotations[0]["argument"],
        injected[0],
        condition,
        prompt_version,
        enable_thinking=enable_thinking,
    )

    print("\n" + "=" * 70)
    print(f"DRY RUN — Qwen causation | condition={condition} | prompt={prompt_version}")
    print("=" * 70)
    print(f"Model: {args.model} ({config['model_id']})")
    print(f"Argument index: {annotations[0]['argumentIndex']}")
    print("\n--- FULL QWEN CHAT-TEMPLATE PROMPT ---\n")
    print(prompt)
    print("\n--- END PROMPT ---")
    print(f"\nPrompt length: {len(prompt)} chars")

    print("\n--- Normalization sanity check ---")
    errors = []
    for i, a in enumerate(injected):
        if a is None:
            continue
        for field in [
            "knowledgeDomain", "reasoningType", "hasAssumption",
            "primaryDomain", "alternativeType", "logicalFallacy",
            "changeDecision", "targetExplanation", "reasoningStructure",
        ]:
            raw = a.get(field, "")
            if raw == "" or raw is None:
                continue
            s = str(raw).strip()
            mapping = _NORM_SCALAR.get(field, {})
            if s not in mapping and s.lower() not in mapping:
                errors.append(
                    f"  [WARN] arg {i} {field}={raw!r} not in normalization map"
                )
    print("\n".join(errors) if errors else "  All categorical fields normalised successfully.")


def run_condition(args, condition: str, prompt_version: str,
                  output_path: str) -> None:
    """Run one causation condition and save LLaMA-compatible output JSON."""
    set_reproducibility(args.seeds[0] if args.seeds else 42, args.deterministic)

    annotations = load_gold_annotations(args.gold_json, args.num_samples)
    n = len(annotations)
    injected = build_injected_annotations(annotations, condition)

    enable_thinking = not args.model.startswith("qwen3")

    print(f"Loaded {n} gold annotations from {args.gold_json}")
    print(f"Condition: {condition} | prompt_version: {prompt_version}")

    model, tokenizer, device, model_id = load_qwen_model_and_tokenizer(
        args.model,
        load_in_4bit=not args.no_4bit,
    )

    results = {
        "config": {
            "model": f"{args.model}_causation",
            "condition": condition,
            "noise": "rotation_i+1" if condition == "noise" else None,
            "prompt_version": prompt_version,
            "model_id": model_id,
            "seeds": args.seeds,
            "num_samples": n,
            "max_new_tokens": args.max_new_tokens,
            "gold_json": args.gold_json,
            "qwen_notes": {
                "chat_template": "tokenizer.apply_chat_template",
                "tokenizer_use_fast": get_model_config(args.model)
                .get("tokenizer_kwargs", {})
                .get("use_fast"),
                "quantization": "4-bit NF4" if not args.no_4bit else "fp16",
            },
        },
        "per_sample": [],
        "aggregate": {},
    }

    all_metrics = []
    for i, (ann_arg, ann_inject) in enumerate(zip(annotations, injected)):
        argument = ann_arg["argument"]
        arg_idx = ann_arg.get("argumentIndex")
        arg_id = ann_arg.get("argument_id", ann_arg.get("id", i))

        print(f"\n{'=' * 60}")
        print(f"Argument {arg_id} [{i + 1}/{n}]: {argument[:80]}...")
        if condition == "noise" and ann_inject is not None:
            print(f"  Injecting annotation from argumentIndex {ann_inject.get('argumentIndex')}")
        print(f"{'=' * 60}")

        sample_result = {
            "argument_index": arg_idx,
            "argument_id": arg_id,
            "argument": argument,
            "per_seed": {},
            "seed_aggregate": {},
        }

        if condition == "noise":
            sample_result["noise_source"] = {
                "injected_argumentIndex": injected[i].get("argumentIndex"),
            }

        seed_metrics = []
        for seed in args.seeds:
            print(f"  Seed {seed}:")
            result = run_qwen_single(
                model,
                tokenizer,
                argument,
                ann_inject,
                seed=seed,
                max_new_tokens=args.max_new_tokens,
                device=device,
                condition=condition,
                prompt_version=prompt_version,
                deterministic=args.deterministic,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                enable_thinking=enable_thinking,
            )
            m = result["metrics"]
            print(f"    Socratic Qs: {m['num_socratic_questions']}")
            print(
                f"    Time:        {m['generation_time_sec']:.1f}s "
                f"({m['num_tokens_generated']} tokens)"
            )
            sample_result["per_seed"][str(seed)] = {
                "generated_text": result["generated_text"],
                "metrics": m,
            }
            seed_metrics.append(m)
            all_metrics.append(m)

        sample_result["seed_aggregate"] = aggregate_metrics(seed_metrics)
        results["per_sample"].append(sample_result)

    results["aggregate"] = aggregate_metrics(all_metrics)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {output_path}")


def run_all_conditions(args) -> None:
    """
    Run the four outputs used by the LLaMA 2 semantic-causation analysis:
      vanilla:  no phases, minimal v3 prompt
      noise:    expert annotation of the next argument as notes, full v2 prompt
      baseline: no gold notes, full v2 prompt
      oracle:   gold notes, full v2 prompt
    """
    tag = args.date_tag or date.today().isoformat()
    prefix = args.output_prefix or args.model
    jobs = [
        ("vanilla", "vanilla", "v3", f"{prefix}_vanilla_{args.num_samples}args_{tag}.json"),
        ("noise", "noise", "v2", f"{prefix}_noise_{args.num_samples}args_{tag}.json"),
        ("baseline", "vanilla", "v2", f"{prefix}_baseline_{args.num_samples}args_{tag}.json"),
        ("oracle", "gold", "v2", f"{prefix}_oracle_{args.num_samples}args_{tag}.json"),
    ]

    for label, condition, prompt_version, filename in jobs:
        print("\n" + "#" * 72)
        print(f"RUN-ALL: {label} -> condition={condition}, prompt={prompt_version}")
        print("#" * 72)
        output_path = os.path.join(args.output_dir, filename)
        run_condition(args, condition, prompt_version, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen causation experiment matching the LLaMA 2 pipeline"
    )
    parser.add_argument("--model", default="qwen", choices=list_models(),
                        help="Model key from model_configs.py: llama2, llama3, qwen, mistral, olmo, qwen3")
    parser.add_argument("--gold-json", required=True,
                        help="Gold annotation JSON")
    parser.add_argument("--condition", choices=["gold", "noise", "vanilla"], default="gold")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--prompt-version", choices=["v2", "v3"], default="v2")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--deterministic", action="store_true",
                        help="Best-effort deterministic torch/cudnn settings")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization and load fp16")
    parser.add_argument("--output", default=None,
                        help="Output JSON for single-condition mode")
    parser.add_argument("--output-dir", default="results/causation",
                        help="Output directory for --run-all")
    parser.add_argument("--output-prefix", default=None,
                        help="Filename prefix for --run-all outputs")
    parser.add_argument("--date-tag", default=None,
                        help="Filename date tag for --run-all")
    parser.add_argument("--run-all", action="store_true",
                        help="Run vanilla/noise/baseline/oracle outputs for direct comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first prompt and exit without loading model weights")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        condition = "vanilla" if args.run_all else args.condition
        prompt_version = "v3" if args.run_all else args.prompt_version
        dry_run(args, condition, prompt_version)
        return

    if args.run_all:
        run_all_conditions(args)
        return

    output_path = args.output or os.path.join(
        args.output_dir,
        f"{args.model}_{args.condition}_{args.num_samples}args_{date.today().isoformat()}.json",
    )
    run_condition(args, args.condition, args.prompt_version, output_path)


if __name__ == "__main__":
    main()
