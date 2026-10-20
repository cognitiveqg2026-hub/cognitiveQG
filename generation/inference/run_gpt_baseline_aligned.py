#!/usr/bin/env python3
"""
run_gpt_baseline_aligned.py

GPT-4o / GPT-5.5 P4-structured baseline for CognitiveQG, with an
argument-order alignment guard before any paid API call.
Mirrors the open-model baseline pipeline exactly:
  - Same P4_complete_20260704.txt system prompt as the open-source models
  - Same raw_output schema (per_sample → per_seed → generated_text)
  - Feeds into parse_model_output.py → preprocess_run2 → judge pipeline unchanged

Usage:
    python inference/run_gpt_baseline_aligned.py \\
        --model gpt-4o \\
        --gold-json path/to/gold_annotations.json \\
        --prompt-file prompts/P4_complete_20260704.txt \\
        --seeds 44 45 46 \\
        --output results/gpt-4o/raw_output.json

    # Smoke: first 5 args, seed 44 only
    python inference/run_gpt_baseline_aligned.py \\
        --model gpt-4o --gold-json path/to/gold_annotations.json \\
        --num-samples 5 --seeds 44 \\
        --output results/gpt-4o/raw_output.json

    # Dry-run (no API call)
    python inference/run_gpt_baseline_aligned.py --model gpt-4o --gold-json path/to/gold_annotations.json --dry-run --num-samples 1

Set API key:
    export OPENAI_API_KEY=sk-...
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not found. Install: pip install openai")

BASE_DIR     = Path(__file__).resolve().parents[1]  # package root
DEFAULT_GOLD = BASE_DIR / "data/gold_annotations.json"
DEFAULT_PROMPT = BASE_DIR / "prompts/P4_complete_20260704.txt"  # prompt recorded in the published GPT-4o/GPT-5.5 runs

SUPPORTED_MODELS = ["gpt-4o", "gpt-5", "gpt-5.5"]

# Models that use max_completion_tokens instead of max_tokens
COMPLETION_TOKENS_MODELS = {"gpt-5", "gpt-5.5", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"}

# Models that do not support custom temperature
FIXED_TEMP_MODELS = {"gpt-5", "gpt-5.5", "o1", "o1-mini", "o3", "o3-mini"}

MODEL_TEMP   = {"gpt-4o": 0.3}
TEMP_DEFAULT = 0.2
MAX_TOKENS        = 1500   # gpt-4o
MAX_TOKENS_LARGE  = 8000   # gpt-5/gpt-5.5: reasoning models need large budget
LARGE_OUTPUT_MODELS = {"gpt-5", "gpt-5.5"}
MAX_RETRIES  = 3
RETRY_DELAY  = 5.0


def get_temperature(model: str) -> float:
    return MODEL_TEMP.get(model, TEMP_DEFAULT)


def call_api(client: OpenAI, model: str, messages: list, seed: int) -> str:
    tokens_key = "max_completion_tokens" if model in COMPLETION_TOKENS_MODELS else "max_tokens"
    budget = MAX_TOKENS_LARGE if model in LARGE_OUTPUT_MODELS else MAX_TOKENS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(model=model, messages=messages, **{tokens_key: budget})
            if model not in FIXED_TEMP_MODELS:
                kwargs["temperature"] = get_temperature(model)
            kwargs["seed"] = seed
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] API error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",       required=True, choices=SUPPORTED_MODELS)
    ap.add_argument("--gold-json",   default=str(DEFAULT_GOLD))
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    ap.add_argument("--seeds",       type=int, nargs="+", default=[44, 45, 46])
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Limit to first N arguments (default: all 60)")
    ap.add_argument("--output",      required=True,
                    help="Output JSON path (raw_output_run2_20260525.json)")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Print prompt for first arg without calling the API")
    ap.add_argument("--delay",        type=float, default=0.3,
                    help="Seconds between API calls (default: 0.3)")
    ap.add_argument("--system-prefix", default=None,
                    help="Text to prepend to system prompt (e.g. academic framing)")
    ap.add_argument("--arg-ids",      type=int, nargs="+", default=None,
                    help="Only run these specific argument IDs (space-separated)")
    ap.add_argument("--retry-refused", action="store_true",
                    help="Re-run only args where all seeds produced <200 chars")
    args = ap.parse_args()

    # Load prompt
    system_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if args.system_prefix:
        system_prompt = args.system_prefix.strip() + "\n\n" + system_prompt
    print(f"Prompt: {args.prompt_file} ({len(system_prompt)} chars)")

    # Load gold arguments
    gold = json.loads(Path(args.gold_json).read_text(encoding="utf-8"))
    annotations = gold["annotations"]
    # BULLETPROOF alignment: sort by the CANONICAL argumentIndex (NOT argument_id, which
    # historically collided 1-60x2 and scrambled GPT's order). Then HARD-ABORT before any
    # paid API call if the written argument_index would not equal the gold argumentIndex.
    annotations.sort(key=lambda x: int(x["argumentIndex"]))
    if args.num_samples:
        annotations = annotations[:args.num_samples]
    n = len(annotations)
    for pos, ann in enumerate(annotations):
        if int(ann["argumentIndex"]) != pos:
            raise SystemExit(
                f"ALIGNMENT ABORT (no spend): position {pos} has argumentIndex "
                f"{ann['argumentIndex']} (expected {pos}). Gold not contiguous/ordered.")
    print(f"[alignment OK] argument_index == argumentIndex for all {n} args (0..{n-1})")
    print(f"Arguments: {n} | Seeds: {args.seeds} | Model: {args.model}")

    # Load existing output — always needed for merge / retry logic
    out_path = Path(args.output)
    existing: dict = {}
    if out_path.exists() and not args.dry_run:
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for s in prev.get("per_sample", []):
                existing[int(s["argument_id"])] = s
            print(f"Loaded existing — {len(existing)} samples on disk")
        except Exception:
            pass

    # Build set of arg_ids to (re-)run
    if args.retry_refused:
        refused = {
            aid for aid, s in existing.items()
            if all(len(v["generated_text"]) <= 200
                   for v in s.get("per_seed", {}).values())
        }
        target_ids = refused
        print(f"Retry-refused mode: {len(target_ids)} args to re-run → {sorted(target_ids)}")
    elif args.arg_ids:
        target_ids = set(args.arg_ids)
        print(f"Targeting specific arg_ids: {sorted(target_ids)}")
    else:
        target_ids = None  # run all

    client = None
    if not args.dry_run:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)

    per_sample = []
    for i, ann in enumerate(annotations, 1):
        arg_id   = int(ann.get("argument_id", ann.get("argumentIndex", i)))
        argument = ann["argument"]

        # Skip if not in targeted set
        if target_ids is not None and arg_id not in target_ids:
            if arg_id in existing:
                per_sample.append(existing[arg_id])
            continue

        # Resume: skip if all requested seeds already done (and not forcing retry)
        if arg_id in existing and target_ids is None:
            done_seeds = set(existing[arg_id].get("per_seed", {}).keys())
            needed = {str(s) for s in args.seeds}
            if needed.issubset(done_seeds):
                print(f"[{i:2}/{n}] arg_id={arg_id} — skipped (already complete)")
                per_sample.append(existing[arg_id])
                continue

        print(f"[{i:2}/{n}] arg_id={arg_id}: {argument[:80]}...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Argument:\n{argument}"},
        ]

        if args.dry_run:
            print("  SYSTEM (first 300 chars):", system_prompt[:300])
            print("  USER:", messages[1]["content"])
            break

        per_seed: dict = {}
        for seed in args.seeds:
            t0  = time.time()
            raw = call_api(client, args.model, messages, seed)
            elapsed = time.time() - t0
            print(f"  seed={seed} → {len(raw)} chars ({elapsed:.1f}s)")
            per_seed[str(seed)] = {"generated_text": raw}
            if args.delay > 0:
                time.sleep(args.delay)

        per_sample.append({
            "argument_index": i - 1,
            "argument_id":    arg_id,
            "argument":       argument,
            "per_seed":       per_seed,
        })

        # Save progress to the output file after each argument
        _save(args, per_sample, n, system_prompt, out_path)

    if not args.dry_run:
        _save(args, per_sample, n, system_prompt, out_path)
        print(f"\nDone. Saved → {out_path}")


def _save(args, per_sample, n, system_prompt, out_path):
    output = {
        "config": {
            "model":        args.model,
            "model_id":     args.model,
            "prompt_file":  args.prompt_file,
            "seeds":        args.seeds,
            "num_samples":  n,
            "max_new_tokens": MAX_TOKENS,
            "generated_at": datetime.now().isoformat(),
        },
        "per_sample": per_sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
