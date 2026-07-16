#!/usr/bin/env python3
"""
Build judge-compatible socratic_format JSON from a causation inference output file.

Causation outputs store raw generated_text in per_sample[].per_seed[seed].
This script extracts Socratic questions using a 5-pattern extractor that handles
all model output formats observed across llama3, mistral, olmo, and qwen3.

Special handling:
  - Qwen3 thinking mode: strips content before </think> tag before extraction
  - JSON output format: parses "SOCRATIC QUESTIONS" array from JSON outputs
  - Bold format: **question**: text?  (llama3 oracle/noise)
  - Bullet format: - text?  (Mistral)
  - Numbered format: 1. text?

Usage:
  python scripts/build_socratic_format_causation.py \\
      --input  results/causation/llama3_oracle_60args_2026-05-28.json \\
      --output results/causation_judge/llama3_oracle_60args.json

  python scripts/build_socratic_format_causation.py \\
      --input  results/causation/llama3_oracle_60args_2026-05-28.json \\
      --output results/causation_judge/llama3_oracle_smoke5.json \\
      --limit  5
"""

import argparse
import json
import re
from pathlib import Path


def extract_questions(text: str) -> list:
    """Extract up to 3 Socratic questions from generated text.

    Handles all formats observed in causation outputs:
      1. Qwen3 think-block stripping: content after </think>
      2. JSON with 'SOCRATIC QUESTIONS' array
      3. **question**: text?   (llama3 oracle/noise bold format)
      4. *question: text*
      5. question: text?       (llama3 plain format)
      6. 1. text?              (numbered)
      7. - text?               (Mistral bullet, including indented)
    """
    # Strip Qwen3 thinking block — content before </think> is internal reasoning
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]

    # JSON output format (Qwen3 sometimes outputs structured JSON)
    try:
        obj = json.loads(text.strip())
        sq = (obj.get("SOCRATIC QUESTIONS")
              or obj.get("socraticQuestions")
              or obj.get("socratic_questions"))
        if isinstance(sq, list):
            qs = []
            for item in sq:
                if isinstance(item, dict):
                    q = str(item.get("question", "") or "").strip()
                else:
                    q = str(item or "").strip()
                if "?" in q:
                    qs.append(q)
            if qs:
                return qs[:3]
    except Exception:
        pass

    patterns = [
        r"\*\*question\*\*:\s*(.+?)(?:\n|$)",   # **question**: text?
        r"\*question:\s*(.+?)\*",               # *question: text*
        r"question:\s*(.+?)(?:\n|$)",            # question: text?
        r"\d+\.\s+\*?([^*\n]+\?)\*?",           # 1. text?
        r"^\s*[\-•]\s+(.+\?)",                   # - text?  (bullet, Mistral)
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
        qs = [m.strip().strip("* ") for m in matches if "?" in m]
        if qs:
            return qs[:3]
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Convert causation inference output to judge-compatible socratic_format JSON"
    )
    ap.add_argument("--input",  required=True, help="Causation output JSON")
    ap.add_argument("--output", required=True, help="Output socratic_format JSON path")
    ap.add_argument("--seed",   default="44",  help="Seed key to read from per_seed (default: 44)")
    ap.add_argument("--best-of-seeds", action="store_true",
                    help="Per arg, use whichever available seed yields the most questions "
                         "(tie-break: 44, 42, 43, ...). Raises coverage for multi-seed runs.")
    ap.add_argument("--limit",  type=int, default=None, help="First N arguments only (smoke test)")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    samples = data.get("per_sample", [])

    if args.limit:
        samples = samples[:args.limit]

    # Pre-check for duplicate arg IDs (GPT outputs reset per batch); if found, use global position
    raw_ids = [sample.get("argument_id", sample.get("arg_index", -1)) for sample in samples]
    use_global_idx = len(set(raw_ids)) < len(raw_ids)
    if use_global_idx:
        print("  [info] duplicate arg_ids detected — using global list position as arg_index")

    results = []
    empty_q1 = 0
    for global_pos, sample in enumerate(samples):
        if use_global_idx:
            arg_id = global_pos + 1  # gold uses 1-based argument_id (1–60)
        else:
            arg_id = int(sample.get("argument_id", sample.get("arg_index", -1)))
        argument = sample.get("argument", "")
        per_seed = sample.get("per_seed", {})

        if args.best_of_seeds:
            # Per arg, pick the available seed whose extraction yields the most questions.
            order = ["44", "42", "43"] + [k for k in per_seed if k not in ("44", "42", "43")]
            qs = []
            for seed_key in order:
                sd = per_seed.get(seed_key)
                if not sd:
                    continue
                cand = [q for q in extract_questions(sd.get("generated_text", "")) if q and str(q).strip()]
                if len(cand) > len(qs):
                    qs = cand
        else:
            # Try requested seed, then fallback
            text = ""
            for seed_key in [args.seed, "44", "42", "43"]:
                sd = per_seed.get(seed_key)
                if sd:
                    text = sd.get("generated_text", "")
                    break
            qs = extract_questions(text)
        # Pad to exactly 3 slots
        while len(qs) < 3:
            qs.append("")

        results.append({
            "arg_index": arg_id,
            "argument":  argument,
            "questions": qs[:3],
        })
        if not qs[0]:
            empty_q1 += 1

    results.sort(key=lambda r: r["arg_index"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    n = len(results)
    has_q = n - empty_q1
    print(f"[{Path(args.input).name}]")
    print(f"  {has_q}/{n} args with ≥1 question  |  {empty_q1}/{n} empty Q1")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
