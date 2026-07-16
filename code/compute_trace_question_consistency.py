#!/usr/bin/env python3
"""
compute_trace_question_consistency.py

Within-model trace-question consistency metric for RQ2.

For each GPT arg in the trace condition, scores whether the generated
question targets the same reasoning element (coreClaim, missingComponent,
logicalFallacy) identified in the model's own pipeline_output.

Two-stage scoring:
  Stage 1: BERT cosine similarity (question vs model's own field value)
  Stage 2: GPT-4o judge — FOLLOWS / BYPASSES / CONTRADICTS

Outputs:
  results/rq2/trace_question_consistency_{model}.json  — per-arg detail
  results/rq2/trace_question_consistency_summary.csv   — summary table
  results/figures/trace_question_consistency_2026-05-19.png

Usage:
  python scripts/compute_trace_question_consistency.py --model gpt4o
  python scripts/compute_trace_question_consistency.py --model gpt55
  python scripts/compute_trace_question_consistency.py --model gpt4o --skip-judge
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE    = Path(__file__).resolve().parent.parent
GOLD    = BASE / "data/Dev_combined/a1_dev_combined_2026-02-26.json"
OUT_DIR = BASE / "results/rq2"
FIG_DIR = BASE / "results/figures"
CACHE   = BASE / ".judge_cache_tq_consistency"

MODEL_CONFIGS = {
    "gpt4o": {
        "label":      "GPT-4o",
        "trace_file": "results/causation/gpt_gpt-4o_trace_20args_2026-05-19.json",
        "judge_model": "gpt-4o",
    },
    "gpt55": {
        "label":      "GPT-5.5",
        "trace_file": "results/causation/gpt_gpt-5-5_trace_20args_2026-05-19.json",
        "judge_model": "gpt-4o",   # always use gpt-4o as judge for comparability
    },
}

# Only args that were in the causation experiment (had human focus annotation)
TARGET_ARGS = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18, 19]

LABELS = ["FOLLOWS", "BYPASSES", "CONTRADICTS"]

SYSTEM_PROMPT = """\
You are evaluating trace-question consistency for a study on Socratic question generation.

You will be given:
1. A model's intermediate reasoning trace (what the model identified in the argument)
2. A generated Socratic question produced after that trace

Your task is to classify whether the generated question targets the same reasoning
element that was identified in the trace.

Classify as exactly ONE of:
  FOLLOWS     — the question directly probes the claim, gap, or fallacy the trace identified
  BYPASSES    — the question ignores the trace content and addresses a surface feature of
                the argument not captured in the trace fields
  CONTRADICTS — the question targets a different reasoning element than what the trace identified

Output valid JSON only:
{"label": "FOLLOWS|BYPASSES|CONTRADICTS", "reason": "<one sentence>"}"""

USER_TEMPLATE = """\
Argument: {argument}

Model's reasoning trace:
  Core claim identified:       {core_claim}
  Missing component identified: {missing_component}
  Logical fallacy identified:  {logical_fallacy}
  Has assumption:              {has_assumption}

Generated question: "{question}"

Classify whether the question targets the same reasoning element as the trace."""


def _cache_key(parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:32]


def _call_judge(argument, core_claim, missing_component, logical_fallacy,
                has_assumption, question, judge_model, cache_dir):
    """Call GPT-4o to classify trace-question consistency. Returns (label, reason)."""
    key = _cache_key([argument, core_claim, missing_component, logical_fallacy,
                      has_assumption, question, judge_model])
    cache_file = cache_dir / f"{key}.json"

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return cached["label"], cached["reason"]

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    user_msg = USER_TEMPLATE.format(
        argument=argument[:400],
        core_claim=core_claim or "(not identified)",
        missing_component=missing_component or "(none)",
        logical_fallacy=logical_fallacy or "(none)",
        has_assumption=has_assumption or "(unknown)",
        question=question,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=128,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(raw.strip())
            label  = parsed.get("label", "BYPASSES").upper()
            reason = parsed.get("reason", "")
            if label not in LABELS:
                label = "BYPASSES"
            cache_file.write_text(json.dumps({"label": label, "reason": reason}))
            return label, reason
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  Judge failed: {e}")
                return "BYPASSES", "judge error"


def flatten_pipeline(po):
    out = {}
    for phase_val in po.values():
        if isinstance(phase_val, dict):
            out.update(phase_val)
    return out


def compute_bert_sims(model_str, question, gold_str, encoder):
    """Cosine similarity between question and each string."""
    vecs = encoder.encode([question, model_str or "", gold_str or ""],
                          normalize_embeddings=True)
    sim_model = float(np.dot(vecs[0], vecs[1]))
    sim_gold  = float(np.dot(vecs[0], vecs[2]))
    return sim_model, sim_gold


def run_model(model_key, skip_judge, encoder, gold_lookup):
    cfg   = MODEL_CONFIGS[model_key]
    data  = json.load(open(BASE / cfg["trace_file"]))
    CACHE.mkdir(parents=True, exist_ok=True)

    results = []

    for sample in data["per_sample"]:
        arg_idx = sample["argument_index"]
        if arg_idx not in TARGET_ARGS:
            continue

        argument = sample["argument"]
        gold     = gold_lookup.get(arg_idx, {})

        seed_data = sample["per_seed"].get("42", sample["per_seed"].get(42, {}))
        po = seed_data.get("pipeline_output", {})
        questions = seed_data.get("questions", [])

        if not po or not questions:
            continue

        flat = flatten_pipeline(po)

        # Model's own trace fields
        model_core_claim    = str(flat.get("coreClaim", ""))
        model_missing       = str(flat.get("missingComponent", ""))
        model_fallacy       = str(flat.get("logicalFallacy", ""))
        model_assumption    = str(flat.get("hasAssumption", ""))

        # Normalize fallacy
        if model_fallacy.lower() in ("no logical fallacy", "none", "no fallacy"):
            model_fallacy_display = "(none)"
        else:
            model_fallacy_display = model_fallacy

        # Gold fields
        gold_core_claim  = str(gold.get("coreClaim", ""))
        gold_missing     = str(gold.get("missingComponent", ""))
        gold_fallacy     = str(gold.get("logicalFallacy", ""))

        q_results = []
        for q in questions[:3]:
            # Stage 1: BERT similarity
            sim_model_cc, sim_gold_cc = compute_bert_sims(
                model_core_claim, q, gold_core_claim, encoder)

            # Pick primary trace field: prefer missingComponent if non-empty, else coreClaim
            if model_missing and model_missing not in ("", "(none)"):
                primary_field = model_missing
                primary_name  = "missingComponent"
            elif model_fallacy_display != "(none)":
                primary_field = model_fallacy
                primary_name  = "logicalFallacy"
            else:
                primary_field = model_core_claim
                primary_name  = "coreClaim"

            sim_model_primary, _ = compute_bert_sims(primary_field, q, "", encoder)

            # Stage 2: LLM judge
            if skip_judge:
                label  = "BYPASSES"   # placeholder
                reason = "skipped"
            else:
                label, reason = _call_judge(
                    argument, model_core_claim, model_missing,
                    model_fallacy_display, model_assumption, q,
                    cfg["judge_model"], CACHE,
                )
                print(f"  Arg{arg_idx} Q: '{q[:60]}...' → {label}")

            q_results.append({
                "question":           q,
                "label":              label,
                "reason":             reason,
                "primary_field_name": primary_name,
                "primary_field_val":  primary_field[:80],
                "sim_model_primary":  round(sim_model_primary, 4),
                "sim_model_coreClaim":round(sim_model_cc, 4),
                "sim_gold_coreClaim": round(sim_gold_cc, 4),
            })

        # Majority label across 3 questions
        label_counts = Counter(r["label"] for r in q_results)
        majority_label = label_counts.most_common(1)[0][0]
        consistency_score = label_counts.get("FOLLOWS", 0) / max(len(q_results), 1)

        results.append({
            "arg_index":        arg_idx,
            "argument":         argument[:120],
            "model_coreClaim":  model_core_claim[:80],
            "model_missing":    model_missing[:80],
            "model_fallacy":    model_fallacy_display[:60],
            "gold_coreClaim":   gold_core_claim[:80],
            "majority_label":   majority_label,
            "consistency_score":round(consistency_score, 4),
            "per_question":     q_results,
        })

    return results


def save_outputs(all_results):
    """Save JSON per model and summary CSV."""
    for model_key, results in all_results.items():
        out_json = OUT_DIR / f"trace_question_consistency_{model_key}.json"
        out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Saved: {out_json}")

    # Summary CSV
    csv_path = OUT_DIR / "trace_question_consistency_summary.csv"
    fieldnames = ["arg_index", "model", "majority_label", "consistency_score",
                  "model_coreClaim", "gold_coreClaim", "model_missing", "model_fallacy"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model_key, results in all_results.items():
            for r in results:
                w.writerow({
                    "arg_index":         r["arg_index"],
                    "model":             MODEL_CONFIGS[model_key]["label"],
                    "majority_label":    r["majority_label"],
                    "consistency_score": r["consistency_score"],
                    "model_coreClaim":   r["model_coreClaim"],
                    "gold_coreClaim":    r["gold_coreClaim"],
                    "model_missing":     r["model_missing"],
                    "model_fallacy":     r["model_fallacy"],
                })
    print(f"Saved: {csv_path}")


def plot_figure(all_results):
    """Two-panel figure: stacked bar + consistency_score vs model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    colors = {"FOLLOWS": "#2ca25f", "BYPASSES": "#fc8d59", "CONTRADICTS": "#d73027"}

    # Panel A: stacked bar chart
    ax = axes[0]
    model_keys = list(all_results.keys())
    labels_display = [MODEL_CONFIGS[k]["label"] for k in model_keys]

    bottoms = np.zeros(len(model_keys))
    for lbl in LABELS:
        vals = []
        for model_key, results in all_results.items():
            counts = Counter(r["majority_label"] for r in results)
            total  = max(len(results), 1)
            vals.append(counts.get(lbl, 0) / total * 100)
        bars = ax.bar(labels_display, vals, bottom=bottoms, color=colors[lbl],
                      label=lbl, width=0.5, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if v > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottoms += np.array(vals)

    ax.set_ylabel("% of arguments", fontsize=10)
    ax.set_title("Trace-Question Coupling\n(majority label per argument)", fontsize=10, pad=6)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.06, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # Panel B: consistency_score per arg per model (scatter / box)
    ax2 = axes[1]
    positions = np.arange(len(model_keys))
    for i, (model_key, results) in enumerate(all_results.items()):
        scores = [r["consistency_score"] for r in results]
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(scores))
        ax2.scatter([i + j for j in jitter], scores, s=45, alpha=0.75,
                    color=list(colors.values())[i % 3], edgecolors="white",
                    linewidths=0.5, zorder=3)
        ax2.plot([i - 0.18, i + 0.18], [np.mean(scores)] * 2,
                 lw=2.5, color="black", zorder=4)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_display, fontsize=10)
    ax2.set_ylabel("Consistency score\n(fraction of Qs labelled FOLLOWS)", fontsize=9.5)
    ax2.set_title("Within-model consistency score per argument", fontsize=10, pad=6)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(0.5, color="#cccccc", lw=0.8, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(-0.14, 1.06, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    fig.suptitle(
        "RQ2: Trace-Question Consistency — Do models follow their own reasoning traces?\n"
        "FOLLOWS = question targets same element as trace · BYPASSES = surface heuristic · "
        "CONTRADICTS = different element",
        fontsize=10, fontweight="bold", y=1.01,
    )

    out = FIG_DIR / "trace_question_consistency_2026-05-19.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge calls (BERT stage only, labels will be placeholders)")
    args = parser.parse_args()

    # Load BERT encoder
    print("Loading SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load gold annotations
    gold_data = json.load(open(GOLD))
    gold_lookup = {}
    for entry in gold_data["annotations"]:
        idx = entry["argumentIndex"]
        gold_lookup[idx] = {
            "coreClaim":       entry.get("coreClaim", ""),
            "missingComponent":entry.get("missingComponent", ""),
            "logicalFallacy":  entry.get("logicalFallacy", ""),
        }

    model_keys = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_key in model_keys:
        print(f"\n=== Processing {MODEL_CONFIGS[model_key]['label']} ===")
        results = run_model(model_key, args.skip_judge, encoder, gold_lookup)
        all_results[model_key] = results

        # Print per-arg summary
        print(f"\n{'Arg':>4} | {'Label':>11} | {'Score':>5} | Model coreClaim (first 50)")
        print("-" * 75)
        for r in sorted(results, key=lambda x: x["arg_index"]):
            print(f"{r['arg_index']:>4} | {r['majority_label']:>11} | "
                  f"{r['consistency_score']:>5.2f} | {r['model_coreClaim'][:50]}")

    # Load existing all_results if running only one model
    if args.model != "all":
        for other_key in MODEL_CONFIGS:
            if other_key not in all_results:
                other_json = OUT_DIR / f"trace_question_consistency_{other_key}.json"
                if other_json.exists():
                    all_results[other_key] = json.load(open(other_json))

    save_outputs({k: v for k, v in all_results.items()
                  if k in MODEL_CONFIGS and v})
    plot_figure({k: v for k, v in all_results.items()
                 if k in MODEL_CONFIGS and v})

    # Print aggregate stats
    print("\n=== Summary ===")
    for model_key, results in all_results.items():
        if not results:
            continue
        counts = Counter(r["majority_label"] for r in results)
        n = len(results)
        label = MODEL_CONFIGS[model_key]["label"]
        print(f"{label} (n={n}):")
        for lbl in LABELS:
            print(f"  {lbl}: {counts.get(lbl,0)}/{n} = {counts.get(lbl,0)/n*100:.0f}%")
        mean_cs = np.mean([r["consistency_score"] for r in results])
        print(f"  Mean consistency score: {mean_cs:.3f}")


if __name__ == "__main__":
    main()
