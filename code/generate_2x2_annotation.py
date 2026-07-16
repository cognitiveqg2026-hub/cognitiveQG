"""
Generate 2x2 error analysis annotation data for dev1-6, seed 44, all 8 models.

2x2 framework:
  Rows: trace_jac >= 0.240 (GROUNDED / GOOD TRACE)
  Cols: GPT-Sem norm >= 0.339 (ALIGNED / GOOD QUESTION)
  Case 2: Grounded + Aligned   (GG) — both correct
  Case 3: Grounded + Misaligned (GB) — good trace, bad question   ← focus
  Case 4: Ungrounded + Aligned  (BG) — bad trace, good question   ← focus
  Case 1: Ungrounded + Misaligned (BB) — both fail

Output: cognitiveQG/data/2x2_annotation_seed44.json
"""

import json
import csv
import os
import sys
import time
from collections import defaultdict
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS     = os.path.join(BASE, "results", "baseline_complete_dev1-12")
OUT_PATH    = os.path.join(BASE, "data", "2x2_annotation_seed44.json")
CHECKPOINT  = os.path.join(BASE, "data", "2x2_annotation_seed44_checkpoint.json")

MODELS = ["llama2", "llama3", "qwen", "qwen3", "mistral", "olmo", "gpt-4o", "gpt-5.5"]
CORRELATED_MODEL = "llama2"
DEV16_MAX = 59          # arg_index 0–59 = dev1-6
SEED = "44"
THRESH_TRACE_JAC = 0.240
THRESH_SE_NORM   = 0.339
ANNOT_MODEL = "gpt-4o"  # annotation model

# ── Phase field grouping (Self-Regulation excluded) ────────────────────────
PHASE_FIELDS = {
    "Interpretation": [
        "initialUnderstandingTarget", "stanceTarget", "knowledgeDomain", "targetExplanation",
    ],
    "Analysis": [
        "coreClaim", "minorClaim", "premise", "paraphrasingUnderstanding",
        "reasoningStructure", "reasoningType",
        "span1", "span2", "span3",
        "deductiveTermX", "deductiveTermY", "inductiveTermX", "inductiveTermY",
        "hasAssumption", "missingComponent",
    ],
    "Inference": [
        "positiveConsequences", "negativeConsequences",
        "goodEvidence", "badEvidence",
        "primaryDomain", "alternativeType", "alternativeKeywords",
    ],
    "Evaluation": [
        "inferenceStrength", "credibilityFactors",
        "logicalFallacy", "trustworthiness", "trustExplanation",
    ],
    "Explanation": [
        "fallacySpan1", "fallacySpan2", "noFallacyExplanation",
    ],
}

# ── Load trace scores ───────────────────────────────────────────────────────
def load_trace_jac():
    tj = defaultdict(dict)
    with open(os.path.join(RESULTS, "reasoning_trace_scores.csv")) as f:
        for row in csv.DictReader(f):
            idx = int(row["arg_index"])
            if idx <= DEV16_MAX:
                tj[row["model"]][idx] = float(row["trace_jac"])
    return tj

# ── Load SE scores per (model, arg_index, question_position) ──────────────
def load_se_scores():
    """Returns: se[model][arg_index][question_position] = (raw_se, norm_se)"""
    se = defaultdict(lambda: defaultdict(dict))
    for m in MODELS:
        p = os.path.join(RESULTS, m, "judge_run2_judge.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for pair in d.get("scored_pairs", []):
            idx = int(pair["arg_index"])
            if idx > DEV16_MAX:
                continue
            pos = int(pair["question_position"])
            if pair.get("model_missing"):
                se[m][idx][pos] = None
                continue
            j = pair.get("judge", {})
            raw = j.get("semantic_equivalence")
            if raw is None or j.get("_skipped"):
                se[m][idx][pos] = None
                continue
            se[m][idx][pos] = (int(raw), (int(raw) - 1) / 4.0)
    return se

# ── Load gold questions per (model, arg_index, question_position) ──────────
def load_gold_questions():
    """Returns: gq[model][arg_index][question_position] = gold_question_str"""
    gq = defaultdict(lambda: defaultdict(dict))
    for m in MODELS:
        p = os.path.join(RESULTS, m, "judge_run2_judge.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for pair in d.get("scored_pairs", []):
            idx = int(pair["arg_index"])
            if idx > DEV16_MAX:
                continue
            pos = int(pair["question_position"])
            gq[m][idx][pos] = pair.get("gold_question", "")
    return gq

# ── Load model questions from parsed.json ──────────────────────────────────
def load_model_questions_and_fields():
    """Returns: pf[model][arg_index] = {fields: {phase: {...}}, questions: {1: q, 2: q, 3: q}}"""
    pf = {}
    for m in MODELS:
        p = os.path.join(RESULTS, "parsed", m, "parsed.json")
        if not os.path.exists(p):
            print(f"  WARNING: no parsed.json for {m}")
            continue
        d = json.load(open(p))
        pf[m] = {}
        for i, sample in enumerate(d["samples"]):
            if i > DEV16_MAX:
                continue
            seed_data = sample.get("per_seed", {}).get(SEED, {})
            if not seed_data:
                continue
            parsed = seed_data.get("parsed", {})
            # Organize by phase
            phase_data = {}
            for phase, fields in PHASE_FIELDS.items():
                phase_data[phase] = {f: parsed.get(f, "") for f in fields}
            # Extract questions
            qs = {
                1: parsed.get("socraticQuestion1", ""),
                2: parsed.get("socraticQuestion2", ""),
                3: parsed.get("socraticQuestion3", ""),
            }
            pf[m][i] = {
                "trace_fields": phase_data,
                "questions": qs,
                "argument": sample.get("argument", ""),
            }
    return pf

# ── Load gold fields from annotation_review.json ──────────────────────────
def load_gold_fields():
    """Returns: gf[arg_index] = {fieldName: {gold: val, model: val, ...}}"""
    gf = {}
    with open(os.path.join(RESULTS, "annotation_review.json")) as f:
        entries = json.load(f)
    for e in entries:
        idx = int(e["arg_index"])
        if idx <= DEV16_MAX:
            # Extract gold values only from the fields dict
            gold = {}
            for fname, fdata in e["fields"].items():
                if isinstance(fdata, dict) and "gold" in fdata:
                    gold[fname] = fdata["gold"]
            gf[idx] = {"gold": gold, "argument": e["argument"]}
    return gf

# ── Build annotation prompt ─────────────────────────────────────────────────
ANNOTATION_SYSTEM = """You are a trace-question consistency annotator.
Your task: Given a model's reasoning trace (organized by analysis phase) and one Socratic question,
determine if the question is CONSISTENT or INCONSISTENT with what the trace identified.

CONSISTENT = The question probes the weakness target (fallacy, missing premise, reasoning gap,
questionable assumption) that the trace fields point to. The question's focus is derivable from
the trace content. Different wording is fine as long as the logical target matches.

INCONSISTENT = The question's focus cannot be derived from the trace. This includes:
- Trace diagnosed a fallacy but question targets a completely different weakness
- Trace said no-fallacy (logicalFallacy="none") but question asks about a fallacy
- Key trace fields (fallacySpan, missingComponent, premise) are empty/generic, so question has no trace anchor
- Question is a generic meta-probe ("Are there logical flaws?") not grounded in specific trace content
- Trace misidentified the core claim, and question targets a component the trace didn't identify

When INCONSISTENT, identify the primary breakdown_field (the field that should have grounded
the question but did not): one of logicalFallacy, fallacySpan1, fallacySpan2, missingComponent,
coreClaim, premise, hasAssumption, or another specific field name.

Return ONLY valid JSON: {"label": "CONSISTENT" or "INCONSISTENT", "breakdown_field": field_name_or_null, "rationale": "1-2 concise sentences"}"""

def build_prompt(argument, trace_fields, question):
    lines = [f"ARGUMENT:\n{argument}\n\nMODEL TRACE (seed 44, all phases except Self-Regulation):"]
    for phase, fields in trace_fields.items():
        lines.append(f"\n[{phase}]")
        for fname, val in fields.items():
            if val and val not in ("", "[]", "[Not available.]", "[None detected.]", "[No fallacy present.]"):
                if isinstance(val, list):
                    lines.append(f"  {fname}: {', '.join(str(v) for v in val)}")
                else:
                    lines.append(f"  {fname}: {val}")
    lines.append(f"\nGENERATED QUESTION:\n{question}")
    lines.append("\nIs this question CONSISTENT or INCONSISTENT with the trace?")
    return "\n".join(lines)

# ── GPT annotation ──────────────────────────────────────────────────────────
client = OpenAI()

def annotate(argument, trace_fields, question, retries=3):
    prompt = build_prompt(argument, trace_fields, question)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=ANNOT_MODEL,
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.0,
            )
            result = json.loads(resp.choices[0].message.content)
            label = result.get("label", "").upper()
            if label not in ("CONSISTENT", "INCONSISTENT"):
                label = "INCONSISTENT"
            breakdown = result.get("breakdown_field") if label == "INCONSISTENT" else None
            rationale = result.get("rationale", "")
            return label, breakdown, rationale
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return "INCONSISTENT", "api_error", "Annotation failed after retries."

# ── Compute mean SE per (model, arg_index) ─────────────────────────────────
def compute_mean_se(se_scores, model, arg_idx):
    by_pos = se_scores.get(model, {}).get(arg_idx, {})
    vals = [v[1] for v in by_pos.values() if v is not None]
    return sum(vals) / len(vals) if vals else None

# ── Case assignment ─────────────────────────────────────────────────────────
def assign_case(is_grounded, is_aligned):
    if is_grounded and is_aligned:
        return 2, "Both Correct (GG)"
    if is_grounded and not is_aligned:
        return 3, "Good Trace, Bad Question (GB)"
    if not is_grounded and is_aligned:
        return 4, "Bad Trace, Good Question (BG)"
    return 1, "Both Fail (BB)"

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)

    print("Loading data...")
    tj      = load_trace_jac()
    se      = load_se_scores()
    gq      = load_gold_questions()
    pf      = load_model_questions_and_fields()
    gf      = load_gold_fields()

    # Load checkpoint if exists
    checkpoint_data = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            checkpoint_data = json.load(f)
        print(f"  Resuming from checkpoint ({len(checkpoint_data)} annotations cached)")

    output = {
        "meta": {
            "threshold_trace_jac": THRESH_TRACE_JAC,
            "threshold_gpt_sem_norm": THRESH_SE_NORM,
            "seed": int(SEED),
            "n_args": DEV16_MAX + 1,
            "models": MODELS,
            "annotation_model": ANNOT_MODEL,
        },
        "arguments": []
    }

    total_calls = 0
    total_skipped = 0

    for arg_idx in range(DEV16_MAX + 1):
        if arg_idx % 10 == 0:
            print(f"  Processing arg {arg_idx}/{DEV16_MAX}...")

        gold_entry = gf.get(arg_idx, {})
        argument_text = gold_entry.get("argument", pf.get("llama2", {}).get(arg_idx, {}).get("argument", ""))

        arg_record = {
            "arg_index": arg_idx,
            "argument": argument_text,
            "gold_fields": gold_entry.get("gold", {}),
            "models": {}
        }

        for model in MODELS:
            trace_info = pf.get(model, {}).get(arg_idx)
            if trace_info is None:
                continue

            trace_jac_val = tj.get(model, {}).get(arg_idx)
            if trace_jac_val is None:
                continue

            mean_se = compute_mean_se(se, model, arg_idx)
            is_grounded = trace_jac_val >= THRESH_TRACE_JAC
            is_aligned  = (mean_se is not None) and (mean_se >= THRESH_SE_NORM)
            case_num, case_label = assign_case(is_grounded, is_aligned)

            question_records = []
            for pos in [1, 2, 3]:
                model_q = trace_info["questions"].get(pos, "")
                gold_q  = gq.get(model, {}).get(arg_idx, {}).get(pos, "")
                se_info = se.get(model, {}).get(arg_idx, {}).get(pos)

                if not model_q:
                    question_records.append({
                        "position": pos,
                        "model_question": "",
                        "gold_question": gold_q,
                        "model_missing": True,
                        "gpt_sem_raw": None,
                        "gpt_sem_norm": None,
                        "is_aligned": None,
                        "consistency_label": None,
                        "breakdown_field": None,
                        "rationale": None,
                    })
                    total_skipped += 1
                    continue

                if se_info is not None:
                    raw_se, norm_se = se_info
                    q_aligned = norm_se >= THRESH_SE_NORM
                else:
                    raw_se, norm_se = None, None
                    q_aligned = None

                # Check checkpoint
                ck_key = f"{model}_{arg_idx}_{pos}"
                if ck_key in checkpoint_data:
                    label, breakdown, rationale = checkpoint_data[ck_key]
                else:
                    label, breakdown, rationale = annotate(
                        argument_text, trace_info["trace_fields"], model_q
                    )
                    checkpoint_data[ck_key] = (label, breakdown, rationale)
                    total_calls += 1
                    # Save checkpoint every 50 calls
                    if total_calls % 50 == 0:
                        with open(CHECKPOINT, "w") as f:
                            json.dump(checkpoint_data, f)
                        print(f"    Checkpoint saved ({total_calls} API calls so far)")

                question_records.append({
                    "position": pos,
                    "model_question": model_q,
                    "gold_question": gold_q,
                    "model_missing": False,
                    "gpt_sem_raw": raw_se,
                    "gpt_sem_norm": round(norm_se, 4) if norm_se is not None else None,
                    "is_aligned": q_aligned,
                    "consistency_label": label,
                    "breakdown_field": breakdown,
                    "rationale": rationale,
                })

            arg_record["models"][model] = {
                "is_correlated_model": model == CORRELATED_MODEL,
                "trace_jac": round(trace_jac_val, 4),
                "mean_se_norm": round(mean_se, 4) if mean_se is not None else None,
                "is_grounded": is_grounded,
                "is_aligned": is_aligned,
                "case": case_num,
                "case_label": case_label,
                "trace_fields": trace_info["trace_fields"],
                "questions": question_records,
            }

        output["arguments"].append(arg_record)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Remove checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\nDone. {total_calls} API calls, {total_skipped} skipped (model_missing).")
    print(f"Output: {OUT_PATH}")

    # Summary stats
    case_counts = defaultdict(int)
    consistency_counts = defaultdict(lambda: defaultdict(int))
    for arg_rec in output["arguments"]:
        for m, mrec in arg_rec["models"].items():
            case_counts[mrec["case"]] += 1
            for q in mrec["questions"]:
                if q.get("consistency_label"):
                    consistency_counts[m][q["consistency_label"]] += 1

    print("\n── 2×2 Case Distribution (across all models, arg level) ──")
    total = sum(case_counts.values())
    for case_num in sorted(case_counts):
        labels = {1:"BB Both Fail", 2:"GG Both Correct", 3:"GB Good Trace Bad Q", 4:"BG Bad Trace Good Q"}
        print(f"  Case {case_num} ({labels[case_num]}): {case_counts[case_num]} / {total} = {100*case_counts[case_num]/total:.1f}%")

    print("\n── Consistency per model ──")
    for m in MODELS:
        c = consistency_counts[m]
        tot = c["CONSISTENT"] + c["INCONSISTENT"]
        pct = 100 * c["CONSISTENT"] / tot if tot > 0 else 0
        print(f"  {m:12s}: CONSISTENT={c['CONSISTENT']}, INCONSISTENT={c['INCONSISTENT']}, consistent%={pct:.1f}%")


if __name__ == "__main__":
    main()
