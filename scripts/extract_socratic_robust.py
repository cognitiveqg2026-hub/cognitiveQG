#!/usr/bin/env python3
"""
Robust Socratic-question extractor + socratic_format rebuilder.

Motivation: OSS models emit the 3 Socratic questions in MANY layout variants
(e.g. "1) question: Q / target / sourceField", "1. Q / Question target / Source field",
"Question 1: Q / Target / Source Field", inline "(target: ...; sourceField: ...)").
The template-strict parser in parse_model_output.py only recognises the exact
"question:/target:/sourceField:" labels, so it recovered questions for ~5/120 args.
The question TEXT is fully present in the raw output (outputs are ~600-1200 tokens,
well under the 1500 cap — this is a PARSE gap, not truncation).

This standalone post-processor re-reads each model's raw_output.json and rebuilds
  socratic_format.json  {"results":[{"arg_index":N,"questions":[q1,q2,q3]}]}
(the exact structure evaluate_llm_judge_fixed.py / the correlation figure read),
and optionally patches parsed.json's socraticQuestion1/2/3 for the field table.

Duplicate-don't-modify: does NOT touch parse_model_output.py or build_gold_vs_pred.py.
"""

import argparse
import json
import os
import re

# A line that starts a question block: optional "Question" word, a digit 1-3,
# then . ) or : , then the rest of the line.
_QSTART = re.compile(r'^\s*(?:question\s*)?([1-3])\s*[.):]\s*(.*)$', re.IGNORECASE)
# A metadata line (target / source field), any spacing/casing, optional "Question" prefix.
_META = re.compile(r'^\s*(?:question\s*)?(?:target|source\s*field|sourcefield)\s*[:]\s*',
                   re.IGNORECASE)
# Leading "question:" label to strip from the first line after the number.
_QLABEL = re.compile(r'^\s*question\s*[:]\s*', re.IGNORECASE)
# Trailing inline parenthetical metadata: "... (target: ...; sourceField: ...)"
_INLINE_META = re.compile(r'\(\s*target\s*:.*?\)\s*$', re.IGNORECASE)


def extract_questions_robust(text: str) -> list[str]:
    """Return up to 3 Socratic question strings from generated text."""
    if not text:
        return []
    # Isolate the Socratic section (last occurrence of 'socratic'); fall back to tail.
    low = text.lower()
    idx = low.rfind('socratic')
    seg = text[idx:] if idx >= 0 else text[-1400:]
    lines = seg.splitlines()

    # Find question-start lines.
    starts = []  # (line_index, first_line_remainder)
    for i, ln in enumerate(lines):
        m = _QSTART.match(ln)
        if m:
            # Skip lines that are actually metadata mislabeled with a number (rare).
            if _META.match(ln):
                continue
            starts.append((i, m.group(2)))
    if not starts:
        return []

    questions = []
    for k, (li, first_remainder) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        # First line: strip a leading "question:" label + trailing inline metadata.
        first = _QLABEL.sub('', first_remainder)
        first = _INLINE_META.sub('', first).strip()
        buff = [first] if first else []
        # Continuation lines until a metadata marker (target/source) appears.
        for ln in lines[li + 1:end]:
            if _META.match(ln):
                break
            s = ln.strip()
            if not s:
                # blank line: stop only if we already have some text
                if buff:
                    break
                continue
            # a stray "sourceField:"-only or bracket line
            buff.append(s)
        q = re.sub(r'\s+', ' ', ' '.join(buff)).strip()
        # Drop empties / pure metadata residue.
        if q and not _META.match(q):
            questions.append(q)
    return questions[:3]


def _raw_samples(raw: dict) -> list:
    return raw.get("per_sample", raw.get("results", raw.get("samples", [])))


def _seed_text(sample: dict, seed: str) -> str:
    per = sample.get("per_seed", {})
    if seed in per:
        return per[seed].get("generated_text", "") or ""
    # fall back to any available seed
    for k in sorted(per.keys()):
        return per[k].get("generated_text", "") or ""
    return sample.get("generated_text", "") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw_output.json")
    ap.add_argument("--out", required=True, help="socratic_format.json to (re)write")
    ap.add_argument("--seed", default="44", help="seed to use for judge socratic (matches build_gold_vs_pred --seed)")
    ap.add_argument("--patch-parsed", default=None,
                    help="parsed.json to patch socraticQuestion1/2/3 (uses each sample's best_seed)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    raw = json.load(open(args.raw))
    samples = _raw_samples(raw)

    results = []
    n_ge1 = n_ge3 = 0
    for s in samples:
        aidx = int(s.get("argument_index", s.get("arg_index", len(results))))
        qs = extract_questions_robust(_seed_text(s, args.seed))
        if len(qs) >= 1:
            n_ge1 += 1
        if len(qs) >= 3:
            n_ge3 += 1
        results.append({"arg_index": aidx, "questions": qs})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"results": results}, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}: {len(results)} args | >=1 q: {n_ge1}/{len(results)} | >=3 q: {n_ge3}/{len(results)} (seed {args.seed})")

    if args.patch_parsed and os.path.exists(args.patch_parsed):
        parsed = json.load(open(args.patch_parsed))
        psamps = parsed.get("samples", [])
        patched = 0
        for i, ps in enumerate(psamps):
            if i >= len(samples):
                break
            bseed = str(ps.get("best_seed", args.seed))
            qs = extract_questions_robust(_seed_text(samples[i], bseed))
            agg = ps.setdefault("aggregated", {})
            agg["socraticQuestion1"] = qs[0] if len(qs) > 0 else ""
            agg["socraticQuestion2"] = qs[1] if len(qs) > 1 else ""
            agg["socraticQuestion3"] = qs[2] if len(qs) > 2 else ""
            if qs:
                patched += 1
        json.dump(parsed, open(args.patch_parsed, "w"), ensure_ascii=False, indent=2)
        print(f"Patched {args.patch_parsed}: socraticQuestion1/2/3 non-empty for {patched}/{len(psamps)} samples")


if __name__ == "__main__":
    main()
