#!/usr/bin/env python3
"""
Standalone LLM-as-a-Judge evaluation script.

Runs GPT-based scoring on model-generated Socratic questions vs. a gold annotation
file, independently of the full text-similarity pipeline (BERT/ROUGE/METEOR).
Useful for:
  - Cost-controlled judge runs (no local model loading required)
  - Human-vs-human ceiling runs (pass --human-vs-human)
  - Quick iteration on prompts without re-running all metrics

Outputs:
  {prefix}_judge.json       — full scored pairs + aggregate stats
  {prefix}_judge_span.csv   — per-position judge scores table

Usage:
  # Baseline: LLaMA2 model vs annotator P
  python scripts/evaluate_llm_judge.py \\
      --model-output results/misc/direct_generation_llama2_2026-02-27.json \\
      --gold-file data/Dev_combined/a1_dev_combined_2026-02-26.json \\
      --model-name llama2 \\
      --output results/llm_judge/llama2_vs_p

  # Human vs human ceiling
  python scripts/evaluate_llm_judge.py \\
      --model-output data/Dev_combined/a2_dev_combined_2026-02-26.json \\
      --gold-file data/Dev_combined/a1_dev_combined_2026-02-26.json \\
      --human-vs-human \\
      --model-name human_k \\
      --output results/llm_judge/human_k_vs_p
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iaa_core.llm_judge import SocraticJudge, JUDGE_DIMENSIONS

QUESTION_POSITIONS = [1, 2, 3]

_JUDGE_CSV_FIELDNAMES = [
    'model', 'annotator', 'question_position',
    'coverage_valid', 'coverage_total',
    'Judge_ReasoningDepth',       'Judge_ReasoningDepth_std',
    'Judge_ContextualRelevance',  'Judge_ContextualRelevance_std',
    'Judge_CriticalThinking',     'Judge_CriticalThinking_std',
    'Judge_PedagogicalEquiv',     'Judge_PedagogicalEquiv_std',
    'Judge_SemanticEquiv',        'Judge_SemanticEquiv_std',
    'Judge_FocusAppropriateness', 'Judge_FocusAppropriateness_std',
    'Judge_Composite',            'Judge_Composite_std',
]

_DIM_TO_CSV = {
    'reasoning_depth':               ('Judge_ReasoningDepth',       'Judge_ReasoningDepth_std'),
    'contextual_relevance':          ('Judge_ContextualRelevance',  'Judge_ContextualRelevance_std'),
    'critical_thinking_stimulation': ('Judge_CriticalThinking',     'Judge_CriticalThinking_std'),
    'pedagogical_equivalence':       ('Judge_PedagogicalEquiv',     'Judge_PedagogicalEquiv_std'),
    'semantic_equivalence':          ('Judge_SemanticEquiv',        'Judge_SemanticEquiv_std'),
    'focus_appropriateness':         ('Judge_FocusAppropriateness', 'Judge_FocusAppropriateness_std'),
    'composite':                     ('Judge_Composite',             'Judge_Composite_std'),
}


def _fmt(v) -> str:
    if v is None:
        return ''
    try:
        return f'{float(v):.4f}'
    except (TypeError, ValueError):
        return ''


def _infer_annotator(path: str) -> str:
    base = os.path.basename(path).lower()
    if re.search(r'(^|[_/])a2[_/]|a2_dev', base):
        return 'k'
    if re.search(r'(^|[_/])a1[_/]|a1_dev', base):
        return 'p'
    return 'unknown'


def load_model_output(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', [])
    return sorted(results, key=lambda r: int(r['arg_index']))


def load_gold_annotations(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data.get('annotations', data.get('items', []))
    # Use argument_id (1-based, unique 1–60). argumentIndex cycles 0–4 and is NOT unique.
    return sorted(items, key=lambda x: int(x.get('argument_id', x.get('argumentIndex', 0))))


def load_gold_as_model_output(path: str) -> List[dict]:
    """Convert gold annotation format to inference output format."""
    items = load_gold_annotations(path)
    results = []
    for item in items:
        results.append({
            'arg_index': int(item.get('argument_id', item.get('argumentIndex', 0))),
            'argument':  item.get('argument', ''),
            'questions': [
                str(item.get('socraticQuestion1', '') or '').strip(),
                str(item.get('socraticQuestion2', '') or '').strip(),
                str(item.get('socraticQuestion3', '') or '').strip(),
            ],
        })
    return results


def _jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def align_pairs(model_results: List[dict], gold_annotations: List[dict]) -> List[dict]:
    gold_by_idx  = {int(item.get('argument_id', item.get('argumentIndex', 0))): item for item in gold_annotations}
    model_by_idx = {int(r['arg_index']): r       for r in model_results}
    aligned = []
    for arg_idx in sorted(gold_by_idx.keys()):
        gold_item     = gold_by_idx[arg_idx]
        model_item    = model_by_idx.get(arg_idx)
        model_questions = []
        if model_item:
            model_questions = [q.strip() for q in model_item.get('questions', [])]
        for pos in QUESTION_POSITIONS:
            zero_pos = pos - 1
            model_q  = model_questions[zero_pos] if zero_pos < len(model_questions) else ''
            gold_q   = str(gold_item.get(f'socraticQuestion{pos}', '') or '').strip()
            aligned.append({
                'arg_index':             arg_idx,
                'argument':              gold_item.get('argument', ''),
                'question_position':     pos,
                'matched_gold_position': pos,   # positional: same as question_position
                'model_question':        model_q,
                'gold_question':         gold_q,
                'model_missing':         model_q == '',
                'gold_missing':          gold_q  == '',
                'jaccard_similarity':    _jaccard(model_q, gold_q),
            })
    return aligned


def align_pairs_hungarian(
    model_results: List[dict],
    gold_annotations: List[dict],
) -> List[dict]:
    """
    Align model/K questions to gold/P questions using Hungarian optimal matching.
    Builds a 3x3 BERT cosine similarity matrix and finds the 1-to-1 assignment
    that maximises total semantic similarity. Falls back to Jaccard if the
    SentenceTransformer model is unavailable.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # Load BERT model for semantic matching
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading SentenceTransformer for BERT-based matching...")
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        def _sim(a, b):
            if not a.strip() or not b.strip():
                return 0.0
            emb = _bert_model.encode([a, b])
            return float(cos_sim([emb[0]], [emb[1]])[0][0])
        sim_label = "BERT cosine"
    except Exception:
        print("  SentenceTransformer unavailable — falling back to Jaccard matching")
        _sim = _jaccard
        sim_label = "Jaccard"

    gold_by_idx  = {int(item.get('argument_id', item.get('argumentIndex', 0))): item for item in gold_annotations}
    model_by_idx = {int(r['arg_index']): r for r in model_results}
    aligned = []

    for arg_idx in sorted(gold_by_idx.keys()):
        gold_item       = gold_by_idx[arg_idx]
        model_item      = model_by_idx.get(arg_idx)
        model_questions = []
        if model_item:
            model_questions = [q.strip() for q in model_item.get('questions', [])]

        # Pad to 3 questions
        model_qs = [model_questions[i] if i < len(model_questions) else '' for i in range(3)]
        gold_qs  = [str(gold_item.get(f'socraticQuestion{i+1}', '') or '').strip() for i in range(3)]

        # Build 3x3 BERT similarity matrix
        sim = np.array([[_sim(model_qs[i], gold_qs[j]) for j in range(3)] for i in range(3)])

        # Hungarian: maximise similarity = minimise negative similarity
        row_ind, col_ind = linear_sum_assignment(-sim)

        pos_total = sum(sim[i, i] for i in range(3))
        hun_total = sum(sim[r, c] for r, c in zip(row_ind, col_ind))
        if hun_total > pos_total + 0.01:
            print(f"  arg={arg_idx}: Hungarian improved matching "
                  f"({pos_total:.3f} → {hun_total:.3f} total {sim_label})", flush=True)

        for k_pos_zero, p_pos_zero in zip(row_ind, col_ind):
            model_q = model_qs[k_pos_zero]
            gold_q  = gold_qs[p_pos_zero]
            aligned.append({
                'arg_index':             arg_idx,
                'argument':              gold_item.get('argument', ''),
                'question_position':     int(k_pos_zero) + 1,   # K's question index (1-based)
                'matched_gold_position': int(p_pos_zero) + 1,   # P's matched question index (1-based)
                'model_question':        model_q,
                'gold_question':         gold_q,
                'model_missing':         model_q == '',
                'gold_missing':          gold_q  == '',
                'matching_similarity':   float(sim[k_pos_zero, p_pos_zero]),
            })

    return aligned


def write_judge_csv(judge_agg: dict, model: str, annotator: str, output_prefix: str):
    rows = []
    for pos in QUESTION_POSITIONS:
        pos_data = judge_agg.get('per_position', {}).get(pos, {})
        cov      = pos_data.get('coverage', {})
        row = {
            'model':              model,
            'annotator':          annotator,
            'question_position':  str(pos),
            'coverage_valid':     cov.get('valid_pairs', ''),
            'coverage_total':     cov.get('total_pairs', ''),
        }
        for dim, (mean_col, std_col) in _DIM_TO_CSV.items():
            row[mean_col] = _fmt(pos_data.get(dim, {}).get('mean'))
            row[std_col]  = _fmt(pos_data.get(dim, {}).get('std'))
        rows.append(row)

    overall  = judge_agg.get('overall', {})
    cov      = overall.get('coverage', {})
    row = {
        'model':              model,
        'annotator':          annotator,
        'question_position':  'overall',
        'coverage_valid':     cov.get('valid_pairs', ''),
        'coverage_total':     cov.get('total_pairs', ''),
    }
    for dim, (mean_col, std_col) in _DIM_TO_CSV.items():
        row[mean_col] = _fmt(overall.get(dim, {}).get('mean'))
        row[std_col]  = _fmt(overall.get(dim, {}).get('std'))
    rows.append(row)

    csv_path = f"{output_prefix}_judge_span.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_JUDGE_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Judge CSV saved:    {csv_path}")


def _print_summary(judge_agg: dict):
    overall = judge_agg.get('overall', {})
    print()
    print("RESULTS (overall)")
    print("-" * 50)
    for dim in JUDGE_DIMENSIONS:
        d = overall.get(dim, {})
        label = dim.replace('_', ' ').title()
        print(f"  {label:<38} {d.get('mean', 0):.4f}  ±{d.get('std', 0):.4f}")
    comp = overall.get('composite', {})
    print(f"  {'Composite':<38} {comp.get('mean', 0):.4f}  ±{comp.get('std', 0):.4f}")


def _save_outputs(scored_pairs: list, judge_agg: dict, output_prefix: str, metadata: dict):
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_data = {
        'metadata':        metadata,
        'judge_aggregate': judge_agg,
        'scored_pairs': [
            {k: v for k, v in p.items() if k not in ('llm_judge',)}
            | {'judge': p.get('judge', p.get('llm_judge', {}))}
            for p in scored_pairs
        ],
    }
    json_path = f"{output_prefix}_judge.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved:         {json_path}")
    model_name = metadata.get('model_name', 'unknown')
    annotator  = _infer_annotator(metadata.get('gold_file', ''))
    write_judge_csv(judge_agg, model_name, annotator, output_prefix)


def main():
    parser = argparse.ArgumentParser(
        description='LLM-as-a-Judge standalone evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model-output',   type=str, default=None,
                        help='Path to model output JSON (required unless --rescore)')
    parser.add_argument('--gold-file',       type=str, default=None,
                        help='Path to gold annotation JSON (required unless --rescore)')
    parser.add_argument('--model-name',      type=str, default=None)
    parser.add_argument('--output',          type=str, default=None)
    parser.add_argument('--human-vs-human',  action='store_true', default=False,
                        help='Treat --model-output as a gold annotation JSON')
    parser.add_argument('--judge-model',     type=str, default='gpt-5.5')
    parser.add_argument('--judge-mode',      type=str, default='scalar',
                        choices=['scalar', 'equivalence', 'pairwise'])
    parser.add_argument('--judge-cache-dir', type=str, default='.judge_cache')
    parser.add_argument('--judge-consistency-runs', type=int, default=1)
    parser.add_argument('--hungarian',        action='store_true', default=False,
                        help='Use Hungarian optimal matching (maximises Jaccard similarity) '
                             'instead of positional alignment. Recommended for --human-vs-human.')
    parser.add_argument('--rescore',         action='store_true', default=False,
                        help='Rescore an existing judge JSON with the current (updated) prompt. '
                             'Requires --judge-json.')
    parser.add_argument('--judge-json',      type=str, default=None,
                        help='Path to existing _judge.json for --rescore mode')
    parser.add_argument('--limit',           type=int, default=None,
                        help='Only score first N arguments (smoke test)')

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # RESCORE MODE
    # -----------------------------------------------------------------------
    if args.rescore:
        if not args.judge_json:
            print("Error: --rescore requires --judge-json")
            sys.exit(1)
        if not os.path.exists(args.judge_json):
            print(f"Error: File not found (--judge-json): {args.judge_json}")
            sys.exit(1)

        print("=" * 70)
        print("LLM-AS-A-JUDGE RESCORE (updated prompt v2)")
        print("=" * 70)
        print(f"Input JSON:    {args.judge_json}")
        print(f"Judge model:   {args.judge_model}")
        print(f"Cache dir:     {args.judge_cache_dir}")
        print()

        with open(args.judge_json, 'r', encoding='utf-8') as f:
            existing = json.load(f)

        old_pairs = existing.get('scored_pairs', [])
        for p in old_pairs:
            if 'llm_judge' in p and 'judge' not in p:
                p['judge'] = p['llm_judge']
        print(f"  Loaded {len(old_pairs)} pairs")

        from collections import Counter
        old_se = Counter(p.get('judge', {}).get('semantic_equivalence')
                         for p in old_pairs if not p.get('judge', {}).get('_skipped'))
        print(f"  Old SE distribution: {dict(sorted(old_se.items()))}")
        print()

        judge = SocraticJudge(
            model=args.judge_model,
            mode='scalar',
            cache_dir=args.judge_cache_dir,
            temperature=1.0,
        )
        rescored = judge.rescore_pairs(old_pairs)
        judge_agg = judge.aggregate_scores(rescored)

        new_se = Counter(p.get('judge', {}).get('semantic_equivalence')
                         for p in rescored if not p.get('judge', {}).get('_skipped'))
        print(f"\n  New SE distribution: {dict(sorted(new_se.items()))}")

        _print_summary(judge_agg)

        if args.output:
            output_prefix = args.output
        else:
            stem = os.path.splitext(args.judge_json)[0]
            output_prefix = stem.replace('_judge', '') + '_judge_v2'

        meta = dict(existing.get('metadata', {}))
        meta['rescored_at']    = datetime.now().isoformat()
        meta['rescore_model']  = args.judge_model
        meta['prompt_version'] = 'v2_strategy_agnostic'
        _save_outputs(rescored, judge_agg, output_prefix, meta)

        print()
        print("=" * 70)
        print("RESCORE COMPLETE")
        print("=" * 70)
        return

    # -----------------------------------------------------------------------
    # NORMAL MODE
    # -----------------------------------------------------------------------
    if not args.model_output or not args.gold_file:
        print("Error: --model-output and --gold-file are required (or use --rescore with --judge-json)")
        sys.exit(1)

    for path, label in [(args.model_output, '--model-output'), (args.gold_file, '--gold-file')]:
        if not os.path.exists(path):
            print(f"Error: File not found ({label}): {path}")
            sys.exit(1)

    model_name = args.model_name
    if not model_name:
        basename   = os.path.basename(args.model_output)
        model_name = re.sub(r'^direct_generation_|_\d{4}-\d{2}-\d{2}\.json$', '', basename)

    annotator = _infer_annotator(args.gold_file)

    print("=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)
    print(f"Model output:  {args.model_output}")
    if args.human_vs_human:
        print("Mode:          human-vs-human")
    if args.hungarian:
        print("Alignment:     Hungarian optimal matching (Jaccard-based)")
    print(f"Gold file:     {args.gold_file}")
    print(f"Judge model:   {args.judge_model}  mode={args.judge_mode}")
    print(f"Cache dir:     {args.judge_cache_dir}")
    print()

    if args.human_vs_human:
        model_results = load_gold_as_model_output(args.model_output)
    else:
        model_results = load_model_output(args.model_output)
    if args.limit:
        model_results = model_results[:args.limit]
    gold_items = load_gold_annotations(args.gold_file)
    print(f"  Model results:    {len(model_results)} arguments")
    print(f"  Gold annotations: {len(gold_items)} arguments")

    if args.hungarian:
        print("  Running Hungarian matching...")
        aligned_pairs = align_pairs_hungarian(model_results, gold_items)
    else:
        aligned_pairs = align_pairs(model_results, gold_items)
    print(f"  Aligned pairs:    {len(aligned_pairs)}")
    print()

    judge = SocraticJudge(
        model=args.judge_model,
        mode=args.judge_mode,
        cache_dir=args.judge_cache_dir,
        temperature=1.0,
        n_consistency_runs=args.judge_consistency_runs,
    )
    scored_pairs = judge.score_aligned_pairs(aligned_pairs)
    for p in scored_pairs:
        if 'llm_judge' in p and 'judge' not in p:
            p['judge'] = p['llm_judge']
    judge_agg = judge.aggregate_scores(scored_pairs)

    _print_summary(judge_agg)

    if args.output:
        output_prefix = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"results/llm_judge/{model_name}_vs_{annotator}_{ts}"

    meta = {
        'model_name':        model_name,
        'model_output_file': os.path.abspath(args.model_output),
        'gold_file':         os.path.abspath(args.gold_file),
        'human_vs_human':    args.human_vs_human,
        'use_hungarian':     args.hungarian,
        'judge_model':       args.judge_model,
        'judge_mode':        args.judge_mode,
        'evaluated_at':      datetime.now().isoformat(),
        'total_arguments':   len(gold_items),
        'prompt_version':    'v2_strategy_agnostic',
    }
    _save_outputs(scored_pairs, judge_agg, output_prefix, meta)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)
