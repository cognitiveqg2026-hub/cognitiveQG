#!/usr/bin/env python3
"""
Load pruned judge-score files and compute per-argument GPT-Sem.

GPT-Sem normalization: (s-1)/4, s in {1..5} -- the formula stated in the
manuscript's methods section and used in Table 6/21/9/20/34/35/43/Figure 2.
(The causation-section condition-means pipeline uses a different, s/5,
normalization elsewhere in the original repo -- out of scope here, since
none of Tables 5/6/9/20/21/34/35/43/Figure 2 go through that pipeline.)
"""
import json
from collections import defaultdict
from pathlib import Path


def load_judge_se(path: Path) -> dict:
    """Returns {arg_index: mean_(s-1)/4_across_question_positions}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    by_arg = defaultdict(list)
    for p in d["scored_pairs"]:
        if p.get("model_missing") or p.get("gold_missing"):
            continue
        se = p.get("semantic_equivalence")
        if se is None:
            continue
        by_arg[int(p["arg_index"])].append((float(se) - 1.0) / 4.0)
    return {idx: sum(v) / len(v) for idx, v in by_arg.items()}


def load_judge_se_by_position(path: Path) -> dict:
    """Returns {question_position: [normalized scores across all arguments]},
    for the per-slot aggregation used by the Hungarian-matching table."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    by_pos = defaultdict(list)
    for p in d["scored_pairs"]:
        if p.get("model_missing") or p.get("gold_missing"):
            continue
        se = p.get("semantic_equivalence")
        if se is None:
            continue
        by_pos[int(p["question_position"])].append((float(se) - 1.0) / 4.0)
    return dict(by_pos)
