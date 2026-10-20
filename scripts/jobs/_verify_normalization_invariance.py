#!/usr/bin/env python3
"""
One-off verification (not part of run.sh): imports the real
analyze_causation_judge_120_stats.py module, runs its actual
compute_all_condition_scores()/compute_contrasts_table() pipeline twice --
once with the shipped (s-1)/4 normalization, once with the old s/5 --
both with the new round(..., 10), and diffs every Wilcoxon-family stat.
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

spec = importlib.util.spec_from_file_location(
    "cstats", str(ROOT / "scripts/analyze_causation_judge_120_stats.py"))
cstats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cstats)


def run_with_norm(norm_fn):
    orig = cstats.per_arg_scores

    def patched(pairs):
        by = defaultdict(list)
        for pr in pairs or []:
            if pr.get("model_missing") or pr.get("gold_missing"):
                continue
            j = pr.get("judge")
            if not isinstance(j, dict):
                continue
            se = j.get("semantic_equivalence")
            if se is None:
                continue
            by[int(pr["arg_index"])].append(norm_fn(se))
        return {k: round(float(sum(v) / len(v)), 10) for k, v in by.items() if v}

    cstats.per_arg_scores = patched
    try:
        data = cstats.compute_all_condition_scores()
        rows = cstats.compute_contrasts_table(data, alpha=0.05, n_boot=10000,
                                               seed=42, low_cov_n=90)
    finally:
        cstats.per_arg_scores = orig
    return rows


def main():
    rows_new = run_with_norm(lambda s: (s - 1) / 4.0)   # shipped formula
    rows_old = run_with_norm(lambda s: s / 5.0)          # old formula, for comparison only

    new_by_key = {(r["contrast"], r["model"]): r for r in rows_new}
    old_by_key = {(r["contrast"], r["model"]): r for r in rows_old}
    assert set(new_by_key) == set(old_by_key)

    cols = ["wilcoxon_p", "wilcoxon_p_holm", "wilcoxon_p_holm_reject", "rank_biserial_r"]
    mismatches = []
    for key in new_by_key:
        if new_by_key[key].get("skipped"):
            continue
        a, b = new_by_key[key], old_by_key[key]
        for c in cols:
            av, bv = a[c], b[c]
            if isinstance(av, float):
                if abs(av - bv) > 1e-9:
                    mismatches.append((key, c, bv, av))
            elif av != bv:
                mismatches.append((key, c, bv, av))

    print(f"Compared {len(new_by_key)} rows x {len(cols)} cols "
          f"= {len(new_by_key) * len(cols)} comparisons")
    print(f"Mismatches (s/5-rounded vs (s-1)/4-rounded): {len(mismatches)}")
    for m in mismatches:
        print(" ", m)

    if not mismatches:
        print("\nCONFIRMED: with round(..., 10), every Wilcoxon p, Holm p, "
              "reject flag and r_rb is bit-identical between the two scales.")


if __name__ == "__main__":
    main()
