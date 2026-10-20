#!/usr/bin/env python3
"""
Table 34 rank agreement for S2 (dev3-12, A2 gold, n=100) and S3
(dev3-12, A1 gold, n=100), independently recomputed in this repo using
the bundled Combined Trace Score (src/score_v41/src/score_trace.py).

S3 is a straight filter of annotation_review.json, excluding the dev1-2
calibration indices {35..54} (confirmed: generate_baseline_field_table_dev3-12_a2.py
uses the identical exclusion set for the A1 variant).

S2 reuses scripts/jobs/_weighting_sensitivity_canonical.py's
load_a2_dev3_12_review() (unmodified) -- this rebuilds per-seed reviews
from A2 gold + pooled raw predictions for the original 39-field scheme,
already patched with deductive/inductive Term X/Y. This script additionally
patches in revisionPhases and revisionReason (real A2 gold values, empty
model predictions for every model -- matching the verified S1 behavior:
these fields never appear in 7/8 models' raw text, and the one model whose
raw text occasionally contains them (Qwen3) still merges to 0/960 non-empty
cells in the real S1 review, so empty-string predictions reproduce the
actual published behavior rather than introducing a new, S2-only parse).

Usage: python scripts/compute_table34_s2_s3.py
Writes results/out/table34_s2_s3_rank_agreement.csv
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "score_v41" / "src"))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "wsc", str(ROOT / "scripts/jobs/_weighting_sensitivity_canonical.py"))
wsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wsc)

from score_trace import score_review  # noqa: E402

MODELS = wsc.MODELS
A2_GOLD_PATH = ROOT / "data/dev1-12_a2/k_dev1-12_gold.json"
EXCLUDE_DEV1_2 = set(range(35, 55))
SCORE_VARIANTS = ["A", "A_eq", "A_eq_noSR", "A_old", "A_abl", "A_eq_abl"]


def compute_s3():
    review = json.loads((ROOT / "results/baseline_complete_dev1-12/annotation_review.json").read_text())
    kept = [e for e in review if int(e["arg_index"]) not in EXCLUDE_DEV1_2]
    assert len(kept) == 100, len(kept)
    return score_review(kept, MODELS)


def compute_s2():
    per_seed_reviews, kept_indices = wsc.load_a2_dev3_12_review()
    assert len(kept_indices) == 100

    a2_gold = json.loads(A2_GOLD_PATH.read_text())["annotations"]
    gold_by_idx = {int(a["argumentIndex"]): a for a in a2_gold}

    all_rows = []
    for seed, review in per_seed_reviews.items():
        for entry in review:
            idx = entry["arg_index"]
            ann = gold_by_idx[idx]
            for field in ("revisionPhases", "revisionReason"):
                gval = ann.get(field, "")
                fe = {"gold": gval}
                for m in MODELS:
                    fe[m] = ""  # see module docstring
                entry["fields"][field] = fe
        for row in score_review(review, MODELS):
            row["seed"] = seed
            all_rows.append(row)

    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["arg_index"], r["model"])].append(r)
    avg_rows = []
    for (idx, m), rs in grouped.items():
        avg = {"arg_index": idx, "model": m}
        for v in SCORE_VARIANTS:
            avg[v] = sum(r[v] for r in rs) / len(rs)
        avg_rows.append(avg)
    return avg_rows


def model_summary(rows):
    import statistics
    by_model = {}
    for m in MODELS:
        mrows = [r for r in rows if r["model"] == m]
        by_model[m] = {v: statistics.mean(r[v] for r in mrows) for v in SCORE_VARIANTS}
    return by_model


def rank_agreement(summary):
    from scipy import stats as scipy_stats
    a_vals = [summary[m]["A"] for m in MODELS]
    out = {}
    for v in SCORE_VARIANTS:
        if v == "A":
            continue
        v_vals = [summary[m][v] for m in MODELS]
        rho, _ = scipy_stats.spearmanr(a_vals, v_vals)
        tau, _ = scipy_stats.kendalltau(a_vals, v_vals)
        out[v] = (float(rho), float(tau))
    return out


def main():
    out_dir = ROOT / "results/out"
    out_dir.mkdir(parents=True, exist_ok=True)

    s2_rows = compute_s2()
    s3_rows = compute_s3()

    s2_summary = model_summary(s2_rows)
    s3_summary = model_summary(s3_rows)
    s2_rank = rank_agreement(s2_summary)
    s3_rank = rank_agreement(s3_summary)

    with open(out_dir / "table34_s2_s3_rank_agreement.csv", "w") as f:
        f.write("setting,variant,spearman_rho,kendall_tau\n")
        for setting, rank in [("S2", s2_rank), ("S3", s3_rank)]:
            for v in SCORE_VARIANTS:
                if v == "A":
                    continue
                rho, tau = rank[v]
                f.write(f"{setting},{v},{rho!r},{tau!r}\n")

    with open(out_dir / "table34_s2_s3_model_means.csv", "w") as f:
        f.write("setting,model," + ",".join(SCORE_VARIANTS) + "\n")
        for setting, summ in [("S2", s2_summary), ("S3", s3_summary)]:
            for m in MODELS:
                f.write(f"{setting},{m}," + ",".join(repr(summ[m][v]) for v in SCORE_VARIANTS) + "\n")

    print("=== S2 (dev3-12, A2 gold, n=100) model means (A) ===")
    for m in MODELS:
        print(f"  {m:10s} A={s2_summary[m]['A']:.4f}")
    print("GPT-4o ranks first in S2:", max(MODELS, key=lambda m: s2_summary[m]["A"]) == "gpt-4o")
    print("\n=== S2 rank agreement with A ===")
    for v in SCORE_VARIANTS:
        if v == "A":
            continue
        rho, tau = s2_rank[v]
        print(f"  {v:12s} rho={rho:.4f}  tau={tau:.4f}")

    print("\n=== S3 (dev3-12, A1 gold, n=100) model means (A) ===")
    for m in MODELS:
        print(f"  {m:10s} A={s3_summary[m]['A']:.4f}")
    print("GPT-4o ranks first in S3:", max(MODELS, key=lambda m: s3_summary[m]["A"]) == "gpt-4o")
    print("\n=== S3 rank agreement with A ===")
    for v in SCORE_VARIANTS:
        if v == "A":
            continue
        rho, tau = s3_rank[v]
        print(f"  {v:12s} rho={rho:.4f}  tau={tau:.4f}")

    print(f"\nWrote {out_dir / 'table34_s2_s3_rank_agreement.csv'}")
    print(f"Wrote {out_dir / 'table34_s2_s3_model_means.csv'}")


if __name__ == "__main__":
    main()
