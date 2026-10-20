#!/usr/bin/env python3
"""
Table 34 (and the corrected Table 5, which equals Table 34's A_abl column):
model-level means/ranks for A, A_eq, A_eq_noSR, A_old, A_abl, A_eq_abl, and
Spearman rho / Kendall tau of each alternative against A.

A_old is the previously-published score (0.390/0.098/0.512 over 12/4/23
fields), kept only for side-by-side comparison -- it replaces the former
A_fs column, which doesn't carry independent meaning here: under Eq. (2),
the primary score A already IS field-share-weighted (16/41, 4/41, 21/41
are exactly 16, 4 and 21 fields' share of 41), so a separate field-share
variant would be identical to A.
"""
import csv
import sys
from pathlib import Path

from scipy import stats as scipy_stats

MODELS = ["llama2", "llama3", "qwen", "qwen3", "mistral", "olmo", "gpt-4o", "gpt-5.5"]
MODEL_LABELS = {
    "llama2": "LLaMA-2", "llama3": "LLaMA-3", "qwen": "Qwen2.5", "qwen3": "Qwen3-8B",
    "mistral": "Mistral", "olmo": "OLMo-2", "gpt-4o": "GPT-4o", "gpt-5.5": "GPT-5.5",
}
SCORE_VARIANTS = ["A", "A_eq", "A_eq_noSR", "A_old", "A_abl", "A_eq_abl"]


def load_model_means(scores_csv: Path) -> dict:
    sums = {m: {v: 0.0 for v in SCORE_VARIANTS} for m in MODELS}
    counts = {m: 0 for m in MODELS}
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if m not in sums:
                continue
            for v in SCORE_VARIANTS:
                sums[m][v] += float(row[v])
            counts[m] += 1
    return {m: {v: sums[m][v] / counts[m] for v in SCORE_VARIANTS} for m in MODELS if counts[m]}


def ranks(means: dict) -> dict:
    """model -> {variant: rank}, rank 1 = highest."""
    out = {m: {} for m in means}
    for v in SCORE_VARIANTS:
        order = sorted(means.keys(), key=lambda m: -means[m][v])
        for i, m in enumerate(order, 1):
            out[m][v] = i
    return out


def main(scores_csv: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    means = load_model_means(Path(scores_csv))
    rnk = ranks(means)

    print("=== Table 34: model means and ranks ===")
    with open(out_dir / "table34_weighting_sensitivity.csv", "w") as f:
        f.write("model," + ",".join(SCORE_VARIANTS) + "," + ",".join(f"{v}_rank" for v in SCORE_VARIANTS) + "\n")
        for m in MODELS:
            vals = [means[m][v] for v in SCORE_VARIANTS]
            rks = [rnk[m][v] for v in SCORE_VARIANTS]
            f.write(f"{m}," + ",".join(repr(v) for v in vals) + "," + ",".join(str(r) for r in rks) + "\n")
            print(f"  {m:10s} " + "  ".join(f"{v}={means[m][v]:.4f}(#{rnk[m][v]})" for v in SCORE_VARIANTS))

    print("\n=== Rank agreement with A (Spearman rho / Kendall tau) ===")
    with open(out_dir / "table34_rank_agreement.csv", "w") as f:
        f.write("variant,spearman_rho,kendall_tau\n")
        a_vals = [means[m]["A"] for m in MODELS]
        for v in SCORE_VARIANTS:
            if v == "A":
                continue
            v_vals = [means[m][v] for m in MODELS]
            rho, _ = scipy_stats.spearmanr(a_vals, v_vals)
            tau, _ = scipy_stats.kendalltau(a_vals, v_vals)
            rho, tau = float(rho), float(tau)
            print(f"  {v:12s} rho={rho:.4f}  tau={tau:.4f}")
            f.write(f"{v},{rho!r},{tau!r}\n")

    # Table 5 replacement = Table 34's A_abl column, closed/open blocks.
    print("\n=== Table 5 (corrected) = Table 34's A_abl column ===")
    with open(out_dir / "table5_ablated_score.csv", "w") as f:
        f.write("model,model_label,A_abl,rank\n")
        order = sorted(MODELS, key=lambda m: -means[m]["A_abl"])
        for m in order:
            print(f"  {MODEL_LABELS[m]:10s} A_abl={means[m]['A_abl']:.4f}")
            f.write(f"{m},{MODEL_LABELS[m]},{means[m]['A_abl']!r},{rnk[m]['A_abl']}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    main(args.scores_csv, args.out_dir)
