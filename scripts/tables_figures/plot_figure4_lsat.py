#!/usr/bin/env python3
"""
Figure 4: Combined Trace Score (Eq. 2) vs GPT-5.5 semantic equivalence on
the LSAT sample (n=20), 6 open-source models only (matches the published
figure's panel count -- GPT-4o/GPT-5.5 were added to the LSAT per-field
table later but were never part of this figure). Scores the LSAT
annotation review with the same bundled score_trace.score_review() used
for Figure 2 / Tables 5-43; judge scores read directly from the raw
judge_s44_judge.json files in data/lsat/<model>/.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src/score_v41/src"))
from score_trace import score_review  # noqa: E402

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3"]
MODEL_LABELS = {"llama2": "LLaMA-2-7B", "llama3": "LLaMA-3.1-8B", "qwen": "Qwen2.5-7B",
                "mistral": "Mistral-7B-v0.3", "olmo": "OLMo-2-7B", "qwen3": "Qwen3-8B"}
POINT_COLOR, LINE_COLOR, CI_COLOR = "#2166ac", "#d73027", "#d73027"


def load_judge_se_raw(path):
    d = json.load(open(path))
    by_arg = defaultdict(list)
    for p in d["scored_pairs"]:
        if p.get("model_missing", False):
            continue
        j = p.get("judge", p.get("llm_judge", {}))
        if j.get("_skipped"):
            continue
        se = j.get("semantic_equivalence")
        if se is not None:
            by_arg[int(p["arg_index"])].append((int(se) - 1) / 4.0)
    return {a: sum(v) / len(v) for a, v in by_arg.items()}


def ci95_band(xs, ys, x_line, slope, intercept):
    n = len(xs)
    x_mean = xs.mean()
    ss_xx = np.sum((xs - x_mean) ** 2)
    y_hat = slope * xs + intercept
    mse = np.sum((ys - y_hat) ** 2) / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (x_line - x_mean) ** 2 / ss_xx))
    t_val = stats.t.ppf(0.975, df=n - 2)
    y_line = slope * x_line + intercept
    return y_line - t_val * se_fit, y_line + t_val * se_fit


def draw_subplot(ax, xs, ys, arg_ids, model_label):
    slope, intercept, r, p_val, _ = stats.linregress(xs, ys)
    rho, _ = stats.spearmanr(xs, ys)
    x_line = np.linspace(0.0, 0.55, 200)
    y_line = slope * x_line + intercept
    lo, hi = ci95_band(xs, ys, x_line, slope, intercept)
    lo, hi = np.clip(lo, -0.05, 1.05), np.clip(hi, -0.05, 1.05)
    ax.fill_between(x_line, lo, hi, alpha=0.15, color=CI_COLOR, zorder=1)
    ax.plot(x_line, y_line, color=LINE_COLOR, lw=1.8, linestyle="--", zorder=4)
    ax.scatter(xs, ys, c=POINT_COLOR, s=52, alpha=0.82, edgecolors="white", linewidths=0.6, zorder=5)
    for xi, yi, ai in zip(xs, ys, arg_ids):
        ax.annotate(str(ai), (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=5.5, zorder=6)
    ax.text(0.97, 0.97, f"Pearson r={r:+.3f}, p={p_val:.3f}\nSpearman ρ={rho:+.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, style="italic", color="#444444")
    ax.axhline(ys.mean(), color="#636363", lw=0.9, linestyle=":", alpha=0.7)
    ax.set_title(model_label, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Combined Trace Score", fontsize=8)
    ax.set_ylabel("Semantic Equivalence (norm. 0–1)\n[GPT-5.5 judge]", fontsize=7.5)
    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return r, p_val, rho, len(xs)


def main():
    review = json.loads((ROOT / "data/lsat/annotation_review.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    print(f"{'Model':14s} {'n':>4s} {'r':>8s} {'p':>8s} {'rho':>8s}")
    for ax, m in zip(axes.flat, MODELS):
        rows = score_review(review, [m])
        scores = {r["arg_index"]: r["A"] for r in rows}
        se_by_arg = load_judge_se_raw(ROOT / f"data/lsat/{m}/judge_s44_judge.json")
        common = sorted(set(scores) & set(se_by_arg))
        xs = np.array([scores[i] for i in common])
        ys = np.array([se_by_arg[i] for i in common])
        r, p, rho, n = draw_subplot(ax, xs, ys, common, MODEL_LABELS[m])
        print(f"{MODEL_LABELS[m]:14s} {n:4d} {r:8.3f} {p:8.4f} {rho:8.3f}")

    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    out = ROOT / "results/figures/figure4_lsat_combined_trace_score.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
