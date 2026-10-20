#!/usr/bin/env python3
"""
Figure 2: Combined Trace Score (Eq. 2) vs GPT-5.5 semantic equivalence, all
8 models, dev1-12 (n=120). Uses the bundled score_v41 package directly
(src/score_v41/data/annotation_review.json + data/judge/<model>_gpt55_judge.json)
-- same draw_subplot style (regression line, 95% CI band, arg-index point
labels, Pearson/Spearman stats box, mean line) as the original manuscript
figure script.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src/score_v41/src"))

from judge_io import load_judge_se  # noqa: E402
from score_trace import score_review  # noqa: E402

MODELS = ["llama2", "llama3", "qwen", "mistral", "gpt-4o", "gpt-5.5", "olmo", "qwen3"]
MODEL_LABELS = {"llama2": "LLaMA-2-7B", "llama3": "LLaMA-3.1-8B", "qwen": "Qwen2.5-7B",
                "mistral": "Mistral-7B-v0.3", "gpt-4o": "GPT-4o", "gpt-5.5": "GPT-5.5",
                "olmo": "OLMo-2-7B", "qwen3": "Qwen3-8B"}
POINT_COLOR, LINE_COLOR, CI_COLOR = "#2166ac", "#d73027", "#d73027"


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


def draw_subplot(ax, xs_raw, ys_raw, arg_ids, model_label):
    ys = np.clip(ys_raw, 0.0, 1.0)  # already (s-1)/4 normalized by judge_io
    xs = xs_raw
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
        ax.annotate(str(ai), (xi, yi), textcoords="offset points", xytext=(4, 3),
                    fontsize=5.5, color="#333333", zorder=6)
    p_str = ("p=%.4f" % p_val) if p_val >= 0.0001 else "p<0.0001"
    sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
    ax.text(0.97, 0.97, f"Pearson r={r:+.3f}{sig}, {p_str}\nSpearman ρ={rho:+.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, style="italic", color="#444444")
    ax.axhline(ys.mean(), color="#636363", lw=0.9, linestyle=":", alpha=0.7)
    ax.text(0.02, ys.mean() + 0.02, "Mean=%.2f" % ys.mean(), transform=ax.get_yaxis_transform(),
            fontsize=6.5, color="#636363")
    ax.set_title(model_label, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Combined Trace Score", fontsize=8)
    ax.set_ylabel("Semantic Equivalence (norm. 0–1)\n[GPT-5.5 judge]", fontsize=7.5)
    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return r, p_val, rho, len(xs)


def main():
    review = json.loads((ROOT / "src/score_v41/data/annotation_review.json").read_text())
    scores_by_model = {}
    for m in MODELS:
        rows = score_review(review, [m])
        scores_by_model[m] = {r["arg_index"]: r["A"] for r in rows}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    print(f"{'Model':14s} {'n':>4s} {'r':>8s} {'p':>8s} {'rho':>8s}")
    for ax, m in zip(axes.flat, MODELS):
        se_by_arg = load_judge_se(ROOT / f"src/score_v41/data/judge/{m}_gpt55_judge.json")
        common = sorted(set(scores_by_model[m]) & set(se_by_arg))
        xs = np.array([scores_by_model[m][i] for i in common])
        ys = np.array([se_by_arg[i] for i in common])
        r, p, rho, n = draw_subplot(ax, xs, ys, common, MODEL_LABELS[m])
        print(f"{MODEL_LABELS[m]:14s} {n:4d} {r:8.3f} {p:8.4f} {rho:8.3f}")

    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    out = ROOT / "results/figures/figure2_combined_trace_score.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
