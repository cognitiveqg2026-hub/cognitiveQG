#!/usr/bin/env python3
"""
Tables 9 and 20: 2x2 trace-grounding x semantic-alignment diagnostic on
dev1-6 (arg_index <= 59).

Table 9 (3-model subset: llama2, olmo, qwen3) at the published thresholds
(T_trace=0.240, T_SE=0.339).
Table 20 (all 8 models) at the published thresholds plus a threshold-
sensitivity sweep (pooled mean/median of the trace score and GPT-Sem over
the same dev1-6 x 8-model pool, and +/-20% of the published thresholds).

Cases: 1=ungrounded&misaligned, 2=grounded&aligned, 3=grounded&misaligned,
4=ungrounded&aligned. "Grounded" = trace score >= T_trace;
"aligned" = GPT-Sem >= T_SE.
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge_io import load_judge_se

DEV16_MAX = 59
T_TRACE_PUBLISHED = 0.240
T_SE_PUBLISHED = 0.339

MODELS_8 = ["llama2", "llama3", "qwen", "qwen3", "mistral", "olmo", "gpt-4o", "gpt-5.5"]
MODELS_3 = ["llama2", "olmo", "qwen3"]


def load_per_arg_A(scores_csv: Path) -> dict:
    out = {}
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["model"], {})[int(row["arg_index"])] = float(row["A"])
    return out


def pooled_pairs(trace_by_model, se_by_model, models):
    """Returns list of (trace, se) pairs, dev1-6, for the given models."""
    pairs = []
    for m in models:
        tj = trace_by_model.get(m, {})
        se = se_by_model.get(m, {})
        for i in range(0, DEV16_MAX + 1):
            if i in tj and i in se:
                pairs.append((tj[i], se[i]))
    return pairs


def classify(pairs, t_trace, t_se):
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for trace, se in pairs:
        grounded = trace >= t_trace
        aligned = se >= t_se
        if not grounded and not aligned:
            counts[1] += 1
        elif grounded and aligned:
            counts[2] += 1
        elif grounded and not aligned:
            counts[3] += 1
        else:
            counts[4] += 1
    n = len(pairs)
    pct = {k: (100.0 * v / n if n else float("nan")) for k, v in counts.items()}
    return counts, pct, n


def main(scores_csv: str, judge_dir: str, out_dir: str):
    scores_csv = Path(scores_csv)
    judge_dir = Path(judge_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_by_model = load_per_arg_A(scores_csv)
    se_by_model = {m: load_judge_se(judge_dir / f"{m}_gpt55_judge.json") for m in MODELS_8}

    # --- Table 9: 3-model subset, published thresholds ---
    pairs3 = pooled_pairs(trace_by_model, se_by_model, MODELS_3)
    counts3, pct3, n3 = classify(pairs3, T_TRACE_PUBLISHED, T_SE_PUBLISHED)
    print(f"=== Table 9 (3-model: llama2/olmo/qwen3, dev1-6, N={n3}) ===")
    for k in (1, 2, 3, 4):
        print(f"  Case{k}: {counts3[k]:3d}  ({pct3[k]:.1f}%)")
    with open(out_dir / "table9_2x2_3model.csv", "w") as f:
        f.write("case,count,pct\n")
        for k in (1, 2, 3, 4):
            f.write(f"{k},{counts3[k]},{pct3[k]!r}\n")

    # --- Table 20: 8-model, multiple threshold regimes ---
    pairs8 = pooled_pairs(trace_by_model, se_by_model, MODELS_8)
    traces8 = [t for t, _ in pairs8]
    ses8 = [s for _, s in pairs8]
    pooled_mean_trace = statistics.mean(traces8)
    pooled_median_trace = statistics.median(traces8)
    pooled_mean_se = statistics.mean(ses8)
    pooled_median_se = statistics.median(ses8)

    regimes = {
        "published": (T_TRACE_PUBLISHED, T_SE_PUBLISHED),
        "pooled_mean": (pooled_mean_trace, pooled_mean_se),
        "pooled_median": (pooled_median_trace, pooled_median_se),
        "published_minus20pct": (T_TRACE_PUBLISHED * 0.8, T_SE_PUBLISHED * 0.8),
        "published_plus20pct": (T_TRACE_PUBLISHED * 1.2, T_SE_PUBLISHED * 1.2),
    }
    print(f"\n=== Table 20 (8-model, dev1-6, N={len(pairs8)}) ===")
    print(f"pooled mean trace={pooled_mean_trace:.4f}  pooled median trace={pooled_median_trace:.4f}")
    print(f"pooled mean SE={pooled_mean_se:.4f}  pooled median SE={pooled_median_se:.4f}")
    with open(out_dir / "table20_2x2_sensitivity.csv", "w") as f:
        f.write("regime,T_trace,T_se,case1_pct,case2_pct,case3_pct,case4_pct,case3plus4_pct,n\n")
        for name, (t_trace, t_se) in regimes.items():
            counts, pct, n = classify(pairs8, t_trace, t_se)
            c34 = pct[3] + pct[4]
            print(f"  {name:22s} T=({t_trace:.3f},{t_se:.3f})  "
                  f"C1={pct[1]:.1f}% C2={pct[2]:.1f}% C3={pct[3]:.1f}% C4={pct[4]:.1f}%  C3+4={c34:.1f}%")
            f.write(f"{name},{t_trace!r},{t_se!r},{pct[1]!r},{pct[2]!r},{pct[3]!r},{pct[4]!r},{c34!r},{n}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    main(args.scores_csv, args.judge_dir, args.out_dir)
