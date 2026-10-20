#!/usr/bin/env python3
"""
Orchestrates the full pipeline: scorer -> correlation/2x2/weighting analyses.
Deterministic: fixes every random seed to 42, forces single-threaded BLAS.
Writes out/ (full-precision CSVs) and out/env.txt (machine/env fingerprint).
"""
import os

# Must be set BEFORE numpy/scipy are imported anywhere, including by the
# modules imported below -- single-threaded BLAS keeps floating-point
# results identical across machines with different core counts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import platform
import subprocess
import sys
from pathlib import Path

SEED = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

MODELS = ["llama2", "llama3", "qwen", "qwen3", "mistral", "olmo", "gpt-4o", "gpt-5.5"]


def write_env_txt():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"seed = {SEED}",
        f"python = {sys.version}",
        f"platform = {platform.platform()}",
        f"machine = {platform.machine()}",
        f"processor = {platform.processor()}",
        f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}",
    ]
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True).stdout
    except Exception as e:
        freeze = f"(pip freeze failed: {e})"
    (OUT / "env.txt").write_text("\n".join(lines) + "\n\n--- pip freeze ---\n" + freeze, encoding="utf-8")
    print(f"Wrote {OUT / 'env.txt'}")


def main():
    write_env_txt()

    sys.path.insert(0, str(HERE))
    import score_trace

    print("=" * 70)
    print("STEP 1/4: score_trace.py -- Combined Trace Score, all variants")
    print("=" * 70)
    review = json.loads((DATA / "annotation_review.json").read_text(encoding="utf-8"))
    rows = score_trace.score_review(review, MODELS)
    scores_csv = OUT / "per_argument_scores.csv"
    score_trace.write_csv(rows, scores_csv)
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        mean_a = sum(r["A"] for r in model_rows) / len(model_rows)
        print(f"  {model:10s} n={len(model_rows):3d}  A_mean={mean_a:.6f}")
    print(f"Wrote {scores_csv}\n")

    print("=" * 70)
    print("STEP 2/4: analysis_correlation.py -- Tables 6/21/35/43, Figure 2")
    print("=" * 70)
    import analysis_correlation
    analysis_correlation.main(str(scores_csv), str(DATA / "judge"), str(OUT))

    print("\n" + "=" * 70)
    print("STEP 3/4: analysis_2x2.py -- Tables 9/20")
    print("=" * 70)
    import analysis_2x2
    analysis_2x2.main(str(scores_csv), str(DATA / "judge"), str(OUT))

    print("\n" + "=" * 70)
    print("STEP 4/4: analysis_weighting.py -- Table 34, Table 5 (corrected)")
    print("=" * 70)
    import analysis_weighting
    analysis_weighting.main(str(scores_csv), str(OUT))

    print("\nAll outputs written to", OUT)


if __name__ == "__main__":
    main()
