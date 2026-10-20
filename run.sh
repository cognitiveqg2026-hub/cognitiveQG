#!/usr/bin/env bash
# One-command reproduction: fresh venv -> every table/figure built from the released data.
set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "=== venv setup ==="
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "=== relative-path aliases (scripts hardcode these paths; symlinks keep them unmodified) ==="
mkdir -p results/out results/figures
ln -sfn "$ROOT/data/baseline_complete_dev1-12" results/baseline_complete_dev1-12
ln -sfn "$ROOT/data/causation/raw_judge_dev1-6" results/causation_judge_fixed
ln -sfn "$ROOT/data/causation/raw_judge_dev7-12" results/causation_judge_dev7-12_60

echo "=== Tables 5/6/9/20/21/34(S1)/35/43 + per-argument scores ==="
(cd src/score_v41 && python3 src/run_all.py)
(cd src/score_v41 && python3 -m pytest tests/ -v)

echo "=== Table 34, S2/S3 rank agreement (independent A2/dev3-12 review rebuild) ==="
python3 scripts/compute_table34_s2_s3.py

echo "=== Figure 2 ==="
python3 scripts/tables_figures/plot_figure2.py

echo "=== Figure 4 (LSAT) ==="
python3 scripts/tables_figures/plot_figure4_lsat.py

echo "=== Table 4 (400-pair human judge validation) ==="
python3 scripts/analyze_semeq_human_judge_agreement.py \
  --ratings-file data/human_validation/semeq_dev1-12_8models_seed44_ratings_MAIN.json \
  --sample-file data/human_validation/dev1-12_semeq_sample_seed44.json \
  --output results/out/human_vs_3judges_FRESH.csv

echo "=== Tables 23-26 (causation / trace-substitution) ==="
python3 scripts/analyze_causation_judge_120_stats.py --output-dir results/out/causation_judge_120_stats

echo "=== repo-wide tests ==="
python3 -m pytest tests/ -v

echo ""
echo "=== DONE ==="
echo "Outputs: src/score_v41/out/, results/out/, results/figures/"
echo ""
echo "NOT run by default (slow BERTScore batches, Tables 3/11 per-field scores):"
echo "  python3 scripts/generate_baseline_field_table_complete.py --help"
echo "  python3 scripts/compute_field_metrics_jac_gwet_lsat6.py --help"
