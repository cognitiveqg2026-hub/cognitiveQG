#!/usr/bin/env python3
"""
Repo-wide checks beyond src/score_v41/tests/test_reproduce.py: Table 34
S2/S3, Table 4, and the causation (trace-substitution) headline claim.
Run after `./run.sh` (or the equivalent manual steps in README.md) have
populated results/out/.
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "out"


def _rows(name):
    path = OUT / name
    assert path.exists(), f"missing {path} -- run run.sh first"
    with open(path) as f:
        return list(csv.DictReader(f))


def test_table34_s2_a_eq_rank_agreement():
    rows = _rows("table34_s2_s3_rank_agreement.csv")
    row = next(r for r in rows if r["setting"] == "S2" and r["variant"] == "A_eq")
    assert round(float(row["spearman_rho"]), 2) == 0.74
    assert round(float(row["kendall_tau"]), 2) == 0.64


def test_table34_s3_a_eq_rank_agreement():
    rows = _rows("table34_s2_s3_rank_agreement.csv")
    row = next(r for r in rows if r["setting"] == "S3" and r["variant"] == "A_eq")
    assert round(float(row["spearman_rho"]), 2) == 0.90
    assert round(float(row["kendall_tau"]), 2) == 0.79


def test_gpt4o_ranks_first_s2_and_s3():
    rows = _rows("table34_s2_s3_model_means.csv")
    for setting in ("S2", "S3"):
        setting_rows = [r for r in rows if r["setting"] == setting]
        best = max(setting_rows, key=lambda r: float(r["A"]))
        assert best["model"] == "gpt-4o", f"{setting}: expected gpt-4o first, got {best['model']}"


def test_causation_oracle_vs_noise_headline():
    path = OUT / "causation_judge_120_stats" / "causation_judge_120_summary.json"
    assert path.exists(), f"missing {path} -- run run.sh first"
    d = json.loads(path.read_text())
    contrasts = [r for r in d["contrasts"] if r["contrast"] == "oracle_vs_noise"]
    assert len(contrasts) == 8
    assert all(r["mean_diff"] > 0 for r in contrasts), "Oracle should exceed Noise in all 8 models"
    n_sig = sum(1 for r in contrasts if r["wilcoxon_p_holm_reject"])
    assert n_sig == 6, f"expected 6/8 Holm-significant, got {n_sig}"
    mean_diff = sum(r["mean_diff"] for r in contrasts) / len(contrasts)
    assert round(mean_diff, 2) == 0.11  # GPT-Sem on the (s-1)/4 scale


def test_causation_condition_means_validate_against_published():
    path = OUT / "causation_judge_120_stats" / "causation_judge_120_summary.json"
    assert path.exists(), f"missing {path} -- run run.sh first"
    d = json.loads(path.read_text())
    cm = d["condition_means"]
    assert len(cm) == 32
    assert all(r["validation_status"] == "PASS" for r in cm)
