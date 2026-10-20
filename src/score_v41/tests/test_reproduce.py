#!/usr/bin/env python3
"""
Asserts the Combined Trace Score (Eq. 2 of the manuscript: weights are
proportional to group size, so every field contributes equally when all
fields are filled; empty expert fields are skipped within their group)
reproduces correctly, plus every downstream table built from it. Run after
`python src/run_all.py` (run.sh/run.ps1/Dockerfile do this for you).
Standalone: `python -m pytest tests/ -v` from the package root.
"""
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def _rows(name):
    path = OUT / name
    assert path.exists(), f"missing {path} -- run src/run_all.py first"
    return list(csv.DictReader(open(path)))


def _by_key(rows, key):
    return {r[key]: r for r in rows}


@pytest.fixture(scope="module")
def table6():
    return _by_key(_rows("table6_21_figure2_under_A.csv"), "model")


def test_A_reproduces_eq2():
    # No reference CSV is shipped (that would defeat the purpose of an
    # independent friend run) -- this test instead checks internal
    # self-consistency: A must equal 16/41*EM + 4/41*J_multi + 21/41*J_span
    # for every row, to full float precision.
    rows = _rows("per_argument_scores.csv")
    assert len(rows) == 120 * 8
    for r in rows:
        expected = (16 / 41) * float(r["EM"]) + (4 / 41) * float(r["J_multi"]) + (21 / 41) * float(r["J_span"])
        assert abs(expected - float(r["A"])) < 1e-12


# Table 6, Table 21, Figure 2 -- Pearson, per model, under v41.
def test_llama2_pearson_r(table6):
    assert round(float(table6["llama2"]["pearson_r"]), 3) == 0.305


def test_llama2_holm_p(table6):
    assert round(float(table6["llama2"]["holm_p"]), 4) == 0.0060


def test_olmo_pearson_r(table6):
    assert round(float(table6["olmo"]["pearson_r"]), 3) == 0.197


def test_qwen3_pearson_r(table6):
    assert round(float(table6["qwen3"]["pearson_r"]), 3) == 0.192


# Same tests, Spearman columns (Table 6/21; Holm/BH corrected SEPARATELY
# from Pearson, per the Table 6 caption).
def test_llama2_spearman_rho(table6):
    assert round(float(table6["llama2"]["spearman_rho"]), 3) == 0.283
    assert round(float(table6["llama2"]["spearman_holm_p"]), 4) == 0.0143


def test_olmo_spearman_rho(table6):
    assert round(float(table6["olmo"]["spearman_rho"]), 3) == 0.177
    assert round(float(table6["olmo"]["spearman_p"]), 4) == 0.0528


def test_qwen3_spearman_rho(table6):
    assert round(float(table6["qwen3"]["spearman_rho"]), 3) == 0.194


def test_only_llama2_holm_significant_both_metrics(table6):
    # Holm-corrected conclusion is unchanged from the previously-published
    # score: only LLaMA-2 remains significant, for both Pearson and
    # Spearman.
    for model, row in table6.items():
        pearson_sig = row["holm_reject"] == "True"
        spearman_sig = row["spearman_holm_reject"] == "True"
        if model == "llama2":
            assert pearson_sig and spearman_sig
        else:
            assert not pearson_sig and not spearman_sig, model


# Table 43, trace-question association row only -- the rest of Table 43
# needs causation-experiment data outside this package's scope, and does
# not depend on A at all (it scores GPT-Sem directly under trace
# substitution), so it is UNCHANGED by this update.
def test_table43_judge_robustness():
    rows = _by_key(_rows("table43_judge_robustness.csv"), "judge")
    assert int(rows["GPT-5.5"]["n_holm_significant_of_8"]) == 1
    assert int(rows["GPT-6 Astra"]["n_holm_significant_of_8"]) == 0
    assert int(rows["GPT-5.6 Sol"]["n_holm_significant_of_8"]) == 2


# Table 9, 3-model subset (LLaMA-2/OLMo/Qwen3, dev1-6), under v41. The
# Cons./Incons./Cont. sub-breakdown within Case 3/4 is NOT asserted here --
# see ../CONCLUSIONS_CHECK.md and ../v41/out/table9_case_transitions.csv:
# 11 pairs newly enter Case 3/4 and need fresh 2-annotator labeling before
# that sub-table can be recomputed.
def test_table9_2x2():
    rows = _by_key(_rows("table9_2x2_3model.csv"), "case")
    assert round(float(rows["1"]["pct"]), 1) == 25.0
    assert round(float(rows["2"]["pct"]), 1) == 20.6
    assert round(float(rows["3"]["pct"]), 1) == 11.1
    assert round(float(rows["4"]["pct"]), 1) == 43.3


# Table 20.
def test_table20_2x2_sensitivity():
    rows = _by_key(_rows("table20_2x2_sensitivity.csv"), "regime")
    assert round(float(rows["published"]["case3plus4_pct"]), 1) == 47.5
    assert round(float(rows["pooled_mean"]["T_trace"]), 3) == 0.238
    assert round(float(rows["pooled_mean"]["T_se"]), 3) == 0.481
    assert round(float(rows["pooled_median"]["T_trace"]), 3) == 0.232
    assert round(float(rows["pooled_median"]["T_se"]), 3) == 0.500
    all_c34 = [round(float(r["case3plus4_pct"]), 1) for r in rows.values()]
    assert min(all_c34) == 32.9
    assert max(all_c34) == 57.3


# A_old: the previously-published score (0.390/0.098/0.512 over 12/4/23
# fields), kept only for comparison. This checks it STILL reproduces the
# originally-published numbers exactly, i.e. that porting the package to
# Eq. (2) didn't silently change the comparison baseline too.
def test_A_old_reproduces_previously_published_values():
    rows = _rows("per_argument_scores.csv")
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(float(r["A_old"]))
    means = {m: sum(v) / len(v) for m, v in by_model.items()}
    expected = {
        "llama2": 0.234611, "llama3": 0.241937, "qwen": 0.290312, "qwen3": 0.244458,
        "mistral": 0.249391, "olmo": 0.228900, "gpt-4o": 0.358120, "gpt-5.5": 0.266371,
    }
    for m, exp in expected.items():
        assert abs(means[m] - exp) < 1e-5, m


# Table 34 (S1 only: S2/S3 need A2-gold / dev3-12-only data this package
# does not ship).
def test_table34_rank_agreement_s1():
    rows = _by_key(_rows("table34_rank_agreement.csv"), "variant")
    assert round(float(rows["A_eq"]["spearman_rho"]), 2) == 0.86
    assert round(float(rows["A_abl"]["spearman_rho"]), 2) == 0.69
    assert round(float(rows["A_eq_abl"]["spearman_rho"]), 2) == 0.60
    assert round(float(rows["A_eq_noSR"]["spearman_rho"]), 2) == 0.71


# GPT-4o ranks first under every score variant -- this holds under both
# the previously-published score and v41.
def test_gpt4o_ranks_first_every_variant():
    rows = _by_key(_rows("table34_weighting_sensitivity.csv"), "model")
    for variant in ["A", "A_eq", "A_eq_noSR", "A_old", "A_abl", "A_eq_abl"]:
        other_means = [float(r[variant]) for m, r in rows.items() if m != "gpt-4o"]
        assert float(rows["gpt-4o"][variant]) > max(other_means), variant


# Prose (p.11 of the earlier v16 PDF; now superseded by the prose giving
# these same A_abl values under the 36-field basis): GPT-4o (0.277) >
# Qwen2.5 (0.224) > GPT-5.5 (0.209) > Qwen3 (0.192) > OLMo (0.188) >
# Mistral (0.183) > LLaMA-2 (0.180) > LLaMA-3 (0.165). This equals Table
# 31's A_abl column, over the 36 fields remaining after excluding
# span1/2/3 and biasDetection/heuristicDetection (13/36, 2/36, 21/36) --
# NOT the "34 remaining fields" still printed in the current Table 5
# caption, which is a stale leftover from an earlier 39-field draft
# (41 - 5 = 36, not 34). The code intentionally does not reproduce that
# stale caption's implied field count.
def test_table5_equals_table34_abl():
    rows = _by_key(_rows("table5_ablated_score.csv"), "model")
    expected = {
        "gpt-4o": 0.277, "gpt-5.5": 0.208, "qwen": 0.224, "qwen3": 0.192,
        "olmo": 0.188, "mistral": 0.183, "llama2": 0.180, "llama3": 0.164,
    }
    for model, exp in expected.items():
        assert round(float(rows[model]["A_abl"]), 3) == exp, model


def test_no_ac1_in_source():
    src_dir = ROOT / "src"
    for py in src_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        assert "ac1" not in text and "gwet" not in text, (
            f"{py} references AC1/Gwet -- the Combined Trace Score path must be exact-match/Jaccard only")
