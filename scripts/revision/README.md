# Revision scripts

Scripts for the tables and figures added or changed in the revision. Paths are repo-relative
(`scripts/revision/paths.py`). Intermediate results go to `results/revision/`. Scripts that write LaTeX
tables write them to `manuscript/table/` by default; set `CQG_MANUSCRIPT` or `CQG_TABLE_DIR` to change this.

BERTScore numbers were computed with `bert-score 0.3.12`, `transformers 5.9.0` and `tokenizers 0.22.2`
(roberta-large, `rescale_with_baseline=True`). Other transformers versions give different BERTScore values
for the same text (about 5 to 7 points with older 4.x releases), so use the versions in `requirements.txt`.

## What produces what

| Paper item | Script | Inputs | Output |
|---|---|---|---|
| Tables 23-26 (trace substitution, GPT-Sem on (s-1)/4) | `normalization/exact_causation.py` | `data/causation/`, `data/baseline_complete_dev1-12/` | `results/revision/causation_exact_s-1over4/*.csv` |
| Tables 23-26 as LaTeX | `normalization/build_spare_tables.py` | the CSVs above | the four tables |
| Fig. 3 (trace substitution) | `normalization/plot_fig3_s-1over4.py` | the CSVs above | `results/revision/causation_judge_120_FIXED.pdf` |
| Human column, Wt. F1 (both directions) | `human_ceiling/human_wf1.py` | `data/a1_dev1-12_gold.json`, `data/a2_dev1-12_gold.json` | `results/revision/human_wtf1_dev1-12.json` |
| Span-role F1 (models and human, dev1-12 and dev3-12) | `span_roles/compute_span_roles_table3.py` | raw model outputs, both gold files | `results/revision/span_roles_table3.json` (`span_roles_dev1-12.csv` is a rounded copy for reading) |
| Tables 27-28 (dev1-12 against A1 / A2, with Human column) | `tables/build_dev1-12_annotator_tables.py` | per-annotator tables, IAA table, the two files above | Tables 27 and 28 |
| Table 3 (mean over the two references) | `tables/build_table3_meanref.py` | Tables 27-28, span-role results | Table 3 |
| Tables 29-30 (dev3-12, span-role rows) | `tables/update_dev3-12_spanroles.py` | `span_roles_table3.json` | updates Tables 29 and 30 |
| LSAT span roles | `span_roles/compute_span_roles_lsat.py` | LSAT parsed outputs (`CQG_PROD_ROOT`, not redistributed) | `results/revision/span_roles_lsat.{json,csv}` |
| Table 11 and Tables 32-33 (LSAT) | `lsat/build_lsat_meanref_tables.py` | `results/revision/lsat_*_result.json` | the three LSAT tables |

Run order (CPU only): `human_ceiling/human_wf1.py` -> `span_roles/compute_span_roles_table3.py` ->
`tables/build_dev1-12_annotator_tables.py` -> `tables/build_table3_meanref.py` ->
`tables/update_dev3-12_spanroles.py` -> `lsat/build_lsat_meanref_tables.py`.

Each builder was checked against the tables in the manuscript: Tables 3, 11, 23-30 and 32-33 are
reproduced exactly.

## Results computed on the GPU machine

The `results/revision/lsat_*_result.json` files and `task1_iaa_results.json` were computed on the GPU
machine because they need the LSAT argument text or BERTScore on a GPU: LSAT A1/A2 scoring, Wt. F1 for the
multi-label LSAT fields, Term X/Y rows, Socratic questions (same extractor as dev1-12), and a BERTScore
check of the IAA tables (all 42 values within 0.05 of the published tables).

## Other files

- `human_ceiling/human_wf1_reverse.py`, `human_ceiling/span_roles_human.py`: cross-checks of the human values.
- `scripts/jobs/_verify_normalization_invariance.py`, `scripts/jobs/_wilcoxon_raw_scale.py`: check that the
  rank statistics are the same on the s/5 and (s-1)/4 scales.
