# CognitiveQG

Data, prompts and code for the paper:

> **CognitiveQG: A Diagnostic Benchmark for Reasoning-Trace Evaluation in Socratic Question Generation**
> *IEEE Access* (under review), 2026

CognitiveQG tests whether language models write reasoning traces that agree with expert annotations
under Facione's critical-thinking framework, and whether the Socratic questions they generate follow
from those traces.

**At a glance:** 120 argumentative texts (FOCUS development split, dev1-dev12), two expert annotators,
44 annotation fields per argument (41 reasoning-trace fields over Facione's six phases, plus up to three
Socratic questions), and a 20-argument LSAT extension.

## Repository layout

```
data/            expert annotations (A1, A2), guidelines, annotation tool, model outputs, judge scores
prompts/         baseline prompt (P1) and the GPT-5.5 judge rubric
generation/      code that generates the model outputs (baseline and trace substitution); see generation/README.md
iaa_core/        metric code (exact match, Jaccard, Gwet's AC1, judge helpers)
src/score_v41/   Combined Trace Score (Eq. 2) and the tables built from it; see src/score_v41/README.md
scripts/         one script per remaining table or figure; scripts/revision/ covers the revision tables
tests/           checks of the headline numbers
results/         reference outputs and figures
run.sh           runs the evaluation end to end
DATA.md          source and license of each dataset
```

## Reproducing the tables

No GPU or API key is needed. Everything runs from the model outputs and judge scores in `data/`.

```bash
./run.sh
```

This creates a virtual environment, installs `requirements.txt`, runs the scripts below and the tests.
It takes a few minutes. Outputs go to `src/score_v41/out/`, `results/out/` and `results/figures/`.

| Paper item | Command |
|---|---|
| Tables 5, 6, 9, 20, 21, 34 (S1), 35, 43 | `cd src/score_v41 && python3 src/run_all.py` |
| Table 34 (S2, S3) | `python3 scripts/compute_table34_s2_s3.py` |
| Table 4 (400-pair human validation of the judge) | `python3 scripts/analyze_semeq_human_judge_agreement.py --ratings-file data/human_validation/semeq_dev1-12_8models_seed44_ratings_MAIN.json --sample-file data/human_validation/dev1-12_semeq_sample_seed44.json --output results/out/human_vs_3judges.csv` |
| Tables 23-26 (trace substitution) | `python3 scripts/analyze_causation_judge_120_stats.py --output-dir results/out/causation_judge_120_stats` |
| Figure 2 | `python3 scripts/tables_figures/plot_figure2.py` |
| Figure 4 (LSAT) | `python3 scripts/tables_figures/plot_figure4_lsat.py` |
| Tables 3, 11, 27-30, 32-33 | see `scripts/revision/README.md` |

Tables 3 and 11 use BERTScore and are not run by `run.sh`. The BERTScore values depend on the
`transformers` version; the versions used in the paper are pinned in `requirements.txt`.

## Generating model outputs

`generation/` has the code that produced the outputs in `data/`: baseline reasoning traces and questions
(Tier 1) and the Vanilla, Oracle and Noise conditions (Tier 2). It needs a GPU for the open-source models
and an OpenAI API key for GPT-4o and GPT-5.5. See `generation/README.md`.

## Source arguments

The argument texts are not redistributed. The `argument` field is blank in every data file. The
arguments come from the FOCUS development split (Pothong et al., IJCNLP-AACL 2025), which reuses
SoQG-2023 (Ang et al., EACL 2023); the LSAT items come from AGIEval. Obtain them from those releases and
join them by `argumentIndex`. See `data/README.md` and `DATA.md`.

## License

- Annotations, guidelines and other data in `data/`: CC BY 4.0 (`data/LICENSE`).
- Code and the annotation tool: MIT (`LICENSE`).
