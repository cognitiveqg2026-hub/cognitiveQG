# Results

Model outputs and scores that reproduce the numbers, tables, and figures in the paper.
Unless noted, results are for the final **120-argument** CognitiveQG corpus (dev1–dev12).

> **Note on source text:** the FOCUS `argument` field is blank throughout (see `../data/README.md`).
> Model outputs here still quote *fragments* of the arguments they processed (in generated traces,
> questions, and judge rationales) — that is inherent to the task and is model-generated content, not
> the FOCUS dataset.

## `baseline_dev1-12/` — RQ1 (cognitive alignment + trace–question consistency)

Zero-shot **P1** generations and scores for all eight models on 120 arguments (3 seeds each).

| Path | Contents |
|---|---|
| `<model>/judge_run2_judge.json` | GPT-5.5 Semantic-Equivalence judge output (per question) |
| `reasoning_trace_scores.csv` | per-(model, argument) Combined Trace Score + components |
| `parsed/<model>/parsed.json` | parsed, field-structured model traces |
| `annotation_review.json` | model-vs-gold field comparison used for the RQ1a metrics |
| `comparison/` | per-field comparison intermediates |

Models: `llama2`, `llama3`, `qwen` (Qwen2.5), `mistral`, `olmo`, `qwen3`, `gpt-4o`, `gpt-5.5`.

**Reproduces:** the per-field baseline table, the ablated-score table, the
trace-vs-Semantic-Equivalence figure, and the 2×2 diagnostic (Table 5).

## RQ2 — functional trace influence (oracle / noise / vanilla)

| Path | Contents |
|---|---|
| `causation_judge/` | GPT-5.5 judge scores per model × condition for dev1–6 (`*_60args_judge.json`) + the run2 FIXED bar figures |
| `causation/` | oracle / noise / vanilla generations and comparison data |

The released judged outputs cover the dev1–6 arguments (60 per model × condition); the paper's
n = 120 figure additionally uses dev7–12 runs, whose judge scores can be regenerated with
`code/evaluate_llm_judge.py` (requires `OPENAI_API_KEY`).
