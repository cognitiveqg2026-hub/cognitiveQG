# CognitiveQG generation code

Code that generates the model outputs used in the paper:
(1) baseline reasoning traces and Socratic questions (Tier 1), and
(2) the trace-substitution conditions Vanilla, Oracle and Noise (Tier 2).

All open-source models are used off the shelf (no training of any kind),
loaded in 4-bit NF4 quantization with bitsandbytes.

## Layout

Run the commands below from this `generation/` folder. This code has its own `requirements.txt`
(separate from the evaluation code at the repository root).

```
model_configs.py               HF repo IDs, 4-bit NF4 settings and chat templates for the six open-source models
inference/
  inference.py                 baseline generation, open-source models
  run_gpt_baseline_aligned.py  baseline generation, GPT-4o / GPT-5.5 (OpenAI API)
causation/
  run_oracle_causation.py      trace substitution for LLaMA-2, and the shared prompt / condition builders
  run_qwen_causation.py        trace substitution for the other five open-source models
  run_gpt_causation.py         trace substitution for GPT-4o / GPT-5.5 (OpenAI API)
prompts/
  P4_complete_20260704.txt     the baseline prompt ("P1" in the paper)
requirements.txt
```

"P1" is the name used in the paper. The file is `prompts/P4_complete_20260704.txt`;
the recorded configuration of the published baseline runs lists this file for
all eight models, including GPT-4o and GPT-5.5.

## Settings used in the paper

| | Value |
|---|---|
| Open-source models | LLaMA-2-7B-Chat, LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, OLMo-2-7B-Instruct, Qwen3-8B |
| Quantization | 4-bit NF4, double quantization, float16 compute |
| Decoding (open-source) | sampling, temperature 0.1, top-p 0.9, repetition penalty 1.15, max_new_tokens 1500 |
| Seeds | 44, 45, 46 |
| Arguments | 120 (dev1-12) |

These are the script defaults, so the commands below need no extra flags for them.

## 1. Baseline generation

Open-source models:

```bash
python inference/inference.py \
    --model llama2 \
    --prompt-file prompts/P4_complete_20260704.txt \
    --test-csv path/to/arguments.csv \
    --output results/llama2/raw_output.json
```

`--test-csv` needs the columns `argumentIndex` and `argument`. The argument texts
come from the FOCUS / SoQG-2023 releases and are not redistributed here.
Gated models need a Hugging Face token in `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN`.
`--dry-run` checks the token, model access, input file and GPU without loading the model.

GPT-4o / GPT-5.5:

```bash
export OPENAI_API_KEY=...
python inference/run_gpt_baseline_aligned.py \
    --model gpt-4o \
    --gold-json path/to/gold_annotations.json \
    --output results/gpt-4o/raw_output.json
```

The script checks that the arguments are in canonical order before any API call.

## 2. Trace substitution (Vanilla / Oracle / Noise)

- **Vanilla**: the model reads the argument and writes up to three Socratic questions, with no trace (`--prompt-version v3`).
- **Oracle**: the expert annotation for the same argument is given as "pre-analysis notes" before generation (`--condition gold`, `--prompt-version v2`).
- **Noise**: an adversarial trace, mismatched to the argument. For argument i, the expert annotation of the next argument, (i+1) mod N, is given as "pre-analysis notes" in the same format as Oracle, so only the content differs (`--condition noise`, `--prompt-version v2`). The same rule is used for all eight models; nothing is generated for this condition.

Oracle and Noise notes contain 9 fields of the annotation: stance, knowledge domain, core claim, reasoning type, assumption, primary domain, inference strength, logical fallacy and change decision.

LLaMA-2:

```bash
python causation/run_oracle_causation.py --gold-json path/to/dev1-12_gold.json --condition vanilla --prompt-version v3 --output results/llama2_vanilla.json
python causation/run_oracle_causation.py --gold-json path/to/dev1-12_gold.json --condition gold    --prompt-version v2 --output results/llama2_oracle.json
python causation/run_oracle_causation.py --gold-json path/to/dev1-12_gold.json --condition noise   --prompt-version v2 --output results/llama2_noise.json
```

Other open-source models: the same flags with `causation/run_qwen_causation.py --model <name>`
(`llama3`, `qwen`, `mistral`, `olmo`, `qwen3`).

GPT-4o / GPT-5.5 (conditions are named `vanilla`, `human_trace` = Oracle, `intervention` = Noise):

```bash
python causation/run_gpt_causation.py --model gpt-4o --condition vanilla      --gold-json path/to/dev1-12_gold.json --output-dir results/causation
python causation/run_gpt_causation.py --model gpt-4o --condition human_trace  --gold-json path/to/dev1-12_gold.json --output-dir results/causation
python causation/run_gpt_causation.py --model gpt-4o --condition intervention --gold-json path/to/dev1-12_gold.json --output-dir results/causation
```

## Environment

See `requirements.txt`. Required environment variables: `OPENAI_API_KEY` for
the GPT scripts, and `HUGGING_FACE_HUB_TOKEN` (or `HF_TOKEN`) for the gated
open-source models.
