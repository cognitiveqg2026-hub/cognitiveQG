#!/usr/bin/env python3
"""Build the main Table 3 (mean over the two expert references) for dev1-12.

Each model cell = mean of the model's score against A1 and against A2, taken
from the two per-annotator appendix tables (Tables 27 and 28). Where one
reference has no value ("--"), the other reference is used. Span-role rows are
averaged from the unrounded F1 values in results/revision/span_roles_table3.json.
Human = mean of the two directions (A2 vs A1, A1 vs A2).
Phase averages = mean of the (unrounded) cells of the phase; Explanation uses
only Target Explanation, Reasoning Structure, Good Evidence, Bad Evidence.
Bold = best model per row, ties included; Human never bold.

Inputs : TABLE_DIR/model-baselinescore-dev1-12_A1_human_spanroles.tex
         TABLE_DIR/model_baselinescore_dev1-12_kim_rescaled_human_spanroles.tex
         OUT/span_roles_table3.json
Output : TABLE_DIR/table3_meanref_main.tex   (override dir with CQG_TABLE_DIR)
"""
import csv, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import OUT, TABLE_DIR

A1_FILE = TABLE_DIR / "model-baselinescore-dev1-12_A1_human_spanroles.tex"
A2_FILE = TABLE_DIR / "model_baselinescore_dev1-12_kim_rescaled_human_spanroles.tex"
OUT_FILE = TABLE_DIR / "table3_meanref_main.tex"
MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
ROLE = {"Span role: Major claim": "major-claim", "Span role: Minor claim": "minor-claim",
        "Span role: Premise": "premise"}
EXPLANATION_AVG = {"Target Explanation", "Reasoning Structure", "Good Evidence", "Bad Evidence"}

CAPTION = r"""\caption{Per-field model scores averaged over the two expert references (mean over references; dev1--12, $n=120$; P1 prompt, mean over 3 inference seeds). Each model cell is the mean of its score against A1 (Table~\ref{tab:model-baseline-a1-human}) and against A2 (Table~\ref{tab:model-baseline-jac-gwet-RunP1structured-a2}); seed-level standard deviations are reported in those two tables. Human = A2 scored against A1 and A1 scored against A2, averaged (Wt.\,F1 and Span Eval rows computed with the same scoring code as the models; other BERTScore rows from the inter-annotator agreement analysis, Table~\ref{tab:iaa-human}), so models and humans are both scored against one reference at a time. Where one reference left a field empty, the other reference is used; ``--'' = no reference or no human value available. \textbf{Bold} = best model per row (Human excluded). Rescaled BERTScore can be negative when a prediction is less similar to the reference than an average pair of unrelated sentences.}"""
HEADER = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", CAPTION,
          r"\label{tab:model-baseline-jac-gwet-Run2P4structured}",
          r"\begin{tabular}{p{3.2cm} l c c c c c c c c | c}", r"\hline",
          r"\textbf{Annotation Field} & \textbf{Metric} & \multicolumn{6}{c}{\textbf{Open-Source}} & \multicolumn{2}{c}{\textbf{Closed-Source}} & \textbf{Human} \\",
          r"& & \textbf{LLaMA-2} & \textbf{LLaMA-3} & \textbf{Qwen} & \textbf{Mistral} & \textbf{OLMo} & \textbf{Qwen3} & \textbf{GPT-4o} & \textbf{GPT-5.5} & \textbf{A1$\leftrightarrow$A2} \\",
          r"\hline"]


def cells(line):
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", line)
    s = re.sub(r"\{\\tiny\$\\pm\$[0-9.]+\}", "", s)
    return [c.strip() for c in s.strip()[:-2].split("&")]


def body(path):
    """(kind, name, metric, values) for every line inside the tabular, after the header."""
    out, started, hlines = [], False, 0
    for l in open(path, encoding="utf-8").read().split("\n"):
        if l.startswith("%"):
            continue
        if l.startswith(r"\begin{tabular}"):
            started = True
            continue
        if not started:
            continue
        if l.startswith(r"\end{tabular}"):
            break
        if hlines < 2:                      # skip the two header rows and their rules
            hlines += l.strip() == r"\hline"
            continue
        if l.startswith(r"\multicolumn"):
            out.append(("phase", re.search(r"\\textit\{([^}]*)\}", l).group(1), None, None))
        elif l.strip() == r"\hline":
            out.append(("hline", None, None, None))
        elif "Phase avg" in l:
            out.append(("avg", None, None, None))
        elif " & " in l:
            c = cells(l)
            out.append(("row", c[0], c[1], c[2:]))
    return out


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def fmt(x):
    if x is None:
        return "--"
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def bold_row(vals):
    shown = [fmt(v) for v in vals[:8]]
    best = max((float(s) for s in shown if s != "--"), default=None)
    out = [(r"\textbf{%s}" % s) if s != "--" and float(s) == best else s for s in shown]
    return out + [fmt(vals[8])]


def main():
    span = {}
    # unrounded span-role F1 for models and humans (span_roles_table3.json)
    for key, roles in json.load(open(OUT / "span_roles_table3.json")).items():
        model, gold, batch = key.split("|")
        if batch == "dev1-12":
            for role, (f1, _std, _n) in roles.items():
                span[(gold, model, role)] = f1

    a1, a2 = body(A1_FILE), body(A2_FILE)
    a2rows = {n: v for k, n, m, v in a2 if k == "row"}
    lines, phase, acc = list(HEADER), None, []
    for kind, name, metric, v1 in a1:
        if kind == "phase":
            phase, acc = name, []
            lines.append(r"\multicolumn{11}{l}{\textit{%s}} \\" % name)
        elif kind == "hline":
            lines.append(r"\hline")
        elif kind == "avg":
            avg = []
            for i in range(9):
                xs = [r[i] for r in acc if r[i] is not None]
                avg.append(math.fsum(xs) / len(xs) if xs else None)
            lines.append(r"\rowcolor[gray]{0.92} \textit{Phase avg} & & " + " & ".join(bold_row(avg)) + r" \\")
        else:
            if False and name in ROLE:
                vals = [(span[("A1", m, ROLE[name])] + span[("A2", m, ROLE[name])]) / 2 for m in MODELS]
                vals.append((span[("-", "HUMAN_A2vsA1", ROLE[name])] + span[("-", "HUMAN_A1vsA2", ROLE[name])]) / 2)
            else:
                v2 = a2rows[name]
                vals = []
                for x, y in zip(v1, v2):
                    x, y = num(x), num(y)
                    vals.append(None if x is None and y is None else x if y is None else y if x is None else (x + y) / 2)
            lines.append(f"{name} & {metric} & " + " & ".join(bold_row(vals)) + r" \\")
            if phase != "Explanation" or name in EXPLANATION_AVG:
                acc.append(vals)
    lines += [r"\end{tabular}", r"\end{table*}"]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
