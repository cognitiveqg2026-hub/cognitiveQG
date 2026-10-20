#!/usr/bin/env python3
"""Build the two per-annotator dev1-12 appendix tables with a Human column (Tables 27 and 28).

Starting from the original per-field tables (model scores against one annotator, 3-seed mean +/- std),
this script
  * adds a Human column: the other annotator scored against the same gold with the same code
      - Wt. F1 rows   : results/revision/human_wtf1_dev1-12.json   (scripts/revision/human_ceiling/human_wf1.py)
      - BERTScore rows: BERT column of the inter-annotator agreement table (table/cognitiveqg-iaa-dev1-12.tex, x100)
      - span-role rows: results/revision/span_roles_table3.json    (scripts/revision/span_roles/compute_span_roles_table3.py)
  * replaces the three "Span Eval. (span1/2/3)" BERTScore rows with span-role F1 rows (major claim, minor claim,
    premise; one-vs-rest F1 pooled over the three span slots), 3-seed mean +/- std from the same JSON;
  * appends the Human phase average (mean over the phase's Human cells; Explanation uses only
    Target Explanation, Reasoning Structure, Good Evidence, Bad Evidence).
"--" = no human value: the predicting annotator left the field empty (Revision Type, Error Detection),
or the agreement analysis has no separate value (Fallacy Span 2).

Inputs : TABLE_DIR/model-baselinescore-dev1-12_RESCALED.tex        (A1 gold, original Table 3)
         TABLE_DIR/model_baselinescore_dev1-12_kim_rescaled.tex     (A2 gold)
         TABLE_DIR/cognitiveqg-iaa-dev1-12.tex
         OUT/human_wtf1_dev1-12.json, OUT/span_roles_table3.json
Outputs: TABLE_DIR/model-baselinescore-dev1-12_A1_human_spanroles.tex          (Table 27)
         TABLE_DIR/model_baselinescore_dev1-12_kim_rescaled_human_spanroles.tex (Table 28)
Set CQG_TABLE_DIR to read/write another folder.
"""
import csv, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import OUT, TABLE_DIR

MODELS = ["llama2", "llama3", "qwen", "mistral", "olmo", "qwen3", "gpt-4o", "gpt-5.5"]
EXPLANATION_AVG = {"Target Explanation", "Reasoning Structure", "Good Evidence", "Bad Evidence"}
WTF1_KEY = {"Stance Target": "stanceTarget", "Knowledge Domain": "knowledgeDomain", "Reasoning Type": "reasoningType",
            "Has Assumption": "hasAssumption", "Primary Domain Affected": "primaryDomain",
            "Alternative Type": "alternativeType", "Inference Score": "inferenceStrength",
            "Credibility Factors": "credibilityFactors", "Logical Fallacy": "logicalFallacy",
            "Trustworthiness": "trustworthiness", "Target Explanation": "targetExplanation",
            "Reasoning Structure": "reasoningStructure", "Bias Detection": "biasDetection",
            "Heuristic Detection": "heuristicDetection", "Change Decision": "changeDecision"}
NO_HUMAN = {"Revision Type", "Error Detection", "Fallacy Span 2"}     # see docstring
IAA_NAME = {r"Inductive/Ded.\ Term X": "Inductive/Ded. Term X", r"Inductive/Ded.\ Term Y": "Inductive/Ded. Term Y",
            "Fallacy Span 1": "Fallacy Span"}
ROLES = [("Major claim", "major-claim"), ("Minor claim", "minor-claim"), ("Premise", "premise")]
SPAN_NOTE = (" Span role rows: one-vs-rest F1 for each argumentative role, pooled over the three span slots"
             " of the reasoning structure.}")

TABLES = [
    dict(base="model-baselinescore-dev1-12_RESCALED.tex", out="model-baselinescore-dev1-12_A1_human_spanroles.tex",
         gold="A1", direction="A2vsA1", human_header="A2 vs A1", keep_context=True,
         label=r"\label{tab:model-baseline-a1-human}",
         caption=r"""\caption{Per-field model-vs-gold scores by cognitive phase on the dev1--12 batch ($n=120$, A1 gold; P1 prompt, mean\,$\pm$\,std over 3 inference seeds: 44, 45, 46). Wt.\,F1 = support-weighted F1 (weighted macro-F1 over classes; multi-label fields scored over the label-set indicator); BERTScore = BERTScore F1 (roberta-large). \textbf{Bold} = best per row. Shaded rows = phase-level averages over fields with valid predictions. Human = annotator A2 scored against the A1 gold as a reference ceiling (single annotation, no seeds; not counted for bold): Wt.\,F1 and Span Eval rows use the same scoring code as the models, and the other BERTScore rows are the rescaled BERTScore values from the inter-annotator agreement analysis (Table~\ref{tab:iaa-human}). ``--'' = no human reference available (A2 left the field empty, or no BERTScore agreement was computed for that field).}"""),
    dict(base="model_baselinescore_dev1-12_kim_rescaled.tex",
         out="model_baselinescore_dev1-12_kim_rescaled_human_spanroles.tex",
         gold="A2", direction="A1vsA2", human_header="A1 vs A2", keep_context=False,
         label=r"\label{tab:model-baseline-jac-gwet-RunP1structured-a2}",
         caption=r"""\caption{Per-field model-vs-gold scores by cognitive phase on the dev1--12 batch ($n=120$, A2 gold; P1 prompt, mean\,$\pm$\,std over 3 inference seeds: 44, 45, 46). Wt.\,F1 = support-weighted F1 (weighted macro-F1 over classes; multi-label fields scored over the label-set indicator); BERTScore = BERTScore F1 (roberta-large). \textbf{Bold} = best per row. Shaded rows = phase-level averages over fields with valid predictions. Human = annotator A1 scored against the A2 gold (single annotation, no seeds; not counted for bold): Wt.\,F1 and Span Eval rows use the same scoring code as the models, and the other BERTScore rows are the rescaled BERTScore values from the inter-annotator agreement analysis (Table~\ref{tab:iaa-human}). ``--'' = no reference available (A2 left the field empty, or no BERTScore agreement was computed for that field).}"""),
]


SPAN_EVAL = json.load(open(OUT / "span_eval_bertscore.json"))


def iaa_bert():
    vals = {}
    for l in open(TABLE_DIR / "cognitiveqg-iaa-dev1-12.tex", encoding="utf-8"):
        if l.startswith("%") or "& BERT &" not in l:
            continue
        c = [x.strip() for x in l.split("&")]
        name = re.sub(r"^\\rowcolor\[gray\]\{[0-9.]+\}", "", c[0]).strip()
        vals[name] = None if c[6] == "--" else round(float(c[6]) * 100, 1)
    return vals


def strip(s):
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    return re.sub(r"\{\\tiny\$\\pm\$[0-9.]+\}", "", s).strip()


def fmt(x):
    return "--" if x is None else f"{x:.1f}"


def build(t, human_wtf1, iaa, span):
    lines = open(TABLE_DIR / t["base"], encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(r"\begin{table*}"))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith(r"\end{table*}"))
    out, phase, acc, hlines, in_tab = [], None, [], 0, False
    for l in lines[start:end + 1]:
        if l.startswith(r"\caption{"):
            out.append(t["caption"]); continue
        if l.startswith(r"\label{"):
            out.append(t["label"]); continue
        if l.startswith(r"\begin{tabular}"):
            out.append(r"\begin{tabular}{p{3.2cm} l c c c c c c c c | c}"); in_tab = True; continue
        if not in_tab or l.startswith(r"\end{tabular}"):
            out.append(l); in_tab = in_tab and not l.startswith(r"\end{tabular}"); continue
        if hlines < 2 and not l.startswith(r"\multicolumn{10}{l}"):
            if l.startswith(r"\textbf{Annotation Field}"):
                l = l[:-2].rstrip() + r" & \textbf{Human} \\"
            elif l.startswith(r"& & \textbf{LLaMA-2}"):
                l = l[:-2].rstrip() + r" & \textbf{" + t["human_header"] + r"} \\"
            hlines += l.strip() == r"\hline"
            out.append(l); continue
        if l.startswith(r"\multicolumn{10}{l}"):
            phase, acc = re.search(r"\\textit\{([^}]*)\}", l).group(1), []
            out.append(l.replace(r"\multicolumn{10}", r"\multicolumn{11}")); continue
        if l.strip() == r"\hline":
            out.append(l); continue
        if "Phase avg" in l:
            xs = [x for x in acc if x is not None]
            h = math.fsum(xs) / len(xs) if xs else None
            out.append(l[:-2].rstrip() + " & " + fmt(h) + r" \\"); continue
        name = l.split(" & ")[0].strip()
        if name.startswith(r"Span Eval.\ (span"):
            # Span Eval BERTScore row kept; Human = other annotator scored with the same code (span_eval_bertscore.json)
            field = "span" + name[-2]
            hv = SPAN_EVAL["dev1-12"]["HUMAN_" + t["direction"]][field]
            out.append(l[:-2].rstrip() + " & " + fmt(None if hv is None else round(hv, 1)) + r" \\"); continue
        if False:
            if name.endswith("(span1)"):
                for disp, role in ROLES:
                    vals = [span[(t["gold"], m, role)] for m in MODELS]
                    shown = [f"{v[0]:.1f}" for v in vals]
                    best = max(float(s) for s in shown)
                    cells = [(r"\textbf{%s}" % s if float(s) == best else s) + r"{\tiny$\pm$%.1f}" % v[1]
                             for s, v in zip(shown, vals)]
                    hv = span[("-", "HUMAN_" + t["direction"], role)][0]
                    out.append(f"Span role: {disp} & F1 & " + " & ".join(cells) + f" & {hv:.1f} " + "\\\\")
            continue
        if name in NO_HUMAN:
            h = None
        elif name in WTF1_KEY:
            v = human_wtf1[t["direction"]][WTF1_KEY[name]]["wtf1"]
            h = None if v is None else round(v, 1)
        else:
            h = iaa[IAA_NAME.get(name, name)]
        if phase != "Explanation" or name in EXPLANATION_AVG:
            acc.append(h)
        out.append(l[:-2].rstrip() + " & " + fmt(h) + r" \\")
    if t["keep_context"]:
        out = lines[:start] + out + lines[end + 1:]
        text = "\n".join(out)
    else:
        text = "\n".join(out) + "\n"
    (TABLE_DIR / t["out"]).write_text(text, encoding="utf-8")
    print("wrote", TABLE_DIR / t["out"])


def main():
    human_wtf1 = json.load(open(OUT / "human_wtf1_dev1-12.json"))
    span = {}
    # unrounded 3-seed mean/std per model and gold, plus the human values, from span_roles_table3.json
    for key, roles in json.load(open(OUT / "span_roles_table3.json")).items():
        model, gold, batch = key.split("|")
        if batch == "dev1-12":
            for role, (f1, std, _n) in roles.items():
                span[(gold, model, role)] = (f1, std)
    # human cells use the same JSON (keys "HUMAN_A2vsA1|-|dev1-12" and "HUMAN_A1vsA2|-|dev1-12")
    iaa = iaa_bert()
    for t in TABLES:
        build(t, human_wtf1, iaa, span)


if __name__ == "__main__":
    main()
