from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "semantic_error_analysis"
FIG_DIR = OUT_DIR / "figures"
SEM_DIR = (
    ROOT
    / "notebooks"
    / "evaluation"
    / "structures"
    / "evaluation_outputs"
    / "semantic_error_analysis"
)
LENGTH_PATH = (
    ROOT
    / "notebooks"
    / "evaluation"
    / "gold_length"
    / "length_analysis"
    / "generated_summary_token_length_summary.csv"
)

MODEL_NAMES = {
    "aya32b": "Aya-Expanse 32B",
    "gemma27b": "Gemma 3 27B",
    "qwen27b": "Qwen 3.5 27B",
    "aya": "Aya-Expanse 32B",
    "gemma": "Gemma 3 27B",
    "qwen": "Qwen 3.5 27B",
}

PIPELINE_NAMES = {
    "direct": "Direct",
    "summarize_then_translate": "ST",
    "translate_then_summarize": "TS",
    "semantic": "Semantic",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def tex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_pipeline_scores() -> pd.DataFrame:
    rows = []
    for path in (ROOT / "notebooks" / "agents" / "results" / "full").glob(
        "*/*/eval/*omniscore_summary.csv"
    ):
        parts = path.relative_to(ROOT).parts
        pipeline = parts[4]
        model = parts[5]
        row = pd.read_csv(path).query("mode == 'source_grounded'").iloc[0]
        rows.append(
            {
                "pipeline": pipeline,
                "pipeline_name": PIPELINE_NAMES[pipeline],
                "model": model,
                "model_name": MODEL_NAMES[model],
                "informativeness": row["informativeness_mean"],
                "clarity": row["clarity_mean"],
                "plausibility": row["plausibility_mean"],
                "faithfulness": row["faithfulness_mean"],
            }
        )
    return pd.DataFrame(rows)


def load_length_scores() -> pd.DataFrame:
    length = pd.read_csv(LENGTH_PATH, encoding="utf-8-sig")
    gold_len = float(
        length.loc[length["Model"].eq("Golden Length"), "Generated summary length"].iloc[0]
    )
    rows = []
    for _, row in length.iterrows():
        label = row["Model"]
        if "Qwen" in label:
            model = "qwen"
        elif "Gemma" in label:
            model = "gemma"
        elif "Aya" in label:
            model = "aya"
        else:
            continue

        if "(Dir)" in label:
            pipeline = "direct"
        elif "(ST)" in label:
            pipeline = "summarize_then_translate"
        elif "(TS)" in label:
            pipeline = "translate_then_summarize"
        elif "(Sem)" in label:
            pipeline = "semantic"
        else:
            continue

        rows.append(
            {
                "model": model,
                "pipeline": pipeline,
                "avg_length": float(row["Generated summary length"]),
                "gold_abs_diff": abs(float(row["Generated summary length"]) - gold_len),
            }
        )
    return pd.DataFrame(rows), gold_len


def plot_pipeline_informativeness(pipeline: pd.DataFrame) -> None:
    order_models = ["aya", "gemma", "qwen"]
    order_pipes = ["direct", "summarize_then_translate", "translate_then_summarize", "semantic"]
    colors = ["#5B7C99", "#80A05A", "#D08A45", "#B04A5A"]
    x = np.arange(len(order_models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for i, pipe in enumerate(order_pipes):
        vals = [
            float(
                pipeline.query("model == @model and pipeline == @pipe")[
                    "informativeness"
                ].iloc[0]
            )
            for model in order_models
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=PIPELINE_NAMES[pipe], color=colors[i])
    ax.set_ylabel("Source-grounded informativeness")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in order_models], rotation=8, ha="right")
    ax.set_ylim(2.35, 3.15)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "pipeline_informativeness_by_model")


def plot_field_heatmap(score_matrix: pd.DataFrame) -> None:
    fields = [
        "schema_validity",
        "participants",
        "semantic_grounding.overall",
        "semantic_grounding.event_coverage",
        "semantic_grounding.event_granularity",
        "semantic_grounding.speech_act",
        "semantic_grounding.intended_meaning",
        "semantic_grounding.actor",
        "semantic_grounding.action",
        "semantic_grounding.object",
        "semantic_grounding.recipient",
        "semantic_grounding.evidence",
        "final_outcome",
    ]
    labels = [
        "Schema validity",
        "Participants",
        "Overall grounding",
        "Event coverage",
        "Event granularity",
        "Speech act",
        "Intended meaning",
        "Actor",
        "Action",
        "Object",
        "Recipient",
        "Evidence",
        "Final outcome",
    ]
    data = (
        score_matrix.set_index("schema_field")
        .loc[fields, ["aya32b", "gemma27b", "qwen27b"]]
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(data.values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([MODEL_NAMES[c] for c in data.columns], rotation=12, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            ax.text(
                j,
                i,
                fmt(val),
                ha="center",
                va="center",
                color="white" if val < 0.45 else "black",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Quality score")
    fig.tight_layout()
    save_fig(fig, "semantic_schema_field_score_heatmap")


def plot_role_slot_accuracy(slot_summary: pd.DataFrame) -> None:
    roles = ["speaker", "actor", "action", "object", "recipient"]
    x = np.arange(len(roles))
    width = 0.25
    colors = ["#5B7C99", "#80A05A", "#D08A45"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, model in enumerate(["aya32b", "gemma27b", "qwen27b"]):
        row = slot_summary.query("model == @model").iloc[0]
        vals = [float(row[f"{role}_slot_accuracy"]) for role in roles]
        ax.bar(x + (i - 1) * width, vals, width, label=MODEL_NAMES[model], color=colors[i])
    ax.set_ylabel("Slot accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "semantic_role_slot_accuracy")


def plot_error_counts(error_counts: pd.DataFrame) -> None:
    metrics = [
        ("parse_errors", "Parse errors"),
        ("over_segmented_examples", "Over-segmented"),
        ("under_segmented_examples", "Under-segmented"),
        ("examples_with_omitted_events", "Omitted events"),
        ("examples_with_extra_events", "Extra events"),
        ("low_final_outcome_examples", "Low final outcome"),
    ]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#5B7C99", "#80A05A", "#D08A45"]
    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    for i, model in enumerate(["aya32b", "gemma27b", "qwen27b"]):
        row = error_counts.query("model == @model").iloc[0]
        vals = [int(row[m[0]]) for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=MODEL_NAMES[model], color=colors[i])
    ax.set_ylabel("Examples out of 50")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], rotation=18, ha="right")
    ax.set_ylim(0, 52)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "semantic_error_counts_by_model")


def make_table(tabular: list[list[str]], header: list[str], align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in tabular:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_latex(
    summary: pd.DataFrame,
    error_counts: pd.DataFrame,
    slot_summary: pd.DataFrame,
    score_matrix: pd.DataFrame,
    pipeline: pd.DataFrame,
    lengths: pd.DataFrame,
    gold_len: float,
) -> str:
    merged = pipeline.merge(lengths, on=["model", "pipeline"], how="left")
    order_models = ["aya", "gemma", "qwen"]
    order_pipes = ["direct", "summarize_then_translate", "translate_then_summarize", "semantic"]

    rows = []
    for model in order_models:
        for pipe in order_pipes:
            row = merged.query("model == @model and pipeline == @pipe").iloc[0]
            rows.append(
                [
                    MODEL_NAMES[model],
                    PIPELINE_NAMES[pipe],
                    fmt(row["informativeness"]),
                    fmt(row["faithfulness"]),
                    fmt(row["avg_length"], 2),
                    fmt(row["gold_abs_diff"], 2),
                ]
            )
    pipeline_table = make_table(
        rows,
        [
            "Model",
            "Pipeline",
            "Info.",
            "Faith.",
            "Avg. len.",
            "$|\\Delta|$ from gold",
        ],
        "llrrrr",
    )

    rows = []
    for model in ["aya32b", "gemma27b", "qwen27b"]:
        s = summary.query("model == @model").iloc[0]
        c = error_counts.query("model == @model").iloc[0]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(s["parse_error_rate"]),
                fmt(s["participant_f1"]),
                fmt(s["event_coverage"]),
                fmt(s["extra_event_rate"]),
                fmt(s["event_count_diff_mean"]),
                f'{int(c["examples_with_omitted_events"])}/50',
                f'{int(c["examples_with_extra_events"])}/50',
                fmt(s["informativeness"]),
                fmt(s["faithfulness"]),
            ]
        )
    semantic_profile_table = make_table(
        rows,
        [
            "Model",
            "Parse err.",
            "Part. F1",
            "Coverage",
            "Extra rate",
            "Count diff",
            "Omitted",
            "Extra",
            "Info.",
            "Faith.",
        ],
        "lrrrrrrrrr",
    )

    rows = []
    for model in ["aya32b", "gemma27b", "qwen27b"]:
        row = error_counts.query("model == @model").iloc[0]
        rows.append(
            [
                MODEL_NAMES[model],
                f'{int(row["parse_errors"])}/50',
                f'{int(row["over_segmented_examples"])}/50',
                f'{int(row["under_segmented_examples"])}/50',
                f'{int(row["examples_with_omitted_events"])}/50',
                f'{int(row["examples_with_extra_events"])}/50',
                f'{int(row["low_final_outcome_examples"])}/50',
            ]
        )
    error_count_table = make_table(
        rows,
        [
            "Model",
            "Parse",
            "Over-seg.",
            "Under-seg.",
            "Omitted",
            "Extra",
            "Low outcome",
        ],
        "lrrrrrr",
    )

    rows = []
    for model in ["aya32b", "gemma27b", "qwen27b"]:
        row = slot_summary.query("model == @model").iloc[0]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(row["speaker_slot_accuracy"]),
                fmt(row["actor_slot_accuracy"]),
                fmt(row["action_slot_accuracy"]),
                fmt(row["object_slot_accuracy"]),
                fmt(row["recipient_slot_accuracy"]),
                fmt(row["actor_null_rate"]),
                fmt(row["action_null_rate"]),
            ]
        )
    slot_table = make_table(
        rows,
        [
            "Model",
            "Speaker",
            "Actor",
            "Action",
            "Object",
            "Recipient",
            "Actor null",
            "Action null",
        ],
        "lrrrrrrr",
    )

    rows = []
    key_fields = [
        ("semantic_grounding.overall", "Overall grounding"),
        ("semantic_grounding.event_coverage", "Event coverage"),
        ("semantic_grounding.event_granularity", "Event granularity"),
        ("semantic_grounding.speech_act", "Speech act"),
        ("semantic_grounding.intended_meaning", "Intended meaning"),
        ("semantic_grounding.action", "Action role"),
        ("final_outcome", "Final outcome"),
    ]
    mat = score_matrix.set_index("schema_field")
    for field, label in key_fields:
        row = mat.loc[field]
        rows.append(
            [
                label,
                fmt(row["aya32b"]),
                fmt(row["gemma27b"]),
                fmt(row["qwen27b"]),
            ]
        )
    field_table = make_table(
        rows,
        ["Schema field", "Aya", "Gemma", "Qwen"],
        "lrrr",
    )

    aya_sem = merged.query("model == 'aya' and pipeline == 'semantic'").iloc[0]
    gemma_sem = merged.query("model == 'gemma' and pipeline == 'semantic'").iloc[0]
    qwen_sem = merged.query("model == 'qwen' and pipeline == 'semantic'").iloc[0]

    return f"""% Requires: \\usepackage{{booktabs}}

\\subsection{{Weak Semantic Agentic Pipeline}}

One particularly notable finding is that the semantic workflow consistently produces the weakest results among the agentic variants. In the source-grounded Omniscore evaluation, the semantic pipeline obtains the lowest informativeness score for all three instruction-tuned models: {fmt(aya_sem['informativeness'])} for Aya-Expanse 32B, {fmt(gemma_sem['informativeness'])} for Gemma 3 27B, and {fmt(qwen_sem['informativeness'])} for Qwen 3.5 27B. By comparison, the best non-semantic agentic variant reaches 3.066 for Aya, 3.032 for Gemma, and 2.974 for Qwen. Faithfulness shows a similar but less uniform pattern: the semantic pipeline scores {fmt(aya_sem['faithfulness'])} for Aya, {fmt(gemma_sem['faithfulness'])} for Gemma, and {fmt(qwen_sem['faithfulness'])} for Qwen.

This outcome is initially surprising because the semantic pipeline represents the most sophisticated design. It explicitly extracts pragmatic and semantic structures, including participant information, speech acts, semantic roles, evidence grounding, and contextual outcomes before generating summaries. However, the intermediate schema also introduces a bottleneck: once Agent 1 compresses the dialogue into a fixed representation, omitted events, weak role assignments, or incorrect pragmatic labels become difficult for downstream agents to recover.

The length statistics further show that lower end-task quality is not simply caused by verbosity. The gold summaries average {fmt(gold_len, 2)} tokens. Semantic summaries are closest to this target for all three model families, with absolute length gaps of {fmt(aya_sem['gold_abs_diff'], 2)} tokens for Aya, {fmt(gemma_sem['gold_abs_diff'], 2)} for Gemma, and {fmt(qwen_sem['gold_abs_diff'], 2)} for Qwen. Thus, the semantic pipeline produces appropriately concise outputs, but its content selection and semantic preservation remain weaker.

\\begin{{table*}}[t]
\\centering
\\small
\\caption{{Source-grounded Omniscore and length statistics across agentic pipelines. Lower $|\\Delta|$ means closer to the gold-summary average length of {fmt(gold_len, 2)} tokens.}}
\\label{{tab:pipeline-omniscore-length}}
{pipeline_table}
\\end{{table*}}

\\subsection{{Event Extraction Performance Across Models}}

Table~\\ref{{tab:semantic-model-profile}} summarizes the Agent~1 semantic representation errors. All three models identify participants reliably, with participant F1 above 0.989, so speaker discovery is not the main source of downstream weakness. The main differences appear in event grounding and semantic role extraction.

Aya-Expanse 32B has the highest event coverage (0.849), but this comes with severe over-segmentation: 44/50 examples are over-segmented, the extra-event rate is 0.458, and the mean event-count difference is 1.939. In other words, Aya often captures reference events, but it adds too many additional semantic events. This helps explain why its extracted representations are broad but noisy.

Gemma 3 27B has the strongest event granularity score (0.939) and the lowest extra-event rate (0.249), indicating better control over event segmentation. Its weakness is omission: event coverage drops to 0.677, omitted reference events appear in 25/50 examples, and parse errors occur in 6/50 examples. Gemma therefore tends to produce cleaner event boundaries, but misses more gold semantic content.

Qwen 3.5 27B is the most balanced semantic extractor. It has no parse errors, high event coverage (0.821), relatively low extra-event rate (0.271), and the best overall semantic-grounding score (0.810). Its remaining weaknesses are more local: action role accuracy is only 0.424, speech-act accuracy is 0.682, and final-outcome similarity remains low at 0.485.

\\begin{{table*}}[t]
\\centering
\\small
\\caption{{Model-level semantic representation profile for Agent~1. Info. and Faith. are source-grounded Omniscore means for the final semantic-pipeline summaries.}}
\\label{{tab:semantic-model-profile}}
{semantic_profile_table}
\\end{{table*}}

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Major error counts in the semantic representations. Counts are reported as examples out of 50.}}
\\label{{tab:semantic-error-counts}}
{error_count_table}
\\end{{table}}

\\subsection{{Semantic Role and Field-Level Weaknesses}}

The slot-level analysis shows that action extraction is the weakest semantic-role field for all models. Aya is especially weak on action roles, with action slot accuracy of 0.179 and actor slot accuracy of 0.315. Gemma and Qwen avoid missing action fields entirely, but their action labels are still often semantically mismatched, with action accuracies of 0.433 and 0.424 respectively. Qwen is strongest on entity-like roles, especially actor (0.727), object (0.690), and recipient (0.859), while all models remain strong on speaker attribution.

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Slot accuracy and null rates for aligned semantic events.}}
\\label{{tab:semantic-slot-accuracy}}
{slot_table}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\small
\\caption{{Selected schema-field quality scores. Higher is better.}}
\\label{{tab:schema-field-scores}}
{field_table}
\\end{{table}}
"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SEM_DIR / "semantic_model_error_summary_by_model.csv")
    error_counts = pd.read_csv(SEM_DIR / "semantic_model_error_type_counts.csv")
    slot_summary = pd.read_csv(SEM_DIR / "semantic_model_slot_summary.csv")
    score_matrix = pd.read_csv(SEM_DIR / "semantic_schema_field_score_matrix.csv")
    pipeline = load_pipeline_scores()
    lengths, gold_len = load_length_scores()

    plot_pipeline_informativeness(pipeline)
    plot_error_counts(error_counts)
    plot_role_slot_accuracy(slot_summary)
    plot_field_heatmap(score_matrix)

    latex = build_latex(
        summary=summary,
        error_counts=error_counts,
        slot_summary=slot_summary,
        score_matrix=score_matrix,
        pipeline=pipeline,
        lengths=lengths,
        gold_len=gold_len,
    )
    (OUT_DIR / "semantic_error_analysis_latex_fragment.tex").write_text(latex)


if __name__ == "__main__":
    main()
