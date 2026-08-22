"""Create the manuscript assets for the frozen grouped retention curve.

This script does not rerun model selection.  It renders the already completed
20-fold, condition-disjoint evaluation stored in
``results/grouped_budget_curve_all40`` and fails loudly if the expected folds,
budgets, methods, or multiplicity-adjusted conclusions are missing.
"""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({"pdf.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "grouped_budget_curve_all40"
OUTPUT = ROOT / "paper_latex" / "generated"

METHODS = ["MRDS-IS-R", "W2-KMedoids-R", "FacilityLocation-R"]
LABELS = {
    "MRDS-IS-R": r"MRDS-IS-R",
    "W2-KMedoids-R": r"$W_2$-$k$-medoids-R",
    "FacilityLocation-R": r"Facility Location-R",
}
EXPECTED_BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5]


def validate(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    observed_budgets = sorted(summary["budget"].unique().tolist())
    if observed_budgets != EXPECTED_BUDGETS:
        raise ValueError(f"Unexpected budgets: {observed_budgets}")
    if set(summary["method"]) != set(METHODS):
        raise ValueError(f"Unexpected methods: {sorted(summary['method'].unique())}")
    if not (summary["folds"] == 20).all():
        raise ValueError("Every curve point must contain all 20 outer folds")
    if len(paired) != 10:
        raise ValueError("Expected five budgets times two MRDS comparisons")
    if paired["significant_at_0.05"].astype(str).str.lower().eq("true").any():
        raise ValueError("A Holm-significant comparison exists; revise the caption")


def write_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Balanced accuracy across the fixed retention grid over 20 condition-disjoint outer folds. Values are mean $\pm$ standard deviation. Bold marks the highest reduced-set mean in each row and does not indicate statistical significance.}",
        r"\label{tab:grouped-budget-curve}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ccclll}",
        r"\toprule",
        r"Retention & Reduction & Retained & MRDS-IS-R & $W_2$-$k$-medoids-R & Facility Location-R\\",
        r"\midrule",
    ]
    for budget in EXPECTED_BUDGETS:
        block = summary[summary["budget"] == budget].set_index("method")
        leader = block["mean_balanced_accuracy"].idxmax()
        values: list[str] = []
        for method in METHODS:
            row = block.loc[method]
            value = (
                f"${row['mean_balanced_accuracy']:.4f}"
                rf"\pm{row['std_balanced_accuracy']:.4f}$"
            )
            if method == leader:
                value = rf"$\mathbf{{{row['mean_balanced_accuracy']:.4f}\pm{row['std_balanced_accuracy']:.4f}}}$"
            values.append(value)
        lines.append(
            f"{int(100 * budget)}\\% & {int(100 * (1 - budget))}\\% & "
            f"{round(120 * budget):d} & "
            + " & ".join(values)
            + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT / "grouped_budget_curve.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_figure(summary: pd.DataFrame) -> None:
    styles = {
        "MRDS-IS-R": ("#1f77b4", "o"),
        "W2-KMedoids-R": ("#d62728", "s"),
        "FacilityLocation-R": ("#2ca02c", "^"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in METHODS:
        block = summary[summary["method"] == method].sort_values("budget")
        color, marker = styles[method]
        ax.errorbar(
            100 * block["budget"],
            block["mean_balanced_accuracy"],
            yerr=block["std_balanced_accuracy"],
            color=color,
            marker=marker,
            linewidth=2,
            capsize=3,
            label=LABELS[method],
        )
    ax.axhline(
        0.950744,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Full training reference",
    )
    ax.set_xlabel("Retention of training recordings (%)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_ylim(0.50, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT / "grouped_budget_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "aggregate_scores.csv")
    paired = pd.read_csv(RESULTS / "paired_comparisons.csv")
    validate(summary, paired)
    write_table(summary)
    write_figure(summary)
    print(OUTPUT / "grouped_budget_curve.tex")
    print(OUTPUT / "grouped_budget_curve.pdf")


if __name__ == "__main__":
    main()
