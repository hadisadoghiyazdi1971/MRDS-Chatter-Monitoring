"""Render the fold-wise Rényi and MRDS-refinement figure from saved results.

This script reads the completed 20-fold condition-disjoint outputs.  It does
not rerun data preprocessing, subset selection, refinement, or classification.
"""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "grouped_chatter_all40"
OUTPUT = ROOT / "paper_latex" / "generated"

METHODS = [
    "MRDS-IS-R (proposed)",
    "W2-KMedoids-R",
    "FacilityLocation-R",
    "Random-R",
]
METHOD_LABELS = [
    "MRDS-IS-R",
    "$W_2$-$k$-\nmedoids-R",
    "Facility\nLocation-R",
    "Random-R",
]
METHOD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#666666"]
METHOD_MARKERS = ["o", "s", "^", "D"]

STAGE_COLUMNS = [
    "J_projected_uniform",
    "J_projected_optimized_weights",
    "mrds_refined_only_objective",
]
STAGE_LABELS = [
    "Projected MRDS\n(uniform weights)",
    "Mixture-weight\noptimization",
    "One-swap\nrefinement",
]


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    representativeness = pd.read_csv(RESULTS / "representativeness.csv")
    stagewise = pd.read_csv(RESULTS / "stagewise_objectives.csv")

    selected = representativeness[
        representativeness["method"].isin(METHODS)
    ].copy()
    if set(selected["method"]) != set(METHODS):
        raise ValueError("The expected four refined selectors are not all present")
    method_counts = selected.groupby("method")["split"].nunique()
    if not (method_counts == 20).all() or len(selected) != 80:
        raise ValueError("Panel (a) requires one value per method in all 20 folds")
    if selected["renyi_meta_weighted"].isna().any():
        raise ValueError("Missing weighted finite Rényi values")
    if not (selected["renyi_meta_weighted"] > 0).all():
        raise ValueError("The logarithmic axis requires positive Rényi values")

    missing_stages = [column for column in STAGE_COLUMNS if column not in stagewise]
    if missing_stages:
        raise ValueError(f"Missing stagewise columns: {missing_stages}")
    if stagewise["split"].nunique() != 20 or len(stagewise) != 20:
        raise ValueError("Panel (b) requires exactly 20 outer-fold trajectories")
    if stagewise[STAGE_COLUMNS].isna().any().any():
        raise ValueError("Missing finite Rényi stage values")
    if not (stagewise[STAGE_COLUMNS] > 0).all().all():
        raise ValueError("The logarithmic axis requires positive stage values")
    if not (
        stagewise["J_projected_optimized_weights"]
        <= stagewise["J_projected_uniform"] + 1e-12
    ).all():
        raise ValueError("Weight optimization increases the saved objective")
    if not (
        stagewise["mrds_refined_only_objective"]
        <= stagewise["J_projected_optimized_weights"] + 1e-12
    ).all():
        raise ValueError("One-swap refinement increases the saved objective")

    mrds_final = selected[selected["method"] == "MRDS-IS-R (proposed)"][
        ["split", "renyi_meta_weighted"]
    ]
    check = stagewise[["split", "mrds_refined_only_objective"]].merge(
        mrds_final, on="split", validate="one_to_one"
    )
    if not np.allclose(
        check["mrds_refined_only_objective"],
        check["renyi_meta_weighted"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("The final MRDS stage does not match representativeness.csv")

    return selected, stagewise.sort_values("split")


def draw_figure(representativeness: pd.DataFrame, stagewise: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.25))

    method_values = [
        representativeness.loc[
            representativeness["method"] == method, "renyi_meta_weighted"
        ].to_numpy()
        for method in METHODS
    ]
    box = ax_a.boxplot(
        method_values,
        positions=np.arange(1, len(METHODS) + 1),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], METHOD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    jitter = np.linspace(-0.13, 0.13, 20)
    for position, (values, color, marker) in enumerate(
        zip(method_values, METHOD_COLORS, METHOD_MARKERS), start=1
    ):
        ax_a.scatter(
            position + jitter,
            values,
            s=18,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
    ax_a.set_yscale("log")
    ax_a.set_xticks(np.arange(1, len(METHODS) + 1), METHOD_LABELS)
    ax_a.set_ylabel(r"Finite Rényi objective $J_\alpha$")
    ax_a.grid(axis="y", which="both", alpha=0.22)
    ax_a.text(
        0.01,
        0.98,
        "(a)",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12,
    )

    x_stage = np.arange(1, len(STAGE_COLUMNS) + 1)
    stage_values = stagewise[STAGE_COLUMNS].to_numpy()
    for row in stage_values:
        ax_b.plot(
            x_stage,
            row,
            color="#9c9c9c",
            linewidth=0.75,
            alpha=0.55,
            zorder=1,
        )
        ax_b.scatter(
            x_stage,
            row,
            s=13,
            color="#6f6f6f",
            alpha=0.65,
            zorder=2,
        )
    stage_means = stage_values.mean(axis=0)
    ax_b.plot(
        x_stage,
        stage_means,
        color="black",
        linewidth=1.8,
        marker="D",
        markersize=5.3,
        zorder=4,
    )
    ax_b.set_yscale("log")
    ax_b.set_xticks(x_stage, STAGE_LABELS)
    ax_b.set_ylabel(r"Finite Rényi objective $J_\alpha$")
    ax_b.grid(axis="y", which="both", alpha=0.22)
    ax_b.text(
        0.01,
        0.98,
        "(b)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12,
    )

    fig.tight_layout(w_pad=2.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT / "grouped_renyi_refinement_panels.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "grouped_renyi_refinement_panels.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    representativeness, stagewise = load_and_validate()
    draw_figure(representativeness, stagewise)
    print(OUTPUT / "grouped_renyi_refinement_panels.pdf")
    print(OUTPUT / "grouped_renyi_refinement_panels.png")


if __name__ == "__main__":
    main()
