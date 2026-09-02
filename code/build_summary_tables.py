#!/usr/bin/env python3
"""Build the primary and retention tables from supplied fold-wise results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results" / "primary_10pct"
RETENTION = ROOT / "results" / "structured_retention_10_50"
DEFAULT_OUTPUT = ROOT / "outputs" / "summary"
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
BUDGET_TO_SIZE = {0.10: 12, 0.20: 24, 0.30: 36, 0.40: 48, 0.50: 60}
METHODS = ("MRDS", "W2-k-medoids", "Facility Location")
DISPLAY_NAME = {
    "MRDS": "MRDS",
    "W2-k-medoids": "Wasserstein k-medoids",
    "Facility Location": "Facility Location",
}


def aggregate_classification(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_balanced_accuracy": float(frame.balanced_accuracy.mean()),
        "mean_sensitivity": float(frame.sensitivity.mean()),
        "mean_specificity": float(frame.specificity.mean()),
        "minimum_foldwise_balanced_accuracy": float(
            frame.balanced_accuracy.min()
        ),
    }


def primary_table() -> pd.DataFrame:
    classification = pd.read_csv(
        PRIMARY / "structured_foldwise_classification_10pct.csv"
    )
    objectives = pd.read_csv(
        PRIMARY / "structured_foldwise_objectives_10pct.csv"
    )
    random_classification = pd.read_csv(
        PRIMARY / "random_foldwise_classification_10pct.csv"
    )
    random_objectives = pd.read_csv(
        PRIMARY / "random_foldwise_objectives_10pct.csv"
    )

    rows: list[dict[str, Any]] = []
    reference = classification[
        classification.method == "Full training reference"
    ]
    rows.append(
        {
            "method": "Full training",
            "J_alpha_S": np.nan,
            **aggregate_classification(reference),
        }
    )
    for method in METHODS:
        class_rows = classification[classification.method == method]
        objective_rows = objectives[objectives.method == method]
        rows.append(
            {
                "method": DISPLAY_NAME[method],
                "J_alpha_S": float(objective_rows.uniform_J_alpha.mean()),
                **aggregate_classification(class_rows),
            }
        )
    rows.append(
        {
            "method": "Random",
            "J_alpha_S": float(random_objectives.uniform_J_alpha.mean()),
            **aggregate_classification(random_classification),
        }
    )
    return pd.DataFrame(rows)


def retention_table() -> pd.DataFrame:
    classification = pd.read_csv(
        RETENTION / "foldwise_classification.csv"
    )
    objectives = pd.read_csv(RETENTION / "foldwise_objectives.csv")
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for method in METHODS:
            class_rows = classification[
                np.isclose(classification.budget, budget)
                & (classification.method == method)
            ]
            objective_rows = objectives[
                np.isclose(objectives.budget, budget)
                & (objectives.method == method)
            ]
            rows.append(
                {
                    "retention_percent": int(round(100 * budget)),
                    "m": BUDGET_TO_SIZE[budget],
                    "method": DISPLAY_NAME[method],
                    "mean_J_alpha_S": float(
                        objective_rows.uniform_J_alpha.mean()
                    ),
                    "mean_balanced_accuracy": float(
                        class_rows.balanced_accuracy.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    primary_table().to_csv(
        args.output / "table_primary_10pct.csv", index=False
    )
    retention_table().to_csv(
        args.output / "table_retention_10_50.csv", index=False
    )


if __name__ == "__main__":
    main()
