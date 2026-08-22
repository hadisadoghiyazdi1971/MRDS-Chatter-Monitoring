#!/usr/bin/env python3
"""Validate condition-disjoint result artifacts and selected-recording provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def validate(result_dir: Path, task_aware: bool) -> dict[str, object]:
    folds = pd.read_csv(result_dir / "folds_grouped.csv")
    selected = pd.read_csv(result_dir / "selected_recording_ids.csv")
    errors: list[str] = []
    split_reports: list[dict[str, object]] = []

    for split, rows in folds.groupby("split"):
        train = rows[rows["partition"] == "train"]
        test = rows[rows["partition"] == "test"]
        overlap = set(train["condition_id"]) & set(test["condition_id"])
        if overlap:
            errors.append(f"split {split}: condition overlap {sorted(overlap)[:5]}")
        if set(train["recording_id"]) & set(test["recording_id"]):
            errors.append(f"split {split}: recording overlap")
        if len(rows) != 160 or rows["recording_id"].nunique() != 160:
            errors.append(f"split {split}: incomplete recording partition")
        split_reports.append({
            "split": int(split),
            "n_train": len(train),
            "n_test": len(test),
            "n_train_conditions": int(train["condition_id"].nunique()),
            "n_test_conditions": int(test["condition_id"].nunique()),
            "condition_overlap": len(overlap),
        })

    training_keys = set(
        zip(
            folds.loc[folds["partition"] == "train", "split"].astype(int),
            folds.loc[folds["partition"] == "train", "recording_id"].astype(str),
        )
    )
    for row in selected.itertuples(index=False):
        if (int(row.split), str(row.recording_id)) not in training_keys:
            errors.append(
                f"selected test/unknown recording: split={row.split}, id={row.recording_id}"
            )
            break

    duplicate_rank = selected.duplicated(["split", "method", "rank"]).any()
    duplicate_recording = selected.duplicated(
        ["split", "method", "recording_id"]
    ).any()
    if duplicate_rank:
        errors.append("duplicate selection ranks")
    if duplicate_recording:
        errors.append("duplicate selected recordings")

    if task_aware:
        duplicated_condition = selected.duplicated(
            ["split", "method", "condition_id"]
        ).any()
        if duplicated_condition:
            errors.append("task-aware result violates one-recording-per-condition")
        scores = pd.read_csv(result_dir / "task_aware_results.csv")
    else:
        duplicated_condition = None
        scores = pd.read_csv(result_dir / "classification.csv")

    metric_columns = [
        "balanced_accuracy", "macro_f1", "auroc", "sensitivity", "specificity"
    ]
    if scores[metric_columns].isna().any().any():
        errors.append("missing classification metric")
    if ((scores[metric_columns] < 0.0) | (scores[metric_columns] > 1.0)).any().any():
        errors.append("classification metric outside [0,1]")

    report = {
        "result_directory": str(result_dir),
        "task_aware": task_aware,
        "n_splits": int(folds["split"].nunique()),
        "n_fold_rows": len(folds),
        "n_selected_rows": len(selected),
        "n_score_rows": len(scores),
        "duplicate_rank": bool(duplicate_rank),
        "duplicate_recording": bool(duplicate_recording),
        "duplicate_condition_in_task_aware": (
            None if duplicated_condition is None else bool(duplicated_condition)
        ),
        "splits": split_reports,
        "errors": errors,
        "valid": not errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--task-aware", action="store_true")
    args = parser.parse_args()
    report = validate(args.result_dir, args.task_aware)
    output = args.result_dir / "validation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "errors": report["errors"]}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

