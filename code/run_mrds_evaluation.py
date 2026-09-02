#!/usr/bin/env python3
"""Reproduce the condition-disjoint MRDS evaluation from processed features."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

from mrds_core import (
    EmpiricalMeasure,
    compress_bag,
    facility_location,
    finite_meta_p,
    median_bandwidth,
    pairwise_emd_exact,
    pam_kmedoids,
    rbf_kernel_from_distance,
    uniform_subset_objective,
)
from synthetic_projection import UnresolvedCapError, synthetic_projection


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_feature_matrices"
FOLDS_FILE = ROOT / "reproducibility" / "folds_grouped.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "reproduction"

ALPHA = 2.0
SUPPORT_ATOMS = 8
BUDGET_TO_SIZE = {0.10: 12, 0.20: 24, 0.30: 36, 0.40: 48, 0.50: 60}
STRUCTURED_METHODS = ("MRDS", "W2-k-medoids", "Facility Location")
RANDOM_SEED_BY_REPEAT = {0: 101, 1: 211, 2: 307, 3: 401, 4: 503}
SYNTHETIC_RELATIVE_TOLERANCE = 1e-3
SYNTHETIC_REQUIRED_SWEEPS = 3
SYNTHETIC_SAFETY_CAP = 30
EXPECTED_FOLDS_SHA256 = (
    "147d95906e49863ccb711528246c66f2479b6f07f71d61556990359a6ef6a179"
)
IGNORED_FIELDS = {"FileName", "Label"}
RECORDING_PATTERN = re.compile(
    r"^(?P<state>[SU])_WP(?P<workpiece>[AB])_L(?P<length>[0-9.]+)"
    r"_DOC(?P<doc>[0-9.]+)_WOC(?P<woc>[0-9.]+)"
    r"_N(?P<speed>[0-9]+)_F(?P<feed>[0-9]+)_R(?P<repetition>[1-4])$",
    flags=re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> float:
    return float(np.asarray(value).squeeze())


def load_feature_matrices(
    directory: Path,
) -> tuple[list[np.ndarray], np.ndarray, list[str], list[str]]:
    """Load the 160 processed recording-level feature matrices."""

    files = sorted(directory.glob("*.mat"))
    if len(files) != 160:
        raise RuntimeError(f"expected 160 MAT files, found {len(files)}")
    first = sio.loadmat(files[0], squeeze_me=True, struct_as_record=False)[
        "SigData"
    ]
    fields = sorted(
        field
        for field in dir(first[0])
        if not field.startswith("_") and field not in IGNORED_FIELDS
    )
    if len(fields) != 40:
        raise RuntimeError(f"expected 40 descriptors, found {len(fields)}")

    matrices: list[np.ndarray] = []
    labels: list[int] = []
    recording_ids: list[str] = []
    for path in files:
        signal_data = sio.loadmat(
            path, squeeze_me=True, struct_as_record=False
        )["SigData"]
        matrix = np.asarray(
            [
                [scalar(getattr(window, field)) for field in fields]
                for window in signal_data
            ],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != 40:
            raise RuntimeError(f"invalid feature matrix in {path.name}: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite feature value in {path.name}")
        matrices.append(matrix)
        labels.append(1 if path.name.upper().startswith("U_") else 0)
        recording_ids.append(path.stem)
    return matrices, np.asarray(labels, dtype=np.int64), recording_ids, fields


def condition_id(recording_id: str) -> str:
    if RECORDING_PATTERN.fullmatch(recording_id) is None:
        raise ValueError(f"unexpected recording identifier: {recording_id}")
    return re.sub(r"_R[1-4]$", "", recording_id, flags=re.IGNORECASE)


def fold_indices(
    folds: pd.DataFrame, split: int
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    rows = folds[folds["split"].astype(int) == split]
    train = np.sort(
        rows[rows.partition == "train"].global_index.to_numpy(np.int64)
    )
    test = np.sort(
        rows[rows.partition == "test"].global_index.to_numpy(np.int64)
    )
    if train.size != 120 or test.size != 40:
        raise RuntimeError(
            f"split {split}: expected 120 training and 40 test recordings"
        )
    return (
        train,
        test,
        int(rows.seed.iloc[0]),
        int(rows.repeat.iloc[0]),
        int(rows.fold.iloc[0]),
    )


def fit_robust_transform(
    training_matrices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate(training_matrices, axis=0)
    center = np.median(pooled, axis=0)
    mad = 1.4826 * np.median(np.abs(pooled - center), axis=0)
    standard_deviation = pooled.std(axis=0)
    scale = np.where(
        mad > 1e-12,
        mad,
        np.where(standard_deviation > 1e-12, standard_deviation, 1.0),
    )
    return center, scale


def apply_transform(
    matrices: list[np.ndarray], center: np.ndarray, scale: np.ndarray
) -> list[np.ndarray]:
    return [(matrix - center) / scale for matrix in matrices]


def recording_representation(matrices: list[np.ndarray]) -> np.ndarray:
    """Construct the 200-dimensional GaussianNB representation."""

    rows: list[np.ndarray] = []
    for matrix in matrices:
        q25, q50, q75 = np.quantile(matrix, [0.25, 0.50, 0.75], axis=0)
        rows.append(
            np.concatenate(
                [matrix.mean(axis=0), matrix.std(axis=0), q25, q50, q75]
            )
        )
    result = np.vstack(rows)
    if result.shape[1] != 200:
        raise RuntimeError(
            f"expected a 200-dimensional representation, found {result.shape[1]}"
        )
    return result


def classification_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    classifier = GaussianNB()
    classifier.fit(x_train[selected], y_train[selected])
    prediction = classifier.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(
        y_test, prediction, labels=[0, 1]
    ).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, prediction)
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "minimum_class_recall": min(sensitivity, specificity),
    }


def random_selection(n_items: int, count: int, repeat: int) -> np.ndarray:
    seed = RANDOM_SEED_BY_REPEAT[repeat]
    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(n_items, size=count, replace=False)
    ).astype(np.int64)


def prepare_fold(
    split: int,
    folds: pd.DataFrame,
    matrices: list[np.ndarray],
    labels: np.ndarray,
    recording_ids: list[str],
) -> dict[str, Any]:
    train_indices, test_indices, seed, repeat, fold = fold_indices(folds, split)
    groups = np.asarray(
        [condition_id(recording_id) for recording_id in recording_ids]
    )
    if set(groups[train_indices]).intersection(set(groups[test_indices])):
        raise RuntimeError(f"split {split}: condition leakage")

    train_raw = [matrices[int(index)] for index in train_indices]
    test_raw = [matrices[int(index)] for index in test_indices]
    center, scale = fit_robust_transform(train_raw)
    train_scaled = apply_transform(train_raw, center, scale)
    test_scaled = apply_transform(test_raw, center, scale)
    measures = [
        compress_bag(matrix, SUPPORT_ATOMS, seed + split * 1000 + local)
        for local, matrix in enumerate(train_scaled)
    ]
    distance = pairwise_emd_exact(measures)
    bandwidth = median_bandwidth(distance)
    kernel = rbf_kernel_from_distance(distance, bandwidth)
    return {
        "split": split,
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "groups": groups,
        "recording_ids": np.asarray(recording_ids),
        "measures": measures,
        "distance": distance,
        "kernel": kernel,
        "p": finite_meta_p(kernel),
        "bandwidth": bandwidth,
        "x_train": recording_representation(train_scaled),
        "x_test": recording_representation(test_scaled),
        "y_train": labels[train_indices],
        "y_test": labels[test_indices],
    }


def select_subset(
    method: str,
    count: int,
    fold_data: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    distance = fold_data["distance"]
    kernel = fold_data["kernel"]
    seed = fold_data["seed"]
    if method == "W2-k-medoids":
        return pam_kmedoids(distance, count, seed=seed), {}
    if method == "Facility Location":
        return facility_location(kernel, count), {}
    if method == "Random":
        return random_selection(120, count, fold_data["repeat"]), {}
    if method != "MRDS":
        raise ValueError(f"unknown method: {method}")

    result = synthetic_projection(
        fold_data["measures"],
        distance,
        kernel,
        n_prototypes=count,
        alpha=ALPHA,
        prototype_atoms=SUPPORT_ATOMS,
        bandwidth=fold_data["bandwidth"],
        seed=seed,
        max_sweeps=SYNTHETIC_SAFETY_CAP,
        movement_tolerance=1e-5,
        relative_improvement_tolerance=SYNTHETIC_RELATIVE_TOLERANCE,
        required_consecutive_sweeps=SYNTHETIC_REQUIRED_SWEEPS,
    )
    return result.selected, {
        "synthetic_objective_history": list(result.objective_history),
        "synthetic_sweeps": result.sweeps_completed,
        "synthetic_stop_reason": result.stop_reason,
        "accepted_updates_by_sweep": list(result.accepted_updates_by_sweep),
        "max_move_by_sweep": list(result.max_move_by_sweep),
        "relative_improvement_by_sweep": list(
            result.relative_improvement_by_sweep
        ),
    }


def evaluate_subset(
    method: str,
    budget: float,
    selected: np.ndarray,
    fold_data: dict[str, Any],
    synthetic_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    count = BUDGET_TO_SIZE[budget]
    selected = np.asarray(selected, dtype=np.int64)
    if selected.shape != (count,) or np.unique(selected).size != count:
        raise RuntimeError("the selected subset is not duplicate-free")

    train_indices = fold_data["train_indices"]
    global_selected = train_indices[selected]
    recording_ids = fold_data["recording_ids"]
    groups = fold_data["groups"]
    y_train = fold_data["y_train"]
    metrics = classification_metrics(
        fold_data["x_train"],
        y_train,
        fold_data["x_test"],
        fold_data["y_test"],
        selected,
    )
    return {
        "split": fold_data["split"],
        "repeat": fold_data["repeat"],
        "fold": fold_data["fold"],
        "outer_seed": fold_data["seed"],
        "random_seed": (
            RANDOM_SEED_BY_REPEAT[fold_data["repeat"]]
            if method == "Random"
            else None
        ),
        "budget": budget,
        "retained_recordings": count,
        "method": method,
        "selected_local_indices": "|".join(map(str, selected.tolist())),
        "selected_global_indices": "|".join(map(str, global_selected.tolist())),
        "recording_ids": "|".join(
            str(recording_ids[index]) for index in global_selected
        ),
        "condition_ids": "|".join(
            str(groups[index]) for index in global_selected
        ),
        "n_unique_conditions": len(
            {str(groups[index]) for index in global_selected}
        ),
        "n_stable": int(np.sum(y_train[selected] == 0)),
        "n_chatter": int(np.sum(y_train[selected] == 1)),
        "finite_J_alpha": uniform_subset_objective(
            fold_data["kernel"], fold_data["p"], selected, ALPHA
        ),
        "uniform_subset_mass": 1.0 / count,
        "classifier": "unweighted GaussianNB",
        **metrics,
        **synthetic_diagnostics,
    }


def full_training_reference(fold_data: dict[str, Any]) -> dict[str, Any]:
    selected = np.arange(120, dtype=np.int64)
    metrics = classification_metrics(
        fold_data["x_train"],
        fold_data["y_train"],
        fold_data["x_test"],
        fold_data["y_test"],
        selected,
    )
    return {
        "split": fold_data["split"],
        "repeat": fold_data["repeat"],
        "fold": fold_data["fold"],
        "outer_seed": fold_data["seed"],
        "budget": 1.0,
        "retained_recordings": 120,
        "method": "Full training reference",
        "classifier": "unweighted GaussianNB",
        **metrics,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def summarize(rows: pd.DataFrame, output: Path) -> None:
    objective_columns = [
        "split",
        "repeat",
        "fold",
        "outer_seed",
        "random_seed",
        "budget",
        "retained_recordings",
        "method",
        "finite_J_alpha",
        "uniform_subset_mass",
        "selected_local_indices",
        "selected_global_indices",
        "recording_ids",
    ]
    classification_columns = [
        "split",
        "repeat",
        "fold",
        "outer_seed",
        "random_seed",
        "budget",
        "retained_recordings",
        "method",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "minimum_class_recall",
        "n_unique_conditions",
        "n_stable",
        "n_chatter",
        "recording_ids",
        "classifier",
    ]
    rows[objective_columns].to_csv(
        output / "foldwise_objectives.csv", index=False
    )
    rows[classification_columns].to_csv(
        output / "foldwise_classification.csv", index=False
    )
    objective_summary = rows.groupby(
        ["budget", "retained_recordings", "method"], as_index=False
    ).agg(
        folds=("split", "count"),
        mean_J_alpha=("finite_J_alpha", "mean"),
        std_J_alpha=("finite_J_alpha", "std"),
    )
    classification_summary = rows.groupby(
        ["budget", "retained_recordings", "method"], as_index=False
    ).agg(
        folds=("split", "count"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        minimum_foldwise_BA=("balanced_accuracy", "min"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_specificity=("specificity", "mean"),
        mean_minimum_class_recall=("minimum_class_recall", "mean"),
    )
    objective_summary.to_csv(output / "objective_summary.csv", index=False)
    classification_summary.to_csv(
        output / "classification_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--splits", type=int, nargs="+", default=list(range(20))
    )
    args = parser.parse_args()

    if sha256(FOLDS_FILE).lower() != EXPECTED_FOLDS_SHA256:
        raise RuntimeError("the frozen folds file does not match the reported study")
    if any(split < 0 or split > 19 for split in args.splits):
        raise ValueError("split indices must be between 0 and 19")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrices, labels, recording_ids, feature_fields = load_feature_matrices(
        DATA_DIR
    )
    folds = pd.read_csv(FOLDS_FILE)
    rows: list[dict[str, Any]] = []
    full_reference_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for split in args.splits:
        fold_data = prepare_fold(
            split, folds, matrices, labels, recording_ids
        )
        full_reference_rows.append(full_training_reference(fold_data))
        for budget, count in BUDGET_TO_SIZE.items():
            for method in STRUCTURED_METHODS:
                try:
                    selected, diagnostics = select_subset(
                        method, count, fold_data
                    )
                except UnresolvedCapError as error:
                    write_json(
                        output / f"split_{split:02d}_synthetic_cap.json",
                        error.details,
                    )
                    raise
                rows.append(
                    evaluate_subset(
                        method, budget, selected, fold_data, diagnostics
                    )
                )
            if np.isclose(budget, 0.10):
                selected, diagnostics = select_subset("Random", count, fold_data)
                rows.append(
                    evaluate_subset(
                        "Random", budget, selected, fold_data, diagnostics
                    )
                )

    reduced = pd.DataFrame(rows).sort_values(["budget", "split", "method"])
    summarize(reduced, output)
    pd.DataFrame(full_reference_rows).sort_values("split").to_csv(
        output / "full_training_reference.csv", index=False
    )
    write_json(
        output / "run_metadata.json",
        {
            "alpha": ALPHA,
            "support_atoms": SUPPORT_ATOMS,
            "retention_budgets": BUDGET_TO_SIZE,
            "synthetic_relative_improvement_threshold": (
                SYNTHETIC_RELATIVE_TOLERANCE
            ),
            "synthetic_required_consecutive_sweeps": (
                SYNTHETIC_REQUIRED_SWEEPS
            ),
            "synthetic_safety_cap": SYNTHETIC_SAFETY_CAP,
            "projection": "minimum-cost one-to-one Wasserstein assignment",
            "subset_mass": "uniform",
            "classifier": "unweighted GaussianNB",
            "classifier_representation_dimensions": 200,
            "window_descriptor_dimensions": 40,
            "folds_sha256": sha256(FOLDS_FILE),
            "random_seed_by_repeat": RANDOM_SEED_BY_REPEAT,
            "splits": args.splits,
            "feature_fields": feature_fields,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
