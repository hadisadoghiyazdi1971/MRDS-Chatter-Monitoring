#!/usr/bin/env python3
"""Resumable retention curve for synthetic-MRDS projection-only subsets.

The evaluated MRDS candidate ends immediately after duplicate-free projection.
This runner contains no call to mixture-weight optimization or swap refinement.
The validated 10% result is imported read-only; 20--50% are computed locally.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB


CODE_DIR = Path(__file__).resolve().parent
RUN_DIR = CODE_DIR.parent
PROJECT = RUN_DIR.parents[1]
RESULTS = PROJECT / "results"
DATA_DIR = PROJECT / "chatterData"
PRODUCTION = RESULTS / "grouped_chatter_all40"
FOLDS_PATH = PRODUCTION / "folds_grouped.csv"
CONVERGENCE_10 = RESULTS / "candidate_convergence_10pct_2026_08_23"
ABLATION_10 = RESULTS / "converged_synthetic_projection_ablation_10pct_2026_08_24"
FINITE_10 = (
    CONVERGENCE_10
    / "r10_window_1e-2_cap200_2026_08_23"
    / "gate1"
    / "checkpoints"
)
ARCHIVED_CURVE = RESULTS / "grouped_budget_curve_all40"
sys.path.insert(0, str(CODE_DIR))

from candidate_synthetic_core import UnresolvedCapError, synthetic_projection_only  # noqa: E402
from meta_renyi_reduction import (  # noqa: E402
    EmpiricalMeasure,
    compress_bag,
    facility_location,
    finite_meta_p,
    median_bandwidth,
    pairwise_emd_exact,
    pam_kmedoids,
    rbf_kernel_from_distance,
)


POLICY = "simplified_projection_only_retention_curve_v1"
ALPHA = 2.0
ATOMS = 8
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
BUDGET_TO_M = {0.10: 12, 0.20: 24, 0.30: 36, 0.40: 48, 0.50: 60}
COMPUTE_BUDGETS = (0.20, 0.30, 0.40, 0.50)
SYN_TOL = 1e-3
SYN_CONSECUTIVE = 3
SYN_CAP = 30
EXPECTED_FOLDS_HASH = "147d95906e49863ccb711528246c66f2479b6f07f71d61556990359a6ef6a179"
EXPECTED_10_MEANS = {
    "MRDS": 0.011883659476120096,
    "W2-k-medoids": 0.012824416628221539,
    "Facility Location": 0.014023503758490392,
}
EXPECTED_10_WINS = {"MRDS": 14, "W2-k-medoids": 4, "Facility Location": 2}
EXPECTED_10_CLASSIFICATION = {
    "MRDS": {
        "balanced_accuracy": 0.822172619047619,
        "sensitivity": 0.6854166666666667,
        "specificity": 0.9589285714285716,
    },
    "W2-k-medoids": {
        "balanced_accuracy": 0.749702380952381,
        "sensitivity": 0.5083333333333333,
        "specificity": 0.9910714285714286,
    },
    "Facility Location": {
        "balanced_accuracy": 0.7329613095238094,
        "sensitivity": 0.4677083333333333,
        "specificity": 0.9982142857142857,
    },
}
METHOD_TO_TRACE = {
    "MRDS": "mrds_projected_all_weight_calls.jsonl.gz",
    "W2-k-medoids": "w2_k_medoids_all_weight_calls.jsonl.gz",
    "Facility Location": "facility_location_all_weight_calls.jsonl.gz",
}
ABLATION_METHOD = {
    "MRDS": "Convergence-projected MRDS",
    "W2-k-medoids": "W2-k-medoids initial",
    "Facility Location": "Facility Location initial",
}
METHODS = ("MRDS", "W2-k-medoids", "Facility Location")
IGNORED_FIELDS = {"FileName", "Label"}
RECORDING_RE = re.compile(
    r"^(?P<state>[SU])_WP(?P<workpiece>[AB])_L(?P<length>[0-9.]+)"
    r"_DOC(?P<doc>[0-9.]+)_WOC(?P<woc>[0-9.]+)"
    r"_N(?P<speed>[0-9]+)_F(?P<feed>[0-9]+)_R(?P<repetition>[1-4])$",
    flags=re.IGNORECASE,
)
CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {
            "policy": POLICY,
            "alpha": ALPHA,
            "atoms": ATOMS,
            "budgets": BUDGET_TO_M,
            "synthetic_tol": SYN_TOL,
            "synthetic_consecutive": SYN_CONSECUTIVE,
            "synthetic_cap": SYN_CAP,
            "folds_hash": EXPECTED_FOLDS_HASH,
            "classifier": "unweighted GaussianNB",
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def scalar(value: Any) -> float:
    return float(np.asarray(value).squeeze())


def load_chatter_mat_dir(path: Path) -> tuple[list[np.ndarray], np.ndarray, list[str], list[str]]:
    files = sorted(path.glob("*.mat"))
    if len(files) != 160:
        raise RuntimeError(f"Expected 160 MAT files, found {len(files)}")
    first = sio.loadmat(files[0], squeeze_me=True, struct_as_record=False)["SigData"]
    fields = sorted(
        field
        for field in dir(first[0])
        if not field.startswith("_") and field not in IGNORED_FIELDS
    )
    bags: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for file in files:
        signal_data = sio.loadmat(file, squeeze_me=True, struct_as_record=False)["SigData"]
        matrix = np.array(
            [[scalar(getattr(row, field)) for field in fields] for row in signal_data],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != 40 or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid frozen feature matrix in {file}: {matrix.shape}")
        bags.append(matrix)
        labels.append(1 if file.name.upper().startswith("U_") else 0)
        names.append(file.stem)
    if len(fields) != 40:
        raise RuntimeError(f"Expected 40 features, found {len(fields)}")
    return bags, np.asarray(labels, dtype=np.int64), names, fields


def condition_id(recording_id: str) -> str:
    if RECORDING_RE.fullmatch(recording_id) is None:
        raise ValueError(f"Unexpected recording identifier: {recording_id}")
    return re.sub(r"_R[1-4]$", "", recording_id, flags=re.IGNORECASE)


def fit_robust_transform(train_bags: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate(train_bags, axis=0)
    center = np.median(pooled, axis=0)
    mad = 1.4826 * np.median(np.abs(pooled - center), axis=0)
    standard_deviation = pooled.std(axis=0)
    scale = np.where(mad > 1e-12, mad, np.where(standard_deviation > 1e-12, standard_deviation, 1.0))
    return center, scale


def apply_transform(
    bags: list[np.ndarray], center: np.ndarray, scale: np.ndarray
) -> list[np.ndarray]:
    return [(matrix - center) / scale for matrix in bags]


def bag_embeddings(bags: list[np.ndarray]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for matrix in bags:
        q25, q50, q75 = np.quantile(matrix, [0.25, 0.50, 0.75], axis=0)
        rows.append(np.concatenate([matrix.mean(0), matrix.std(0), q25, q50, q75]))
    result = np.vstack(rows)
    if result.shape[1] != 200:
        raise RuntimeError(f"Expected 200-D recording vectors, found {result.shape[1]}")
    return result


def fold_indices(
    folds: pd.DataFrame, split: int
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    rows = folds[folds["split"].astype(int) == split]
    train = np.sort(rows[rows.partition == "train"].global_index.to_numpy(np.int64))
    test = np.sort(rows[rows.partition == "test"].global_index.to_numpy(np.int64))
    if train.size != 120 or test.size != 40:
        raise RuntimeError(f"split {split}: expected 120/40 train/test, found {train.size}/{test.size}")
    return train, test, int(rows.seed.iloc[0]), int(rows.repeat.iloc[0]), int(rows.fold.iloc[0])


def probability(values: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float64), eps)
    return result / result.sum()


def uniform_objective(kernel: np.ndarray, p: np.ndarray, selected: np.ndarray) -> float:
    weights = np.full(selected.size, 1.0 / selected.size, dtype=np.float64)
    pp = probability(p)
    qq = probability(kernel[:, selected] @ weights)
    value = np.sum((pp ** ALPHA) * (qq ** (1.0 - ALPHA)))
    return float(np.log(max(float(value), 1e-15)) / (ALPHA - 1.0))


def classify(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    classifier = GaussianNB()
    classifier.fit(x_train[selected], y_train[selected])  # Deliberately unweighted.
    prediction = classifier.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "R_min": min(sensitivity, specificity),
    }


def read_initializer(split: int, method: str) -> tuple[np.ndarray, float, Path]:
    source = CONVERGENCE_10 / "checkpoints" / f"split_{split:02d}" / METHOD_TO_TRACE[method]
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.loads(next(handle))
    if payload.get("role") != "initializer":
        raise RuntimeError(f"{source}: first call is not the initializer")
    selected = np.asarray(payload["selected"], dtype=np.int64)
    if selected.shape != (12,) or np.unique(selected).size != 12:
        raise RuntimeError(f"{source}: invalid 10% initializer subset")
    return selected, float(payload["initial_objective"]), source


def validate_static_runner() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    banned = {
        "optimize_mixture_weights",
        "optimize_weights_audited",
        "refine_observed_subset",
        "refine_subset_candidate",
        "multistart_refinement",
    }
    violations = sorted(called.intersection(banned))
    if violations:
        raise RuntimeError(f"Forbidden weight/swap calls in runner: {violations}")
    synthetic_source = (CODE_DIR / "candidate_synthetic_core.py").read_text(encoding="utf-8")
    required_fragments = [
        "normalization_denominator = max(abs(initial_objective), np.finfo(np.float64).eps)",
        "relative_improvement < relative_improvement_tol",
        "consecutive_small_improvements >= required_consecutive_sweeps",
        "linear_sum_assignment(costs.T)",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in synthetic_source]
    if missing:
        raise RuntimeError(f"Frozen synthetic implementation fragments missing: {missing}")
    return {
        "forbidden_weight_or_swap_calls": violations,
        "mrds_function_imported": "synthetic_projection_only",
        "projection": "duplicate-free linear_sum_assignment(costs.T)",
        "synthetic_stop": "r_syn < 1e-3 for 3 consecutive complete sweeps",
        "synthetic_cap": 30,
    }


def validate_10_percent() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if sha256(FOLDS_PATH) != EXPECTED_FOLDS_HASH:
        raise RuntimeError("Frozen-fold SHA-256 mismatch")
    folds = pd.read_csv(FOLDS_PATH)
    ablation = pd.read_csv(ABLATION_10 / "foldwise_classification.csv")
    objective_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    imported_rows: list[dict[str, Any]] = []
    for split in range(20):
        finite_path = FINITE_10 / f"split_{split:02d}" / "fold_finite_quantities.npz"
        finite = np.load(finite_path, allow_pickle=False)
        kernel = np.asarray(finite["kernel"], dtype=np.float64)
        p = np.asarray(finite["p"], dtype=np.float64)
        train_idx, test_idx, seed, repeat, fold = fold_indices(folds, split)
        if not np.array_equal(train_idx, finite["train_idx"]) or not np.array_equal(test_idx, finite["test_idx"]):
            raise RuntimeError(f"split {split}: finite cache fold indices mismatch")
        synthetic_status_path = CONVERGENCE_10 / "checkpoints" / f"split_{split:02d}" / "status.json"
        synthetic_status = json.loads(synthetic_status_path.read_text(encoding="utf-8"))
        synthetic_final = float(synthetic_status["synthetic"]["final_objective"])
        for method in METHODS:
            selected, archived_objective, source = read_initializer(split, method)
            reconstructed = uniform_objective(kernel, p, selected)
            error = abs(reconstructed - archived_objective)
            if error > 1e-12:
                raise RuntimeError(f"split {split} {method}: 10% objective mismatch {error:.3e}")
            row = ablation[
                (ablation.split.astype(int) == split)
                & (ablation.method == ABLATION_METHOD[method])
            ]
            if len(row) != 1:
                raise RuntimeError(f"split {split} {method}: missing exact 10% classifier row")
            saved_local = np.asarray([int(x) for x in str(row.iloc[0].local_train_indices).split("|")], dtype=np.int64)
            if not np.array_equal(saved_local, selected):
                raise RuntimeError(f"split {split} {method}: classifier subset IDs differ from initializer")
            objective_rows.append(
                {
                    "split": split,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": seed,
                    "budget": 0.10,
                    "m": 12,
                    "method": method,
                    "stage": "uniform observed subset before weights and swaps",
                    "uniform_J_alpha": reconstructed,
                    "J_synthetic_final": synthetic_final if method == "MRDS" else np.nan,
                    "source": "reused validated 10% ablation",
                }
            )
            imported = row.iloc[0].to_dict()
            imported.update(
                {
                    "budget": 0.10,
                    "m": 12,
                    "method": method,
                    "stage": "uniform observed subset before weights and swaps",
                    "source": "reused validated 10% ablation",
                }
            )
            imported_rows.append(imported)
            source_rows.append(
                {
                    "split": split,
                    "method": method,
                    "source": str(source.relative_to(PROJECT)),
                    "sha256": sha256(source),
                    "objective_reconstruction_error": error,
                }
            )
    objectives = pd.DataFrame(objective_rows)
    means = objectives.groupby("method").uniform_J_alpha.mean().to_dict()
    for method, expected in EXPECTED_10_MEANS.items():
        if abs(float(means[method]) - expected) > 5e-13:
            raise RuntimeError(f"10% {method} mean J mismatch")
    winners = (
        objectives.loc[objectives.groupby("split").uniform_J_alpha.idxmin(), "method"]
        .value_counts()
        .reindex(METHODS, fill_value=0)
        .astype(int)
        .to_dict()
    )
    if winners != EXPECTED_10_WINS:
        raise RuntimeError(f"10% winner count mismatch: {winners}")
    imported_frame = pd.DataFrame(imported_rows)
    classification_means: dict[str, dict[str, float]] = {}
    for method, expected_metrics in EXPECTED_10_CLASSIFICATION.items():
        method_rows = imported_frame[imported_frame.method == method]
        classification_means[method] = {}
        for metric, expected in expected_metrics.items():
            observed = float(method_rows[metric].astype(float).mean())
            if abs(observed - expected) > 5e-13:
                raise RuntimeError(
                    f"10% {method} mean {metric}={observed:.16g}, expected={expected:.16g}"
                )
            classification_means[method][metric] = observed
    report = {
        "status": "PASS",
        "means": {key: float(value) for key, value in means.items()},
        "winner_counts": {key: int(value) for key, value in winners.items()},
        "classification_means": classification_means,
        "maximum_objective_reconstruction_error": float(
            max(row["objective_reconstruction_error"] for row in source_rows)
        ),
        "sources": source_rows,
        "classification_source": str((ABLATION_10 / "foldwise_classification.csv").relative_to(PROJECT)),
        "classification_source_sha256": sha256(ABLATION_10 / "foldwise_classification.csv"),
    }
    return report, objectives, imported_frame


def audit_archived_curve() -> dict[str, Any]:
    metadata_path = ARCHIVED_CURVE / "run_metadata.json"
    selections_path = ARCHIVED_CURVE / "selected_recordings.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selections = pd.read_csv(selections_path, usecols=["budget", "method"])
    methods = sorted(selections.method.unique().tolist())
    reusable = bool(
        metadata.get("refinement_passes") == 0
        and all(not str(method).endswith("-R") for method in methods)
    )
    if reusable:
        raise RuntimeError("Archived curve unexpectedly passed reuse gate; source policy requires review")
    return {
        "safe_to_reuse_20_to_50": False,
        "decision": "RECOMPUTE_INITIALIZERS_ONLY",
        "reason": "Archived selected-recording rows are labelled -R and metadata records one refinement pass; initializer-stage IDs are not separately certified.",
        "metadata_refinement_passes": metadata.get("refinement_passes"),
        "archived_methods": methods,
        "metadata_sha256": sha256(metadata_path),
        "selected_recordings_sha256": sha256(selections_path),
    }


def dataset_manifest_hash() -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.mat")):
        rows.append({"file": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, rows


def synthetic_tiny_smoke_test() -> dict[str, Any]:
    """Exercise only the frozen stop and duplicate-free projection on a toy case."""
    measures = [
        EmpiricalMeasure(
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            np.asarray([0.5, 0.5], dtype=np.float64),
        )
        for _ in range(6)
    ]
    result = synthetic_projection_only(
        measures,
        np.zeros((6, 6), dtype=np.float64),
        np.ones((6, 6), dtype=np.float64),
        n_prototypes=2,
        alpha=ALPHA,
        prototype_atoms=2,
        bandwidth=1.0,
        seed=11,
        max_sweeps=SYN_CAP,
        relative_improvement_tol=SYN_TOL,
        required_consecutive_sweeps=SYN_CONSECUTIVE,
    )
    if result.sweeps_completed != 3 or np.unique(result.selected).size != 2:
        raise RuntimeError("Tiny synthetic stop/projection smoke test failed")
    return {
        "status": "PASS",
        "sweeps": result.sweeps_completed,
        "stop_reason": result.stop_reason,
        "selected_distinct": True,
    }


def run_prevalidation(*, write_outputs: bool = True) -> dict[str, Any]:
    folds = pd.read_csv(FOLDS_PATH)
    splits = sorted(folds.split.astype(int).unique().tolist())
    if splits != list(range(20)):
        raise RuntimeError(f"Expected frozen splits 0..19, found {splits}")
    if BUDGET_TO_M != {0.10: 12, 0.20: 24, 0.30: 36, 0.40: 48, 0.50: 60}:
        raise RuntimeError("Retention budget mapping changed")
    static = validate_static_runner()
    ten_report, ten_objectives, ten_classification = validate_10_percent()
    archive = audit_archived_curve()
    dataset_hash, dataset_rows = dataset_manifest_hash()
    full_reference = full_training_reference()
    tiny_smoke = synthetic_tiny_smoke_test()
    report = {
        "status": "PRE_RUN_VALIDATION_PASS",
        "validated_at_utc": utc_now(),
        "folds_sha256": sha256(FOLDS_PATH),
        "n_folds": 20,
        "budget_to_m": {str(key): value for key, value in BUDGET_TO_M.items()},
        "alpha": ALPHA,
        "K": ATOMS,
        "static_execution_audit": static,
        "ten_percent_reuse": ten_report,
        "twenty_to_fifty_baseline_reuse": archive,
        "classifier": "standard unweighted GaussianNB; no sample weights",
        "inner_validation_or_tuning": False,
        "output_isolated": RUN_DIR.name == "simplified_mrds_projection_retention_curve_2026_08_24",
        "resume_checkpoint_test": "PASS",
        "tiny_synthetic_stop_and_projection_smoke_test": tiny_smoke,
        "full_training_reference": {
            "rows": int(len(full_reference)),
            "splits": int(full_reference.split.nunique()),
            "source": str((PRODUCTION / "classification.csv").relative_to(PROJECT)),
            "source_sha256": sha256(PRODUCTION / "classification.csv"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    if not report["output_isolated"]:
        raise RuntimeError(f"Unexpected output directory: {RUN_DIR}")
    if write_outputs:
        atomic_csv(RUN_DIR / "pre_run_10pct_objective_audit.csv", ten_objectives)
        atomic_csv(RUN_DIR / "pre_run_10pct_classification_audit.csv", ten_classification)
        atomic_json(RUN_DIR / "VALIDATION_REPORT.json", report)
        atomic_json(
            RUN_DIR / "PROVENANCE_AUDIT.json",
            {
                "folds_file": str(FOLDS_PATH.relative_to(PROJECT)),
                "folds_sha256": sha256(FOLDS_PATH),
                "dataset_directory": str(DATA_DIR.relative_to(PROJECT)),
                "dataset_manifest_sha256": dataset_hash,
                "dataset_files": dataset_rows,
                "ten_percent": ten_report,
                "archived_curve_audit": archive,
                "code_files": {
                    path.name: sha256(path) for path in sorted(CODE_DIR.glob("*.py"))
                },
            },
        )
    return report


def checkpoint_path(budget: float, split: int, method: str) -> Path:
    slug = method.lower().replace(" ", "_").replace("-", "_")
    return RUN_DIR / "checkpoints" / f"budget_{int(round(100 * budget)):02d}" / f"split_{split:02d}" / f"{slug}.json"


def valid_checkpoint(path: Path, budget: float, split: int, method: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    expected_m = BUDGET_TO_M[budget]
    if (
        payload.get("complete") is True
        and payload.get("config_fingerprint") == CONFIG_FINGERPRINT
        and int(payload.get("split", -1)) == split
        and abs(float(payload.get("budget", -1.0)) - budget) < 1e-12
        and payload.get("method") == method
        and int(payload.get("m", -1)) == expected_m
        and len(payload.get("selected_local_indices", [])) == expected_m
        and len(set(payload.get("selected_local_indices", []))) == expected_m
    ):
        return payload
    return None


def cap_binding_checkpoint(
    path: Path, budget: float, split: int, method: str
) -> dict[str, Any] | None:
    """Recognize a validated cap record so resume never reruns it silently."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        payload.get("complete") is False
        and payload.get("status") == "SYNTHETIC_CAP_BINDING"
        and payload.get("config_fingerprint") == CONFIG_FINGERPRINT
        and int(payload.get("split", -1)) == split
        and abs(float(payload.get("budget", -1.0)) - budget) < 1e-12
        and payload.get("method") == method
        and int(payload.get("m", -1)) == BUDGET_TO_M[budget]
    ):
        return payload
    return None


def geometry_cache_path(split: int) -> Path:
    return RUN_DIR / "checkpoints" / "fold_geometry" / f"split_{split:02d}.npz"


def load_or_build_geometry(split: int) -> dict[str, Any]:
    cache = geometry_cache_path(split)
    if cache.is_file():
        arrays = np.load(cache, allow_pickle=False)
        if (
            str(arrays["config_fingerprint"].item()) == CONFIG_FINGERPRINT
            and np.asarray(arrays["distance"]).shape == (120, 120)
            and np.asarray(arrays["supports"]).shape == (120, 8, 40)
        ):
            print(f"split {split:02d}: FOUND VALID GEOMETRY CHECKPOINT - SKIPPING", flush=True)
            return {key: arrays[key] for key in arrays.files}

    bags, labels, names, _fields = load_chatter_mat_dir(DATA_DIR)
    folds = pd.read_csv(FOLDS_PATH)
    train_idx, test_idx, seed, repeat, fold = fold_indices(folds, split)
    groups = np.asarray([condition_id(name) for name in names], dtype=object)
    if set(groups[train_idx]).intersection(set(groups[test_idx])):
        raise RuntimeError(f"split {split}: condition leakage")
    train_raw = [bags[int(index)] for index in train_idx]
    test_raw = [bags[int(index)] for index in test_idx]
    center, scale = fit_robust_transform(train_raw)
    train_scaled = apply_transform(train_raw, center, scale)
    test_scaled = apply_transform(test_raw, center, scale)
    measures = [
        compress_bag(bag, ATOMS, seed + split * 1000 + local)
        for local, bag in enumerate(train_scaled)
    ]
    distance = pairwise_emd_exact(measures)
    bandwidth = median_bandwidth(distance)
    kernel = rbf_kernel_from_distance(distance, bandwidth)
    p = finite_meta_p(kernel)
    payload = {
        "config_fingerprint": np.asarray(CONFIG_FINGERPRINT),
        "split": np.asarray(split),
        "repeat": np.asarray(repeat),
        "fold": np.asarray(fold),
        "seed": np.asarray(seed),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "supports": np.stack([measure.support for measure in measures]),
        "atom_weights": np.stack([measure.weights for measure in measures]),
        "distance": distance,
        "kernel": kernel,
        "p": p,
        "bandwidth": np.asarray(bandwidth),
        "x_train": bag_embeddings(train_scaled),
        "x_test": bag_embeddings(test_scaled),
        "y_train": labels[train_idx],
        "y_test": labels[test_idx],
        "names": np.asarray(names, dtype="U128"),
        "groups": np.asarray(groups, dtype="U128"),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, cache)
    return payload


def measure_list(payload: dict[str, Any]) -> list[EmpiricalMeasure]:
    supports = np.asarray(payload["supports"], dtype=np.float64)
    weights = np.asarray(payload["atom_weights"], dtype=np.float64)
    return [EmpiricalMeasure(supports[i].copy(), weights[i].copy()) for i in range(120)]


def save_synthetic_trace(split: int, budget: float, synthetic: Any) -> None:
    rows = []
    for sweep, objective in enumerate(synthetic.objective_history):
        rows.append(
            {
                "split": split,
                "budget": budget,
                "sweep": sweep,
                "objective": objective,
                "relative_improvement": np.nan if sweep == 0 else synthetic.relative_improvement_by_sweep[sweep - 1],
                "accepted_updates": 0 if sweep == 0 else synthetic.accepted_updates_by_sweep[sweep - 1],
                "max_move": 0.0 if sweep == 0 else synthetic.max_move_by_sweep[sweep - 1],
            }
        )
    atomic_csv(RUN_DIR / "traces" / f"budget_{int(100 * budget):02d}_split_{split:02d}_synthetic.csv", pd.DataFrame(rows))


def save_cap_trace(split: int, budget: float, details: dict[str, Any]) -> None:
    history = list(details.get("objective_history", []))
    relative = list(details.get("relative_improvement_by_sweep", []))
    accepted = list(details.get("accepted_updates_by_sweep", []))
    moves = list(details.get("max_move_by_sweep", []))
    rows = []
    for sweep, objective in enumerate(history):
        rows.append(
            {
                "split": split,
                "budget": budget,
                "sweep": sweep,
                "objective": objective,
                "relative_improvement": np.nan if sweep == 0 else relative[sweep - 1],
                "accepted_updates": 0 if sweep == 0 else accepted[sweep - 1],
                "max_move": 0.0 if sweep == 0 else moves[sweep - 1],
            }
        )
    atomic_csv(RUN_DIR / "traces" / f"budget_{int(100 * budget):02d}_split_{split:02d}_SYNTHETIC_CAP_BINDING.csv", pd.DataFrame(rows))


def compute_method(split: int, budget: float, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = checkpoint_path(budget, split, method)
    existing = valid_checkpoint(path, budget, split, method)
    if existing is not None:
        print(
            f"budget {int(100 * budget):02d}% split {split:02d} {method}: FOUND VALID CHECKPOINT - SKIPPING",
            flush=True,
        )
        return existing
    existing_cap = cap_binding_checkpoint(path, budget, split, method)
    if existing_cap is not None:
        print(
            f"budget {int(100 * budget):02d}% split {split:02d} {method}: "
            "FOUND VALID SYNTHETIC_CAP_BINDING CHECKPOINT - SKIPPING",
            flush=True,
        )
        return existing_cap

    started = time.perf_counter()
    m = BUDGET_TO_M[budget]
    distance = np.asarray(payload["distance"], dtype=np.float64)
    kernel = np.asarray(payload["kernel"], dtype=np.float64)
    p = np.asarray(payload["p"], dtype=np.float64)
    seed = int(np.asarray(payload["seed"]).item())
    synthetic_final = np.nan
    synthetic_sweeps = 0
    synthetic_stop_reason = "not applicable"
    if method == "W2-k-medoids":
        selected = pam_kmedoids(distance, m, seed=seed)
    elif method == "Facility Location":
        selected = facility_location(kernel, m)
    elif method == "MRDS":
        try:
            synthetic = synthetic_projection_only(
                measure_list(payload),
                distance,
                kernel,
                n_prototypes=m,
                alpha=ALPHA,
                prototype_atoms=ATOMS,
                bandwidth=float(np.asarray(payload["bandwidth"]).item()),
                seed=seed,
                max_sweeps=SYN_CAP,
                tol=1e-5,
                relative_improvement_tol=SYN_TOL,
                required_consecutive_sweeps=SYN_CONSECUTIVE,
                monotonicity_tolerance_factor=64.0,
            )
        except UnresolvedCapError as error:
            save_cap_trace(split, budget, error.details)
            failed = {
                "policy": POLICY,
                "config_fingerprint": CONFIG_FINGERPRINT,
                "complete": False,
                "status": "SYNTHETIC_CAP_BINDING",
                "split": split,
                "budget": budget,
                "m": m,
                "method": method,
                "cap_details": error.details,
                "updated_at_utc": utc_now(),
            }
            atomic_json(path, failed)
            return failed
        selected = synthetic.selected
        synthetic_final = float(synthetic.objective_history[-1])
        synthetic_sweeps = int(synthetic.sweeps_completed)
        synthetic_stop_reason = synthetic.stop_reason
        save_synthetic_trace(split, budget, synthetic)
    else:
        raise ValueError(method)

    selected = np.asarray(selected, dtype=np.int64)
    if selected.shape != (m,) or np.unique(selected).size != m:
        raise RuntimeError(f"split {split} budget {budget} {method}: subset is not duplicate-free")
    train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    names = np.asarray(payload["names"], dtype=str)
    groups = np.asarray(payload["groups"], dtype=str)
    global_selected = train_idx[selected]
    selected_labels = np.asarray(payload["y_train"], dtype=np.int64)[selected]
    metrics = classify(
        np.asarray(payload["x_train"], dtype=np.float64),
        np.asarray(payload["y_train"], dtype=np.int64),
        np.asarray(payload["x_test"], dtype=np.float64),
        np.asarray(payload["y_test"], dtype=np.int64),
        selected,
    )
    result = {
        "policy": POLICY,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "complete": True,
        "status": "COMPLETE",
        "source": "newly computed initializer/projection-only candidate",
        "split": split,
        "repeat": int(np.asarray(payload["repeat"]).item()),
        "fold": int(np.asarray(payload["fold"]).item()),
        "seed": seed,
        "budget": budget,
        "m": m,
        "method": method,
        "stage": "uniform observed subset before weights and swaps",
        "selected_local_indices": selected.tolist(),
        "selected_global_indices": global_selected.tolist(),
        "recording_ids": [str(names[index]) for index in global_selected],
        "condition_ids": [str(groups[index]) for index in global_selected],
        "n_unique_conditions": len(set(str(groups[index]) for index in global_selected)),
        "n_stable": int(np.sum(selected_labels == 0)),
        "n_chatter": int(np.sum(selected_labels == 1)),
        "uniform_J_alpha": uniform_objective(kernel, p, selected),
        "J_synthetic_final": synthetic_final,
        "synthetic_sweeps": synthetic_sweeps,
        "synthetic_stop_reason": synthetic_stop_reason,
        "SYNTHETIC_CAP_BINDING": False,
        "classifier": "standard unweighted GaussianNB",
        "renyi_weights_passed_to_classifier": False,
        **metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at_utc": utc_now(),
    }
    atomic_json(path, result)
    atomic_csv(
        RUN_DIR / "fold_results" / f"budget_{int(100 * budget):02d}_split_{split:02d}_{method.lower().replace(' ', '_').replace('-', '_')}.csv",
        pd.DataFrame([result]),
    )
    print(
        f"budget {int(100 * budget):02d}% split {split:02d} {method}: complete in {result['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return result


def process_split(split: int) -> list[dict[str, Any]]:
    payload = load_or_build_geometry(split)
    rows: list[dict[str, Any]] = []
    for budget in COMPUTE_BUDGETS:
        for method in ("W2-k-medoids", "Facility Location", "MRDS"):
            rows.append(compute_method(split, budget, method, payload))
    return rows


def import_ten_percent_checkpoints() -> list[dict[str, Any]]:
    _report, objectives, classification = validate_10_percent()
    rows: list[dict[str, Any]] = []
    for split in range(20):
        for method in METHODS:
            path = checkpoint_path(0.10, split, method)
            existing = valid_checkpoint(path, 0.10, split, method)
            if existing is not None:
                print(f"budget 10% split {split:02d} {method}: FOUND VALID CHECKPOINT - SKIPPING", flush=True)
                rows.append(existing)
                continue
            objective = objectives[(objectives.split == split) & (objectives.method == method)].iloc[0]
            metric = classification[(classification.split.astype(int) == split) & (classification.method == method)].iloc[0]
            selected = [int(value) for value in str(metric["local_train_indices"]).split("|")]
            payload = {
                "policy": POLICY,
                "config_fingerprint": CONFIG_FINGERPRINT,
                "complete": True,
                "status": "COMPLETE_REUSED_VALIDATED",
                "source": "reused validated 10% ablation",
                "split": split,
                "repeat": int(objective["repeat"]),
                "fold": int(objective["fold"]),
                "seed": int(objective["seed"]),
                "budget": 0.10,
                "m": 12,
                "method": method,
                "stage": "uniform observed subset before weights and swaps",
                "selected_local_indices": selected,
                "selected_global_indices": [int(value) for value in str(metric["global_indices"]).split("|")],
                "recording_ids": str(metric["recording_ids"]).split("|"),
                "condition_ids": [],
                "n_unique_conditions": int(metric["n_unique_conditions"]),
                "n_stable": int(metric["n_stable"]),
                "n_chatter": int(metric["n_chatter"]),
                "uniform_J_alpha": float(objective["uniform_J_alpha"]),
                "J_synthetic_final": float(objective["J_synthetic_final"]) if method == "MRDS" else np.nan,
                "synthetic_sweeps": None,
                "synthetic_stop_reason": "reused validated objective-stopped 10% result" if method == "MRDS" else "not applicable",
                "SYNTHETIC_CAP_BINDING": False,
                "classifier": "standard unweighted GaussianNB",
                "renyi_weights_passed_to_classifier": False,
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "sensitivity": float(metric["sensitivity"]),
                "specificity": float(metric["specificity"]),
                "R_min": float(metric["R_min"]),
                "elapsed_seconds": 0.0,
                "updated_at_utc": utc_now(),
            }
            atomic_json(path, payload)
            rows.append(payload)
    return rows


def full_training_reference() -> pd.DataFrame:
    table = pd.read_csv(PRODUCTION / "classification.csv")
    rows = table[
        (table.method == "Full training set")
        & (table.stage == "no reduction")
        & (table.weight_mode == "uniform")
    ].copy()
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 full-training reference rows, found {len(rows)}")
    return pd.DataFrame(
        {
            "split": rows.split.astype(int),
            "repeat": rows.repeat.astype(int),
            "fold": rows.fold.astype(int),
            "seed": rows.seed.astype(int),
            "budget": 1.0,
            "m": 120,
            "method": "Full training reference",
            "stage": "unreduced reference only",
            "n_selected": 120,
            "n_unique_conditions": 30,
            "n_stable": np.nan,
            "n_chatter": np.nan,
            "balanced_accuracy": rows.balanced_accuracy.astype(float),
            "sensitivity": rows.sensitivity.astype(float),
            "specificity": rows.specificity.astype(float),
            "R_min": np.minimum(rows.sensitivity.astype(float), rows.specificity.astype(float)),
            "recording_ids": "",
            "source": "reused unreduced production reference",
        }
    )


def aggregate(checkpoints: list[dict[str, Any]]) -> None:
    cap_rows = [row for row in checkpoints if row.get("status") == "SYNTHETIC_CAP_BINDING"]
    complete = [row for row in checkpoints if row.get("complete") is True]
    expected_complete = 20 * 5 * 3
    objective_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in complete:
        objective_rows.append(
            {key: row.get(key) for key in [
                "split", "repeat", "fold", "seed", "budget", "m", "method", "stage",
                "uniform_J_alpha", "J_synthetic_final", "source",
            ]}
        )
        classification_rows.append(
            {
                "split": row["split"],
                "repeat": row["repeat"],
                "fold": row["fold"],
                "seed": row["seed"],
                "budget": row["budget"],
                "m": row["m"],
                "method": row["method"],
                "stage": row["stage"],
                "n_selected": row["m"],
                "n_unique_conditions": row["n_unique_conditions"],
                "n_stable": row["n_stable"],
                "n_chatter": row["n_chatter"],
                "balanced_accuracy": row["balanced_accuracy"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "R_min": row["R_min"],
                "recording_ids": "|".join(row["recording_ids"]),
                "source": row["source"],
            }
        )
        if row["method"] == "MRDS":
            projected = float(row["uniform_J_alpha"])
            synthetic = float(row["J_synthetic_final"])
            diagnostics.append(
                {
                    "split": row["split"],
                    "budget": row["budget"],
                    "m": row["m"],
                    "J_synthetic_final": synthetic,
                    "J_projected_uniform": projected,
                    "absolute_projection_change": projected - synthetic,
                    "relative_projection_change": (projected - synthetic) / max(abs(synthetic), np.finfo(np.float64).eps),
                    "synthetic_sweeps": row.get("synthetic_sweeps"),
                    "synthetic_stop_reason": row.get("synthetic_stop_reason"),
                    "SYNTHETIC_CAP_BINDING": row.get("SYNTHETIC_CAP_BINDING", False),
                }
            )
    objectives = pd.DataFrame(objective_rows).sort_values(["budget", "split", "method"])
    classification = pd.DataFrame(classification_rows).sort_values(["budget", "split", "method"])
    classification = pd.concat([classification, full_training_reference()], ignore_index=True, sort=False)
    atomic_csv(RUN_DIR / "foldwise_objectives.csv", objectives)
    atomic_csv(RUN_DIR / "foldwise_classification.csv", classification)
    atomic_csv(RUN_DIR / "synthetic_projection_diagnostics.csv", pd.DataFrame(diagnostics))

    objective_summary = objectives.groupby(["budget", "m", "method"], as_index=False).agg(
        folds=("split", "count"),
        mean_J_alpha=("uniform_J_alpha", "mean"),
        std_J_alpha=("uniform_J_alpha", "std"),
    )
    for budget in BUDGETS:
        block = objectives[np.isclose(objectives.budget.astype(float), budget)]
        if set(block.method) == set(METHODS) and block.split.nunique() == 20:
            pivot = block.pivot(index="split", columns="method", values="uniform_J_alpha")
            mask = np.isclose(objective_summary.budget.astype(float), budget)
            objective_summary.loc[mask, "MRDS_minus_W2_mean_gap"] = float((pivot["MRDS"] - pivot["W2-k-medoids"]).mean())
            objective_summary.loc[mask, "MRDS_minus_Facility_mean_gap"] = float((pivot["MRDS"] - pivot["Facility Location"]).mean())
    atomic_csv(RUN_DIR / "objective_summary.csv", objective_summary)

    class_summary = classification.groupby(["budget", "m", "method"], as_index=False).agg(
        folds=("split", "count"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        minimum_foldwise_BA=("balanced_accuracy", "min"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_specificity=("specificity", "mean"),
        mean_R_min=("R_min", "mean"),
        mean_unique_conditions=("n_unique_conditions", "mean"),
    )
    atomic_csv(RUN_DIR / "classification_summary.csv", class_summary)

    winner_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        objective_block = objectives[np.isclose(objectives.budget.astype(float), budget)]
        class_block = classification[np.isclose(classification.budget.astype(float), budget)]
        if objective_block.split.nunique() != 20 or set(objective_block.method) != set(METHODS):
            continue
        objective_winners = objective_block.loc[
            objective_block.groupby("split").uniform_J_alpha.idxmin(), "method"
        ].value_counts()
        for method in METHODS:
            winner_rows.append(
                {"budget": budget, "comparison_type": "minimum uniform J_alpha", "method_or_comparison": method, "wins": int(objective_winners.get(method, 0)), "ties": np.nan, "losses": np.nan}
            )
        pivot = class_block.pivot(index="split", columns="method", values="balanced_accuracy")
        for comparator in ("W2-k-medoids", "Facility Location"):
            difference = pivot["MRDS"] - pivot[comparator]
            winner_rows.append(
                {
                    "budget": budget,
                    "comparison_type": "balanced accuracy descriptive",
                    "method_or_comparison": f"MRDS vs {comparator}",
                    "wins": int(np.sum(difference > 1e-15)),
                    "ties": int(np.sum(np.abs(difference) <= 1e-15)),
                    "losses": int(np.sum(difference < -1e-15)),
                    "mean_difference": float(difference.mean()),
                }
            )
    atomic_csv(RUN_DIR / "winner_counts.csv", pd.DataFrame(winner_rows))
    status = {
        "status": "COMPLETE" if len(complete) == expected_complete and not cap_rows else "INCOMPLETE_SYNTHETIC_CAP_BINDING",
        "policy": POLICY,
        "completed_method_checkpoints": len(complete),
        "expected_method_checkpoints": expected_complete,
        "synthetic_cap_binding_calls": len(cap_rows),
        "cap_rows": cap_rows,
        "full_retention_curve_run_by_user": True,
        "manuscript_modified": False,
        "finished_at_utc": utc_now(),
    }
    atomic_json(RUN_DIR / "RUN_STATUS.json", status)


def checkpoint_smoke_test() -> None:
    path = RUN_DIR / "checkpoints" / "_smoke" / "method.json"
    payload = {
        "complete": True,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "split": 0,
        "budget": 0.20,
        "m": 24,
        "method": "W2-k-medoids",
        "selected_local_indices": list(range(24)),
    }
    atomic_json(path, payload)
    if valid_checkpoint(path, 0.20, 0, "W2-k-medoids") is None:
        raise RuntimeError("Resume/checkpoint smoke test failed")
    path.unlink()
    cap_payload = {
        "complete": False,
        "status": "SYNTHETIC_CAP_BINDING",
        "config_fingerprint": CONFIG_FINGERPRINT,
        "split": 0,
        "budget": 0.20,
        "m": 24,
        "method": "MRDS",
    }
    atomic_json(path, cap_payload)
    if cap_binding_checkpoint(path, 0.20, 0, "MRDS") is None:
        raise RuntimeError("Cap-binding resume smoke test failed")
    path.unlink()
    path.parent.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    try:
        checkpoint_smoke_test()
        report = run_prevalidation(write_outputs=True)
        if args.validate_only or args.smoke_test:
            atomic_json(
                RUN_DIR / "RUN_STATUS.json",
                {
                    "status": "PREPARED_PENDING_USER_EXECUTION",
                    "pre_run_validation": report["status"],
                    "full_retention_curve_not_run": True,
                    "manuscript_modified": False,
                    "updated_at_utc": utc_now(),
                },
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return

        atomic_json(
            RUN_DIR / "RUN_STATUS.json",
            {
                "status": "RUNNING",
                "policy": POLICY,
                "started_at_utc": utc_now(),
                "workers": args.workers,
                "manuscript_modified": False,
            },
        )
        all_rows = import_ten_percent_checkpoints()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_split, split): split for split in range(20)}
            for future in as_completed(futures):
                split = futures[future]
                rows = future.result()
                all_rows.extend(rows)
                print(f"split {split:02d}: all requested 20--50% method tasks returned", flush=True)
        aggregate(all_rows)
        print(f"COMPLETE: {RUN_DIR}", flush=True)
    except KeyboardInterrupt:
        atomic_json(
            RUN_DIR / "RUN_STATUS.json",
            {
                "status": "INTERRUPTED_RESUMABLE",
                "message": "Restart the same command; valid method checkpoints will be skipped.",
                "updated_at_utc": utc_now(),
                "manuscript_modified": False,
            },
        )
        raise
    except Exception as error:
        atomic_json(
            RUN_DIR / "RUN_STATUS.json",
            {
                "status": "FAILED_RESUMABLE",
                "error": repr(error),
                "updated_at_utc": utc_now(),
                "manuscript_modified": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
