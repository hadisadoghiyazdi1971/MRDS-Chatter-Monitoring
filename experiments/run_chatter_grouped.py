#!/usr/bin/env python3
"""Condition-disjoint chatter evaluation with an auditable MRDS decomposition.

The four repetitions of one machining condition are kept together in every
outer fold.  This runner intentionally separates the synthetic MRDS stage,
projection, observed-subset refinement, and multistart diagnostic so that a
baseline initializer cannot silently be reported as the proposed method.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "experiments"))

from grouped_protocol import (  # noqa: E402
    assert_group_disjoint,
    build_group_manifest,
    fold_manifest_rows,
    repeated_stratified_group_folds,
    selection_group_diagnostics,
)
from meta_renyi_reduction import (  # noqa: E402
    MetaRenyiReducer,
    assignments_and_weights,
    compress_bag,
    evaluate_selection,
    facility_location,
    finite_meta_p,
    median_bandwidth,
    optimize_mixture_weights,
    pairwise_emd_exact,
    pam_kmedoids,
    random_selection,
    rbf_kernel_from_distance,
)
from mrds_projection_refinement_integrated import (  # noqa: E402
    refine_observed_subset,
)
from run_chatter import (  # noqa: E402
    apply_transform,
    bag_embeddings,
    fit_robust_transform,
    load_chatter_mat_dir,
)
from run_chatter_audited import classification_metrics, file_sha256  # noqa: E402


ROBUST_THREE = ("EnR", "Imp_Fact", "Spectral_Energy")
ADDED_THREE = ("CE", "MPE", "wRCMDE")


def select_feature_columns(
    bags: list[np.ndarray], fields: list[str], feature_set: str
) -> tuple[list[np.ndarray], list[str]]:
    if feature_set == "all40":
        return bags, fields
    if feature_set == "original37":
        missing = [name for name in ADDED_THREE if name not in fields]
        if missing:
            raise ValueError(f"Cannot identify the added features; missing fields: {missing}")
        columns = [index for index, name in enumerate(fields) if name not in ADDED_THREE]
        if len(columns) != 37:
            raise ValueError(f"Expected 37 original fields after removal, found {len(columns)}")
        return [bag[:, columns] for bag in bags], [fields[index] for index in columns]
    if feature_set == "added3":
        missing = [name for name in ADDED_THREE if name not in fields]
        if missing:
            raise ValueError(f"The added3 feature set is unavailable; missing: {missing}")
        columns = [fields.index(name) for name in ADDED_THREE]
        return [bag[:, columns] for bag in bags], list(ADDED_THREE)
    if feature_set != "robust3":
        raise ValueError(f"Unknown feature set: {feature_set}")
    missing = [name for name in ROBUST_THREE if name not in fields]
    if missing:
        raise ValueError(
            f"The exploratory robust3 feature set is unavailable; missing fields: {missing}. "
            f"Archive fields are: {fields}"
        )
    columns = [fields.index(name) for name in ROBUST_THREE]
    return [bag[:, columns] for bag in bags], list(ROBUST_THREE)


def _record_classification_rows(
    rows: list[dict[str, object]],
    common: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected: np.ndarray,
    voronoi: np.ndarray,
    renyi_weights: np.ndarray,
) -> None:
    weight_modes = {
        "uniform": None,
        "voronoi": voronoi,
        "renyi": renyi_weights,
    }
    for weight_mode, weights in weight_modes.items():
        rows.append({
            **common,
            "weight_mode": weight_mode,
            **classification_metrics(
                x_train[selected], y_train[selected], x_test, y_test, weights
            ),
        })


def _manifest_with_file_audit(
    manifest: pd.DataFrame,
    input_dir: Path,
    bags: list[np.ndarray],
) -> pd.DataFrame:
    audited = manifest.copy()
    paths = [input_dir / f"{name}.mat" for name in audited["recording_id"]]
    audited["path"] = [str(path) for path in paths]
    audited["sha256"] = [file_sha256(path) for path in paths]
    audited["n_windows"] = [int(bag.shape[0]) for bag in bags]
    audited["archive_feature_dimension"] = [int(bag.shape[1]) for bag in bags]
    audited["missing_values"] = [int(np.isnan(bag).sum()) for bag in bags]
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "chatterData")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "grouped_chatter_all40"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1])
    parser.add_argument(
        "--feature-set",
        choices=["all40", "original37", "added3", "robust3"],
        default="all40",
    )
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--synthetic-iterations", type=int, default=1)
    parser.add_argument("--refinement-passes", type=int, default=1)
    parser.add_argument("--weight-max-iter", type=int, default=10)
    args = parser.parse_args()

    if any(not 0.0 < budget <= 1.0 for budget in args.budgets):
        raise ValueError("Every budget must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)

    archive_bags, labels, names, archive_fields = load_chatter_mat_dir(args.input)
    manifest = build_group_manifest(names, labels)
    audited_manifest = _manifest_with_file_audit(manifest, args.input, archive_bags)
    audited_manifest.to_csv(args.output / "dataset_manifest_grouped.csv", index=False)

    bags, model_fields = select_feature_columns(
        archive_bags, archive_fields, args.feature_set
    )
    (args.output / "feature_fields.json").write_text(
        json.dumps(
            {
                "feature_set": args.feature_set,
                "archive_field_count": len(archive_fields),
                "archive_fields": archive_fields,
                "model_field_count": len(model_fields),
                "model_fields": model_fields,
                "status": {
                    "all40": "primary representation used before provenance clarification",
                    "original37": "chronology-based exploratory ablation",
                    "added3": "chronology-based exploratory ablation",
                    "robust3": "exploratory representation inherited from a prior diagnostic",
                }[args.feature_set],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fold_rows: list[dict[str, object]] = []
    rep_rows: list[dict[str, object]] = []
    cls_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    refine_rows: list[dict[str, object]] = []
    split_summary_rows: list[dict[str, object]] = []

    folds = repeated_stratified_group_folds(
        labels=labels,
        groups=manifest["condition_id"].astype(str).to_numpy(),
        seeds=args.seeds,
        n_splits=args.n_splits,
    )
    for fold in folds:
        fold_start = time.perf_counter()
        train_idx, test_idx = fold.train_idx, fold.test_idx
        groups = manifest["condition_id"].astype(str).to_numpy()
        assert_group_disjoint(train_idx, test_idx, groups)
        fold_rows.extend(fold_manifest_rows(fold, manifest))

        train_raw = [bags[int(i)] for i in train_idx]
        test_raw = [bags[int(i)] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_groups = groups[train_idx]
        center, scale = fit_robust_transform(train_raw)
        train_bags = apply_transform(train_raw, center, scale)
        test_bags = apply_transform(test_raw, center, scale)
        measures = [
            compress_bag(bag, args.max_atoms, fold.seed + fold.split * 1000 + i)
            for i, bag in enumerate(train_bags)
        ]

        distance_start = time.perf_counter()
        distance = pairwise_emd_exact(measures)
        distance_seconds = time.perf_counter() - distance_start
        bandwidth = median_bandwidth(distance)
        kernel = rbf_kernel_from_distance(distance, bandwidth)
        p_full = finite_meta_p(kernel)
        x_train = bag_embeddings(train_bags)
        x_test = bag_embeddings(test_bags)

        split_common = {
            "split": fold.split,
            "repeat": fold.repeat,
            "fold": fold.fold,
            "seed": fold.seed,
            "feature_set": args.feature_set,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_conditions": int(np.unique(train_groups).size),
            "n_test_conditions": int(np.unique(groups[test_idx]).size),
        }
        cls_rows.append({
            **split_common,
            "budget": 1.0,
            "method": "Full training set",
            "stage": "no reduction",
            "n_selected": len(train_idx),
            "weight_mode": "uniform",
            **classification_metrics(x_train, y_train, x_test, y_test, None),
        })

        for budget in args.budgets:
            n_selected = max(2, int(round(float(budget) * len(train_idx))))
            initializers: dict[str, np.ndarray] = {
                "Random": random_selection(len(train_idx), n_selected, fold.seed),
                "FacilityLocation": facility_location(kernel, n_selected),
                "W2-KMedoids": pam_kmedoids(distance, n_selected, seed=fold.seed),
            }

            reducer = MetaRenyiReducer(
                n_prototypes=n_selected,
                alpha=args.alpha,
                bandwidth=bandwidth,
                prototype_atoms=args.max_atoms,
                max_iter=args.synthetic_iterations,
                seed=fold.seed,
                transport_backend="emd_exact",
                refinement_max_passes=0,
                refinement_weight_max_iter=args.weight_max_iter,
            )
            mrds_result = reducer.fit(measures, distance, kernel)
            initializers["MRDS projected"] = np.asarray(
                mrds_result.metadata["selected_indices_before_refinement"],
                dtype=np.int64,
            )

            evaluated: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
            refined_candidates: dict[str, object] = {}
            for name, selected in initializers.items():
                optimized = optimize_mixture_weights(
                    kernel,
                    p_full,
                    selected,
                    args.alpha,
                    max_iter=args.weight_max_iter,
                )
                evaluated[name] = (
                    np.asarray(selected, dtype=np.int64),
                    optimized.weights,
                    "initializer",
                )
                refined = refine_observed_subset(
                    kernel=kernel,
                    p=p_full,
                    initial_selected=selected,
                    alpha=args.alpha,
                    initializer_name=name,
                    optimize_weights=True,
                    max_passes=args.refinement_passes,
                    weight_max_iter=args.weight_max_iter,
                )
                refined_candidates[name] = refined
                refined_name = (
                    "MRDS-IS-R (proposed)"
                    if name == "MRDS projected"
                    else f"{name}-R"
                )
                evaluated[refined_name] = (
                    refined.selected,
                    refined.weights,
                    "same-objective one-swap refinement",
                )
                refine_rows.append({
                    **split_common,
                    "budget": budget,
                    "initializer": name,
                    "refined_method": refined_name,
                    "objective_before": float(optimized.objective),
                    "objective_after": float(refined.objective),
                    "objective_improvement": float(optimized.objective - refined.objective),
                    "accepted_swaps": len(refined.accepted_swaps),
                    "selected_changed": bool(
                        set(map(int, selected)) != set(map(int, refined.selected))
                    ),
                })

            best_name, best_refined = min(
                refined_candidates.items(), key=lambda item: item[1].objective
            )
            evaluated["Best-of-starts-R (diagnostic only)"] = (
                best_refined.selected,
                best_refined.weights,
                f"diagnostic oracle over supplied starts; winner={best_name}",
            )

            for method, (selected, renyi_weights, stage) in evaluated.items():
                selected = np.asarray(selected, dtype=np.int64)
                if np.unique(selected).size != selected.size:
                    raise RuntimeError(f"Duplicate indices returned by {method}")
                _, voronoi = assignments_and_weights(distance, selected)
                group_diagnostics = selection_group_diagnostics(selected, train_groups)
                selection_metrics = evaluate_selection(
                    distance,
                    kernel,
                    selected,
                    p_full,
                    args.alpha,
                    y_train,
                    renyi_weights,
                )
                common = {
                    **split_common,
                    "budget": budget,
                    "method": method,
                    "stage": stage,
                    "n_selected": len(selected),
                    **group_diagnostics,
                }
                rep_rows.append({
                    **common,
                    "reduction_fraction": 1.0 - len(selected) / len(train_idx),
                    "alpha": args.alpha,
                    "max_atoms": args.max_atoms,
                    "bandwidth": bandwidth,
                    "distance_backend": "emd_exact",
                    "distance_seconds": distance_seconds,
                    **selection_metrics,
                })
                _record_classification_rows(
                    cls_rows,
                    common,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    selected,
                    voronoi,
                    renyi_weights,
                )
                for rank, local_idx in enumerate(selected):
                    global_idx = int(train_idx[int(local_idx)])
                    source = manifest.iloc[global_idx]
                    selected_rows.append({
                        **common,
                        "rank": rank + 1,
                        "train_local_index": int(local_idx),
                        "global_index": global_idx,
                        "recording_id": source["recording_id"],
                        "condition_id": source["condition_id"],
                        "repetition": int(source["repetition"]),
                        "label": int(source["label"]),
                        "voronoi_weight": float(voronoi[rank]),
                        "renyi_weight": float(renyi_weights[rank]),
                    })

            stage_rows.append({
                **split_common,
                "budget": budget,
                "n_selected": n_selected,
                **mrds_result.metadata["stagewise_objectives"],
                "mrds_refined_only_objective": float(
                    refined_candidates["MRDS projected"].objective
                ),
                "best_of_starts_objective": float(best_refined.objective),
                "best_of_starts_initializer": best_name,
            })

        split_summary_rows.append({
            **split_common,
            "train_stable": int(np.sum(y_train == 0)),
            "train_chatter": int(np.sum(y_train == 1)),
            "test_stable": int(np.sum(y_test == 0)),
            "test_chatter": int(np.sum(y_test == 1)),
            "condition_overlap": 0,
            "distance_seconds": distance_seconds,
            "total_seconds": time.perf_counter() - fold_start,
        })
        np.savez_compressed(
            args.output / f"split_{fold.split:02d}_cache.npz",
            train_idx=train_idx,
            test_idx=test_idx,
            center=center,
            scale=scale,
            distance=distance,
            kernel=kernel,
        )

    pd.DataFrame(fold_rows).to_csv(args.output / "folds_grouped.csv", index=False)
    pd.DataFrame(split_summary_rows).to_csv(args.output / "split_summary.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(args.output / "representativeness.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(args.output / "classification.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(
        args.output / "selected_recording_ids.csv", index=False
    )
    pd.DataFrame(stage_rows).to_csv(args.output / "stagewise_objectives.csv", index=False)
    pd.DataFrame(refine_rows).to_csv(args.output / "refinement_attribution.csv", index=False)

    metadata = {
        **vars(args),
        "input": str(args.input),
        "output": str(args.output),
        "n_recordings": len(bags),
        "n_conditions": int(manifest["condition_id"].nunique()),
        "n_archive_features": len(archive_fields),
        "n_model_features": len(model_fields),
        "outer_protocol": "repeated stratified 4-fold by machining condition",
        "leakage_unit": "condition_id after removing only _R1--_R4",
        "primary_metric": "balanced_accuracy",
        "status": (
            "primary grouped diagnostic"
            if args.feature_set == "all40"
            else "exploratory grouped feature-ablation protocol"
        ),
        "method_attribution_note": (
            "MRDS-IS-R (proposed) is refinement of the projected MRDS start only; "
            "Best-of-starts-R is retained as a diagnostic and is not attributed to MRDS."
        ),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
