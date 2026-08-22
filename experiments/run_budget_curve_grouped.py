#!/usr/bin/env python3
"""Predeclared 10--50% retention curve on frozen condition-grouped folds.

The curve compares the proposed MRDS start with the two strongest deterministic
starts.  Every method receives the same finite-Renyi one-swap refinement and
uniform-weight Gaussian NB downstream classifier.  Results are checkpointed
after each outer fold so an interrupted long run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "experiments"))

from grouped_protocol import assert_group_disjoint, build_group_manifest  # noqa: E402
from meta_renyi_reduction import (  # noqa: E402
    MetaRenyiReducer,
    compress_bag,
    facility_location,
    finite_meta_p,
    median_bandwidth,
    pairwise_emd_exact,
    pam_kmedoids,
    rbf_kernel_from_distance,
)
from mrds_projection_refinement_integrated import refine_observed_subset  # noqa: E402
from run_chatter import apply_transform, bag_embeddings, fit_robust_transform, load_chatter_mat_dir  # noqa: E402
from run_chatter_audited import classification_metrics  # noqa: E402


METHODS = ("MRDS-IS-R", "FacilityLocation-R", "W2-KMedoids-R")


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[position]))
        adjusted[position] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "chatterData")
    parser.add_argument(
        "--outer-folds", type=Path,
        default=ROOT / "results" / "grouped_chatter_all40" / "folds_grouped.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "grouped_budget_curve_all40",
    )
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--synthetic-iterations", type=int, default=1)
    parser.add_argument("--refinement-passes", type=int, default=1)
    parser.add_argument("--weight-max-iter", type=int, default=10)
    args = parser.parse_args()
    if any(not 0.0 < budget <= 1.0 for budget in args.budgets):
        raise ValueError("budgets must lie in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)

    score_path = args.output / "outer_test_scores.csv"
    selected_path = args.output / "selected_recordings.csv"
    scores = pd.read_csv(score_path).to_dict("records") if score_path.exists() else []
    selected_rows = (
        pd.read_csv(selected_path).to_dict("records") if selected_path.exists() else []
    )
    completed = set(pd.DataFrame(scores).get("split", pd.Series(dtype=int)).astype(int))

    bags, labels, names, _ = load_chatter_mat_dir(args.input)
    manifest = build_group_manifest(names, labels)
    groups = manifest["condition_id"].astype(str).to_numpy()
    folds = pd.read_csv(args.outer_folds)

    for split in sorted(folds["split"].astype(int).unique()):
        if split in completed:
            print(f"skipping completed outer split {split:02d}", flush=True)
            continue
        rows = folds[folds["split"] == split]
        train_idx = np.sort(rows[rows.partition == "train"].global_index.to_numpy(dtype=np.int64))
        test_idx = np.sort(rows[rows.partition == "test"].global_index.to_numpy(dtype=np.int64))
        assert_group_disjoint(train_idx, test_idx, groups)
        seed = int(rows.seed.iloc[0])
        train_raw = [bags[int(i)] for i in train_idx]
        test_raw = [bags[int(i)] for i in test_idx]
        center, scale = fit_robust_transform(train_raw)
        train_bags = apply_transform(train_raw, center, scale)
        test_bags = apply_transform(test_raw, center, scale)
        measures = [
            compress_bag(bag, args.max_atoms, seed + split * 1000 + i)
            for i, bag in enumerate(train_bags)
        ]
        distance = pairwise_emd_exact(measures)
        bandwidth = median_bandwidth(distance)
        kernel = rbf_kernel_from_distance(distance, bandwidth)
        p_full = finite_meta_p(kernel)
        x_train, x_test = bag_embeddings(train_bags), bag_embeddings(test_bags)
        y_train, y_test = labels[train_idx], labels[test_idx]
        common = {
            "split": split,
            "repeat": int(rows.repeat.iloc[0]),
            "fold": int(rows.fold.iloc[0]),
            "seed": seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        scores.append({
            **common,
            "budget": 1.0,
            "n_selected": len(train_idx),
            "method": "Full training set",
            **classification_metrics(x_train, y_train, x_test, y_test, None),
        })

        for budget in args.budgets:
            n_selected = max(2, int(round(budget * len(train_idx))))
            reducer = MetaRenyiReducer(
                n_prototypes=n_selected,
                alpha=args.alpha,
                bandwidth=bandwidth,
                prototype_atoms=args.max_atoms,
                max_iter=args.synthetic_iterations,
                seed=seed,
                transport_backend="emd_exact",
                refinement_max_passes=0,
                refinement_weight_max_iter=args.weight_max_iter,
            )
            mrds = reducer.fit(measures, distance, kernel)
            starts = {
                "MRDS-IS-R": np.asarray(
                    mrds.metadata["selected_indices_before_refinement"], dtype=np.int64
                ),
                "FacilityLocation-R": facility_location(kernel, n_selected),
                "W2-KMedoids-R": pam_kmedoids(distance, n_selected, seed=seed),
            }
            for method, start in starts.items():
                refined = refine_observed_subset(
                    kernel=kernel,
                    p=p_full,
                    initial_selected=start,
                    alpha=args.alpha,
                    initializer_name=method,
                    optimize_weights=True,
                    max_passes=args.refinement_passes,
                    weight_max_iter=args.weight_max_iter,
                )
                selected_y = y_train[refined.selected]
                scores.append({
                    **common,
                    "budget": budget,
                    "n_selected": n_selected,
                    "method": method,
                    "selected_stable": int(np.sum(selected_y == 0)),
                    "selected_chatter": int(np.sum(selected_y == 1)),
                    **classification_metrics(
                        x_train[refined.selected], selected_y, x_test, y_test, None
                    ),
                })
                for rank, local_idx in enumerate(refined.selected):
                    global_idx = int(train_idx[int(local_idx)])
                    selected_rows.append({
                        **common,
                        "budget": budget,
                        "n_selected": n_selected,
                        "method": method,
                        "rank": rank + 1,
                        "recording_id": names[global_idx],
                        "condition_id": groups[global_idx],
                        "label": int(labels[global_idx]),
                    })
        pd.DataFrame(scores).to_csv(score_path, index=False)
        pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
        print(f"checkpointed budget curve for outer split {split:02d}", flush=True)

    frame = pd.DataFrame(scores)
    reduced = frame[frame.method.isin(METHODS)].copy()
    summary = reduced.groupby(["budget", "method"], as_index=False).agg(
        folds=("balanced_accuracy", "count"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        mean_sensitivity=("sensitivity", "mean"),
        mean_specificity=("specificity", "mean"),
        mean_selected_stable=("selected_stable", "mean"),
        mean_selected_chatter=("selected_chatter", "mean"),
    )
    summary.to_csv(args.output / "aggregate_scores.csv", index=False)

    comparisons: list[dict[str, object]] = []
    raw_p: list[float] = []
    for budget in args.budgets:
        pivot = reduced[np.isclose(reduced.budget, budget)].pivot(
            index="split", columns="method", values="balanced_accuracy"
        )
        for competitor in METHODS[1:]:
            paired = pivot[["MRDS-IS-R", competitor]].dropna()
            delta = paired["MRDS-IS-R"] - paired[competitor]
            try:
                p_value = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_value = 1.0
            raw_p.append(p_value)
            comparisons.append({
                "budget": budget,
                "competitor": competitor,
                "mean_delta_balanced_accuracy": float(delta.mean()),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "wilcoxon_p_raw": p_value,
            })
    for row, adjusted in zip(comparisons, holm_adjust(raw_p)):
        row["wilcoxon_p_holm_10_tests"] = adjusted
        row["significant_at_0.05"] = bool(adjusted < 0.05)
    pd.DataFrame(comparisons).to_csv(args.output / "paired_comparisons.csv", index=False)
    (args.output / "run_metadata.json").write_text(
        json.dumps({
            **vars(args),
            "input": str(args.input),
            "outer_folds": str(args.outer_folds),
            "output": str(args.output),
            "protocol": "frozen 20 condition-disjoint folds; uniform downstream weights",
            "multiplicity": "Holm correction across 5 budgets x 2 MRDS comparisons",
        }, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
