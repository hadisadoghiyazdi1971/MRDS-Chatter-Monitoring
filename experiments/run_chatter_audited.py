#!/usr/bin/env python3
"""Repeated exact-EMD chatter evaluation with MRDS-IS-R certificates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "experiments"))

from meta_renyi_reduction import (  # noqa: E402
    MetaRenyiReducer,
    assignments_and_weights,
    compress_bag,
    evaluate_selection,
    facility_location,
    finite_meta_p,
    optimize_mixture_weights,
    pairwise_emd_exact,
    pam_kmedoids,
    random_selection,
    rbf_kernel_from_distance,
    median_bandwidth,
)
from run_chatter import (  # noqa: E402
    add_impulsive_outlier_noise_at_snr,
    apply_transform,
    bag_embeddings,
    fit_robust_transform,
    load_chatter_mat_dir,
)


def classification_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, float]:
    if np.unique(y_train).size < 2:
        return {"balanced_accuracy": np.nan, "macro_f1": np.nan, "auroc": np.nan,
                "sensitivity": np.nan, "specificity": np.nan}
    clf = GaussianNB()
    clf.fit(x_train, y_train, sample_weight=weights)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)[:, list(clf.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "auroc": float(roc_auc_score(y_test, prob)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "chatterData")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "audited_chatter")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--max-atoms", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--max-iter", type=int, default=0)
    ap.add_argument("--refinement-passes", type=int, default=1)
    ap.add_argument("--weight-max-iter", type=int, default=30)
    ap.add_argument("--snr-db", type=float, nargs="*", default=[0, -5, -10, -15, -20])
    ap.add_argument("--impulse-prob", type=float, default=0.05)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bags, labels, names, fields = load_chatter_mat_dir(args.input)
    manifest_rows = []
    for path, bag, label, name in zip(sorted(args.input.glob("*.mat")), bags, labels, names):
        manifest_rows.append({
            "file_id": name,
            "path": str(path),
            "label_source": "filename prefix",
            "label": int(label),
            "n_windows": int(bag.shape[0]),
            "feature_dimension": int(bag.shape[1]),
            "missing_values": int(np.isnan(bag).sum()),
            "sha256": file_sha256(path),
            "excluded": False,
            "exclusion_reason": "",
        })
    pd.DataFrame(manifest_rows).to_csv(args.output / "dataset_manifest.csv", index=False)

    rep_rows: list[dict] = []
    cls_rows: list[dict] = []
    noise_rows: list[dict] = []
    cert_rows: list[dict] = []
    selected_rows: list[dict] = []
    stage_rows: list[dict] = []
    line_search_rows: list[dict] = []

    for split, seed in enumerate(args.seeds):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
        if set(train_idx) & set(test_idx):
            raise RuntimeError("train/test leakage detected")
        train_raw = [bags[i] for i in train_idx]
        test_raw = [bags[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        center, scale = fit_robust_transform(train_raw)
        train_bags = apply_transform(train_raw, center, scale)
        test_bags = apply_transform(test_raw, center, scale)
        measures = [compress_bag(x, args.max_atoms, seed + i) for i, x in enumerate(train_bags)]

        distance_start = time.perf_counter()
        d_pair = pairwise_emd_exact(measures)
        distance_seconds = time.perf_counter() - distance_start
        h = median_bandwidth(d_pair)
        kernel = rbf_kernel_from_distance(d_pair, h)
        p = finite_meta_p(kernel)
        x_train = bag_embeddings(train_bags)
        x_test = bag_embeddings(test_bags)

        full_metrics = classification_metrics(x_train, y_train, x_test, y_test, None)
        cls_rows.append({"split": split, "seed": seed, "budget": 1.0,
                         "method": "Full training set", "n_selected": len(train_idx),
                         **full_metrics})

        for budget in args.budgets:
            m = max(2, int(round(budget * len(train_idx))))
            starts = {
                "Random": random_selection(len(train_idx), m, seed),
                "FacilityLocation-init": facility_location(kernel, m),
                "W2-KMedoids-init": pam_kmedoids(d_pair, m, seed=seed),
            }
            reducer = MetaRenyiReducer(
                n_prototypes=m,
                alpha=args.alpha,
                bandwidth=h,
                prototype_atoms=args.max_atoms,
                max_iter=args.max_iter,
                seed=seed,
                transport_backend="emd_exact",
                refinement_max_passes=args.refinement_passes,
                refinement_weight_max_iter=args.weight_max_iter,
            )
            proposed = reducer.fit(measures, d_pair, kernel)
            projected = np.asarray(
                proposed.metadata["selected_indices_before_refinement"], dtype=np.int64
            )
            methods = {**starts, "MRDS-projected-init": projected,
                       "MRDS-IS-R (proposed)": proposed.selected_indices}

            for method, selected in methods.items():
                selected = np.asarray(selected, dtype=np.int64)
                if np.unique(selected).size != selected.size:
                    raise RuntimeError(f"duplicate selected indices: {method}")
                _, voronoi = assignments_and_weights(d_pair, selected)
                optimized = optimize_mixture_weights(
                    kernel, p, selected, args.alpha, max_iter=args.weight_max_iter
                )
                optimized_objective = (
                    float(proposed.metadata["final_renyi_optimized_weights"])
                    if method == "MRDS-IS-R (proposed)"
                    else float(optimized.objective)
                )
                metrics = evaluate_selection(
                    d_pair, kernel, selected, p, args.alpha, y_train, voronoi
                )
                rep_rows.append({
                    "split": split, "seed": seed, "budget": budget,
                    "reduction_fraction": 1.0 - len(selected) / len(train_idx),
                    "method": method, "n_train": len(train_idx),
                    "n_selected": len(selected), "alpha": args.alpha,
                    "max_atoms": args.max_atoms, "distance_backend": "emd_exact",
                    "distance_seconds": distance_seconds,
                    "renyi_optimized_weights": optimized_objective,
                    **metrics,
                })
                cls = classification_metrics(
                    x_train[selected], y_train[selected], x_test, y_test, voronoi
                )
                cls_rows.append({"split": split, "seed": seed, "budget": budget,
                                 "method": method, "n_selected": len(selected), **cls})
                for rank, local_idx in enumerate(selected):
                    global_idx = int(train_idx[int(local_idx)])
                    selected_rows.append({
                        "split": split, "seed": seed, "budget": budget,
                        "method": method, "rank": rank + 1,
                        "train_local_index": int(local_idx), "global_index": global_idx,
                        "recording_id": names[global_idx], "label": int(labels[global_idx]),
                    })
                for snr in args.snr_db:
                    rng = np.random.default_rng(seed + int(100 * snr) + 9187)
                    noisy = add_impulsive_outlier_noise_at_snr(
                        test_bags, snr, rng, args.impulse_prob
                    )
                    noisy_metrics = classification_metrics(
                        x_train[selected], y_train[selected], bag_embeddings(noisy), y_test, voronoi
                    )
                    noise_rows.append({
                        "split": split, "seed": seed, "budget": budget,
                        "method": method, "snr_db": snr,
                        "impulse_probability": args.impulse_prob, **noisy_metrics,
                    })

            initial_opt = proposed.metadata["initializer_optimized_objectives"]
            certificate = proposed.metadata["dominance_certificate"]
            cert_rows.append({
                "split": split, "seed": seed, "budget": budget,
                "m": m, "alpha": args.alpha,
                "projected_mrds_initial": initial_opt["projected_mrds"],
                "facility_location_initial": initial_opt["facility_location"],
                "kmedoids_initial": initial_opt["kmedoids"],
                "final_objective": proposed.metadata["final_renyi_optimized_weights"],
                "chosen_initializer": proposed.metadata["chosen_initializer"],
                "final_le_projected_mrds": certificate["projected_mrds"],
                "final_le_facility_location": certificate["facility_location"],
                "final_le_kmedoids": certificate["kmedoids"],
                "tolerance": proposed.metadata["dominance_tolerance"],
            })
            stage_rows.append({"split": split, "seed": seed, "budget": budget,
                               **proposed.metadata["stagewise_objectives"]})
            for attempt in proposed.metadata["line_search_audit"]:
                line_search_rows.append({
                    "split": split, "seed": seed, "budget": budget, **attempt
                })

        np.savez_compressed(
            args.output / f"split_{split}_cache.npz", train_idx=train_idx,
            test_idx=test_idx, center=center, scale=scale, distance=d_pair, kernel=kernel,
        )

    pd.DataFrame(rep_rows).to_csv(args.output / "representativeness.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(args.output / "classification.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(args.output / "impulsive_noise.csv", index=False)
    pd.DataFrame(cert_rows).to_csv(args.output / "dominance_certificates.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output / "selected_recording_ids.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(args.output / "stagewise_objectives.csv", index=False)
    pd.DataFrame(line_search_rows).to_csv(args.output / "line_search_audit.csv", index=False)
    metadata = vars(args).copy()
    metadata.update({"input": str(args.input), "output": str(args.output),
                     "n_bags": len(bags), "n_features": len(fields),
                     "status": "confirmatory exact-EMD repeated holdout"})
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
