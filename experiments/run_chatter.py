#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from meta_renyi_reduction import (  # noqa: E402
    MetaRenyiReducer,
    agglomerative_medoids,
    assignments_and_weights,
    augment_time,
    class_conditional_robust_mrds_selection,
    compress_bag,
    display_method_name,
    evaluate_selection,
    facility_location,
    farthest_first,
    finite_meta_p,
    kernel_herding,
    median_bandwidth,
    mmd_critic_prototypes,
    pairwise_sinkhorn,
    pairwise_sliced,
    pam_kmedoids,
    protodash_like,
    random_selection,
    rbf_kernel_from_distance,
)


IGNORED_FIELDS = {"FileName", "Label"}


def scalar(x) -> float:
    arr = np.asarray(x).squeeze()
    return float(arr)


def load_chatter_mat_dir(path: Path) -> tuple[list[np.ndarray], np.ndarray, list[str], list[str]]:
    files = sorted(path.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found under {path}")

    first = sio.loadmat(files[0], squeeze_me=True, struct_as_record=False)["SigData"]
    fields = [f for f in dir(first[0]) if not f.startswith("_") and f not in IGNORED_FIELDS]
    fields = sorted(fields)

    bags: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for f in files:
        sig = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["SigData"]
        x = np.array([[scalar(getattr(row, field)) for field in fields] for row in sig], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(fields):
            raise ValueError(f"Unexpected feature matrix in {f}: {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite feature value in {f}")
        bags.append(x)
        names.append(f.stem)
        labels.append(1 if f.name.upper().startswith("U_") else 0)

    return bags, np.asarray(labels, dtype=np.int64), names, fields


def fit_robust_transform(train_bags: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate(train_bags, axis=0)
    center = np.median(pooled, axis=0)
    mad = 1.4826 * np.median(np.abs(pooled - center), axis=0)
    std = pooled.std(axis=0)
    scale = np.where(mad > 1e-12, mad, np.where(std > 1e-12, std, 1.0))
    return center, scale


def apply_transform(bags: list[np.ndarray], center: np.ndarray, scale: np.ndarray) -> list[np.ndarray]:
    return [(x - center) / scale for x in bags]


def bag_embeddings(bags: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for x in bags:
        q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75], axis=0)
        blocks.append(np.concatenate([x.mean(axis=0), x.std(axis=0), q25, q50, q75]))
    return np.vstack(blocks)


def classifier_scores(x_train, y_train, x_test, y_test, sample_weight=None) -> list[dict]:
    rows: list[dict] = []
    classifiers = {
        "GaussianNB": GaussianNB(),
        "SVM-RBF": SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
    }
    for name, clf in classifiers.items():
        if len(np.unique(y_train)) < 2:
            rows.append({"classifier": name, "status": "failed: one class selected"})
            continue
        start = time.perf_counter()
        try:
            clf.fit(x_train, y_train, sample_weight=sample_weight)
            pred_train = clf.predict(x_train)
            pred_test = clf.predict(x_test)
            rows.append(
                {
                    "classifier": name,
                    "status": "ok",
                    "fit_seconds": time.perf_counter() - start,
                    "train_accuracy": accuracy_score(y_train, pred_train),
                    "train_balanced_accuracy": balanced_accuracy_score(y_train, pred_train),
                    "train_macro_f1": f1_score(y_train, pred_train, average="macro"),
                    "test_accuracy": accuracy_score(y_test, pred_test),
                    "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
                    "test_macro_f1": f1_score(y_test, pred_test, average="macro"),
                }
            )
        except Exception as exc:
            rows.append({"classifier": name, "status": f"failed: {exc}"})
    return rows


def gaussian_nb_balanced_accuracy(x_train, y_train, x_val, y_val, sample_weight=None) -> float:
    if len(np.unique(y_train)) < 2:
        return 0.0
    try:
        clf = GaussianNB()
        clf.fit(x_train, y_train, sample_weight=sample_weight)
        return float(balanced_accuracy_score(y_val, clf.predict(x_val)))
    except Exception:
        return 0.0


def selection_methods(d_pair: np.ndarray, k_full: np.ndarray, m: int, seed: int):
    return [
        ("Random", lambda: (random_selection(d_pair.shape[0], m, seed), None)),
        ("W2-FarthestFirst", lambda: (farthest_first(d_pair, m, seed), None)),
        ("W2-KMedoids", lambda: (pam_kmedoids(d_pair, m, seed=seed), None)),
        ("W2-Agglomerative-Complete", lambda: (agglomerative_medoids(d_pair, m, "complete"), None)),
        ("FacilityLocation", lambda: (facility_location(k_full, m), None)),
        ("MMD-Critic", lambda: (mmd_critic_prototypes(k_full, m), None)),
        ("KernelHerding", lambda: (kernel_herding(k_full, m), None)),
        ("ProtoDash-like", lambda: protodash_like(k_full, m)),
    ]


def class_conditional_robust_mrds(
    measures,
    y_train: np.ndarray,
    d_pair: np.ndarray,
    k_full: np.ndarray,
    total_m: int,
    alpha: float,
    reg: float,
    prototype_atoms: int,
    max_iter: int,
    seed: int,
    correntropy_multiplier: float,
    correntropy_gamma: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    return class_conditional_robust_mrds_selection(
        measures,
        y_train,
        d_pair,
        k_full,
        total_m,
        alpha,
        reg,
        prototype_atoms,
        max_iter,
        seed,
        correntropy_multiplier,
        correntropy_gamma,
    )


def add_impulsive_outlier_noise_at_snr(
    bags: list[np.ndarray],
    snr_db: float,
    rng: np.random.Generator,
    impulse_prob: float,
) -> list[np.ndarray]:
    noisy = []
    factor = 10.0 ** (snr_db / 10.0)
    for x in bags:
        power = float(np.mean(x * x))
        target_noise_power = power / max(factor, 1e-12)
        mask = rng.random(size=x.shape) < impulse_prob
        raw = rng.standard_t(df=2.5, size=x.shape) * mask
        raw_power = float(np.mean(raw * raw))
        if raw_power <= 1e-18:
            raw.flat[int(rng.integers(raw.size))] = 1.0
            raw_power = float(np.mean(raw * raw))
        noise = raw * np.sqrt(target_noise_power / raw_power)
        noisy.append(x + noise)
    return noisy


def scale_noise_to_snr(x: np.ndarray, raw_noise: np.ndarray, snr_db: float) -> np.ndarray:
    factor = 10.0 ** (snr_db / 10.0)
    signal_power = float(np.mean(x * x))
    target_noise_power = signal_power / max(factor, 1e-12)
    raw_power = float(np.mean(raw_noise * raw_noise))
    if raw_power <= 1e-18:
        raw_noise = raw_noise.copy()
        raw_noise.flat[0] = 1.0
        raw_power = float(np.mean(raw_noise * raw_noise))
    return raw_noise * np.sqrt(target_noise_power / raw_power)


def add_gaussian_noise_at_snr(bags: list[np.ndarray], snr_db: float, rng: np.random.Generator) -> list[np.ndarray]:
    noisy = []
    for x in bags:
        noise = scale_noise_to_snr(x, rng.normal(size=x.shape), snr_db)
        noisy.append(x + noise)
    return noisy


def add_gaussian_mixture_noise_at_snr(
    bags: list[np.ndarray],
    snr_db: float,
    rng: np.random.Generator,
    burst_prob: float,
) -> list[np.ndarray]:
    noisy = []
    for x in bags:
        base = rng.normal(size=x.shape)
        bursts = rng.normal(scale=8.0, size=x.shape)
        mask = rng.random(size=x.shape) < burst_prob
        raw = np.where(mask, bursts, base)
        noise = scale_noise_to_snr(x, raw, snr_db)
        noisy.append(x + noise)
    return noisy


def add_noise_at_snr(
    bags: list[np.ndarray],
    snr_db: float,
    rng: np.random.Generator,
    noise_model: str,
    impulse_prob: float,
) -> list[np.ndarray]:
    if noise_model == "impulsive":
        return add_impulsive_outlier_noise_at_snr(bags, snr_db, rng, impulse_prob)
    if noise_model == "gaussian":
        return add_gaussian_noise_at_snr(bags, snr_db, rng)
    if noise_model == "gaussian_mixture":
        return add_gaussian_mixture_noise_at_snr(bags, snr_db, rng, impulse_prob)
    raise ValueError(f"Unknown noise model: {noise_model}")


def estimate_snr_db_from_windows(bags: list[np.ndarray]) -> float:
    estimates: list[float] = []
    for x in bags:
        if x.shape[0] < 5:
            continue
        prev_x = np.roll(x, 1, axis=0)
        next_x = np.roll(x, -1, axis=0)
        smooth = np.median(np.stack([prev_x, x, next_x], axis=0), axis=0)
        smooth[0] = x[0]
        smooth[-1] = x[-1]
        residual = x - smooth
        signal_power = float(np.mean(smooth * smooth))
        noise_power = float(np.mean(residual * residual))
        if noise_power > 1e-18:
            estimates.append(10.0 * np.log10(max(signal_power, 1e-18) / noise_power))
    return float(np.mean(estimates)) if estimates else float("nan")


def tune_alpha_on_training_split(
    train_bags: list[np.ndarray],
    measures,
    y_train: np.ndarray,
    d_pair: np.ndarray,
    k_full: np.ndarray,
    m: int,
    alpha_grid: list[float],
    validation_snr_db: list[float],
    impulse_prob: float,
    reg: float,
    prototype_atoms: int,
    max_iter: int,
    seed: int,
    correntropy_multiplier: float,
    correntropy_gamma: float,
) -> tuple[float, list[dict]]:
    if len(alpha_grid) == 1:
        return float(alpha_grid[0]), []

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 701)
    inner_sel, inner_val = next(splitter.split(np.zeros(len(y_train)), y_train))
    inner_m = max(2, int(round(m * len(inner_sel) / len(y_train))))
    inner_m = min(inner_m, len(inner_sel))
    x_all = bag_embeddings(train_bags)
    x_inner = x_all[inner_sel]
    y_inner = y_train[inner_sel]
    x_val_clean = x_all[inner_val]
    y_val = y_train[inner_val]
    d_inner = d_pair[np.ix_(inner_sel, inner_sel)]
    k_inner = k_full[np.ix_(inner_sel, inner_sel)]
    measures_inner = [measures[int(i)] for i in inner_sel]

    rows: list[dict] = []
    for alpha in alpha_grid:
        start = time.perf_counter()
        selected, weights, _ = class_conditional_robust_mrds_selection(
            measures_inner,
            y_inner,
            d_inner,
            k_inner,
            inner_m,
            float(alpha),
            reg,
            prototype_atoms,
            max_iter,
            seed + int(round(1000 * float(alpha))),
            correntropy_multiplier,
            correntropy_gamma,
        )
        x_sel = x_inner[selected]
        y_sel = y_inner[selected]
        scores = [gaussian_nb_balanced_accuracy(x_sel, y_sel, x_val_clean, y_val, weights)]
        for snr_db in validation_snr_db:
            rng = np.random.default_rng(seed + int(round(100 * snr_db)) + int(round(1000 * float(alpha))) + 8801)
            noisy_val = add_impulsive_outlier_noise_at_snr(
                [train_bags[int(i)] for i in inner_val],
                snr_db,
                rng,
                impulse_prob,
            )
            scores.append(gaussian_nb_balanced_accuracy(x_sel, y_sel, bag_embeddings(noisy_val), y_val, weights))
        rows.append(
            {
                "alpha": float(alpha),
                "validation_score": float(np.mean(scores)),
                "validation_balanced_accuracy_clean": float(scores[0]),
                "validation_snr_db": " ".join(f"{x:g}" for x in validation_snr_db),
                "runtime_seconds": time.perf_counter() - start,
            }
        )

    best = max(rows, key=lambda r: (r["validation_score"], -r["alpha"]))
    return float(best["alpha"]), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("chatterData"))
    ap.add_argument("--output", type=Path, default=Path("results/chatter"))
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--distance-backend", choices=["sinkhorn", "sliced"], default="sinkhorn")
    ap.add_argument("--max-atoms", type=int, default=16)
    ap.add_argument("--prototype-atoms", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--alpha-grid", type=float, nargs="*", default=None)
    ap.add_argument("--alpha-validation-snr-db", type=float, nargs="*", default=[0.0, -10.0, -20.0])
    ap.add_argument("--reg", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--projections", type=int, default=128)
    ap.add_argument("--quantiles", type=int, default=128)
    ap.add_argument("--time-beta", type=float, default=0.0)
    ap.add_argument("--correntropy-multiplier", type=float, default=1.0)
    ap.add_argument("--correntropy-gamma", type=float, default=1.0)
    ap.add_argument("--snr-db", type=float, nargs="*", default=[30.0, 20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0])
    ap.add_argument("--noise-models", choices=["impulsive", "gaussian", "gaussian_mixture"], nargs="+", default=["impulsive", "gaussian", "gaussian_mixture"])
    ap.add_argument("--impulse-prob", type=float, default=0.05)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    bags, y, names, fields = load_chatter_mat_dir(args.input)
    np.savez_compressed(args.output / "chatter_bags.npz", X=np.asarray(bags, dtype=object), y=y, names=np.asarray(names))

    manifest = {
        "dataset": "chatterData",
        "n_bags": len(bags),
        "feature_dimension": len(fields),
        "bag_size_min": int(min(x.shape[0] for x in bags)),
        "bag_size_median": float(np.median([x.shape[0] for x in bags])),
        "bag_size_max": int(max(x.shape[0] for x in bags)),
        "class_counts": {"stable": int(np.sum(y == 0)), "chatter": int(np.sum(y == 1))},
        "label_rule": "filename starts with U_ => chatter/unstable; filename starts with S_ => stable",
        "feature_fields": fields,
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    representation_rows: list[dict] = []
    classification_rows: list[dict] = []
    noise_rows: list[dict] = []
    selected_rows: list[dict] = []
    alpha_rows: list[dict] = []

    for split_id, seed in enumerate(args.seeds):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
        train_bags_raw = [bags[i] for i in train_idx]
        test_bags_raw = [bags[i] for i in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        center, scale = fit_robust_transform(train_bags_raw)
        train_bags = augment_time(apply_transform(train_bags_raw, center, scale), args.time_beta)
        test_bags = augment_time(apply_transform(test_bags_raw, center, scale), args.time_beta)
        measures = [compress_bag(x, args.max_atoms, seed + i) for i, x in enumerate(train_bags)]

        distance_start = time.perf_counter()
        if args.distance_backend == "sinkhorn":
            d_pair = pairwise_sinkhorn(measures, reg=args.reg, debias=False)
        else:
            d_pair = pairwise_sliced(measures, args.projections, args.quantiles, seed)
        distance_seconds = time.perf_counter() - distance_start
        h = median_bandwidth(d_pair)
        k_full = rbf_kernel_from_distance(d_pair, h)
        p = finite_meta_p(k_full)
        m = max(2, int(round(args.budget * len(train_idx))))
        alpha_grid = args.alpha_grid if args.alpha_grid else [args.alpha]
        selected_alpha, alpha_diagnostics = tune_alpha_on_training_split(
            train_bags,
            measures,
            y_train,
            d_pair,
            k_full,
            m,
            [float(a) for a in alpha_grid],
            [float(s) for s in args.alpha_validation_snr_db],
            args.impulse_prob,
            args.reg,
            args.prototype_atoms,
            args.max_iter,
            seed,
            args.correntropy_multiplier,
            args.correntropy_gamma,
        )
        for row in alpha_diagnostics:
            alpha_rows.append({"split": split_id, "seed": seed, "selected_alpha": selected_alpha, **row})

        x_train_emb = bag_embeddings(train_bags)
        x_test_emb = bag_embeddings(test_bags)
        for row in classifier_scores(x_train_emb, y_train, x_test_emb, y_test):
            classification_rows.append(
                {
                    "split": split_id,
                    "seed": seed,
                    "budget": 1.0,
                    "method": "Full training set",
                    "n_selected": len(train_idx),
                    **row,
                }
            )
        for noise_model in args.noise_models:
            for snr_db in args.snr_db:
                rng_noise = np.random.default_rng(seed + int(round(100 * snr_db)) + 54321 + 1009 * args.noise_models.index(noise_model))
                noisy_test = add_noise_at_snr(test_bags, snr_db, rng_noise, noise_model, args.impulse_prob)
                estimated_snr_db = estimate_snr_db_from_windows(noisy_test)
                x_test_noisy = bag_embeddings(noisy_test)
                for row in classifier_scores(x_train_emb, y_train, x_test_noisy, y_test):
                    if row.get("status") == "ok":
                        noise_rows.append(
                            {
                                "split": split_id,
                                "seed": seed,
                                "noise_model": noise_model,
                                "snr_db": snr_db,
                                "estimated_snr_db": estimated_snr_db,
                                "outlier_degree": args.impulse_prob if noise_model != "gaussian" else 1.0,
                                "method": "Full training set",
                                "classifier": row["classifier"],
                                "n_selected": len(train_idx),
                                "test_balanced_accuracy": row["test_balanced_accuracy"],
                                "test_macro_f1": row["test_macro_f1"],
                            }
                        )

        methods = selection_methods(d_pair, k_full, m, seed)
        if m <= len(train_idx):
            methods.append(
                (
                    "MRDS-exact-finite",
                    lambda: (
                        MetaRenyiReducer(
                            n_prototypes=m,
                            alpha=selected_alpha,
                            reg=args.reg,
                            bandwidth=h,
                            prototype_atoms=args.prototype_atoms,
                            max_iter=args.max_iter,
                            seed=seed,
                        ).fit(measures, d_pair, k_full),
                        None,
                    ),
                )
            )
            methods.append(
                (
                    "MRDS-robust-class-conditional",
                    lambda: class_conditional_robust_mrds(
                        measures,
                        y_train,
                        d_pair,
                        k_full,
                        m,
                        selected_alpha,
                        args.reg,
                        args.prototype_atoms,
                        args.max_iter,
                        seed,
                        args.correntropy_multiplier,
                        args.correntropy_gamma,
                    ),
                )
            )

        for raw_name, selector in methods:
            start = time.perf_counter()
            selected_weights = None
            objective_history: list[float] = []
            if raw_name == "MRDS-exact-finite":
                result, _ = selector()
                selected = result.selected_indices
                selected_weights = result.representative_weights
                objective_history = result.objective_history
                runtime_seconds = result.runtime_seconds
            elif raw_name == "MRDS-robust-class-conditional":
                selected, selected_weights, robust_metadata = selector()
                runtime_seconds = time.perf_counter() - start
                objective_history = [
                    v
                    for class_run in robust_metadata.get("class_runs", [])
                    for v in class_run.get("objective_history", [])
                ]
            else:
                selected, selected_weights = selector()
                runtime_seconds = time.perf_counter() - start
            method = (
                "Robust class-conditional MRDS (proposed)"
                if raw_name == "MRDS-robust-class-conditional"
                else display_method_name(raw_name)
            )
            _, rep_w = assignments_and_weights(d_pair, selected)
            eval_weights = selected_weights if selected_weights is not None else rep_w
            metrics = evaluate_selection(d_pair, k_full, selected, p, selected_alpha, y_train, eval_weights)
            representation_rows.append(
                {
                    "split": split_id,
                    "seed": seed,
                    "alpha": selected_alpha,
                    "method": method,
                    "budget": args.budget,
                    "n_train": len(train_idx),
                    "n_selected": len(selected),
                    "distance_runtime_seconds": distance_seconds,
                    "runtime_seconds": runtime_seconds,
                    **metrics,
                }
            )

            selected_local = np.asarray(selected, dtype=int)
            selected_global = train_idx[selected_local]
            for rank, (local_i, global_i, weight) in enumerate(zip(selected_local, selected_global, eval_weights)):
                selected_rows.append(
                    {
                        "split": split_id,
                        "seed": seed,
                        "alpha": selected_alpha,
                        "method": method,
                        "rank": rank + 1,
                        "train_local_index": int(local_i),
                        "global_index": int(global_i),
                        "recording_id": names[int(global_i)],
                        "label": int(y[int(global_i)]),
                        "representative_weight": float(weight),
                    }
                )

            x_sel = x_train_emb[selected_local]
            y_sel = y_train[selected_local]
            for row in classifier_scores(x_sel, y_sel, x_test_emb, y_test, sample_weight=eval_weights):
                classification_rows.append(
                    {
                        "split": split_id,
                        "seed": seed,
                        "budget": args.budget,
                        "alpha": selected_alpha,
                        "method": method,
                        "n_selected": len(selected),
                        **row,
                    }
                )

            for noise_model in args.noise_models:
                for snr_db in args.snr_db:
                    rng_noise = np.random.default_rng(seed + int(round(100 * snr_db)) + 12345 + 1009 * args.noise_models.index(noise_model))
                    noisy_test = add_noise_at_snr(test_bags, snr_db, rng_noise, noise_model, args.impulse_prob)
                    estimated_snr_db = estimate_snr_db_from_windows(noisy_test)
                    x_test_noisy = bag_embeddings(noisy_test)
                    for row in classifier_scores(x_sel, y_sel, x_test_noisy, y_test, sample_weight=eval_weights):
                        if row.get("status") == "ok":
                            noise_rows.append(
                                {
                                    "split": split_id,
                                    "seed": seed,
                                    "noise_model": noise_model,
                                    "snr_db": snr_db,
                                    "estimated_snr_db": estimated_snr_db,
                                    "outlier_degree": args.impulse_prob if noise_model != "gaussian" else 1.0,
                                    "method": method,
                                    "alpha": selected_alpha,
                                    "classifier": row["classifier"],
                                    "n_selected": len(selected),
                                    "test_balanced_accuracy": row["test_balanced_accuracy"],
                                    "test_macro_f1": row["test_macro_f1"],
                                }
                            )

            if objective_history:
                (args.output / f"objective_split{split_id}_{method.replace(' ', '_').replace('/', '-')}.json").write_text(
                    json.dumps({"objective_history": objective_history}, indent=2),
                    encoding="utf-8",
                )

    rep = pd.DataFrame(representation_rows)
    cls = pd.DataFrame(classification_rows)
    noise = pd.DataFrame(noise_rows)
    sel = pd.DataFrame(selected_rows)
    rep.to_csv(args.output / "representation_summary.csv", index=False)
    cls.to_csv(args.output / "classification_summary.csv", index=False)
    noise.to_csv(args.output / "noise_snr_summary.csv", index=False)
    sel.to_csv(args.output / "selected_recording_ids.csv", index=False)
    if alpha_rows:
        pd.DataFrame(alpha_rows).to_csv(args.output / "alpha_selection.csv", index=False)

    print("Representation summary")
    print(rep.groupby("method")[["renyi_meta", "coverage_mean", "mmd2", "runtime_seconds"]].mean().sort_values("renyi_meta"))
    print("\nClassification summary")
    ok = cls[cls["status"] == "ok"]
    print(ok.groupby(["method", "classifier"])[["test_balanced_accuracy", "test_macro_f1"]].mean().sort_values("test_balanced_accuracy", ascending=False))
    if not noise.empty:
        print("\nNoise/SNR summary")
        print(noise.groupby(["method", "classifier", "snr_db"])[["test_balanced_accuracy"]].mean())


if __name__ == "__main__":
    main()
