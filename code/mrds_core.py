#!/usr/bin/env python3
"""Core numerical routines for the reported MRDS implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import ot
from numpy.typing import NDArray
from sklearn.cluster import KMeans


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class EmpiricalMeasure:
    """A finite probability measure with vector-valued support atoms."""

    support: FloatArray
    weights: FloatArray

    def validate(self) -> None:
        if self.support.ndim != 2:
            raise ValueError("support must have shape (n_atoms, n_features)")
        if self.weights.ndim != 1 or self.weights.size != self.support.shape[0]:
            raise ValueError("weights must have shape (n_atoms,)")
        if np.any(self.weights < 0.0) or not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite and nonnegative")
        total = float(self.weights.sum())
        if total <= 0.0:
            raise ValueError("weights must have positive total mass")
        if not np.isfinite(self.support).all():
            raise ValueError("support contains a non-finite value")
        self.weights = self.weights / total


def compress_bag(features: FloatArray, max_atoms: int, seed: int) -> EmpiricalMeasure:
    """Compress one recording with seeded K-means and cluster masses."""

    n_rows = features.shape[0]
    if n_rows <= max_atoms:
        result = EmpiricalMeasure(
            features.copy(), np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
        )
        result.validate()
        return result
    model = KMeans(
        n_clusters=max_atoms,
        n_init=5,
        random_state=seed,
        max_iter=200,
    )
    labels = model.fit_predict(features)
    counts = np.bincount(labels, minlength=max_atoms).astype(np.float64)
    result = EmpiricalMeasure(
        model.cluster_centers_.astype(np.float64), counts / counts.sum()
    )
    result.validate()
    return result


def squared_cost(left: FloatArray, right: FloatArray) -> FloatArray:
    """Pairwise squared Euclidean ground cost."""

    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


def emd_exact_coupling(
    source: EmpiricalMeasure,
    target: EmpiricalMeasure,
    max_iter: int = 100000,
) -> tuple[FloatArray, float]:
    """Exact balanced transport coupling and squared transport cost."""

    cost = squared_cost(source.support, target.support)
    coupling = np.asarray(
        ot.emd(source.weights, target.weights, cost, numItermax=max_iter),
        dtype=np.float64,
    )
    return coupling, max(float(np.sum(coupling * cost)), 0.0)


def barycentric_projection(
    coupling: FloatArray,
    target_support: FloatArray,
    source_weights: FloatArray,
) -> FloatArray:
    """Project target support through a source-to-target coupling."""

    return (coupling @ target_support) / np.maximum(
        source_weights[:, None], 1e-300
    )


def pairwise_emd_exact(measures: Sequence[EmpiricalMeasure]) -> FloatArray:
    """Pairwise exact Wasserstein-2 distances."""

    n_measures = len(measures)
    distance = np.zeros((n_measures, n_measures), dtype=np.float64)
    for left in range(n_measures):
        for right in range(left + 1, n_measures):
            _, cost = emd_exact_coupling(measures[left], measures[right])
            distance[left, right] = distance[right, left] = math.sqrt(cost)
    return distance


def median_bandwidth(distance: FloatArray, eps: float = 1e-12) -> float:
    """Median positive off-diagonal distance."""

    values = distance[np.triu_indices_from(distance, k=1)]
    values = values[values > eps]
    return float(np.median(values)) if values.size else 1.0


def rbf_kernel_from_distance(
    distance: FloatArray,
    bandwidth: float,
    kernel_floor: float = 1e-12,
) -> FloatArray:
    """Finite meta-kernel used by MRDS and the structured selectors."""

    if bandwidth <= 0.0 or kernel_floor <= 0.0:
        raise ValueError("bandwidth and kernel_floor must be positive")
    return np.maximum(
        np.exp(-(distance * distance) / (2.0 * bandwidth * bandwidth)),
        kernel_floor,
    )


def probability_from_scores(
    scores: FloatArray, eps: float = 1e-15
) -> FloatArray:
    values = np.maximum(np.asarray(scores, dtype=np.float64), eps)
    return values / values.sum()


def renyi_probability_divergence(
    p: FloatArray,
    q: FloatArray,
    alpha: float,
    eps: float = 1e-15,
) -> float:
    """Rényi divergence between normalized finite probability vectors."""

    if alpha <= 0.0 or np.isclose(alpha, 1.0):
        raise ValueError("alpha must be positive and different from one")
    p_normalized = probability_from_scores(p, eps)
    q_normalized = probability_from_scores(q, eps)
    integral = np.sum(
        (p_normalized**alpha) * (q_normalized ** (1.0 - alpha))
    )
    return float(np.log(max(float(integral), eps)) / (alpha - 1.0))


def finite_meta_p(kernel: FloatArray) -> FloatArray:
    """Finite-anchor probability vector for the training meta-distribution."""

    return probability_from_scores(kernel.mean(axis=1))


def finite_meta_q(anchor_prototype_kernel: FloatArray) -> FloatArray:
    """Finite-anchor probability vector for equally weighted prototypes."""

    return probability_from_scores(anchor_prototype_kernel.mean(axis=1))


def exact_signed_coefficients(
    p: FloatArray,
    q: FloatArray,
    anchor_prototype_kernel: FloatArray,
    alpha: float,
) -> tuple[FloatArray, float]:
    integral = float(np.sum((p**alpha) * (q ** (1.0 - alpha))))
    density_ratio = (p**alpha) * (q ** (-alpha))
    coefficients = (
        density_ratio[:, None] - integral
    ) * anchor_prototype_kernel
    return coefficients, integral


def prototype_kernel_and_projections(
    anchors: Sequence[EmpiricalMeasure],
    prototypes: Sequence[EmpiricalMeasure],
    bandwidth: float,
    need_projections: bool,
) -> tuple[FloatArray, list[list[FloatArray | None]], FloatArray]:
    """Evaluate prototype-to-anchor kernels, projections, and transport costs."""

    n_anchors = len(anchors)
    n_prototypes = len(prototypes)
    kernel = np.empty((n_anchors, n_prototypes), dtype=np.float64)
    costs = np.empty((n_anchors, n_prototypes), dtype=np.float64)
    projections: list[list[FloatArray | None]] = [
        [None for _ in range(n_anchors)] for _ in range(n_prototypes)
    ]
    for prototype_index, prototype in enumerate(prototypes):
        for anchor_index, anchor in enumerate(anchors):
            coupling, cost = emd_exact_coupling(prototype, anchor)
            costs[anchor_index, prototype_index] = cost
            kernel[anchor_index, prototype_index] = max(
                math.exp(-cost / (2.0 * bandwidth * bandwidth)), 1e-12
            )
            if need_projections:
                projections[prototype_index][anchor_index] = barycentric_projection(
                    coupling, anchor.support, prototype.weights
                )
    return kernel, projections, costs


def prototype_kernel_column(
    anchors: Sequence[EmpiricalMeasure],
    prototype: EmpiricalMeasure,
    bandwidth: float,
    need_projections: bool,
) -> tuple[FloatArray, list[FloatArray | None], FloatArray]:
    """Evaluate one prototype column for a Gauss-Seidel update."""

    n_anchors = len(anchors)
    kernel = np.empty(n_anchors, dtype=np.float64)
    costs = np.empty(n_anchors, dtype=np.float64)
    projections: list[FloatArray | None] = [None] * n_anchors
    for anchor_index, anchor in enumerate(anchors):
        coupling, cost = emd_exact_coupling(prototype, anchor)
        costs[anchor_index] = cost
        kernel[anchor_index] = max(
            math.exp(-cost / (2.0 * bandwidth * bandwidth)), 1e-12
        )
        if need_projections:
            projections[anchor_index] = barycentric_projection(
                coupling, anchor.support, prototype.weights
            )
    return kernel, projections, costs


def farthest_first(distance: FloatArray, count: int, seed: int) -> IntArray:
    """Seed a set by farthest-first traversal."""

    generator = np.random.default_rng(seed)
    first = int(generator.integers(distance.shape[0]))
    selected = [first]
    nearest = distance[:, first].copy()
    while len(selected) < count:
        nearest[selected] = -np.inf
        next_index = int(np.argmax(nearest))
        selected.append(next_index)
        nearest = np.minimum(nearest, distance[:, next_index])
    return np.asarray(selected, dtype=np.int64)


def initialize_prototypes(
    measures: Sequence[EmpiricalMeasure],
    distance: FloatArray,
    n_prototypes: int,
    prototype_atoms: int,
    seed: int,
) -> list[EmpiricalMeasure]:
    """Initialize synthetic prototypes from a farthest-first observed set."""

    if not 1 <= n_prototypes <= len(measures):
        raise ValueError("n_prototypes must be between one and the sample size")
    indices = farthest_first(distance, n_prototypes, seed)
    prototypes: list[EmpiricalMeasure] = []
    for offset, index in enumerate(indices):
        source = measures[int(index)]
        if source.support.shape[0] <= prototype_atoms:
            prototypes.append(
                EmpiricalMeasure(source.support.copy(), source.weights.copy())
            )
        else:
            prototypes.append(
                compress_bag(source.support, prototype_atoms, seed + offset)
            )
    return prototypes


def pam_kmedoids(
    distance: FloatArray,
    count: int,
    max_iter: int = 100,
    seed: int = 42,
) -> IntArray:
    """PAM-style best-swap selection on the Wasserstein distance matrix."""

    selected = farthest_first(distance, count, seed).tolist()
    current = float(np.min(distance[:, selected], axis=1).sum())
    for _ in range(max_iter):
        best_gain = 0.0
        best_swap: tuple[int, int] | None = None
        unselected = [
            index for index in range(distance.shape[0]) if index not in selected
        ]
        for position, _old in enumerate(selected):
            for new_index in unselected:
                trial = selected.copy()
                trial[position] = new_index
                cost = float(np.min(distance[:, trial], axis=1).sum())
                gain = current - cost
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_swap = (position, new_index)
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
        current -= best_gain
    return np.asarray(sorted(selected), dtype=np.int64)


def facility_location(kernel: FloatArray, count: int) -> IntArray:
    """Greedy Facility Location selection on the finite meta-kernel."""

    selected: list[int] = []
    current = np.zeros(kernel.shape[0], dtype=np.float64)
    available = np.ones(kernel.shape[0], dtype=bool)
    for _ in range(count):
        candidates = np.flatnonzero(available)
        gains = np.array(
            [
                np.maximum(current, kernel[:, index]).sum() - current.sum()
                for index in candidates
            ]
        )
        best = int(candidates[int(np.argmax(gains))])
        selected.append(best)
        available[best] = False
        current = np.maximum(current, kernel[:, best])
    return np.asarray(selected, dtype=np.int64)


def uniform_subset_objective(
    kernel: FloatArray,
    p: FloatArray,
    selected: IntArray,
    alpha: float,
) -> float:
    """Finite Rényi objective of an equal-mass observed subset."""

    weights = np.full(selected.size, 1.0 / selected.size, dtype=np.float64)
    q = probability_from_scores(kernel[:, selected] @ weights)
    return renyi_probability_divergence(p, q, alpha)
