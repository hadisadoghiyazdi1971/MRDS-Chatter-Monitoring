#!/usr/bin/env python3
"""
Meta-Renyi Distributional Sampling (MRDS) and distribution-valued baselines.

One observation is a bag / empirical probability distribution:
    bag_i = {z_i1, ..., z_iT_i}, z_ir in R^d.

The proposed finite-anchor objective is the exact Renyi divergence between
probability vectors p and q evaluated on the observed meta-samples. In the
confirmatory path, prototype atoms are updated with barycentric projections of
exact balanced OT couplings and an Armijo backtracking line search.

Python Optimal Transport (POT) is required for the confirmatory ``emd_exact``
backend. The optional diagnostic Sinkhorn backend uses POT when available and
otherwise uses a NumPy fallback.

Input formats
-------------
1) NPZ: X can be a dense (N,T,d) array or an object array of (T_i,d) arrays.
        Optional y is a bag-level label vector.
2) CSV: one row per instance/window, with bag_id, optional label, and features.

Example
-------
python meta_renyi_reduction.py synthetic --output out/synthetic --n-bags 60
python meta_renyi_reduction.py run --input chatter.npz --m 30 --output out/chatter
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment, nnls
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from mrds_projection_refinement_integrated import (
    full_meta_probability as refinement_full_meta_probability,
    multistart_refinement,
    optimize_mixture_weights,
    subset_objective as refined_subset_objective,
)

try:
    import ot  # type: ignore
    HAVE_POT = True
except Exception:
    ot = None
    HAVE_POT = False

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class EmpiricalMeasure:
    support: FloatArray
    weights: FloatArray

    def validate(self) -> None:
        if self.support.ndim != 2:
            raise ValueError("support must have shape (n_atoms, d)")
        if self.weights.ndim != 1 or self.weights.size != self.support.shape[0]:
            raise ValueError("weights must have shape (n_atoms,)")
        if np.any(self.weights < 0) or not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite and nonnegative")
        total = float(self.weights.sum())
        if total <= 0:
            raise ValueError("weights must have positive total mass")
        self.weights = self.weights / total
        if not np.isfinite(self.support).all():
            raise ValueError("support contains NaN or infinity")


@dataclass
class ReductionResult:
    method: str
    selected_indices: IntArray
    representative_weights: FloatArray
    objective_history: list[float]
    runtime_seconds: float
    metadata: dict


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_npz(path: Path, x_key: str = "X", y_key: str = "y") -> tuple[list[FloatArray], NDArray | None]:
    data = np.load(path, allow_pickle=True)
    if x_key not in data:
        raise KeyError(f"{x_key!r} not found. Available arrays: {list(data.keys())}")
    raw = data[x_key]
    if raw.dtype == object:
        bags = [np.asarray(x, dtype=np.float64) for x in raw.tolist()]
    elif raw.ndim == 3:
        bags = [np.asarray(raw[i], dtype=np.float64) for i in range(raw.shape[0])]
    else:
        raise ValueError("X must have shape (N,T,d) or be an object array of (T_i,d) arrays")
    labels = np.asarray(data[y_key]) if y_key in data else None
    _validate_bags(bags, labels)
    return bags, labels


def load_mil_csv(
    path: Path,
    bag_id_col: str = "bag_id",
    label_col: str | None = "label",
    feature_cols: Sequence[str] | None = None,
) -> tuple[list[FloatArray], NDArray | None]:
    df = pd.read_csv(path)
    if bag_id_col not in df:
        raise KeyError(f"Missing bag id column {bag_id_col!r}")
    excluded = {bag_id_col}
    if label_col is not None:
        excluded.add(label_col)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns were found")
    bags: list[FloatArray] = []
    labels: list[object] = []
    for _, group in df.groupby(bag_id_col, sort=True):
        bags.append(group.loc[:, feature_cols].to_numpy(dtype=np.float64))
        if label_col is not None and label_col in group:
            unique = group[label_col].dropna().unique()
            if len(unique) != 1:
                raise ValueError("Each bag must have exactly one bag-level label")
            labels.append(unique[0])
    y = np.asarray(labels) if labels else None
    _validate_bags(bags, y)
    return bags, y


def _validate_bags(bags: Sequence[FloatArray], labels: NDArray | None) -> None:
    if len(bags) < 2:
        raise ValueError("At least two bags are required")
    d = bags[0].shape[1]
    for i, bag in enumerate(bags):
        if bag.ndim != 2 or bag.shape[0] < 2 or bag.shape[1] != d:
            raise ValueError(f"Bag {i} has invalid shape {bag.shape}; expected (T_i,{d})")
        if not np.isfinite(bag).all():
            raise ValueError(f"Bag {i} contains NaN or infinite values")
    if labels is not None and len(labels) != len(bags):
        raise ValueError("Number of labels does not match number of bags")


def robust_standardize(bags: Sequence[FloatArray], eps: float = 1e-12) -> tuple[list[FloatArray], FloatArray, FloatArray]:
    pooled = np.concatenate(bags, axis=0)
    center = np.median(pooled, axis=0)
    mad = 1.4826 * np.median(np.abs(pooled - center), axis=0)
    std = pooled.std(axis=0)
    scale = np.where(mad > eps, mad, np.where(std > eps, std, 1.0))
    return [(x - center) / scale for x in bags], center, scale


def augment_time(bags: Sequence[FloatArray], beta: float) -> list[FloatArray]:
    if beta <= 0:
        return [x.copy() for x in bags]
    out: list[FloatArray] = []
    for x in bags:
        t = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float64)[:, None]
        out.append(np.concatenate([x, beta * t], axis=1))
    return out


def compress_bag(x: FloatArray, max_atoms: int, seed: int) -> EmpiricalMeasure:
    """Compress a bag with K-means and retain cluster proportions as masses."""
    n = x.shape[0]
    if n <= max_atoms:
        measure = EmpiricalMeasure(x.copy(), np.full(n, 1.0 / n))
        measure.validate()
        return measure
    model = KMeans(n_clusters=max_atoms, n_init=5, random_state=seed, max_iter=200)
    labels = model.fit_predict(x)
    counts = np.bincount(labels, minlength=max_atoms).astype(np.float64)
    measure = EmpiricalMeasure(model.cluster_centers_.astype(np.float64), counts / counts.sum())
    measure.validate()
    return measure


# ---------------------------------------------------------------------------
# Optimal transport and meta-kernels
# ---------------------------------------------------------------------------

def squared_cost(x: FloatArray, y: FloatArray) -> FloatArray:
    xx = np.sum(x * x, axis=1)[:, None]
    yy = np.sum(y * y, axis=1)[None, :]
    c = xx + yy - 2.0 * x @ y.T
    return np.maximum(c, 0.0)


def sinkhorn_coupling_numpy(
    a: FloatArray,
    b: FloatArray,
    c: FloatArray,
    reg: float,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> FloatArray:
    """Balanced entropic OT using stabilized positive scaling iterations."""
    if reg <= 0:
        raise ValueError("reg must be positive")
    a = np.maximum(a / a.sum(), 1e-300)
    b = np.maximum(b / b.sum(), 1e-300)
    shifted = c - float(c.min())
    k = np.exp(np.clip(-shifted / reg, -700.0, 0.0))
    k = np.maximum(k, 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for it in range(max_iter):
        u_prev = u.copy()
        kv = k @ v
        u = a / np.maximum(kv, 1e-300)
        ktu = k.T @ u
        v = b / np.maximum(ktu, 1e-300)
        if it % 10 == 0 and np.max(np.abs(u - u_prev)) < tol:
            break
    pi = (u[:, None] * k) * v[None, :]
    # A few balancing corrections improve marginal accuracy.
    for _ in range(3):
        pi *= (a / np.maximum(pi.sum(axis=1), 1e-300))[:, None]
        pi *= (b / np.maximum(pi.sum(axis=0), 1e-300))[None, :]
    return pi


def sinkhorn_coupling(
    mu: EmpiricalMeasure,
    nu: EmpiricalMeasure,
    reg: float,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> tuple[FloatArray, float]:
    c = squared_cost(mu.support, nu.support)
    if HAVE_POT:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi = ot.sinkhorn(mu.weights, nu.weights, c, reg, numItermax=max_iter, stopThr=tol)
        pi = np.asarray(pi, dtype=np.float64)
    else:
        pi = sinkhorn_coupling_numpy(mu.weights, nu.weights, c, reg, max_iter=max_iter, tol=tol)
    cost = float(np.sum(pi * c))
    return pi, max(cost, 0.0)


def emd_exact_coupling(
    mu: EmpiricalMeasure,
    nu: EmpiricalMeasure,
    max_iter: int = 100000,
) -> tuple[FloatArray, float]:
    """Exact balanced OT coupling and its matching squared ground cost."""
    if not HAVE_POT:
        raise RuntimeError("emd_exact requires Python Optimal Transport (POT)")
    c = squared_cost(mu.support, nu.support)
    pi = np.asarray(
        ot.emd(mu.weights, nu.weights, c, numItermax=max_iter),
        dtype=np.float64,
    )
    cost = float(np.sum(pi * c))
    return pi, max(cost, 0.0)


def transport_coupling(
    mu: EmpiricalMeasure,
    nu: EmpiricalMeasure,
    backend: str,
    reg: float,
) -> tuple[FloatArray, float]:
    if backend == "emd_exact":
        return emd_exact_coupling(mu, nu)
    if backend == "sinkhorn_cross_cost_diagnostic":
        return sinkhorn_coupling(mu, nu, reg)
    raise ValueError(f"Unsupported prototype transport backend: {backend}")


def barycentric_projection(pi: FloatArray, target_support: FloatArray, source_weights: FloatArray) -> FloatArray:
    return (pi @ target_support) / np.maximum(source_weights[:, None], 1e-300)


def sinkhorn_distance(mu: EmpiricalMeasure, nu: EmpiricalMeasure, reg: float, debias: bool = False) -> float:
    _, cross = sinkhorn_coupling(mu, nu, reg)
    if not debias:
        return math.sqrt(max(cross, 0.0))
    _, self_mu = sinkhorn_coupling(mu, mu, reg)
    _, self_nu = sinkhorn_coupling(nu, nu, reg)
    div = cross - 0.5 * self_mu - 0.5 * self_nu
    return math.sqrt(max(div, 0.0))


def pairwise_sinkhorn(measures: Sequence[EmpiricalMeasure], reg: float, debias: bool = False) -> FloatArray:
    n = len(measures)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dij = sinkhorn_distance(measures[i], measures[j], reg=reg, debias=debias)
            d[i, j] = d[j, i] = dij
    return d


def pairwise_emd_exact(measures: Sequence[EmpiricalMeasure]) -> FloatArray:
    """Pairwise exact W2 distances using the same EMD coupling as prototypes."""
    n = len(measures)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            _, cost = emd_exact_coupling(measures[i], measures[j])
            d[i, j] = d[j, i] = math.sqrt(cost)
    return d


def projected_quantile_embedding(
    measures: Sequence[EmpiricalMeasure],
    n_projections: int,
    n_quantiles: int,
    seed: int,
) -> FloatArray:
    """Fast approximate sliced-W2 embedding, including nonuniform masses."""
    rng = np.random.default_rng(seed)
    d = measures[0].support.shape[1]
    theta = rng.normal(size=(n_projections, d))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    out = np.empty((len(measures), n_projections * n_quantiles), dtype=np.float64)
    for i, measure in enumerate(measures):
        proj = measure.support @ theta.T
        block = np.empty((n_projections, n_quantiles), dtype=np.float64)
        for ell in range(n_projections):
            order = np.argsort(proj[:, ell])
            vals = proj[order, ell]
            weights = measure.weights[order]
            cdf = np.cumsum(weights)
            cdf = np.concatenate([[0.0], cdf])
            vals_ext = np.concatenate([[vals[0]], vals])
            block[ell] = np.interp(q, cdf, vals_ext)
        out[i] = block.ravel() / math.sqrt(n_projections * n_quantiles)
    return out


def pairwise_sliced(
    measures: Sequence[EmpiricalMeasure],
    n_projections: int = 128,
    n_quantiles: int = 128,
    seed: int = 42,
) -> FloatArray:
    emb = projected_quantile_embedding(measures, n_projections, n_quantiles, seed)
    return squareform(pdist(emb, metric="euclidean"))


def median_bandwidth(d: FloatArray, eps: float = 1e-12) -> float:
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > eps]
    return float(np.median(vals)) if vals.size else 1.0


def rbf_kernel_from_distance(
    d: FloatArray,
    h: float,
    kernel_floor: float = 1e-12,
) -> FloatArray:
    """Positive RBF kernel with a declared floor preventing tail underflow."""
    if h <= 0.0 or kernel_floor <= 0.0:
        raise ValueError("h and kernel_floor must be positive")
    return np.maximum(np.exp(-(d * d) / (2.0 * h * h)), kernel_floor)


# ---------------------------------------------------------------------------
# Exact finite-anchor Renyi objective
# ---------------------------------------------------------------------------

def probability_from_scores(scores: FloatArray, eps: float = 1e-15) -> FloatArray:
    x = np.maximum(np.asarray(scores, dtype=np.float64), eps)
    return x / x.sum()


def renyi_probability_divergence(p: FloatArray, q: FloatArray, alpha: float, eps: float = 1e-15) -> float:
    if alpha <= 0 or np.isclose(alpha, 1.0):
        raise ValueError("alpha must be positive and different from 1")
    p = probability_from_scores(p, eps)
    q = probability_from_scores(q, eps)
    a = np.sum((p ** alpha) * (q ** (1.0 - alpha)))
    return float(np.log(max(a, eps)) / (alpha - 1.0))


def finite_meta_p(k_full: FloatArray, anchor_weights: FloatArray | None = None) -> FloatArray:
    if anchor_weights is None:
        return probability_from_scores(k_full.mean(axis=1))
    w = probability_from_scores(anchor_weights)
    return probability_from_scores(k_full @ w)


def prototype_kernel_and_projections(
    anchors: Sequence[EmpiricalMeasure],
    prototypes: Sequence[EmpiricalMeasure],
    reg: float,
    h: float,
    need_projections: bool,
    backend: str = "emd_exact",
) -> tuple[FloatArray, list[list[FloatArray | None]], FloatArray]:
    n, m = len(anchors), len(prototypes)
    k = np.empty((n, m), dtype=np.float64)
    costs = np.empty((n, m), dtype=np.float64)
    projections: list[list[FloatArray | None]] = [[None for _ in range(n)] for _ in range(m)]
    for j, proto in enumerate(prototypes):
        for i, anchor in enumerate(anchors):
            pi, cost = transport_coupling(proto, anchor, backend=backend, reg=reg)
            costs[i, j] = cost
            k[i, j] = max(math.exp(-cost / (2.0 * h * h)), 1e-12)
            if need_projections:
                projections[j][i] = barycentric_projection(pi, anchor.support, proto.weights)
    return k, projections, costs


def prototype_kernel_column(
    anchors: Sequence[EmpiricalMeasure],
    prototype: EmpiricalMeasure,
    reg: float,
    h: float,
    need_projections: bool,
    backend: str = "emd_exact",
) -> tuple[FloatArray, list[FloatArray | None], FloatArray]:
    """Evaluate one prototype column for efficient Gauss-Seidel updates."""
    n = len(anchors)
    kernel = np.empty(n, dtype=np.float64)
    costs = np.empty(n, dtype=np.float64)
    projections: list[FloatArray | None] = [None] * n
    for i, anchor in enumerate(anchors):
        pi, cost = transport_coupling(prototype, anchor, backend=backend, reg=reg)
        costs[i] = cost
        kernel[i] = max(math.exp(-cost / (2.0 * h * h)), 1e-12)
        if need_projections:
            projections[i] = barycentric_projection(
                pi, anchor.support, prototype.weights
            )
    return kernel, projections, costs


def finite_meta_q(k_anchor_proto: FloatArray) -> FloatArray:
    return probability_from_scores(k_anchor_proto.mean(axis=1))


def exact_signed_coefficients(p: FloatArray, q: FloatArray, k_anchor_proto: FloatArray, alpha: float) -> tuple[FloatArray, float]:
    a = float(np.sum((p ** alpha) * (q ** (1.0 - alpha))))
    u = (p ** alpha) * (q ** (-alpha))
    c = (u[:, None] - a) * k_anchor_proto
    return c, a


def clone_prototypes(prototypes: Sequence[EmpiricalMeasure]) -> list[EmpiricalMeasure]:
    return [EmpiricalMeasure(p.support.copy(), p.weights.copy()) for p in prototypes]


class MetaRenyiReducer:
    """Finite normalized MRDS with exact signed first-variation coefficients."""

    def __init__(
        self,
        n_prototypes: int,
        alpha: float = 2.0,
        reg: float = 0.5,
        bandwidth: float | None = None,
        prototype_atoms: int = 24,
        max_iter: int = 30,
        step_size: float = 1.0,
        armijo: float = 1e-4,
        backtrack: float = 0.5,
        min_step: float = 1e-5,
        tol: float = 1e-5,
        seed: int = 42,
        anchor_weights: FloatArray | None = None,
        transport_backend: str = "emd_exact",
        update_schedule: str = "gauss_seidel",
        refinement_max_passes: int = 8,
        refinement_weight_max_iter: int = 50,
    ) -> None:
        self.n_prototypes = n_prototypes
        self.alpha = alpha
        self.reg = reg
        self.bandwidth = bandwidth
        self.prototype_atoms = prototype_atoms
        self.max_iter = max_iter
        self.step_size = step_size
        self.armijo = armijo
        self.backtrack = backtrack
        self.min_step = min_step
        self.tol = tol
        self.seed = seed
        self.anchor_weights = anchor_weights
        self.transport_backend = transport_backend
        self.update_schedule = update_schedule
        self.refinement_max_passes = refinement_max_passes
        self.refinement_weight_max_iter = refinement_weight_max_iter

    def _initialize(self, measures: Sequence[EmpiricalMeasure], d_pair: FloatArray) -> list[EmpiricalMeasure]:
        idx = farthest_first(d_pair, self.n_prototypes, seed=self.seed)
        out: list[EmpiricalMeasure] = []
        for j, i in enumerate(idx):
            src = measures[int(i)]
            if src.support.shape[0] <= self.prototype_atoms:
                out.append(EmpiricalMeasure(src.support.copy(), src.weights.copy()))
            else:
                compressed = compress_bag(src.support, self.prototype_atoms, seed=self.seed + j)
                out.append(compressed)
        return out

    def fit(self, measures: Sequence[EmpiricalMeasure], d_pair: FloatArray, k_full: FloatArray) -> ReductionResult:
        start = time.perf_counter()
        n = len(measures)
        if not 1 <= self.n_prototypes <= n:
            raise ValueError("n_prototypes must be between 1 and N")
        if self.update_schedule != "gauss_seidel":
            raise ValueError("The confirmatory reducer currently requires gauss_seidel updates")
        if self.transport_backend == "emd_exact" and any(
            measure.support.shape[0] != self.prototype_atoms for measure in measures
        ):
            raise ValueError(
                "emd_exact primary runs require max_atoms == prototype_atoms for every compressed bag"
            )
        p = finite_meta_p(k_full, self.anchor_weights)
        h = self.bandwidth if self.bandwidth is not None else median_bandwidth(d_pair)
        prototypes = self._initialize(measures, d_pair)
        history: list[float] = []

        k_ap, projections, _ = prototype_kernel_and_projections(
            measures, prototypes, reg=self.reg, h=h, need_projections=True,
            backend=self.transport_backend,
        )
        q = finite_meta_q(k_ap)
        current = renyi_probability_divergence(p, q, self.alpha)
        history.append(current)
        line_search_audit: list[dict] = []

        for iteration in range(self.max_iter):
            max_move = 0.0
            accepted_any = False

            for j, proto in enumerate(prototypes):
                # Gauss-Seidel: coefficients and projections belong to the
                # current state, including every earlier accepted update.
                c, _ = exact_signed_coefficients(p, q, k_ap, self.alpha)
                direction = np.zeros_like(proto.support)
                normalizer = float(np.sum(np.abs(c[:, j]))) + 1e-15
                for i in range(n):
                    bij = projections[j][i]
                    assert bij is not None
                    direction += c[i, j] * (bij - proto.support)
                direction /= normalizer
                dir_norm_sq = float(np.sum(proto.weights[:, None] * direction * direction))
                if dir_norm_sq <= self.tol * self.tol:
                    continue

                old_support = proto.support.copy()
                fd_step = 1e-6 / max(1.0, math.sqrt(dir_norm_sq))

                def objective_along(signed_step: float) -> float:
                    prototypes[j].support = old_support + signed_step * direction
                    column, _, _ = prototype_kernel_column(
                        measures, prototypes[j], reg=self.reg, h=h,
                        need_projections=False, backend=self.transport_backend,
                    )
                    k_fd = k_ap.copy()
                    k_fd[:, j] = column
                    return renyi_probability_divergence(
                        p, finite_meta_q(k_fd), self.alpha
                    )

                plus = objective_along(fd_step)
                minus = objective_along(-fd_step)
                prototypes[j].support = old_support
                directional_slope = (plus - minus) / (2.0 * fd_step)
                if directional_slope >= 0.0:
                    direction *= -1.0
                    directional_slope *= -1.0
                if directional_slope >= -1e-14:
                    continue

                step = self.step_size
                accepted = False
                backtracks = 0
                objective_before = current
                while step >= self.min_step:
                    prototypes[j].support = old_support + step * direction
                    trial_column, _, _ = prototype_kernel_column(
                        measures, prototypes[j], reg=self.reg, h=h,
                        need_projections=False, backend=self.transport_backend,
                    )
                    k_trial = k_ap.copy()
                    k_trial[:, j] = trial_column
                    q_trial = finite_meta_q(k_trial)
                    trial = renyi_probability_divergence(p, q_trial, self.alpha)
                    armijo_rhs = objective_before + self.armijo * step * directional_slope
                    if trial <= armijo_rhs:
                        k_ap = k_trial
                        q = q_trial
                        current = trial
                        accepted = True
                        accepted_any = True
                        max_move = max(max_move, step * math.sqrt(dir_norm_sq))
                        line_search_audit.append({
                            "iteration": iteration,
                            "prototype": j,
                            "current_objective": float(objective_before),
                            "directional_derivative": float(directional_slope),
                            "accepted_step": float(step),
                            "backtracks": backtracks,
                            "trial_objective": float(trial),
                            "armijo_rhs": float(armijo_rhs),
                            "accepted": True,
                        })
                        break
                    line_search_audit.append({
                        "iteration": iteration,
                        "prototype": j,
                        "current_objective": float(objective_before),
                        "directional_derivative": float(directional_slope),
                        "attempted_step": float(step),
                        "backtracks": backtracks,
                        "trial_objective": float(trial),
                        "armijo_rhs": float(armijo_rhs),
                        "accepted": False,
                    })
                    prototypes[j].support = old_support
                    step *= self.backtrack
                    backtracks += 1

                if not accepted:
                    prototypes[j].support = old_support

                # Projections must correspond to the accepted current prototypes.
                current_column, current_projections, _ = prototype_kernel_column(
                    measures, prototypes[j], reg=self.reg, h=h,
                    need_projections=True, backend=self.transport_backend,
                )
                k_ap[:, j] = current_column
                projections[j] = current_projections
                q = finite_meta_q(k_ap)
                current = renyi_probability_divergence(p, q, self.alpha)

            history.append(current)
            if not accepted_any or max_move < self.tol:
                break

        # Duplicate-free projection of synthetic prototypes to real bags.
        _, _, costs = prototype_kernel_and_projections(
            measures, prototypes, reg=self.reg, h=h, need_projections=False,
            backend=self.transport_backend,
        )
        row, col = linear_sum_assignment(costs.T)
        order = np.argsort(row)
        selected_initial = col[order].astype(np.int64)
        initializers = {
            "projected_mrds": selected_initial,
            "facility_location": facility_location(k_full, self.n_prototypes),
            "kmedoids": pam_kmedoids(d_pair, self.n_prototypes, seed=self.seed),
        }
        initial_uniform = {
            name: refined_subset_objective(k_full, p, subset, self.alpha)
            for name, subset in initializers.items()
        }
        initial_optimized = {
            name: optimize_mixture_weights(
                k_full, p, subset, self.alpha,
                max_iter=self.refinement_weight_max_iter,
            ).objective
            for name, subset in initializers.items()
        }
        best_refined, refined_runs = multistart_refinement(
            k_full, p, initializers, self.alpha, optimize_weights=True,
            max_passes=self.refinement_max_passes, improvement_tol=1e-10,
            weight_max_iter=self.refinement_weight_max_iter,
        )
        selected = best_refined.selected
        final_objective = float(best_refined.objective)
        tolerance = 1e-9
        dominance = {
            name: bool(final_objective <= value + tolerance)
            for name, value in initial_optimized.items()
        }
        if not all(dominance.values()):
            raise RuntimeError(f"MRDS-IS-R dominance certificate failed: {dominance}")
        assignments, weights = assignments_and_weights(d_pair, selected)
        runtime = time.perf_counter() - start
        return ReductionResult(
            method="MRDS-IS-R",
            selected_indices=selected,
            representative_weights=weights,
            objective_history=history,
            runtime_seconds=runtime,
            metadata={
                "alpha": self.alpha,
                "reg": self.reg,
                "bandwidth": h,
                "prototype_atoms": self.prototype_atoms,
                "iterations": len(history) - 1,
                "pot_backend": HAVE_POT,
                "pairwise_backend": self.transport_backend,
                "prototype_backend": self.transport_backend,
                "update_schedule": self.update_schedule,
                "line_search_audit": line_search_audit,
                "selected_indices_before_refinement": selected_initial.tolist(),
                "initializer_uniform_objectives": initial_uniform,
                "initializer_optimized_objectives": initial_optimized,
                "refined_objectives": {
                    name: float(result.objective) for name, result in refined_runs.items()
                },
                "chosen_initializer": best_refined.initializer,
                "final_renyi_optimized_weights": final_objective,
                "final_renyi_uniform": refined_subset_objective(
                    k_full, p, selected, self.alpha
                ),
                "optimized_mixture_weights": best_refined.weights.tolist(),
                "accepted_swaps": [list(pair) for pair in best_refined.accepted_swaps],
                "dominance_tolerance": tolerance,
                "dominance_certificate": dominance,
                "stagewise_objectives": {
                    "J_init": float(history[0]),
                    "J_synthetic": float(history[-1]),
                    "J_projected_uniform": float(initial_uniform["projected_mrds"]),
                    "J_projected_optimized_weights": float(initial_optimized["projected_mrds"]),
                    "J_refined": final_objective,
                },
                "projection_refinement": "authoritative multistart MRDS-IS-R",
            },
        )


# ---------------------------------------------------------------------------
# Baseline subset selectors on the common meta-distance / kernel
# ---------------------------------------------------------------------------

def assignments_and_weights(d: FloatArray, selected: Sequence[int]) -> tuple[IntArray, FloatArray]:
    sel = np.asarray(selected, dtype=np.int64)
    local = np.argmin(d[:, sel], axis=1)
    counts = np.bincount(local, minlength=len(sel)).astype(np.float64)
    return local.astype(np.int64), counts / counts.sum()


def random_selection(n: int, m: int, seed: int) -> IntArray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=m, replace=False)).astype(np.int64)


def farthest_first(d: FloatArray, m: int, seed: int = 42) -> IntArray:
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    selected = [first]
    nearest = d[:, first].copy()
    while len(selected) < m:
        nearest[selected] = -np.inf
        nxt = int(np.argmax(nearest))
        selected.append(nxt)
        nearest = np.minimum(nearest, d[:, nxt])
    return np.asarray(selected, dtype=np.int64)


def pam_kmedoids(d: FloatArray, m: int, max_iter: int = 100, seed: int = 42) -> IntArray:
    selected = farthest_first(d, m, seed=seed).tolist()
    current = float(np.min(d[:, selected], axis=1).sum())
    for _ in range(max_iter):
        best_gain = 0.0
        best_swap: tuple[int, int] | None = None
        unselected = [i for i in range(d.shape[0]) if i not in selected]
        for pos, old in enumerate(selected):
            for new in unselected:
                trial = selected.copy()
                trial[pos] = new
                cost = float(np.min(d[:, trial], axis=1).sum())
                gain = current - cost
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_swap = (pos, new)
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
        current -= best_gain
    return np.asarray(sorted(selected), dtype=np.int64)


def agglomerative_medoids(d: FloatArray, m: int, linkage_method: str = "complete") -> IntArray:
    if m == d.shape[0]:
        return np.arange(m, dtype=np.int64)
    z = linkage(squareform(d, checks=False), method=linkage_method)
    labels = fcluster(z, t=m, criterion="maxclust")
    selected: list[int] = []
    for cid in np.unique(labels):
        members = np.flatnonzero(labels == cid)
        sub = d[np.ix_(members, members)]
        selected.append(int(members[np.argmin(sub.sum(axis=1))]))
    if len(selected) < m:  # distance ties can collapse maxclust output
        remain = [i for i in range(d.shape[0]) if i not in selected]
        while len(selected) < m:
            score = np.min(d[np.ix_(remain, selected)], axis=1)
            selected.append(remain.pop(int(np.argmax(score))))
    return np.asarray(sorted(selected[:m]), dtype=np.int64)


def facility_location(k: FloatArray, m: int) -> IntArray:
    n = k.shape[0]
    selected: list[int] = []
    current = np.zeros(n, dtype=np.float64)
    available = np.ones(n, dtype=bool)
    for _ in range(m):
        candidates = np.flatnonzero(available)
        gains = np.array([np.maximum(current, k[:, c]).sum() - current.sum() for c in candidates])
        best = int(candidates[int(np.argmax(gains))])
        selected.append(best)
        available[best] = False
        current = np.maximum(current, k[:, best])
    return np.asarray(selected, dtype=np.int64)


def mmd_critic_prototypes(k: FloatArray, m: int) -> IntArray:
    n = k.shape[0]
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    colsum = 2.0 * k.sum(axis=0) / n
    for _ in range(m):
        candidates = np.flatnonzero(available)
        s1 = colsum[candidates].copy()
        if not selected:
            s1 -= np.abs(np.diag(k)[candidates])
        else:
            temp = k[np.ix_(selected, candidates)]
            s1 -= (2.0 * temp.sum(axis=0) + np.diag(k)[candidates]) / (len(selected) + 1)
        best = int(candidates[int(np.argmax(s1))])
        selected.append(best)
        available[best] = False
    return np.asarray(selected, dtype=np.int64)


def kernel_herding(k: FloatArray, m: int) -> IntArray:
    n = k.shape[0]
    mean_feature = k.mean(axis=0)
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    running = np.zeros(n, dtype=np.float64)
    for t in range(m):
        candidates = np.flatnonzero(available)
        score = mean_feature[candidates] - running[candidates] / max(t, 1)
        best = int(candidates[int(np.argmax(score))])
        selected.append(best)
        available[best] = False
        running += k[best]
    return np.asarray(selected, dtype=np.int64)


def protodash_like(k: FloatArray, m: int) -> tuple[IntArray, FloatArray]:
    """Greedy nonnegative MMD prototype selection using the supplied kernel."""
    n = k.shape[0]
    mu = k.mean(axis=0)
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    weights = np.empty(0, dtype=np.float64)
    residual = mu.copy()
    for _ in range(m):
        candidates = np.flatnonzero(available)
        if selected:
            k_ss = k[np.ix_(selected, selected)] + 1e-10 * np.eye(len(selected))
            weights, _ = nnls(k_ss, mu[selected])
            residual = mu - k[:, selected] @ weights
        score = residual[candidates]
        best = int(candidates[int(np.argmax(score))])
        selected.append(best)
        available[best] = False
    k_ss = k[np.ix_(selected, selected)] + 1e-10 * np.eye(len(selected))
    weights, _ = nnls(k_ss, mu[selected])
    if weights.sum() <= 0:
        weights = np.ones(len(selected))
    weights /= weights.sum()
    return np.asarray(selected, dtype=np.int64), weights


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def subset_q_from_kernel(k: FloatArray, selected: Sequence[int], weights: FloatArray | None = None) -> FloatArray:
    sel = np.asarray(selected, dtype=np.int64)
    if weights is None:
        weights = np.full(len(sel), 1.0 / len(sel))
    return probability_from_scores(k[:, sel] @ weights)


def mmd2_from_kernel(k: FloatArray, selected: Sequence[int], weights: FloatArray | None = None) -> float:
    n = k.shape[0]
    sel = np.asarray(selected, dtype=np.int64)
    if weights is None:
        weights = np.full(len(sel), 1.0 / len(sel))
    full = np.full(n, 1.0 / n)
    return float(full @ k @ full + weights @ k[np.ix_(sel, sel)] @ weights - 2.0 * full @ k[:, sel] @ weights)


def finite_renyi_subset_objective(k: FloatArray, p: FloatArray, selected: Sequence[int], alpha: float) -> float:
    return renyi_probability_divergence(p, subset_q_from_kernel(k, selected, None), alpha)


def refine_subset_by_finite_renyi(
    k: FloatArray,
    selected: Sequence[int],
    p: FloatArray,
    alpha: float,
    max_passes: int = 5,
) -> IntArray:
    """Swap-refine an observed subset against the exact finite uniform objective."""
    current = np.asarray(selected, dtype=np.int64).copy()
    n = k.shape[0]
    best_obj = finite_renyi_subset_objective(k, p, current, alpha)
    for _ in range(max_passes):
        improved = False
        selected_set = set(int(i) for i in current)
        candidates = [i for i in range(n) if i not in selected_set]
        best_swap: tuple[int, int, float] | None = None
        for pos in range(len(current)):
            old = int(current[pos])
            for new in candidates:
                trial = current.copy()
                trial[pos] = new
                obj = finite_renyi_subset_objective(k, p, trial, alpha)
                if obj < best_obj - 1e-12:
                    best_obj = obj
                    best_swap = (pos, new, obj)
                    improved = True
        if not improved or best_swap is None:
            break
        current[best_swap[0]] = best_swap[1]
    return current.astype(np.int64)


def correntropy_density_weights(d: FloatArray, multiplier: float = 1.0, gamma: float = 1.0) -> FloatArray:
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > 1e-12]
    sigma = float(np.median(vals)) * multiplier if vals.size else 1.0
    sigma = max(sigma, 1e-12)
    density = np.exp(-(d * d) / (2.0 * sigma * sigma)).sum(axis=1) - 1.0
    density = np.maximum(density, 1e-12) ** gamma
    return probability_from_scores(density)


def class_conditional_robust_mrds_selection(
    measures: Sequence[EmpiricalMeasure],
    labels: NDArray,
    d_pair: FloatArray,
    k_full: FloatArray,
    m: int,
    alpha: float,
    reg: float,
    prototype_atoms: int,
    max_iter: int,
    seed: int,
    correntropy_multiplier: float = 1.0,
    correntropy_gamma: float = 1.0,
    transport_backend: str = "emd_exact",
) -> tuple[IntArray, FloatArray, dict]:
    classes, counts = np.unique(labels, return_counts=True)
    allocation = np.maximum(1, np.round(m * counts / counts.sum()).astype(int))
    while allocation.sum() > m:
        pos = int(np.argmax(allocation))
        if allocation[pos] > 1:
            allocation[pos] -= 1
        else:
            break
    while allocation.sum() < m:
        allocation[int(np.argmax(counts / allocation))] += 1

    selected_parts: list[int] = []
    metadata = {
        "description": "class-conditional MRDS with correntropy local-density anchor weights",
        "correntropy_multiplier": correntropy_multiplier,
        "correntropy_gamma": correntropy_gamma,
        "class_runs": [],
    }
    for cls, m_cls in zip(classes, allocation):
        idx = np.flatnonzero(labels == cls)
        if int(m_cls) >= len(idx):
            selected_cls = idx.astype(np.int64)
            metadata["class_runs"].append({"class": int(cls), "n": int(len(idx)), "m": int(m_cls), "status": "all selected"})
        else:
            d_cls = d_pair[np.ix_(idx, idx)]
            k_cls = k_full[np.ix_(idx, idx)]
            weights_anchor = correntropy_density_weights(d_cls, correntropy_multiplier, correntropy_gamma)
            reducer = MetaRenyiReducer(
                n_prototypes=int(m_cls),
                alpha=alpha,
                reg=reg,
                bandwidth=median_bandwidth(d_cls),
                prototype_atoms=prototype_atoms,
                max_iter=max_iter,
                seed=seed + int(cls) * 997,
                anchor_weights=weights_anchor,
                transport_backend=transport_backend,
            )
            result = reducer.fit([measures[int(i)] for i in idx], d_cls, k_cls)
            selected_cls = idx[result.selected_indices]
            metadata["class_runs"].append(
                {
                    "class": int(cls),
                    "n": int(len(idx)),
                    "m": int(m_cls),
                    "anchor_weight_min": float(weights_anchor.min()),
                    "anchor_weight_max": float(weights_anchor.max()),
                    "objective_history": result.objective_history,
                    **result.metadata,
                }
            )
        selected_parts.extend(int(i) for i in selected_cls)

    selected = np.asarray(selected_parts, dtype=np.int64)
    _, rep_w = assignments_and_weights(d_pair, selected)
    return selected, rep_w, metadata


def nearest_subset_predict(d_train: FloatArray, y_train: NDArray, selected: Sequence[int]) -> NDArray:
    sel = np.asarray(selected, dtype=np.int64)
    nearest = np.argmin(d_train[:, sel], axis=1)
    return y_train[sel[nearest]]


def evaluate_selection(
    d: FloatArray,
    k: FloatArray,
    selected: Sequence[int],
    p: FloatArray,
    alpha: float,
    labels: NDArray | None = None,
    representative_weights: FloatArray | None = None,
) -> dict:
    sel = np.asarray(selected, dtype=np.int64)
    q_uniform = subset_q_from_kernel(k, sel, None)
    q_weighted = subset_q_from_kernel(k, sel, representative_weights)
    result = {
        "m": int(len(sel)),
        "coverage_mean": float(np.min(d[:, sel], axis=1).mean()),
        "coverage_max": float(np.min(d[:, sel], axis=1).max()),
        "renyi_meta": renyi_probability_divergence(p, q_uniform, alpha),
        "renyi_meta_uniform": renyi_probability_divergence(p, q_uniform, alpha),
        "renyi_meta_weighted": renyi_probability_divergence(p, q_weighted, alpha),
        "mmd2": mmd2_from_kernel(k, sel, representative_weights),
    }
    if labels is not None:
        pred = nearest_subset_predict(d, labels, sel)
        result["subset_1nn_accuracy"] = float(accuracy_score(labels, pred))
        result["subset_1nn_balanced_accuracy"] = float(balanced_accuracy_score(labels, pred))
        result["subset_1nn_macro_f1"] = float(f1_score(labels, pred, average="macro"))
        classes, counts = np.unique(labels, return_counts=True)
        classes_s, counts_s = np.unique(labels[sel], return_counts=True)
        full_prop = {str(c): v / len(labels) for c, v in zip(classes, counts)}
        sub_prop = {str(c): v / len(sel) for c, v in zip(classes_s, counts_s)}
        result["class_proportion_l1"] = float(sum(abs(full_prop.get(str(c), 0.0) - sub_prop.get(str(c), 0.0)) for c in classes))
    return result


def run_baselines(d: FloatArray, k: FloatArray, m: int, seed: int) -> dict[str, tuple[IntArray, FloatArray | None]]:
    methods: dict[str, tuple[IntArray, FloatArray | None]] = {}
    methods["Random"] = (random_selection(d.shape[0], m, seed), None)
    methods["W2-FarthestFirst"] = (farthest_first(d, m, seed), None)
    methods["W2-KMedoids"] = (pam_kmedoids(d, m, seed=seed), None)
    methods["W2-Agglomerative-Complete"] = (agglomerative_medoids(d, m, "complete"), None)
    methods["FacilityLocation"] = (facility_location(k, m), None)
    methods["MMD-Critic"] = (mmd_critic_prototypes(k, m), None)
    methods["KernelHerding"] = (kernel_herding(k, m), None)
    pd_idx, pd_w = protodash_like(k, m)
    methods["ProtoDash-like"] = (pd_idx, pd_w)
    return methods


def display_method_name(name: str) -> str:
    if name in {"MRDS-exact-finite", "MRDS-IS-R"}:
        return "MRDS-IS-R (proposed)"
    if name == "ProtoDash-like":
        return "ProtoDash-like"
    return name


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

def make_synthetic_meta_gaussians(
    n_bags: int,
    bag_size: int,
    d: int,
    n_classes: int,
    seed: int,
) -> tuple[list[FloatArray], IntArray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_bags) % n_classes
    rng.shuffle(labels)
    bags: list[FloatArray] = []
    base_directions = rng.normal(size=(n_classes, d))
    base_directions /= np.linalg.norm(base_directions, axis=1, keepdims=True)
    for c in labels:
        mean = 2.5 * base_directions[c] + rng.normal(scale=0.25, size=d)
        diag = 0.3 + 0.25 * (c + 1) / n_classes + rng.uniform(0.0, 0.15, size=d)
        latent = rng.normal(size=(bag_size, d)) * diag
        # Add class-dependent mixture structure so means alone are insufficient.
        sign = rng.choice([-1.0, 1.0], size=(bag_size, 1))
        mixture_shift = sign * (0.4 + 0.15 * c) * base_directions[c]
        bags.append(mean + latent + mixture_shift)
    return bags, labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def prepare_measures(
    bags: Sequence[FloatArray],
    max_atoms: int,
    time_beta: float,
    seed: int,
) -> tuple[list[EmpiricalMeasure], FloatArray, FloatArray]:
    standardized, center, scale = robust_standardize(bags)
    standardized = augment_time(standardized, time_beta)
    measures = [compress_bag(x, max_atoms, seed + i) for i, x in enumerate(standardized)]
    return measures, center, scale


def save_result(output: Path, result: ReductionResult, evaluation: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(result),
        "selected_indices": result.selected_indices.tolist(),
        "representative_weights": result.representative_weights.tolist(),
        "evaluation": evaluation,
    }
    (output / f"{result.method}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(
    bags: Sequence[FloatArray],
    labels: NDArray | None,
    output: Path,
    m: int,
    alpha: float,
    max_atoms: int,
    prototype_atoms: int,
    reg: float,
    distance_backend: str,
    projections: int,
    quantiles: int,
    time_beta: float,
    max_iter: int,
    seed: int,
    run_proposed: bool,
    run_robust_class_conditional: bool = True,
    correntropy_multiplier: float = 1.0,
    correntropy_gamma: float = 1.0,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    measures, center, scale = prepare_measures(bags, max_atoms, time_beta, seed)
    np.save(output / "feature_center.npy", center)
    np.save(output / "feature_scale.npy", scale)

    start = time.perf_counter()
    if distance_backend == "emd_exact":
        d_pair = pairwise_emd_exact(measures)
    elif distance_backend == "sinkhorn":
        d_pair = pairwise_sinkhorn(measures, reg=reg, debias=False)
    elif distance_backend == "sliced":
        d_pair = pairwise_sliced(measures, projections, quantiles, seed)
    else:
        raise ValueError("distance_backend must be 'emd_exact', 'sinkhorn', or 'sliced'")
    if run_proposed and distance_backend != "emd_exact":
        raise ValueError(
            "Confirmatory MRDS requires distance_backend='emd_exact'; "
            "sliced and raw Sinkhorn cross-cost are diagnostics only"
        )
    if run_proposed and prototype_atoms != max_atoms:
        raise ValueError("Confirmatory MRDS requires prototype_atoms == max_atoms")
    distance_time = time.perf_counter() - start
    h = median_bandwidth(d_pair)
    k_full = rbf_kernel_from_distance(d_pair, h)
    p = finite_meta_p(k_full)
    np.save(output / "meta_distance.npy", d_pair)
    np.save(output / "meta_kernel.npy", k_full)

    rows: list[dict] = []
    for name, selector in [
        ("Random", lambda: (random_selection(d_pair.shape[0], m, seed), None)),
        ("W2-FarthestFirst", lambda: (farthest_first(d_pair, m, seed), None)),
        ("W2-KMedoids", lambda: (pam_kmedoids(d_pair, m, seed=seed), None)),
        ("W2-Agglomerative-Complete", lambda: (agglomerative_medoids(d_pair, m, "complete"), None)),
        ("FacilityLocation", lambda: (facility_location(k_full, m), None)),
        ("MMD-Critic", lambda: (mmd_critic_prototypes(k_full, m), None)),
        ("KernelHerding", lambda: (kernel_herding(k_full, m), None)),
        ("ProtoDash-like", lambda: protodash_like(k_full, m)),
    ]:
        method_start = time.perf_counter()
        idx, weights = selector()
        method_time = time.perf_counter() - method_start
        _, rep_w = assignments_and_weights(d_pair, idx)
        eval_weights = weights if weights is not None else rep_w
        metrics = evaluate_selection(d_pair, k_full, idx, p, alpha, labels, eval_weights)
        rows.append({"method": display_method_name(name), "runtime_seconds": method_time, **metrics})
        result = ReductionResult(display_method_name(name), idx, eval_weights, [], method_time, {"bandwidth": h})
        save_result(output, result, metrics)

    if run_proposed:
        reducer = MetaRenyiReducer(
            n_prototypes=m,
            alpha=alpha,
            reg=reg,
            bandwidth=h,
            prototype_atoms=prototype_atoms,
            max_iter=max_iter,
            seed=seed,
            transport_backend="emd_exact",
        )
        proposed = reducer.fit(measures, d_pair, k_full)
        metrics = evaluate_selection(
            d_pair,
            k_full,
            proposed.selected_indices,
            p,
            alpha,
            labels,
            proposed.representative_weights,
        )
        proposed.method = display_method_name(proposed.method)
        rows.append({"method": proposed.method, "runtime_seconds": proposed.runtime_seconds, **metrics})
        save_result(output, proposed, metrics)

    if run_robust_class_conditional and labels is not None:
        start_method = time.perf_counter()
        idx, weights, meta = class_conditional_robust_mrds_selection(
            measures,
            labels,
            d_pair,
            k_full,
            m,
            alpha,
            reg,
            prototype_atoms,
            max_iter,
            seed,
            correntropy_multiplier,
            correntropy_gamma,
            "emd_exact",
        )
        runtime = time.perf_counter() - start_method
        metrics = evaluate_selection(d_pair, k_full, idx, p, alpha, labels, weights)
        method_name = "Robust class-conditional MRDS (proposed)"
        rows.append({"method": method_name, "runtime_seconds": runtime, **metrics})
        result = ReductionResult(method_name, idx, weights, [], runtime, {"bandwidth": h, **meta})
        save_result(output, result, metrics)

    table = pd.DataFrame(rows).sort_values(["renyi_meta", "coverage_mean"], ascending=True)
    table.to_csv(output / "summary.csv", index=False)
    metadata = {
        "N": len(measures),
        "d": int(measures[0].support.shape[1]),
        "m": m,
        "alpha": alpha,
        "max_atoms": max_atoms,
        "prototype_atoms": prototype_atoms,
        "reg": reg,
        "distance_backend": distance_backend,
        "distance_runtime_seconds": distance_time,
        "bandwidth": h,
        "time_beta": time_beta,
        "seed": seed,
        "pot_available": HAVE_POT,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


def add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--m", type=int, default=20)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--max-atoms", type=int, default=8)
    p.add_argument("--prototype-atoms", type=int, default=8)
    p.add_argument("--reg", type=float, default=0.5)
    p.add_argument(
        "--distance-backend",
        choices=["emd_exact", "sliced", "sinkhorn"],
        default="emd_exact",
    )
    p.add_argument("--projections", type=int, default=64)
    p.add_argument("--quantiles", type=int, default=64)
    p.add_argument("--time-beta", type=float, default=0.0)
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-proposed", action="store_true")
    p.add_argument("--skip-robust-class-conditional", action="store_true")
    p.add_argument("--correntropy-multiplier", type=float, default=1.0)
    p.add_argument("--correntropy-gamma", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="Generate and run a synthetic meta-distribution benchmark")
    add_common_run_args(syn)
    syn.add_argument("--n-bags", type=int, default=60)
    syn.add_argument("--bag-size", type=int, default=80)
    syn.add_argument("--dimension", type=int, default=8)
    syn.add_argument("--classes", type=int, default=3)

    run = sub.add_parser("run", help="Run on NPZ or MIL CSV input")
    add_common_run_args(run)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--format", choices=["npz", "csv"], default="npz")
    run.add_argument("--x-key", default="X")
    run.add_argument("--y-key", default="y")
    run.add_argument("--bag-id-col", default="bag_id")
    run.add_argument("--label-col", default="label")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "synthetic":
        bags, labels = make_synthetic_meta_gaussians(
            args.n_bags, args.bag_size, args.dimension, args.classes, args.seed
        )
    else:
        if args.format == "npz":
            bags, labels = load_npz(args.input, args.x_key, args.y_key)
        else:
            bags, labels = load_mil_csv(args.input, args.bag_id_col, args.label_col)

    run_experiment(
        bags=bags,
        labels=labels,
        output=args.output,
        m=args.m,
        alpha=args.alpha,
        max_atoms=args.max_atoms,
        prototype_atoms=args.prototype_atoms,
        reg=args.reg,
        distance_backend=args.distance_backend,
        projections=args.projections,
        quantiles=args.quantiles,
        time_beta=args.time_beta,
        max_iter=args.max_iter,
        seed=args.seed,
        run_proposed=not args.skip_proposed,
        run_robust_class_conditional=not args.skip_robust_class_conditional,
        correntropy_multiplier=args.correntropy_multiplier,
        correntropy_gamma=args.correntropy_gamma,
    )


if __name__ == "__main__":
    main()
