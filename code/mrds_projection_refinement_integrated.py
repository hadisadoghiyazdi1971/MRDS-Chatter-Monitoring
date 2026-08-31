#!/usr/bin/env python3
"""Projection-aware Rényi-monotone refinement for observed distribution subsets.

This module operates on a precomputed positive meta-kernel K between complete
observations (bags/distributions).  It repairs the final stage of MRDS:
synthetic prototypes may be projected to an observed subset, but projection can
increase the finite meta-Rényi objective.  The routines below refine the
observed subset directly and accept only objective-decreasing weight or swap
updates.

The module is intentionally independent of the OT backend.  The same K must be
used for MRDS, Facility Location, k-medoids evaluation, and this refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class WeightOptimizationResult:
    weights: FloatArray
    objective: float
    history: tuple[float, ...]
    converged: bool


@dataclass(frozen=True)
class RefinementResult:
    selected: IntArray
    weights: FloatArray
    objective: float
    history: tuple[float, ...]
    accepted_swaps: tuple[tuple[int, int], ...]
    initializer: str


def _validate_alpha(alpha: float) -> None:
    if alpha <= 0.0 or np.isclose(alpha, 1.0):
        raise ValueError("alpha must be positive and different from one")


def _validate_kernel(kernel: FloatArray) -> FloatArray:
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if not np.isfinite(k).all() or np.any(k < 0.0):
        raise ValueError("kernel must be finite and nonnegative")
    if np.any(k.sum(axis=0) <= 0.0):
        raise ValueError("every kernel column must have positive mass")
    return k


def probability_from_scores(scores: FloatArray, eps: float = 1e-15) -> FloatArray:
    x = np.maximum(np.asarray(scores, dtype=np.float64), eps)
    total = float(x.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("scores cannot be normalized")
    return x / total


def full_meta_probability(kernel: FloatArray, eps: float = 1e-15) -> FloatArray:
    """Finite-anchor p used by the manuscript: normalized row-mean scores."""
    k = _validate_kernel(kernel)
    return probability_from_scores(k.mean(axis=1), eps=eps)


def subset_probability(
    kernel: FloatArray,
    selected: Sequence[int],
    weights: FloatArray | None = None,
    eps: float = 1e-15,
) -> FloatArray:
    """Compute q(S,v) using the exact global normalization in the manuscript."""
    k = _validate_kernel(kernel)
    sel = np.asarray(selected, dtype=np.int64)
    if sel.ndim != 1 or sel.size == 0:
        raise ValueError("selected must be a nonempty one-dimensional index set")
    if np.unique(sel).size != sel.size:
        raise ValueError("selected indices must be unique")
    if np.any(sel < 0) or np.any(sel >= k.shape[0]):
        raise IndexError("selected index out of range")
    if weights is None:
        v = np.full(sel.size, 1.0 / sel.size, dtype=np.float64)
    else:
        v = np.asarray(weights, dtype=np.float64)
        if v.shape != (sel.size,):
            raise ValueError("weights shape does not match selected subset")
        if np.any(v < 0.0) or not np.isfinite(v).all():
            raise ValueError("weights must be finite and nonnegative")
        v = probability_from_scores(v, eps=eps)
    scores = k[:, sel] @ v
    return probability_from_scores(scores, eps=eps)


def renyi_divergence(
    p: FloatArray,
    q: FloatArray,
    alpha: float,
    eps: float = 1e-15,
) -> float:
    _validate_alpha(alpha)
    pp = probability_from_scores(p, eps=eps)
    qq = probability_from_scores(q, eps=eps)
    a = float(np.sum((pp**alpha) * (qq ** (1.0 - alpha))))
    return float(np.log(max(a, eps)) / (alpha - 1.0))


def subset_objective(
    kernel: FloatArray,
    p: FloatArray,
    selected: Sequence[int],
    alpha: float,
    weights: FloatArray | None = None,
) -> float:
    return renyi_divergence(p, subset_probability(kernel, selected, weights), alpha)


def weight_objective_and_gradient(
    kernel: FloatArray,
    p: FloatArray,
    selected: Sequence[int],
    weights: FloatArray,
    alpha: float,
    eps: float = 1e-15,
) -> tuple[float, FloatArray]:
    """Exact objective and Euclidean gradient with global q normalization.

    Let B=K[:,S], s=Bv, z=1^T s, q=s/z.  With
    A=sum p_i^alpha q_i^(1-alpha) and u_i=p_i^alpha q_i^-alpha,

        grad_v D_alpha = -(B^T u - A B^T 1)/(A z).
    """
    _validate_alpha(alpha)
    k = _validate_kernel(kernel)
    sel = np.asarray(selected, dtype=np.int64)
    v = probability_from_scores(np.asarray(weights, dtype=np.float64), eps=eps)
    if v.shape != (sel.size,):
        raise ValueError("weights shape does not match selected subset")

    b = k[:, sel]
    s = np.maximum(b @ v, eps)
    z = float(s.sum())
    q = s / z
    pp = probability_from_scores(p, eps=eps)
    a = float(np.sum((pp**alpha) * (q ** (1.0 - alpha))))
    obj = float(np.log(max(a, eps)) / (alpha - 1.0))
    u = (pp**alpha) * (q ** (-alpha))
    column_mass = b.sum(axis=0)
    grad = -(b.T @ u - a * column_mass) / max(a * z, eps)
    return obj, np.asarray(grad, dtype=np.float64)


def optimize_mixture_weights(
    kernel: FloatArray,
    p: FloatArray,
    selected: Sequence[int],
    alpha: float,
    initial: FloatArray | None = None,
    max_iter: int = 300,
    initial_step: float = 1.0,
    backtrack: float = 0.5,
    armijo: float = 1e-4,
    tol: float = 1e-10,
    min_step: float = 1e-12,
) -> WeightOptimizationResult:
    """Monotone exponentiated-gradient optimization on the simplex.

    Uniform weights are the default initial point.  A proposal is accepted only
    when it decreases the same finite Rényi objective; hence the returned value
    is never worse than the initial weights up to numerical tolerance.
    """
    sel = np.asarray(selected, dtype=np.int64)
    if initial is None:
        v = np.full(sel.size, 1.0 / sel.size, dtype=np.float64)
    else:
        v = probability_from_scores(np.asarray(initial, dtype=np.float64))
    obj, grad = weight_objective_and_gradient(kernel, p, sel, v, alpha)
    history = [obj]
    converged = False

    for _ in range(max_iter):
        # Centering does not change exponentiated-gradient normalization and
        # improves numerical stability.
        centered_grad = grad - float(v @ grad)
        if float(np.max(np.abs(centered_grad))) < tol:
            converged = True
            break

        step = initial_step
        accepted = False
        while step >= min_step:
            logits = np.log(np.maximum(v, 1e-300)) - step * centered_grad
            logits -= float(np.max(logits))
            trial_v = np.exp(logits)
            trial_v /= trial_v.sum()
            trial_obj, trial_grad = weight_objective_and_gradient(
                kernel, p, sel, trial_v, alpha
            )
            delta = trial_v - v
            directional = float(grad @ delta)
            # The exact Armijo right side uses the actual directional slope.
            if directional < 0.0 and trial_obj <= obj + armijo * directional:
                v, obj, grad = trial_v, trial_obj, trial_grad
                history.append(obj)
                accepted = True
                break
            step *= backtrack

        if not accepted:
            converged = True
            break
        if len(history) >= 2 and abs(history[-2] - history[-1]) <= tol * max(1.0, abs(history[-2])):
            converged = True
            break

    return WeightOptimizationResult(v, obj, tuple(history), converged)


def refine_observed_subset(
    kernel: FloatArray,
    p: FloatArray,
    initial_selected: Sequence[int],
    alpha: float,
    initializer_name: str = "unspecified",
    optimize_weights: bool = True,
    max_passes: int = 100,
    improvement_tol: float = 1e-10,
    weight_max_iter: int = 300,
) -> RefinementResult:
    """Best-improvement one-swap refinement on the exact finite objective.

    Every accepted swap strictly decreases the same objective used for final
    evaluation.  Because the number of m-subsets is finite, the procedure
    terminates at a one-swap local optimum (up to improvement_tol).
    """
    k = _validate_kernel(kernel)
    selected = np.asarray(initial_selected, dtype=np.int64).copy()
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("initial_selected must be nonempty")
    if np.unique(selected).size != selected.size:
        raise ValueError("initial_selected contains duplicates")
    n = k.shape[0]
    if np.any(selected < 0) or np.any(selected >= n):
        raise IndexError("initial_selected contains out-of-range indices")

    if optimize_weights:
        wres = optimize_mixture_weights(
            k, p, selected, alpha, max_iter=weight_max_iter
        )
        weights, current = wres.weights, wres.objective
    else:
        weights = np.full(selected.size, 1.0 / selected.size)
        current = subset_objective(k, p, selected, alpha, weights)
    history = [current]
    swaps: list[tuple[int, int]] = []

    for _ in range(max_passes):
        selected_set = set(int(x) for x in selected)
        outside = [i for i in range(n) if i not in selected_set]
        best_obj = current
        best_selected: IntArray | None = None
        best_weights: FloatArray | None = None
        best_swap: tuple[int, int] | None = None

        for pos, old_idx in enumerate(selected.tolist()):
            for new_idx in outside:
                candidate = selected.copy()
                candidate[pos] = new_idx
                if optimize_weights:
                    # Uniform is always feasible.  Starting from the previous
                    # weights is faster, while monotone weight optimization
                    # protects the candidate objective.
                    wres = optimize_mixture_weights(
                        k, p, candidate, alpha, initial=weights,
                        max_iter=weight_max_iter,
                    )
                    cand_w, cand_obj = wres.weights, wres.objective
                else:
                    cand_w = np.full(candidate.size, 1.0 / candidate.size)
                    cand_obj = subset_objective(k, p, candidate, alpha, cand_w)
                if cand_obj < best_obj - improvement_tol:
                    best_obj = cand_obj
                    best_selected = candidate
                    best_weights = cand_w
                    best_swap = (int(old_idx), int(new_idx))

        if best_selected is None or best_weights is None or best_swap is None:
            break
        selected = best_selected
        weights = best_weights
        current = best_obj
        history.append(current)
        swaps.append(best_swap)

    order = np.argsort(selected)
    return RefinementResult(
        selected=selected[order],
        weights=weights[order],
        objective=current,
        history=tuple(history),
        accepted_swaps=tuple(swaps),
        initializer=initializer_name,
    )


def multistart_refinement(
    kernel: FloatArray,
    p: FloatArray,
    initializers: Mapping[str, Sequence[int]],
    alpha: float,
    optimize_weights: bool = True,
    max_passes: int = 100,
    improvement_tol: float = 1e-10,
    weight_max_iter: int = 300,
) -> tuple[RefinementResult, dict[str, RefinementResult]]:
    """Refine each initializer and return the best final Rényi objective.

    Including Facility Location, k-medoids, and projected MRDS as initializers
    yields an immediate dominance certificate on the *same* Rényi objective:
    the returned objective cannot exceed the objective of any supplied
    initializer, because each run is monotone and the best run is retained.
    """
    if not initializers:
        raise ValueError("at least one initializer is required")
    results: dict[str, RefinementResult] = {}
    for name, selected in initializers.items():
        results[name] = refine_observed_subset(
            kernel=kernel,
            p=p,
            initial_selected=selected,
            alpha=alpha,
            initializer_name=name,
            optimize_weights=optimize_weights,
            max_passes=max_passes,
            improvement_tol=improvement_tol,
            weight_max_iter=weight_max_iter,
        )
    best = min(results.values(), key=lambda result: result.objective)
    return best, results


def stagewise_subset_audit(
    kernel: FloatArray,
    p: FloatArray,
    stages: Mapping[str, Sequence[int]],
    alpha: float,
) -> dict[str, dict[str, float]]:
    """Report uniform and weight-optimized Rényi for observed subsets."""
    report: dict[str, dict[str, float]] = {}
    for name, selected in stages.items():
        sel = np.asarray(selected, dtype=np.int64)
        uniform = subset_objective(kernel, p, sel, alpha)
        optimized = optimize_mixture_weights(kernel, p, sel, alpha)
        report[name] = {
            "renyi_uniform": float(uniform),
            "renyi_optimized_weights": float(optimized.objective),
            "weight_iterations": float(max(0, len(optimized.history) - 1)),
        }
    return report


def _smoke_test() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    sq = np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2 * x @ x.T
    sq = np.maximum(sq, 0.0)
    h = np.median(np.sqrt(sq[np.triu_indices(30, 1)]))
    k = np.exp(-sq / (2.0 * h * h))
    p = full_meta_probability(k)
    initializers = {
        "init_a": np.array([0, 1, 2, 3, 4]),
        "init_b": np.array([5, 6, 7, 8, 9]),
    }
    initial_best = min(
        subset_objective(k, p, sel, alpha=2.0) for sel in initializers.values()
    )
    best, all_results = multistart_refinement(k, p, initializers, alpha=2.0)
    assert best.objective <= initial_best + 1e-10
    assert np.unique(best.selected).size == best.selected.size
    assert np.isclose(best.weights.sum(), 1.0)
    for result in all_results.values():
        assert all(b <= a + 1e-12 for a, b in zip(result.history, result.history[1:]))
    print("projection refinement smoke test passed")
    print({name: result.objective for name, result in all_results.items()})


if __name__ == "__main__":
    _smoke_test()
