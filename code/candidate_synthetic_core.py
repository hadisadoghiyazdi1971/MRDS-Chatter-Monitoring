#!/usr/bin/env python3
"""Frozen stopping-policy primitives for the candidate MRDS rerun.

This module intentionally implements a projection-only synthetic MRDS path and
audited weight/swap refinement.  It does not compute downstream metrics.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from meta_renyi_reduction import (
    EmpiricalMeasure,
    MetaRenyiReducer,
    exact_signed_coefficients,
    finite_meta_q,
    finite_meta_p,
    median_bandwidth,
    prototype_kernel_and_projections,
    prototype_kernel_column,
    renyi_probability_divergence,
)
from mrds_projection_refinement_integrated import (
    probability_from_scores,
    subset_objective,
    weight_objective_and_gradient,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class UnresolvedCapError(RuntimeError):
    """Raised immediately when a safety cap, rather than a stop rule, ends a stage."""

    def __init__(self, stage: str, details: dict[str, object]):
        super().__init__(f"unresolved {stage} safety cap: {details}")
        self.stage = stage
        self.details = details


class InvalidTerminationError(RuntimeError):
    """Raised when a numerical validity audit fails before a stage terminates."""

    def __init__(self, stage: str, details: dict[str, object]):
        super().__init__(f"invalid {stage} termination status: {details}")
        self.stage = stage
        self.details = details


@dataclass(frozen=True)
class SyntheticProjectionResult:
    selected: IntArray
    objective_history: tuple[float, ...]
    sweeps_completed: int
    stop_reason: str
    cap_hit: bool
    last_max_move: float
    accepted_updates_by_sweep: tuple[int, ...]
    max_move_by_sweep: tuple[float, ...]
    absolute_decrease_by_sweep: tuple[float, ...]
    relative_improvement_by_sweep: tuple[float, ...]
    final_consecutive_small_improvements: int
    relative_improvement_threshold: float
    required_consecutive_sweeps: int
    normalization_denominator: float
    monotonicity_valid: bool
    monotonicity_tolerance_factor: float
    tiny_roundoff_increases: int
    runtime_seconds: float


@dataclass(frozen=True)
class WeightResult:
    weights: FloatArray
    objective: float
    history: tuple[float, ...]
    iterations: int
    stop_reason: str
    cap_hit: bool
    monotonicity_valid: bool


@dataclass
class WeightAudit:
    max_iter: int
    calls: int = 0
    total_iterations: int = 0
    max_iterations_observed: int = 0
    stop_reasons: Counter[str] = field(default_factory=Counter)
    role_counts: Counter[str] = field(default_factory=Counter)
    iteration_histogram: Counter[int] = field(default_factory=Counter)
    cap_hits: int = 0
    monotonicity_failures: int = 0

    def record(self, result: WeightResult, role: str) -> None:
        self.calls += 1
        self.total_iterations += result.iterations
        self.max_iterations_observed = max(self.max_iterations_observed, result.iterations)
        self.stop_reasons[result.stop_reason] += 1
        self.role_counts[role] += 1
        self.iteration_histogram[result.iterations] += 1
        self.cap_hits += int(result.cap_hit)
        self.monotonicity_failures += int(not result.monotonicity_valid)

    def as_dict(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "total_iterations": self.total_iterations,
            "mean_iterations": self.total_iterations / max(self.calls, 1),
            "max_iterations_observed": self.max_iterations_observed,
            "stop_reasons": dict(sorted(self.stop_reasons.items())),
            "role_counts": dict(sorted(self.role_counts.items())),
            "iteration_histogram": {
                str(key): value for key, value in sorted(self.iteration_histogram.items())
            },
            "cap_hits": self.cap_hits,
            "monotonicity_failures": self.monotonicity_failures,
            "all_calls_resolved": self.cap_hits == 0,
            "all_objective_histories_monotone": self.monotonicity_failures == 0,
            "configured_cap": self.max_iter,
        }


@dataclass(frozen=True)
class RefinementResultAudited:
    selected: IntArray
    weights: FloatArray
    objective: float
    initial_weight_objective: float
    objective_history: tuple[float, ...]
    accepted_swaps: tuple[tuple[int, int], ...]
    passes_completed: int
    stop_reason: str
    cap_hit: bool
    initializer: str
    weight_audit: dict[str, object]
    objective_monotonicity_valid: bool


def synthetic_projection_only(
    measures: Sequence[EmpiricalMeasure],
    d_pair: FloatArray,
    k_full: FloatArray,
    *,
    n_prototypes: int,
    alpha: float,
    prototype_atoms: int,
    bandwidth: float | None,
    seed: int,
    max_sweeps: int,
    tol: float = 1e-5,
    step_size: float = 1.0,
    armijo: float = 1e-4,
    backtrack: float = 0.5,
    min_step: float = 1e-5,
    relative_improvement_tol: float = 1e-3,
    required_consecutive_sweeps: int = 3,
    monotonicity_tolerance_factor: float = 64.0,
) -> SyntheticProjectionResult:
    """Run only synthetic Gauss--Seidel updates and duplicate-free projection."""
    start = time.perf_counter()
    reducer = MetaRenyiReducer(
        n_prototypes=n_prototypes,
        alpha=alpha,
        bandwidth=bandwidth,
        prototype_atoms=prototype_atoms,
        max_iter=max_sweeps,
        step_size=step_size,
        armijo=armijo,
        backtrack=backtrack,
        min_step=min_step,
        tol=tol,
        seed=seed,
        transport_backend="emd_exact",
        update_schedule="gauss_seidel",
        refinement_max_passes=0,
    )
    n = len(measures)
    p = finite_meta_p(k_full)
    h = bandwidth if bandwidth is not None else median_bandwidth(d_pair)
    prototypes = reducer._initialize(measures, d_pair)
    k_ap, projections, _ = prototype_kernel_and_projections(
        measures,
        prototypes,
        reg=reducer.reg,
        h=h,
        need_projections=True,
        backend="emd_exact",
    )
    q = finite_meta_q(k_ap)
    current = float(renyi_probability_divergence(p, q, alpha))
    initial_objective = current
    normalization_denominator = max(abs(initial_objective), np.finfo(np.float64).eps)
    history = [current]
    accepted_by_sweep: list[int] = []
    max_move_by_sweep: list[float] = []
    absolute_decrease_by_sweep: list[float] = []
    relative_improvement_by_sweep: list[float] = []
    stop_reason: str | None = None
    last_max_move = float("nan")
    consecutive_small_improvements = 0
    tiny_roundoff_increases = 0

    for _iteration in range(max_sweeps):
        objective_before_sweep = current
        max_move = 0.0
        accepted_count = 0
        for j, proto in enumerate(prototypes):
            c, _ = exact_signed_coefficients(p, q, k_ap, alpha)
            direction = np.zeros_like(proto.support)
            normalizer = float(np.sum(np.abs(c[:, j]))) + 1e-15
            for i in range(n):
                bij = projections[j][i]
                assert bij is not None
                direction += c[i, j] * (bij - proto.support)
            direction /= normalizer
            dir_norm_sq = float(np.sum(proto.weights[:, None] * direction * direction))
            if dir_norm_sq <= tol * tol:
                continue

            old_support = proto.support.copy()
            fd_step = 1e-6 / max(1.0, math.sqrt(dir_norm_sq))

            def objective_along(signed_step: float) -> float:
                prototypes[j].support = old_support + signed_step * direction
                column, _, _ = prototype_kernel_column(
                    measures,
                    prototypes[j],
                    reg=reducer.reg,
                    h=h,
                    need_projections=False,
                    backend="emd_exact",
                )
                trial_kernel = k_ap.copy()
                trial_kernel[:, j] = column
                return renyi_probability_divergence(p, finite_meta_q(trial_kernel), alpha)

            plus = objective_along(fd_step)
            minus = objective_along(-fd_step)
            prototypes[j].support = old_support
            directional_slope = (plus - minus) / (2.0 * fd_step)
            if directional_slope >= 0.0:
                direction *= -1.0
                directional_slope *= -1.0
            if directional_slope >= -1e-14:
                continue

            step = step_size
            accepted = False
            objective_before = current
            while step >= min_step:
                prototypes[j].support = old_support + step * direction
                trial_column, _, _ = prototype_kernel_column(
                    measures,
                    prototypes[j],
                    reg=reducer.reg,
                    h=h,
                    need_projections=False,
                    backend="emd_exact",
                )
                trial_kernel = k_ap.copy()
                trial_kernel[:, j] = trial_column
                trial = renyi_probability_divergence(p, finite_meta_q(trial_kernel), alpha)
                if trial <= objective_before + armijo * step * directional_slope:
                    k_ap = trial_kernel
                    q = finite_meta_q(k_ap)
                    current = float(trial)
                    accepted = True
                    accepted_count += 1
                    max_move = max(max_move, step * math.sqrt(dir_norm_sq))
                    break
                prototypes[j].support = old_support
                step *= backtrack

            if not accepted:
                prototypes[j].support = old_support

            current_column, current_projections, _ = prototype_kernel_column(
                measures,
                prototypes[j],
                reg=reducer.reg,
                h=h,
                need_projections=True,
                backend="emd_exact",
            )
            k_ap[:, j] = current_column
            projections[j] = current_projections
            q = finite_meta_q(k_ap)
            current = float(renyi_probability_divergence(p, q, alpha))

        monotonicity_tolerance = (
            monotonicity_tolerance_factor
            * np.finfo(np.float64).eps
            * max(1.0, abs(objective_before_sweep), abs(current))
        )
        objective_increase = current - objective_before_sweep
        if objective_increase > monotonicity_tolerance:
            raise InvalidTerminationError(
                "synthetic_monotonicity",
                {
                    "sweep": _iteration + 1,
                    "previous_objective": objective_before_sweep,
                    "current_objective": current,
                    "objective_increase": objective_increase,
                    "allowed_numerical_tolerance": monotonicity_tolerance,
                    "objective_history": history + [current],
                },
            )
        if objective_increase > 0.0:
            tiny_roundoff_increases += 1
        absolute_decrease = max(0.0, objective_before_sweep - current)
        relative_improvement = absolute_decrease / normalization_denominator
        if relative_improvement < relative_improvement_tol:
            consecutive_small_improvements += 1
        else:
            consecutive_small_improvements = 0

        history.append(current)
        accepted_by_sweep.append(accepted_count)
        max_move_by_sweep.append(float(max_move))
        absolute_decrease_by_sweep.append(float(absolute_decrease))
        relative_improvement_by_sweep.append(float(relative_improvement))
        last_max_move = max_move
        if consecutive_small_improvements >= required_consecutive_sweeps:
            stop_reason = "three_consecutive_relative_objective_improvements_below_1e-3"
            break

    cap_hit = stop_reason is None
    if cap_hit:
        stop_reason = "safety_cap_exhausted"

    if cap_hit:
        raise UnresolvedCapError(
            "synthetic",
            {
                "configured_cap": max_sweeps,
                "sweeps_completed": len(accepted_by_sweep),
                "last_max_move": last_max_move,
                "last_accepted_updates": accepted_by_sweep[-1] if accepted_by_sweep else None,
                "final_consecutive_small_improvements": consecutive_small_improvements,
                "relative_improvement_threshold": relative_improvement_tol,
                "required_consecutive_sweeps": required_consecutive_sweeps,
                "normalization_denominator": normalization_denominator,
                "objective_history": history,
                "absolute_decrease_by_sweep": absolute_decrease_by_sweep,
                "relative_improvement_by_sweep": relative_improvement_by_sweep,
                "max_move_by_sweep": max_move_by_sweep,
                "accepted_updates_by_sweep": accepted_by_sweep,
                "tiny_roundoff_increases": tiny_roundoff_increases,
            },
        )

    _, _, costs = prototype_kernel_and_projections(
        measures,
        prototypes,
        reg=reducer.reg,
        h=h,
        need_projections=False,
        backend="emd_exact",
    )
    row, col = linear_sum_assignment(costs.T)
    selected = col[np.argsort(row)].astype(np.int64)
    result = SyntheticProjectionResult(
        selected=selected,
        objective_history=tuple(history),
        sweeps_completed=len(accepted_by_sweep),
        stop_reason=stop_reason,
        cap_hit=cap_hit,
        last_max_move=float(last_max_move),
        accepted_updates_by_sweep=tuple(accepted_by_sweep),
        max_move_by_sweep=tuple(max_move_by_sweep),
        absolute_decrease_by_sweep=tuple(absolute_decrease_by_sweep),
        relative_improvement_by_sweep=tuple(relative_improvement_by_sweep),
        final_consecutive_small_improvements=consecutive_small_improvements,
        relative_improvement_threshold=relative_improvement_tol,
        required_consecutive_sweeps=required_consecutive_sweeps,
        normalization_denominator=normalization_denominator,
        monotonicity_valid=True,
        monotonicity_tolerance_factor=monotonicity_tolerance_factor,
        tiny_roundoff_increases=tiny_roundoff_increases,
        runtime_seconds=time.perf_counter() - start,
    )
    return result


def optimize_weights_audited(
    kernel: FloatArray,
    p: FloatArray,
    selected: Sequence[int],
    alpha: float,
    *,
    initial: FloatArray | None,
    max_iter: int,
    audit: WeightAudit,
    role: str,
    initial_step: float = 1.0,
    backtrack: float = 0.5,
    armijo: float = 1e-4,
    tol: float = 1e-10,
    min_step: float = 1e-12,
) -> WeightResult:
    sel = np.asarray(selected, dtype=np.int64)
    if initial is None:
        v = np.full(sel.size, 1.0 / sel.size, dtype=np.float64)
    else:
        v = probability_from_scores(np.asarray(initial, dtype=np.float64))
    obj, grad = weight_objective_and_gradient(kernel, p, sel, v, alpha)
    history = [float(obj)]
    stop_reason: str | None = None

    for _ in range(max_iter):
        centered_grad = grad - float(v @ grad)
        if float(np.max(np.abs(centered_grad))) < tol:
            stop_reason = "centered_gradient_below_tolerance"
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
            if directional < 0.0 and trial_obj <= obj + armijo * directional:
                v, obj, grad = trial_v, float(trial_obj), trial_grad
                history.append(obj)
                accepted = True
                break
            step *= backtrack
        if not accepted:
            stop_reason = "no_accepted_armijo_step"
            break
        if abs(history[-2] - history[-1]) <= tol * max(1.0, abs(history[-2])):
            stop_reason = "objective_change_below_tolerance"
            break

    cap_hit = stop_reason is None
    if cap_hit:
        stop_reason = "safety_cap_exhausted"
    monotonicity_tolerance = 64.0 * np.finfo(np.float64).eps
    monotonicity_valid = all(
        later <= earlier + monotonicity_tolerance * max(1.0, abs(earlier), abs(later))
        for earlier, later in zip(history[:-1], history[1:])
    )
    result = WeightResult(
        weights=v,
        objective=float(obj),
        history=tuple(history),
        iterations=len(history) - 1,
        stop_reason=stop_reason,
        cap_hit=cap_hit,
        monotonicity_valid=monotonicity_valid,
    )
    audit.record(result, role)
    if not monotonicity_valid:
        raise InvalidTerminationError(
            "weights_monotonicity",
            {"role": role, "objective_history": history},
        )
    if cap_hit:
        raise UnresolvedCapError(
            "weights",
            {
                "configured_cap": max_iter,
                "role": role,
                "iterations": result.iterations,
                "last_objective": result.objective,
                "weight_audit": audit.as_dict(),
            },
        )
    return result


def refine_subset_audited(
    kernel: FloatArray,
    p: FloatArray,
    initial_selected: Sequence[int],
    alpha: float,
    *,
    initializer_name: str,
    max_passes: int,
    weight_max_iter: int,
    improvement_tol: float = 1e-10,
) -> RefinementResultAudited:
    selected = np.asarray(initial_selected, dtype=np.int64).copy()
    if np.unique(selected).size != selected.size:
        raise ValueError("initial subset contains duplicates")
    n = kernel.shape[0]
    audit = WeightAudit(max_iter=weight_max_iter)
    initial_weight = optimize_weights_audited(
        kernel,
        p,
        selected,
        alpha,
        initial=None,
        max_iter=weight_max_iter,
        audit=audit,
        role="initializer",
    )
    weights = initial_weight.weights
    current = initial_weight.objective
    history = [current]
    swaps: list[tuple[int, int]] = []
    passes_completed = 0
    stop_reason: str | None = None

    for pass_index in range(max_passes):
        passes_completed += 1
        selected_set = set(int(value) for value in selected)
        outside = [index for index in range(n) if index not in selected_set]
        best_obj = current
        best_selected: IntArray | None = None
        best_weights: FloatArray | None = None
        best_swap: tuple[int, int] | None = None
        for pos, old_idx in enumerate(selected.tolist()):
            for new_idx in outside:
                candidate = selected.copy()
                candidate[pos] = new_idx
                candidate_weight = optimize_weights_audited(
                    kernel,
                    p,
                    candidate,
                    alpha,
                    initial=weights,
                    max_iter=weight_max_iter,
                    audit=audit,
                    role=f"swap_candidate_pass_{pass_index + 1}",
                )
                if candidate_weight.objective < best_obj - improvement_tol:
                    best_obj = candidate_weight.objective
                    best_selected = candidate
                    best_weights = candidate_weight.weights
                    best_swap = (int(old_idx), int(new_idx))
        if best_selected is None or best_weights is None or best_swap is None:
            stop_reason = "complete_pass_no_accepted_improvement"
            break
        selected = best_selected
        weights = best_weights
        current = float(best_obj)
        history.append(current)
        swaps.append(best_swap)

    cap_hit = stop_reason is None
    if cap_hit:
        stop_reason = "safety_cap_exhausted"
    order = np.argsort(selected)
    refinement_monotonicity_tolerance = 64.0 * np.finfo(np.float64).eps
    objective_monotonicity_valid = all(
        later <= earlier + refinement_monotonicity_tolerance * max(1.0, abs(earlier), abs(later))
        for earlier, later in zip(history[:-1], history[1:])
    )
    result = RefinementResultAudited(
        selected=selected[order],
        weights=weights[order],
        objective=current,
        initial_weight_objective=initial_weight.objective,
        objective_history=tuple(history),
        accepted_swaps=tuple(swaps),
        passes_completed=passes_completed,
        stop_reason=stop_reason,
        cap_hit=cap_hit,
        initializer=initializer_name,
        weight_audit=audit.as_dict(),
        objective_monotonicity_valid=objective_monotonicity_valid,
    )
    if not objective_monotonicity_valid:
        raise InvalidTerminationError(
            "swaps_monotonicity",
            {"initializer": initializer_name, "objective_history": history},
        )
    if cap_hit:
        raise UnresolvedCapError(
            "swaps",
            {
                "configured_cap": max_passes,
                "initializer": initializer_name,
                "accepted_swaps": len(swaps),
                "passes_completed": passes_completed,
                "last_objective": current,
                "weight_audit": audit.as_dict(),
            },
        )
    return result


def uniform_subset_objective(
    kernel: FloatArray,
    p: FloatArray,
    selected: Sequence[int],
    alpha: float,
) -> float:
    weights = np.full(len(selected), 1.0 / len(selected), dtype=np.float64)
    return float(subset_objective(kernel, p, selected, alpha, weights))
