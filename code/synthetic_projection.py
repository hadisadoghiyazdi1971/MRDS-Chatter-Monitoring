#!/usr/bin/env python3
"""Synthetic MRDS updates followed by duplicate-free observed projection."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from mrds_core import (
    EmpiricalMeasure,
    exact_signed_coefficients,
    finite_meta_p,
    finite_meta_q,
    initialize_prototypes,
    median_bandwidth,
    prototype_kernel_and_projections,
    prototype_kernel_column,
    renyi_probability_divergence,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class UnresolvedCapError(RuntimeError):
    """Raised when the synthetic-stage safety cap is binding."""

    def __init__(self, details: dict[str, object]):
        super().__init__(f"unresolved synthetic safety cap: {details}")
        self.details = details


class InvalidTerminationError(RuntimeError):
    """Raised when the objective trajectory fails its numerical checks."""

    def __init__(self, details: dict[str, object]):
        super().__init__(f"invalid synthetic objective trajectory: {details}")
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


def synthetic_projection(
    measures: Sequence[EmpiricalMeasure],
    pairwise_distance: FloatArray,
    full_kernel: FloatArray,
    *,
    n_prototypes: int,
    alpha: float,
    prototype_atoms: int,
    bandwidth: float | None,
    seed: int,
    max_sweeps: int,
    movement_tolerance: float = 1e-5,
    step_size: float = 1.0,
    armijo: float = 1e-4,
    backtrack: float = 0.5,
    min_step: float = 1e-5,
    relative_improvement_tolerance: float = 1e-3,
    required_consecutive_sweeps: int = 3,
    monotonicity_tolerance_factor: float = 64.0,
) -> SyntheticProjectionResult:
    """Apply the reported finite synthetic stage and one-to-one projection."""

    started = time.perf_counter()
    if max_sweeps < required_consecutive_sweeps:
        raise ValueError("max_sweeps is smaller than the stopping window")
    if not 1 <= n_prototypes <= len(measures):
        raise ValueError("invalid prototype count")
    if any(measure.support.shape[0] != prototype_atoms for measure in measures):
        raise ValueError(
            "exact-EMD runs require the configured number of atoms in each recording"
        )

    p = finite_meta_p(full_kernel)
    h = bandwidth if bandwidth is not None else median_bandwidth(pairwise_distance)
    prototypes = initialize_prototypes(
        measures,
        pairwise_distance,
        n_prototypes,
        prototype_atoms,
        seed,
    )
    anchor_prototype_kernel, projections, _ = prototype_kernel_and_projections(
        measures,
        prototypes,
        bandwidth=h,
        need_projections=True,
    )
    q = finite_meta_q(anchor_prototype_kernel)
    current = float(renyi_probability_divergence(p, q, alpha))
    initial_objective = current
    denominator = max(abs(initial_objective), np.finfo(np.float64).eps)
    history = [current]
    accepted_by_sweep: list[int] = []
    max_move_by_sweep: list[float] = []
    absolute_decrease_by_sweep: list[float] = []
    relative_improvement_by_sweep: list[float] = []
    consecutive_small_improvements = 0
    tiny_roundoff_increases = 0
    stop_reason: str | None = None
    last_max_move = float("nan")

    for sweep in range(max_sweeps):
        objective_before_sweep = current
        max_move = 0.0
        accepted_count = 0

        for prototype_index, prototype in enumerate(prototypes):
            coefficients, _ = exact_signed_coefficients(
                p, q, anchor_prototype_kernel, alpha
            )
            direction = np.zeros_like(prototype.support)
            normalizer = (
                float(np.sum(np.abs(coefficients[:, prototype_index]))) + 1e-15
            )
            for anchor_index in range(len(measures)):
                projection = projections[prototype_index][anchor_index]
                if projection is None:
                    raise RuntimeError("a required barycentric projection is missing")
                direction += coefficients[anchor_index, prototype_index] * (
                    projection - prototype.support
                )
            direction /= normalizer
            direction_norm_squared = float(
                np.sum(prototype.weights[:, None] * direction * direction)
            )
            if direction_norm_squared <= movement_tolerance * movement_tolerance:
                continue

            old_support = prototype.support.copy()
            finite_difference_step = 1e-6 / max(
                1.0, math.sqrt(direction_norm_squared)
            )

            def objective_along(signed_step: float) -> float:
                prototypes[prototype_index].support = (
                    old_support + signed_step * direction
                )
                column, _, _ = prototype_kernel_column(
                    measures,
                    prototypes[prototype_index],
                    bandwidth=h,
                    need_projections=False,
                )
                trial_kernel = anchor_prototype_kernel.copy()
                trial_kernel[:, prototype_index] = column
                return renyi_probability_divergence(
                    p, finite_meta_q(trial_kernel), alpha
                )

            plus = objective_along(finite_difference_step)
            minus = objective_along(-finite_difference_step)
            prototypes[prototype_index].support = old_support
            directional_slope = (plus - minus) / (2.0 * finite_difference_step)
            if directional_slope >= 0.0:
                direction *= -1.0
                directional_slope *= -1.0
            if directional_slope >= -1e-14:
                continue

            step = step_size
            accepted = False
            objective_before_update = current
            while step >= min_step:
                prototypes[prototype_index].support = old_support + step * direction
                trial_column, _, _ = prototype_kernel_column(
                    measures,
                    prototypes[prototype_index],
                    bandwidth=h,
                    need_projections=False,
                )
                trial_kernel = anchor_prototype_kernel.copy()
                trial_kernel[:, prototype_index] = trial_column
                trial = renyi_probability_divergence(
                    p, finite_meta_q(trial_kernel), alpha
                )
                if (
                    trial
                    <= objective_before_update + armijo * step * directional_slope
                ):
                    anchor_prototype_kernel = trial_kernel
                    q = finite_meta_q(anchor_prototype_kernel)
                    current = float(trial)
                    accepted = True
                    accepted_count += 1
                    max_move = max(
                        max_move, step * math.sqrt(direction_norm_squared)
                    )
                    break
                prototypes[prototype_index].support = old_support
                step *= backtrack

            if not accepted:
                prototypes[prototype_index].support = old_support

            current_column, current_projections, _ = prototype_kernel_column(
                measures,
                prototypes[prototype_index],
                bandwidth=h,
                need_projections=True,
            )
            anchor_prototype_kernel[:, prototype_index] = current_column
            projections[prototype_index] = current_projections
            q = finite_meta_q(anchor_prototype_kernel)
            current = float(renyi_probability_divergence(p, q, alpha))

        numerical_tolerance = (
            monotonicity_tolerance_factor
            * np.finfo(np.float64).eps
            * max(1.0, abs(objective_before_sweep), abs(current))
        )
        objective_increase = current - objective_before_sweep
        if objective_increase > numerical_tolerance:
            raise InvalidTerminationError(
                {
                    "sweep": sweep + 1,
                    "previous_objective": objective_before_sweep,
                    "current_objective": current,
                    "objective_increase": objective_increase,
                    "allowed_numerical_tolerance": numerical_tolerance,
                    "objective_history": history + [current],
                }
            )
        if objective_increase > 0.0:
            tiny_roundoff_increases += 1

        absolute_decrease = max(0.0, objective_before_sweep - current)
        relative_improvement = absolute_decrease / denominator
        if relative_improvement < relative_improvement_tolerance:
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
            stop_reason = "relative_objective_improvement_below_threshold"
            break

    if stop_reason is None:
        raise UnresolvedCapError(
            {
                "configured_cap": max_sweeps,
                "sweeps_completed": len(accepted_by_sweep),
                "last_max_move": last_max_move,
                "last_accepted_updates": (
                    accepted_by_sweep[-1] if accepted_by_sweep else None
                ),
                "final_consecutive_small_improvements": (
                    consecutive_small_improvements
                ),
                "relative_improvement_threshold": (
                    relative_improvement_tolerance
                ),
                "required_consecutive_sweeps": required_consecutive_sweeps,
                "normalization_denominator": denominator,
                "objective_history": history,
                "absolute_decrease_by_sweep": absolute_decrease_by_sweep,
                "relative_improvement_by_sweep": relative_improvement_by_sweep,
                "max_move_by_sweep": max_move_by_sweep,
                "accepted_updates_by_sweep": accepted_by_sweep,
                "tiny_roundoff_increases": tiny_roundoff_increases,
            }
        )

    _, _, projection_cost = prototype_kernel_and_projections(
        measures,
        prototypes,
        bandwidth=h,
        need_projections=False,
    )
    prototype_rows, observed_columns = linear_sum_assignment(projection_cost.T)
    selected = observed_columns[np.argsort(prototype_rows)].astype(np.int64)

    return SyntheticProjectionResult(
        selected=selected,
        objective_history=tuple(history),
        sweeps_completed=len(accepted_by_sweep),
        stop_reason=stop_reason,
        cap_hit=False,
        last_max_move=float(last_max_move),
        accepted_updates_by_sweep=tuple(accepted_by_sweep),
        max_move_by_sweep=tuple(max_move_by_sweep),
        absolute_decrease_by_sweep=tuple(absolute_decrease_by_sweep),
        relative_improvement_by_sweep=tuple(relative_improvement_by_sweep),
        final_consecutive_small_improvements=consecutive_small_improvements,
        relative_improvement_threshold=relative_improvement_tolerance,
        required_consecutive_sweeps=required_consecutive_sweeps,
        normalization_denominator=denominator,
        monotonicity_valid=True,
        monotonicity_tolerance_factor=monotonicity_tolerance_factor,
        tiny_roundoff_increases=tiny_roundoff_increases,
        runtime_seconds=time.perf_counter() - started,
    )
