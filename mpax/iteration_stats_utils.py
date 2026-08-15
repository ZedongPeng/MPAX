from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from mpax.solver_log import display_iteration_stats
from mpax.utils import (
    CachedQuadraticProgramInfo,
    ConvergenceInformation,
    InfeasibilityInformation,
    IterationStats,
    PdhgSolverState,
    PointType,
    QuadraticProgrammingProblem,
    ScaledQpProblem,
    safe_norm,
)


def compute_dual_objective(
    variable_lower_bound: jnp.ndarray,
    variable_upper_bound: jnp.ndarray,
    reduced_costs: jnp.ndarray,
    constraint_lower_bound: jnp.ndarray,
    constraint_upper_bound: jnp.ndarray,
    primal_solution: jnp.ndarray,
    dual_solution: jnp.ndarray,
    primal_obj_product: jnp.ndarray,
    objective_constant: float,
):
    """Compute the dual objective.

    Parameters
    ----------
    variable_lower_bound : jnp.ndarray
        Lower bound of variables.
    variable_upper_bound : jnp.ndarray
        Upper bound of variables.
    reduced_costs : jnp.ndarray
        Reduced costs.
    constraint_lower_bound : jnp.ndarray
        Lower bounds of the constraints.
    constraint_upper_bound : jnp.ndarray
        Upper bounds of the constraints.
    primal_solution : jnp.ndarray
        Primal solution.
    dual_solution : jnp.ndarray
        Dual solution.
    primal_obj_product : jnp.ndarray
        Product of the primal solution and the objective.
    objective_constant : float
        Constant in the objective.

    Returns
    -------
    float
        the dual objective
    """
    dual_objective_contribution_sum = jnp.sum(
        jnp.where(
            reduced_costs > 0.0,
            variable_lower_bound * reduced_costs,
            jnp.where(
                reduced_costs < 0.0,
                variable_upper_bound * reduced_costs,
                0.0,  # Handle the case where reduced_costs == 0
            ),
        )
    )
    lower_finite = jnp.where(
        jnp.isfinite(constraint_lower_bound), constraint_lower_bound, 0.0
    )
    upper_finite = jnp.where(
        jnp.isfinite(constraint_upper_bound), constraint_upper_bound, 0.0
    )
    base_dual_objective = (
        jnp.sum(
            jnp.maximum(dual_solution, 0.0) * lower_finite
            + jnp.minimum(dual_solution, 0.0) * upper_finite
        )
        + objective_constant
        - 0.5 * jnp.dot(primal_solution, primal_obj_product)
    )
    return base_dual_objective + dual_objective_contribution_sum


def compute_reduced_costs_from_primal_gradient(
    primal_gradient: jnp.ndarray,
    isfinite_variable_lower_bound: jnp.ndarray,
    isfinite_variable_upper_bound: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kernel to compute the reduced costs from the primal gradient.

    Parameters
    ----------
    primal_gradient : jnp.ndarray
        Primal gradient vector.
    isfinite_variable_lower_bound : jnp.ndarray
        Boolean array indicating finite lower bounds.
    isfinite_variable_upper_bound : jnp.ndarray
        Boolean array indicating finite upper bounds.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Reduced costs and reduced costs violation vectors.
    """
    reduced_costs = (
        jnp.maximum(primal_gradient, 0.0) * isfinite_variable_lower_bound
        + jnp.minimum(primal_gradient, 0.0) * isfinite_variable_upper_bound
    )
    reduced_costs_violation = primal_gradient - reduced_costs
    return reduced_costs, reduced_costs_violation


def compute_kkt_components(
    problem,
    primal_iterate,
    dual_iterate,
    primal_product,
    dual_product,
    primal_obj_product,
    norm_ord,
):
    """Shared KKT residual components for convergence stats and restart.

    Single source of truth: the restart metric previously recomputed these
    with two defects (gradient missing + primal_obj_product; dual residual
    missing the reduced-cost violation, leaving it identically zero for
    projected iterates).

    Returns
    -------
    tuple
        (primal_residual_norm, dual_residual_norm,
         primal_objective, dual_objective)
    """
    lower_variable_violation = jnp.maximum(
        problem.variable_lower_bound - primal_iterate, 0.0
    )
    upper_variable_violation = jnp.maximum(
        primal_iterate - problem.variable_upper_bound, 0.0
    )
    constraint_violation = primal_product - jnp.clip(
        primal_product, problem.constraint_lower_bound, problem.constraint_upper_bound
    )
    primal_residual_norm = jnp.linalg.norm(
        jnp.concatenate(
            [constraint_violation, lower_variable_violation, upper_variable_violation]
        ),
        ord=norm_ord,
    )

    primal_objective = (
        problem.objective_constant
        + jnp.dot(problem.objective_vector, primal_iterate)
        + 0.5 * jnp.dot(primal_iterate, primal_obj_product)
    )

    reduced_costs, reduced_costs_violation = compute_reduced_costs_from_primal_gradient(
        problem.objective_vector - dual_product + primal_obj_product,
        problem.isfinite_variable_lower_bound,
        problem.isfinite_variable_upper_bound,
    )
    dual_objective = compute_dual_objective(
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        reduced_costs,
        problem.constraint_lower_bound,
        problem.constraint_upper_bound,
        primal_iterate,
        dual_iterate,
        primal_obj_product,
        problem.objective_constant,
    )
    dual_residual = jnp.where(
        jnp.isfinite(problem.constraint_lower_bound),
        0.0,
        jnp.maximum(dual_iterate, 0.0),
    ) + jnp.where(
        jnp.isfinite(problem.constraint_upper_bound),
        0.0,
        jnp.maximum(-dual_iterate, 0.0),
    )
    dual_residual_norm = jnp.linalg.norm(
        jnp.concatenate([dual_residual, reduced_costs_violation]), ord=norm_ord
    )
    return (primal_residual_norm, dual_residual_norm, primal_objective, dual_objective)


# Note: the order of the calculations can be improved.
def compute_convergence_information(
    problem: QuadraticProgrammingProblem,
    qp_cache: CachedQuadraticProgramInfo,
    primal_iterate: jnp.ndarray,
    dual_iterate: jnp.ndarray,
    eps_ratio: float,
    primal_product: jnp.ndarray,
    dual_product: jnp.ndarray,
    primal_obj_product: jnp.ndarray,
    norm_ord: int,
) -> ConvergenceInformation:
    """
    Compute convergence information of the given primal and dual solutions.

    Relative versions of the residuals are defined as
      relative_residual = residual / (eps_ratio + norm),
    where
      eps_ratio = eps_abs / eps_rel
      residual = one of the residuals (l{2,_inf}_{primal,dual}_residual)
      norm = the relative norm (l{2,_inf} norm of
             {constraint_bounds,primal_linear_objective} respectively).

    1. If eps_rel = 0.0, these will all be 0.0.
    2. If eps_rel > 0.0, the absolute and relative termination
    criteria translate to relative_residual <= eps_rel.

    NOTE: The usefulness of these relative residuals is based on their
    relationship to TerminationCriteria. If the TerminationCriteria change
    consider adding additional iteration measures here.


    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        Quadratic programming problem instance.
    qp_cache : CachedQuadraticProgramInfo
        Cached quadratic program information.
    primal_iterate : jnp.ndarray
        Primal iterate vector.
    dual_iterate : jnp.ndarray
        Dual iterate vector.
    eps_ratio : float
        Epsilon ratio for relative measures.
    primal_product : jnp.ndarray
        Primal product vector.
    dual_product : jnp.ndarray
        Dual product vector.
    primal_obj_product : jnp.ndarray
        Primal objective product vector.

    Returns
    -------
    ConvergenceInformation
        Computed convergence information.
    """
    (primal_residual_norm, dual_residual_norm, primal_objective, dual_objective) = (
        compute_kkt_components(
            problem,
            primal_iterate,
            dual_iterate,
            primal_product,
            dual_product,
            primal_obj_product,
            norm_ord,
        )
    )
    primal_solution_norm = jnp.linalg.norm(primal_iterate, ord=norm_ord)
    dual_solution_norm = safe_norm(dual_iterate, ord=norm_ord)
    relative_primal_residual_norm = primal_residual_norm / (
        eps_ratio
        + jnp.maximum(
            qp_cache.constraint_bound_norm, safe_norm(primal_product, ord=norm_ord)
        )
    )
    relative_dual_residual_norm = dual_residual_norm / (
        eps_ratio
        + jnp.maximum(
            jnp.maximum(
                qp_cache.primal_linear_objective_norm,
                jnp.linalg.norm(primal_obj_product, ord=norm_ord),
            ),
            jnp.linalg.norm(dual_product, ord=norm_ord),
        )
    )
    corrected_dual_obj_value = jax.lax.cond(
        dual_residual_norm == 0.0, lambda: dual_objective, lambda: -jnp.inf
    )
    absolute_optimality_gap = jnp.abs(primal_objective - dual_objective)
    relative_optimality_gap = absolute_optimality_gap / (
        eps_ratio + jnp.maximum(abs(primal_objective), abs(dual_objective))
    )
    return ConvergenceInformation(
        PointType.POINT_TYPE_AVERAGE_ITERATE,
        primal_objective,
        dual_objective,
        corrected_dual_obj_value,
        primal_residual_norm,
        dual_residual_norm,
        relative_primal_residual_norm,
        relative_dual_residual_norm,
        absolute_optimality_gap,
        relative_optimality_gap,
        primal_solution_norm,
        dual_solution_norm,
    )


def compute_cupdlpx_convergence_information(
    scaled_problem,
    qp_cache: CachedQuadraticProgramInfo,
    solver_state,
    norm_ord: float = 2,
) -> ConvergenceInformation:
    """cuPDLP-x's convergence measures, for the r2HPDHG LP path.

    Four things differ from `compute_convergence_information`, and each one
    moves the point at which the solver stops:

    * evaluated at the pure PDHG iterate, not the Halpern average;
    * the dual residual is `c - A'y - dual_slack`, using the bound
      multiplier the primal projection implies, rather than a reduced-cost
      violation reconstructed from the gradient;
    * the relative denominators are static (norms of the original data)
      instead of growing with ||Ax|| and ||A'y||;
    * the gap denominator is `1 + |p| + |d|`, not `1 + max(|p|, |d|)`.
    """
    problem = scaled_problem.scaled_qp
    primal_iterate = solver_state.pdhg_primal_solution
    dual_iterate = solver_state.pdhg_dual_solution
    # Both products are formed here rather than carried: this runs once per
    # evaluation window, so two matvecs are cheaper than maintaining them on
    # every one of the ~200 steps in between.
    primal_product = problem.matvec(primal_iterate)
    dual_product = problem.matvec_t(dual_iterate)
    dual_slack = solver_state.dual_slack

    # Undo the Ruiz/Pock-Chambolle scaling before measuring, as the
    # reference does; the objective pairing is scale-invariant already.
    primal_residual = (
        primal_product
        - jnp.clip(
            primal_product,
            problem.constraint_lower_bound,
            problem.constraint_upper_bound,
        )
    ) * scaled_problem.constraint_rescaling
    primal_residual_norm = (
        jnp.linalg.norm(primal_residual, ord=norm_ord)
        / scaled_problem.constraint_bound_rescaling
    )

    dual_residual = (
        problem.objective_vector - dual_product - dual_slack
    ) * scaled_problem.variable_rescaling
    dual_residual_norm = (
        jnp.linalg.norm(dual_residual, ord=norm_ord)
        / scaled_problem.objective_vector_rescaling
    )

    lower_finite = jnp.where(
        jnp.isfinite(problem.constraint_lower_bound),
        problem.constraint_lower_bound,
        0.0,
    )
    upper_finite = jnp.where(
        jnp.isfinite(problem.constraint_upper_bound),
        problem.constraint_upper_bound,
        0.0,
    )

    objective_unscale = (
        scaled_problem.constraint_bound_rescaling
        * scaled_problem.objective_vector_rescaling
    )
    primal_objective = (
        jnp.dot(problem.objective_vector, primal_iterate) / objective_unscale
        + problem.objective_constant
    )
    dual_objective = (
        jnp.sum(
            jnp.maximum(dual_iterate, 0.0) * lower_finite
            + jnp.minimum(dual_iterate, 0.0) * upper_finite
        )
        + jnp.dot(primal_iterate, dual_slack)
    ) / objective_unscale + problem.objective_constant

    relative_primal_residual_norm = primal_residual_norm / (
        1 + qp_cache.constraint_bound_norm
    )
    relative_dual_residual_norm = dual_residual_norm / (
        1 + qp_cache.primal_linear_objective_norm
    )
    absolute_optimality_gap = jnp.abs(primal_objective - dual_objective)
    relative_optimality_gap = absolute_optimality_gap / (
        1 + jnp.abs(primal_objective) + jnp.abs(dual_objective)
    )
    corrected_dual_obj_value = jax.lax.cond(
        dual_residual_norm == 0.0, lambda: dual_objective, lambda: -jnp.inf
    )

    return ConvergenceInformation(
        PointType.POINT_TYPE_CURRENT_ITERATE,
        primal_objective,
        dual_objective,
        corrected_dual_obj_value,
        primal_residual_norm,
        dual_residual_norm,
        relative_primal_residual_norm,
        relative_dual_residual_norm,
        absolute_optimality_gap,
        relative_optimality_gap,
        jnp.linalg.norm(primal_iterate, ord=norm_ord),
        safe_norm(dual_iterate, ord=norm_ord),
    )


def compute_infeasibility_information(
    problem: QuadraticProgrammingProblem,
    primal_ray_estimate: jnp.ndarray,
    dual_ray_estimate: jnp.ndarray,
    primal_ray_estimate_product: jnp.ndarray,
    dual_ray_estimate_product: jnp.ndarray,
    primal_ray_estimate_obj_product: jnp.ndarray,
):
    """
    Compute infeasibility information of the given primal and dual solutions.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        Quadratic programming problem instance.
    primal_ray_estimate : jnp.ndarray
        Primal ray estimate vector.
    dual_ray_estimate : jnp.ndarray
        Dual ray estimate vector.
    primal_ray_estimate_product : jnp.ndarray
        Primal ray estimate product vector.
    dual_ray_estimate_product : jnp.ndarray
        Dual ray estimate product vector.
    primal_ray_estimate_obj_product : jnp.ndarray
        Primal ray estimate objective product vector.

    Returns
    -------
    InfeasibilityInformation
        Computed infeasibility information.
    """
    # Assume InfeasibilityInformation is a namedtuple
    primal_ray_inf_norm = jnp.linalg.norm(primal_ray_estimate, ord=jnp.inf)
    scaled_primal_ray_estimate, scaled_primal_ray_estimate_product = jax.lax.cond(
        primal_ray_inf_norm == 0.0,
        lambda _: (primal_ray_estimate, primal_ray_estimate_product),
        lambda _: (
            primal_ray_estimate / primal_ray_inf_norm,
            primal_ray_estimate_product / primal_ray_inf_norm,
        ),
        operand=None,
    )

    lower_variable_violation = jnp.maximum(
        (-1 / problem.isfinite_variable_lower_bound + 1) - scaled_primal_ray_estimate,
        0.0,
    )
    upper_variable_violation = jnp.maximum(
        scaled_primal_ray_estimate - (1 / problem.isfinite_variable_upper_bound - 1),
        0.0,
    )

    constraint_violation = jnp.where(
        jnp.isfinite(problem.constraint_upper_bound),
        jnp.maximum(scaled_primal_ray_estimate_product, 0.0),
        0.0,
    ) + jnp.where(
        jnp.isfinite(problem.constraint_lower_bound),
        jnp.maximum(-scaled_primal_ray_estimate_product, 0.0),
        0.0,
    )

    max_primal_ray_infeasibility = jnp.linalg.norm(
        jnp.concatenate(
            [constraint_violation, lower_variable_violation, upper_variable_violation]
        ),
        ord=jnp.inf,
    )
    primal_ray_linear_objective = jnp.dot(
        problem.objective_vector, scaled_primal_ray_estimate
    )
    reduced_costs, reduced_costs_violation = compute_reduced_costs_from_primal_gradient(
        -dual_ray_estimate_product,
        problem.isfinite_variable_lower_bound,
        problem.isfinite_variable_upper_bound,
    )
    dual_objective = compute_dual_objective(
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        reduced_costs,
        problem.constraint_lower_bound,
        problem.constraint_upper_bound,
        primal_ray_estimate,
        dual_ray_estimate,
        primal_ray_estimate_obj_product,
        problem.objective_constant,
    )
    dual_residual = jnp.where(
        jnp.isfinite(problem.constraint_lower_bound),
        0.0,
        jnp.maximum(dual_ray_estimate, 0.0),
    ) + jnp.where(
        jnp.isfinite(problem.constraint_upper_bound),
        0.0,
        jnp.maximum(-dual_ray_estimate, 0.0),
    )
    l_inf_dual_residual = jnp.linalg.norm(
        jnp.concatenate([dual_residual, reduced_costs_violation]), ord=jnp.inf
    )

    scaling_factor = jnp.maximum(
        jnp.linalg.norm(scaled_primal_ray_estimate, ord=jnp.inf),
        jnp.linalg.norm(reduced_costs, ord=jnp.inf),
    )
    max_dual_ray_infeasibility, dual_ray_objective = jax.lax.cond(
        scaling_factor == 0.0,
        lambda: (0.0, 0.0),
        lambda: (l_inf_dual_residual / scaling_factor, dual_objective / scaling_factor),
    )

    return InfeasibilityInformation(
        PointType.POINT_TYPE_AVERAGE_ITERATE,
        max_primal_ray_infeasibility,
        primal_ray_linear_objective,
        max_dual_ray_infeasibility,
        dual_ray_objective,
    )


def evaluate_unscaled_iteration_stats(
    scaled_problem: ScaledQpProblem,
    qp_cache: CachedQuadraticProgramInfo,
    solver_state: PdhgSolverState,
    cumulative_time: float,
    eps_ratio: float,
    norm_ord: float,
    average: bool = True,
    infeasibility_detection: bool = True,
):
    """
    Compute the iteration stats of the unscaled primal and dual solutions.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        Scaled quadratic programming problem instance.
    qp_cache : CachedQuadraticProgramInfo
        Cached quadratic program information.
    solver_state : PdhgSolverState
        The current solver state.
    cumulative_time : float
        Cumulative time in seconds.
    eps_ratio : float
        eps_abs / eps_rel
    norm_ord : float
        Order of the norm.
    average : bool
        Whether to use the average solution.
    infeasibility_detection : bool
        Whether to detect infeasibility.

    Returns
    -------
    IterationStats
        Computed iteration statistics for the unscaled problem.
    """
    (
        unscaled_primal_solution,
        unscaled_dual_solution,
        unscaled_primal_product,
        unscaled_dual_product,
        unscaled_primal_obj_product,
    ) = jax.lax.cond(
        average == True,
        lambda: (
            solver_state.avg_primal_solution / scaled_problem.variable_rescaling,
            solver_state.avg_dual_solution / scaled_problem.constraint_rescaling,
            solver_state.avg_primal_product * scaled_problem.constraint_rescaling,
            solver_state.avg_dual_product * scaled_problem.variable_rescaling,
            solver_state.avg_primal_obj_product * scaled_problem.variable_rescaling,
        ),
        lambda: (
            solver_state.current_primal_solution / scaled_problem.variable_rescaling,
            solver_state.current_dual_solution / scaled_problem.constraint_rescaling,
            solver_state.current_primal_product * scaled_problem.constraint_rescaling,
            solver_state.current_dual_product * scaled_problem.variable_rescaling,
            solver_state.current_primal_obj_product * scaled_problem.variable_rescaling,
        ),
    )
    # bound_objective_rescaling's scalar factors (1.0 when that pass is off,
    # so raPDHG paths are untouched): bounds were multiplied by s_b and the
    # objective by s_c, so primal-side quantities carry 1/s_b and dual-side
    # quantities 1/s_c back to original units. Without this, the polishing
    # feasibility checks measure an s_b-shrunk iterate against the original
    # bounds and can never reach their tolerance.
    s_b = scaled_problem.constraint_bound_rescaling
    s_c = scaled_problem.objective_vector_rescaling
    unscaled_primal_solution = unscaled_primal_solution / s_b
    unscaled_primal_product = unscaled_primal_product / s_b
    unscaled_dual_solution = unscaled_dual_solution / s_c
    unscaled_dual_product = unscaled_dual_product / s_c
    # Qx is a c-unit term of the dual residual; identically zero on the only
    # path that enables the scalar pass (r2HPDHG is LP-only).
    unscaled_primal_obj_product = unscaled_primal_obj_product / s_c
    convergence_information = compute_convergence_information(
        scaled_problem.original_qp,
        qp_cache,
        unscaled_primal_solution,
        unscaled_dual_solution,
        eps_ratio,
        unscaled_primal_product,
        unscaled_dual_product,
        unscaled_primal_obj_product,
        norm_ord,
    )
    # TODO: improve the cond for vmap
    infeasibility_information = jax.lax.cond(
        infeasibility_detection,
        lambda: compute_infeasibility_information(
            scaled_problem.original_qp,
            unscaled_primal_solution,
            unscaled_dual_solution,
            unscaled_primal_product,
            unscaled_dual_product,
            unscaled_primal_obj_product,
        ),
        lambda: InfeasibilityInformation(
            PointType.POINT_TYPE_AVERAGE_ITERATE, 1.0, 1.0, 1.0, 1.0
        ),
    )
    current_iteration_stats = IterationStats(
        iteration_number=solver_state.num_iterations,
        convergence_information=convergence_information,
        infeasibility_information=infeasibility_information,
        cumulative_rejected_steps=0,  # cumulative_rejected_steps
        cumulative_time_sec=cumulative_time,
        step_size=solver_state.step_size,
        primal_weight=solver_state.primal_weight,
        method_specific_stats={},
    )
    display_iteration_stats(current_iteration_stats, solver_state)
    return current_iteration_stats


def should_log_iteration_status(iteration: int, params: NamedTuple) -> bool:
    """
    Determine if the iteration statistics should be printed based on the
    termination status, current iteration number, and display frequency.

    Parameters
    ----------
    iteration : int
        Current iteration number.
    params : NamedTuple
        Parameters for the solver.

    Returns
    -------
    bool
        Whether to print the iteration stats.
    """
    num_of_evaluations = (iteration - 1) // params.termination_evaluation_frequency
    # Print stats every display_frequency * termination_evaluation_frequency iterations
    return num_of_evaluations % params.display_frequency == 0
