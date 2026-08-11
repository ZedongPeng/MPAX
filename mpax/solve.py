"""Unified entry point dispatching to the recommended solver per problem class."""

from mpax.r2hpdhg import r2HPDHG
from mpax.rapdhg import raPDHG
from mpax.utils import SaddlePointOutput

# Options shared by both solver classes with algorithm-agnostic meaning.
# Algorithm-specific tuning (restart thresholds, step-size exponents, ...)
# deliberately requires instantiating raPDHG / r2HPDHG directly.
_COMMON_OPTIONS = frozenset(
    {
        "eps_abs",
        "eps_rel",
        "eps_primal_infeasible",
        "eps_dual_infeasible",
        "iteration_limit",
        "verbose",
        "debug",
        "display_frequency",
        "jit",
        "unroll",
        "warm_start",
        "feasibility_polishing",
        "eps_feas_polish",
    }
)


def solve(
    problem, initial_primal_solution=None, initial_dual_solution=None, **options
) -> SaddlePointOutput:
    """Solve an LP or QP with the recommended solver.

    Dispatches on the static `problem.is_lp` flag: LPs go to r2HPDHG,
    QPs to raPDHG. The branch is resolved at trace time, so `solve`
    works under `jax.jit` and `vmap`.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        Problem from create_lp / create_qp / create_qp_from_gurobi.
    initial_primal_solution, initial_dual_solution : jnp.ndarray, optional
        Warm-start point, used when `warm_start=True` is passed.
    **options
        Algorithm-agnostic solver options only: eps_abs, eps_rel,
        eps_primal_infeasible, eps_dual_infeasible, iteration_limit,
        verbose, debug, display_frequency, jit, unroll, warm_start,
        feasibility_polishing, eps_feas_polish.

    Returns
    -------
    SaddlePointOutput
    """
    unknown = set(options) - _COMMON_OPTIONS
    if unknown:
        raise TypeError(
            f"solve() got unsupported options {sorted(unknown)}; "
            "algorithm-specific tuning requires instantiating raPDHG or "
            "r2HPDHG directly."
        )
    if options.get("warm_start") and (
        initial_primal_solution is None or initial_dual_solution is None
    ):
        raise ValueError(
            "warm_start=True requires both initial_primal_solution and "
            "initial_dual_solution"
        )
    solver_cls = r2HPDHG if problem.is_lp else raPDHG
    solver = solver_cls(**options)
    return solver.optimize(problem, initial_primal_solution, initial_dual_solution)
