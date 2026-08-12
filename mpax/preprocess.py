import logging
from copy import deepcopy
from typing import Tuple, Union

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR

from mpax.solver_log import (
    display_problem_details,
    get_col_l_inf_norms,
    get_row_l_inf_norms,
    get_row_l2_norms,
    get_col_l2_norms,
)
from mpax.utils import QuadraticProgrammingProblem, ScaledQpProblem

logger = logging.getLogger(__name__)


def validate(p: QuadraticProgrammingProblem) -> bool:
    """
    Check that the QuadraticProgrammingProblem is valid.

    Parameters
    ----------
    p : QuadraticProgrammingProblem
        The quadratic programming problem to validate.

    Returns
    -------
    bool
        True if the problem is valid, otherwise raises an error.
    """
    error_found = False

    if len(p.variable_lower_bound) != len(p.variable_upper_bound):
        logger.error(
            "%d != %d", len(p.variable_lower_bound), len(p.variable_upper_bound)
        )
        error_found = True

    if len(p.variable_lower_bound) != len(p.objective_vector):
        logger.error("%d != %d", len(p.variable_lower_bound), len(p.objective_vector))
        error_found = True

    if p.constraint_matrix.shape[0] != len(
        p.constraint_lower_bound
    ) or p.constraint_matrix.shape[0] != len(p.constraint_upper_bound):
        logger.error(
            "%d != %d != %d",
            p.constraint_matrix.shape[0],
            len(p.constraint_lower_bound),
            len(p.constraint_upper_bound),
        )
        error_found = True

    if p.constraint_matrix.shape[1] != len(p.objective_vector):
        logger.error("%d != %d", p.constraint_matrix.shape[1], len(p.objective_vector))
        error_found = True

    if p.objective_matrix is not None and p.objective_matrix.shape != (
        len(p.objective_vector),
        len(p.objective_vector),
    ):
        logger.error(
            "%s is not square with length %d",
            p.objective_matrix.shape,
            len(p.objective_vector),
        )
        error_found = True

    if jnp.any(p.variable_lower_bound == jnp.inf):
        logger.error(
            "sum(p.variable_lower_bound == Inf) = %s",
            jnp.sum(jnp.isinf(p.variable_lower_bound)),
        )
        error_found = True

    if jnp.any(p.variable_upper_bound == -jnp.inf):
        logger.error(
            "sum(p.variable_upper_bound == -Inf) = %s",
            jnp.sum(jnp.isinf(p.variable_upper_bound)),
        )
        error_found = True

    if jnp.any(jnp.isnan(p.variable_lower_bound)) or jnp.any(
        jnp.isnan(p.variable_upper_bound)
    ):
        logger.error("NaN found in variable bounds of QuadraticProgrammingProblem.")
        error_found = True

    if jnp.any(jnp.isnan(p.constraint_lower_bound)) or jnp.any(
        jnp.isnan(p.constraint_upper_bound)
    ):
        logger.error("NaN found in constraint bounds of QuadraticProgrammingProblem.")
        error_found = True

    if jnp.any(p.constraint_lower_bound == jnp.inf):
        logger.error("constraint_lower_bound must not contain +inf entries.")
        error_found = True

    if jnp.any(p.constraint_upper_bound == -jnp.inf):
        logger.error("constraint_upper_bound must not contain -inf entries.")
        error_found = True

    if jnp.any(p.constraint_lower_bound > p.constraint_upper_bound):
        logger.error("constraint_lower_bound must be <= constraint_upper_bound.")
        error_found = True

    if jnp.any(jnp.isinf(p.objective_vector)) or jnp.any(jnp.isnan(p.objective_vector)):
        logger.error(
            "NaN or Inf found in objective vector of QuadraticProgrammingProblem."
        )
        error_found = True

    if jnp.any(jnp.isinf(p.constraint_matrix.data)) or jnp.any(
        jnp.isnan(p.constraint_matrix.data)
    ):
        logger.error(
            "NaN or Inf found in constraint matrix of QuadraticProgrammingProblem."
        )
        error_found = True

    if p.objective_matrix is not None and (
        jnp.any(jnp.isinf(p.objective_matrix.data))
        or jnp.any(jnp.isnan(p.objective_matrix.data))
    ):
        logger.error(
            "NaN or Inf found in objective matrix of QuadraticProgrammingProblem."
        )
        error_found = True

    if error_found:
        raise ValueError(
            "Error found when validating QuadraticProgrammingProblem. See log statements for details."
        )

    return True


def scale_problem(
    problem: QuadraticProgrammingProblem,
    constraint_rescaling: jnp.ndarray,
    variable_rescaling: jnp.ndarray,
) -> None:
    """
    Rescales `problem` in place.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.
    constraint_rescaling : jnp.ndarray
        The rescaling factors for the constraints.
    variable_rescaling : jnp.ndarray
        The rescaling factors for the variables.
    """
    # Scale the objective vector
    problem.objective_vector /= variable_rescaling

    # Scale the objective matrix (None for LPs — nothing to scale)
    if problem.objective_matrix is None:
        pass
    elif isinstance(problem.objective_matrix, jnp.ndarray):
        # Scale the matrix along the rows
        # variable_rescaling[:, None] reshapes variable_rescaling from (n,) to (n, 1),
        # enabling broadcasting along the rows. Each element in row i is divided by variable_rescaling[i].
        problem.objective_matrix = (
            problem.objective_matrix / variable_rescaling[:, None]
        )
        # Scale the matrix along the columns
        # variable_rescaling (with shape (n,)) is broadcasted along the columns,
        # so each element in column j is divided by variable_rescaling[j].
        problem.objective_matrix = problem.objective_matrix / variable_rescaling
    elif isinstance(problem.objective_matrix, BCOO):
        scaled_data = (
            problem.objective_matrix.data
            * (1.0 / variable_rescaling)[problem.objective_matrix.indices[:, 0]]
            * (1.0 / variable_rescaling)[problem.objective_matrix.indices[:, 1]]
        )
        problem.objective_matrix.data = scaled_data

    # Scale variable bounds
    problem.variable_upper_bound *= variable_rescaling
    problem.variable_lower_bound *= variable_rescaling

    # Scale the constraint bounds (inf entries stay inf: rescaling > 0)
    problem.constraint_lower_bound /= constraint_rescaling
    problem.constraint_upper_bound /= constraint_rescaling

    # Scale the constraint matrix
    if isinstance(problem.constraint_matrix, jnp.ndarray):
        problem.constraint_matrix = (
            problem.constraint_matrix / constraint_rescaling[:, None]
        )
        problem.constraint_matrix = problem.constraint_matrix / variable_rescaling
    elif isinstance(problem.constraint_matrix, BCOO):
        scaled_data = (
            problem.constraint_matrix.data
            * (1.0 / constraint_rescaling)[problem.constraint_matrix.indices[:, 0]]
            * (1.0 / variable_rescaling)[problem.constraint_matrix.indices[:, 1]]
        )
        problem.constraint_matrix.data = scaled_data
    problem.constraint_matrix_t = problem.constraint_matrix.T


def unscale_problem(
    problem: QuadraticProgrammingProblem,
    constraint_rescaling: jnp.ndarray,
    variable_rescaling: jnp.ndarray,
) -> None:
    """
    Recovers the original problem from the scaled problem.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.
    constraint_rescaling : jnp.ndarray
        The rescaling factors for the constraints.
    variable_rescaling : jnp.ndarray
        The rescaling factors for the variables.
    """
    scale_problem(problem, 1.0 / constraint_rescaling, 1.0 / variable_rescaling)


def l2_norm_rescaling(
    problem: QuadraticProgrammingProblem,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rescales a quadratic programming problem by dividing each row and column of the constraint matrix by the sqrt of its respective L2 norm.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The input quadratic programming problem. This is modified in place.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple of vectors containing the row and column rescaling factors.
    """
    # Calculate L2 norms of rows and columns
    norm_of_rows = get_row_l2_norms(problem.constraint_matrix)
    objective_col_l2 = (
        jnp.zeros(problem.constraint_matrix.shape[1])
        if problem.objective_matrix is None
        else get_col_l2_norms(problem.objective_matrix)
    )
    norm_of_columns = jnp.sqrt(
        jnp.square(get_col_l2_norms(problem.constraint_matrix))
        + jnp.square(objective_col_l2)
    )
    # Avoid division by zero by setting norms to 1 where they are 0
    norm_of_rows = jnp.where(norm_of_rows == 0, 1.0, norm_of_rows)
    norm_of_columns = jnp.where(norm_of_columns == 0, 1.0, norm_of_columns)

    # Compute the rescaling factors as the square roots of the norms
    row_rescale_factor = jnp.sqrt(norm_of_rows)
    column_rescale_factor = jnp.sqrt(norm_of_columns)

    # Scale the problem using these factors
    scale_problem(problem, row_rescale_factor, column_rescale_factor)

    return row_rescale_factor, column_rescale_factor


def rescale_problem(
    l_inf_ruiz_iterations: int,
    l2_norm_rescaling_flag: bool,
    pock_chambolle_alpha: Union[float, None],
    original_problem: QuadraticProgrammingProblem,
    bound_objective_rescaling: bool = False,
) -> ScaledQpProblem:
    """
    Preprocesses and rescales the original problem, returning a ScaledQpProblem struct.

    Parameters
    ----------
    l_inf_ruiz_iterations : int
        The number of iterations for L_inf Ruiz rescaling.
    l2_norm_rescaling_flag : bool
        Whether to apply L2 norm rescaling.
    pock_chambolle_alpha : Union[float, None]
        The exponent parameter for Pock-Chambolle rescaling. Set to None to skip.
    original_problem : QuadraticProgrammingProblem
        The original quadratic programming problem.

    Returns
    -------
    ScaledQpProblem
        A struct containing the scaled problem and rescaling factors.
    """
    # Convert to BCOO format for easier manipulation
    if isinstance(original_problem.constraint_matrix, BCSR):
        original_problem.constraint_matrix = (
            original_problem.constraint_matrix.to_bcoo()
        )
    elif isinstance(original_problem.constraint_matrix, (BCOO, jnp.ndarray)):
        pass
    else:
        raise ValueError("Unsupported matrix format.")

    if isinstance(original_problem.constraint_matrix_t, BCSR):
        original_problem.constraint_matrix_t = (
            original_problem.constraint_matrix_t.to_bcoo()
        )
    elif isinstance(original_problem.constraint_matrix_t, (BCOO, jnp.ndarray)):
        pass
    else:
        raise ValueError("Unsupported matrix format.")

    problem = deepcopy(original_problem)

    num_constraints, num_variables = problem.constraint_matrix.shape
    constraint_rescaling = jnp.ones(num_constraints)
    variable_rescaling = jnp.ones(num_variables)

    if l_inf_ruiz_iterations > 0:
        con_rescale, var_rescale = ruiz_rescaling(
            problem, l_inf_ruiz_iterations, jnp.inf
        )
        constraint_rescaling *= con_rescale
        variable_rescaling *= var_rescale

    if l2_norm_rescaling_flag:
        con_rescale, var_rescale = l2_norm_rescaling(problem)
        constraint_rescaling *= con_rescale
        variable_rescaling *= var_rescale

    if pock_chambolle_alpha is not None:
        con_rescale, var_rescale = pock_chambolle_rescaling(
            problem, pock_chambolle_alpha
        )
        constraint_rescaling *= con_rescale
        variable_rescaling *= var_rescale

    if l_inf_ruiz_iterations == 0 and not l2_norm_rescaling_flag:
        logger.info("No rescaling applied.")
    else:
        logger.info(
            "Problem after rescaling (Ruiz iterations = %d, l2_norm_rescaling = %s):",
            l_inf_ruiz_iterations,
            l2_norm_rescaling_flag,
        )
    display_problem_details(problem)

    # Benchmark-backed choice (P6, 2026-08-10): BCOO beats BCSR on both matvec
    # directions on 4/5 instances (see benchmarks/results/2026-08-10-matvec.txt),
    # so the constraint matrices stay BCOO end-to-end instead of converting
    # back to BCSR here.
    # cuPDLP-x's final scalar pass, applied after Ruiz/Pock-Chambolle: divide
    # the bounds and the objective through by their own norms so both land at
    # order 1. Equality rows are excluded from the bound norm (their lower and
    # upper bound would otherwise be counted twice).
    constraint_bound_rescaling = 1.0
    objective_vector_rescaling = 1.0
    if bound_objective_rescaling:
        lower_masked = jnp.where(
            jnp.isfinite(problem.constraint_lower_bound)
            & (problem.constraint_lower_bound != problem.constraint_upper_bound),
            problem.constraint_lower_bound,
            0.0,
        )
        upper_masked = jnp.where(
            jnp.isfinite(problem.constraint_upper_bound),
            problem.constraint_upper_bound,
            0.0,
        )
        constraint_bound_rescaling = 1 / (
            jnp.linalg.norm(jnp.concatenate([lower_masked, upper_masked]), ord=2) + 1
        )
        objective_vector_rescaling = 1 / (
            jnp.linalg.norm(problem.objective_vector, ord=2) + 1
        )
        problem.variable_lower_bound *= constraint_bound_rescaling
        problem.variable_upper_bound *= constraint_bound_rescaling
        problem.constraint_lower_bound *= constraint_bound_rescaling
        problem.constraint_upper_bound *= constraint_bound_rescaling
        problem.objective_vector *= objective_vector_rescaling

    scaled_problem = ScaledQpProblem(
        original_qp=original_problem,
        scaled_qp=problem,
        constraint_rescaling=constraint_rescaling,
        variable_rescaling=variable_rescaling,
        constraint_bound_rescaling=constraint_bound_rescaling,
        objective_vector_rescaling=objective_vector_rescaling,
    )

    return scaled_problem


def ruiz_rescaling(
    problem, num_iterations: int, p: float = float("inf")
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Uses a modified Ruiz rescaling algorithm to rescale the matrix M=[Q,A';A,0]
    where Q is objective_matrix and A is constraint_matrix, and returns the
    cumulative scaling vectors.

    Reference:
    https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem. This is modified to store the transformed problem.
    num_iterations : int
        The number of iterations to run the Ruiz rescaling algorithm. Must be positive.
    p : float
        Which norm to use. Must be 2 or Inf.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
        the original problem is recovered by `unscale_problem`.
    """
    num_constraints, num_variables = problem.constraint_matrix.shape
    cum_constraint_rescaling = jnp.ones(num_constraints)
    cum_variable_rescaling = jnp.ones(num_variables)

    for _ in range(num_iterations):
        constraint_matrix = problem.constraint_matrix
        objective_matrix = problem.objective_matrix

        # Determine variable rescaling
        if p == float("inf"):
            constraint_col_max = get_col_l_inf_norms(constraint_matrix)
            objective_col_max = (
                jnp.zeros(num_variables)
                if problem.is_lp
                else get_col_l_inf_norms(objective_matrix)
            )
            variable_rescaling = jnp.sqrt(
                jnp.maximum(constraint_col_max, objective_col_max)
            )
        elif p == 2:
            objective_col_l2 = (
                jnp.zeros(num_variables)
                if problem.is_lp
                else get_col_l2_norms(objective_matrix)
            )
            variable_rescaling = jnp.sqrt(
                jnp.sqrt(
                    jnp.square(get_col_l2_norms(constraint_matrix))
                    + jnp.square(objective_col_l2)
                )
            )
        else:
            raise ValueError("Norm must be 2 or Inf.")

        # Avoid division by zero by setting zero values to 1.0
        variable_rescaling = jnp.where(
            variable_rescaling == 0.0, 1.0, variable_rescaling
        )

        # Determine constraint rescaling
        if num_constraints == 0:
            constraint_rescaling = jnp.array([])
        else:
            if p == float("inf"):
                constraint_row_max = get_row_l_inf_norms(constraint_matrix)
                constraint_rescaling = jnp.sqrt(constraint_row_max)
            elif p == 2:
                norm_of_rows = get_row_l2_norms(problem.constraint_matrix)

                # Determine the target row norm
                target_row_norm = jnp.sqrt(num_variables / num_constraints)
                if problem.is_lp:
                    # LP case
                    target_row_norm = jnp.sqrt(num_variables / num_constraints)
                else:
                    # QP case
                    target_row_norm = jnp.sqrt(
                        num_variables / (num_constraints + num_variables)
                    )

                constraint_rescaling = jnp.sqrt(norm_of_rows / target_row_norm)
            else:
                raise ValueError("Norm must be 2 or inf.")

            # Avoid division by zero
            constraint_rescaling = jnp.where(
                constraint_rescaling == 0, 1.0, constraint_rescaling
            )

        # Apply scaling to the problem
        scale_problem(problem, constraint_rescaling, variable_rescaling)

        # Accumulate the cumulative scaling factors
        cum_constraint_rescaling *= constraint_rescaling
        cum_variable_rescaling *= variable_rescaling

    return cum_constraint_rescaling, cum_variable_rescaling


def pock_chambolle_rescaling(
    qp: QuadraticProgrammingProblem, alpha: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies the rescaling proposed by Pock and Cambolle (2011),
    "Diagonal preconditioning for first order primal-dual algorithms
    in convex optimization"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126441&tag=1

    Although presented as a form of diagonal preconditioning, it can be
    equivalently implemented by rescaling the problem data.

    Each column of the constraint matrix is divided by
    sqrt(sum_{elements e in the column} |e|^(2 - alpha))
    and each row of the constraint matrix is divided by
    sqrt(sum_{elements e in the row} |e|^alpha)

    Lemma 2 in Pock and Chambolle demonstrates that this rescaling causes the
    operator norm of the rescaled constraint matrix to be less than or equal to
    one, which is a desirable property for PDHG.

    Parameters
    ----------
    qp : QuadraticProgrammingProblem
        The quadratic programming problem.
    alpha : float
        Exponent parameter in the range [0, 2].

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        The constraint and variable rescaling factors.
    """
    assert 0 <= alpha <= 2

    constraint_matrix = qp.constraint_matrix
    objective_matrix = qp.objective_matrix
    if isinstance(qp.constraint_matrix, jnp.ndarray):
        objective_term = (
            0.0
            if qp.objective_matrix is None
            else jnp.sum(jnp.abs(objective_matrix) ** (2 - alpha), axis=0)
        )
        variable_rescaling = jnp.sqrt(
            jnp.sum(jnp.abs(constraint_matrix) ** (2 - alpha), axis=0) + objective_term
        )
        constraint_rescaling = jnp.sqrt(
            jnp.sum(jnp.abs(constraint_matrix) ** (2 - alpha), axis=1)
        )
    elif isinstance(qp.constraint_matrix, BCOO):
        # TODO: improve the code here, instead of using jnp.bincount.
        # Use BCOO.sum or use the sparsify() transform.
        objective_term = (
            0.0
            if qp.objective_matrix is None
            else jnp.bincount(
                objective_matrix.indices[:, 1],
                weights=jnp.abs(objective_matrix.data) ** (2 - alpha),
                length=objective_matrix.shape[1],
            )
        )
        variable_rescaling = jnp.sqrt(
            jnp.bincount(
                constraint_matrix.indices[:, 1],
                weights=jnp.abs(constraint_matrix.data) ** (2 - alpha),
                length=constraint_matrix.shape[1],
            )
            + objective_term
        )
        constraint_rescaling = jnp.sqrt(
            jnp.bincount(
                constraint_matrix.indices[:, 0],
                weights=jnp.abs(constraint_matrix.data) ** (alpha),
                length=constraint_matrix.shape[0],
            )
        )

    variable_rescaling = jnp.where(variable_rescaling == 0, 1.0, variable_rescaling)
    constraint_rescaling = jnp.where(
        constraint_rescaling == 0, 1.0, constraint_rescaling
    )

    scale_problem(qp, constraint_rescaling, variable_rescaling)

    return constraint_rescaling, variable_rescaling
