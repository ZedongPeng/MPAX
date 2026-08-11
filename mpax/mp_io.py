import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR, bcoo_concatenate
from scipy.sparse import sparray, spmatrix

from mpax.utils import QuadraticProgrammingProblem


def transform_to_bcoo(input_matrix):
    """Transform the input matrix to the BCOO format.

    Parameters
    ----------
    input_matrix :
        The input matrix to be transformed.

    Returns
    -------
    BCOO
        The matrix in BCOO format.

    Raises
    ------
    ValueError
        The input matrix format is not supported.
    """
    if isinstance(input_matrix, BCSR):
        bcoo_matrix = input_matrix.to_bcoo()
    elif isinstance(input_matrix, (sparray, spmatrix)):
        bcoo_matrix = BCOO.from_scipy_sparse(input_matrix)
    elif isinstance(input_matrix, BCOO):
        bcoo_matrix = input_matrix
    else:
        raise ValueError(
            "Unsupported matrix format. "
            "The sparse constraint matrix must be one of the following types: "
            "scipy.sparse.sparray, scipy.sparse.spmatrix, BCOO, or BCSR."
        )
    return bcoo_matrix


def transform_to_jnp_array(input_matrix):
    """Transform the input matrix to the BCOO format.

    Parameters
    ----------
    input_matrix :
        The input matrix to be transformed.

    Returns
    -------
    jnp.ndarray
        The matrix in jnp.ndarray format.

    Raises
    ------
    ValueError
        The input matrix format is not supported.
    """
    if isinstance(input_matrix, (BCOO, BCSR)):
        output_matrix = input_matrix.todense()
    elif isinstance(input_matrix, np.ndarray):
        output_matrix = jnp.array(input_matrix)
    elif isinstance(input_matrix, (sparray, spmatrix)):
        output_matrix = jnp.array(input_matrix.toarray())
    elif isinstance(input_matrix, jnp.ndarray):
        output_matrix = input_matrix
    else:
        raise ValueError(
            "Unsupported matrix format. "
            "The constraint matrix must be one of the following types: "
            "jnp.ndarray, BCOO, or BCSR."
        )
    return output_matrix


def _validate_dimensions(c, A, lc, uc, l, u, Q=None):
    """Shape-consistency checks with readable errors (issue #27).

    Only static shapes are inspected, so this is safe for traced inputs
    under jit/vmap. Data-dependent validity (lc <= uc, no NaNs) is NOT
    checked here — it cannot raise a Python error at trace time.
    """
    if len(A.shape) != 2:
        raise ValueError(
            f"constraint matrix must be 2-dimensional, got shape {A.shape}"
        )
    n = c.shape[0]
    if A.shape[1] != n:
        raise ValueError(
            f"constraint matrix has {A.shape[1]} columns but the "
            f"objective vector has {n} entries"
        )
    m = A.shape[0]
    if lc.shape[0] != m or uc.shape[0] != m:
        raise ValueError(
            f"constraint bounds have lengths {lc.shape[0]} (lower) and "
            f"{uc.shape[0]} (upper) but the constraint matrix has {m} rows"
        )
    if l.shape[0] != n or u.shape[0] != n:
        raise ValueError(
            f"variable bounds have lengths {l.shape[0]} (lower) and "
            f"{u.shape[0]} (upper) but the objective vector has {n} entries"
        )
    if Q is not None and Q.shape != (n, n):
        raise ValueError(f"objective matrix has shape {Q.shape}; expected ({n}, {n})")


def create_lp(c, A, lc, uc, l, u, use_sparse_matrix=True):
    """Create a boxed linear program with two-sided constraints.
            min  c'x
            s.t. lc <= Ax <= uc
                 l <= x <= u

    Row classes are encoded by the bounds: equality rows have lc == uc,
    `>=` rows have uc = +inf, `<=` rows have lc = -inf.

    Parameters
    ----------
    c : jnp.ndarray
        The objective vector.
    A : jnp.ndarray, BCOO or BCSR
        The constraint matrix.
    lc : jnp.ndarray
        The lower bounds of the constraints (-inf for unbounded below).
    uc : jnp.ndarray
        The upper bounds of the constraints (+inf for unbounded above).
    l : jnp.ndarray
        The lower bound of the variables.
    u : jnp.ndarray
        The upper bound of the variables.
    use_sparse_matrix : bool
        Whether to use sparse matrix format, by default True.

    Returns
    -------
    QuadraticProgrammingProblem
        The boxed linear program.
    """
    _validate_dimensions(c, A, lc, uc, l, u)
    if use_sparse_matrix:
        constraint_matrix = transform_to_bcoo(A)
    else:
        constraint_matrix = transform_to_jnp_array(A)

    return QuadraticProgrammingProblem(
        num_variables=c.shape[0],
        num_constraints=constraint_matrix.shape[0],
        variable_lower_bound=jnp.array(l),
        variable_upper_bound=jnp.array(u),
        isfinite_variable_lower_bound=jnp.isfinite(l),
        isfinite_variable_upper_bound=jnp.isfinite(u),
        objective_matrix=None,
        objective_vector=jnp.array(c),
        objective_constant=0.0,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix.T,
        constraint_lower_bound=jnp.array(lc),
        constraint_upper_bound=jnp.array(uc),
        is_lp=True,
    )


def create_qp(Q, c, A, lc, uc, l, u, use_sparse_matrix=True):
    """Create a boxed quadratic program with two-sided constraints.
            min  1/2 x'Qx + c'x
            s.t. lc <= Ax <= uc
                 l <= x <= u

    Parameters are as in create_lp, plus Q (jnp.ndarray, BCOO or BCSR),
    the positive semidefinite quadratic objective matrix. A Q with no
    nonzeros produces an LP (objective_matrix None, is_lp True).
    """
    _validate_dimensions(c, A, lc, uc, l, u, Q=Q)
    if use_sparse_matrix:
        constraint_matrix = transform_to_bcoo(A)
        objective_matrix = transform_to_bcoo(Q)
        is_lp = bool(objective_matrix.nse == 0)
    else:
        constraint_matrix = transform_to_jnp_array(A)
        objective_matrix = transform_to_jnp_array(Q)
        is_lp = bool(jnp.all(objective_matrix == 0))
    if is_lp:
        objective_matrix = None

    return QuadraticProgrammingProblem(
        num_variables=c.shape[0],
        num_constraints=constraint_matrix.shape[0],
        variable_lower_bound=jnp.array(l),
        variable_upper_bound=jnp.array(u),
        isfinite_variable_lower_bound=jnp.isfinite(l),
        isfinite_variable_upper_bound=jnp.isfinite(u),
        objective_matrix=objective_matrix,
        objective_vector=jnp.array(c),
        objective_constant=0.0,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix.T,
        constraint_lower_bound=jnp.array(lc),
        constraint_upper_bound=jnp.array(uc),
        is_lp=is_lp,
    )


def _standard_form_to_two_sided(A, b, G, h, use_sparse_matrix):
    """Concatenate old-form (Ax = b, Gx >= h) blocks into two-sided data."""
    if use_sparse_matrix:
        constraint_matrix = bcoo_concatenate(
            [transform_to_bcoo(A), transform_to_bcoo(G)], dimension=0
        )
    else:
        constraint_matrix = jnp.concatenate(
            [transform_to_jnp_array(A), transform_to_jnp_array(G)], axis=0
        )
    lc = jnp.concatenate([jnp.array(b), jnp.array(h)])
    uc = jnp.concatenate([jnp.array(b), jnp.full(len(h), jnp.inf)])
    return constraint_matrix, lc, uc


def create_lp_standard_form(c, A, b, G, h, l, u, use_sparse_matrix=True):
    """Deprecated pre-v0.3 constructor: min c'x s.t. Ax = b, Gx >= h, l <= x <= u."""
    warnings.warn(
        "create_lp_standard_form is deprecated and will be removed in v0.4; "
        "use create_lp(c, A, lc, uc, l, u) with two-sided constraint bounds.",
        DeprecationWarning,
        stacklevel=2,
    )
    constraint_matrix, lc, uc = _standard_form_to_two_sided(
        A, b, G, h, use_sparse_matrix
    )
    return create_lp(c, constraint_matrix, lc, uc, l, u, use_sparse_matrix)


def create_qp_standard_form(Q, c, A, b, G, h, l, u, use_sparse_matrix=True):
    """Deprecated pre-v0.3 constructor: min 1/2 x'Qx + c'x s.t. Ax = b, Gx >= h."""
    warnings.warn(
        "create_qp_standard_form is deprecated and will be removed in v0.4; "
        "use create_qp(Q, c, A, lc, uc, l, u) with two-sided constraint bounds.",
        DeprecationWarning,
        stacklevel=2,
    )
    constraint_matrix, lc, uc = _standard_form_to_two_sided(
        A, b, G, h, use_sparse_matrix
    )
    return create_qp(Q, c, constraint_matrix, lc, uc, l, u, use_sparse_matrix)


def create_qp_from_gurobi(
    model, use_sparse_matrix=True, sharding=None
) -> QuadraticProgrammingProblem:
    """Build a two-sided QuadraticProgrammingProblem from a gurobipy model.

    Parameters
    ----------
    model : gurobipy.Model
        The gurobi model to transform.
    use_sparse_matrix : bool
        Whether to use sparse matrix format, by default True.
    sharding : jax.sharding.Sharding
        The sharding to use, by default None.

    Returns
    -------
    QuadraticProgrammingProblem
        The problem in two-sided form (constraint senses map directly to
        bounds; `<=` rows are NOT negated, so their duals are nonpositive).
    """
    constraint_sense = np.array(model.getAttr("Sense", model.getConstrs()))
    constraint_rhs = np.array(model.getAttr("RHS", model.getConstrs()))
    constraint_lower_bound = jnp.array(
        np.where(constraint_sense == "<", -np.inf, constraint_rhs)
    )
    constraint_upper_bound = jnp.array(
        np.where(constraint_sense == ">", np.inf, constraint_rhs)
    )

    if use_sparse_matrix:
        constraint_matrix = BCOO.from_scipy_sparse(model.getA())
    else:
        constraint_matrix = jnp.array(model.getA().toarray())

    is_lp = model.getQ().nnz == 0
    if is_lp:
        objective_matrix = None
    elif use_sparse_matrix:
        objective_matrix = 2 * BCOO.from_scipy_sparse(
            (model.getQ() + model.getQ().T) / 2
        )
    else:
        objective_matrix = 2 * jnp.array(
            ((model.getQ() + model.getQ().T) / 2).toarray()
        )
    if sharding is not None:
        constraint_matrix = jax.device_put(constraint_matrix, sharding)

    var_lb = jnp.array(model.getAttr("LB", model.getVars()))
    var_ub = jnp.array(model.getAttr("UB", model.getVars()))
    objective_vector = jnp.array(model.getAttr("Obj", model.getVars()))
    objective_constant = model.ObjCon

    return QuadraticProgrammingProblem(
        num_variables=len(var_lb),
        num_constraints=constraint_matrix.shape[0],
        variable_lower_bound=var_lb,
        variable_upper_bound=var_ub,
        isfinite_variable_lower_bound=jnp.isfinite(var_lb),
        isfinite_variable_upper_bound=jnp.isfinite(var_ub),
        objective_matrix=objective_matrix,
        objective_vector=objective_vector,
        objective_constant=objective_constant,
        constraint_matrix=constraint_matrix,
        constraint_matrix_t=constraint_matrix.T,
        constraint_lower_bound=constraint_lower_bound,
        constraint_upper_bound=constraint_upper_bound,
        is_lp=is_lp,
    )
