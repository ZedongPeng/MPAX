import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp
from mpax.r2hpdhg import r2HPDHG
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus

# Fail fast as ITERATION_LIMIT instead of hanging if detection breaks.
_LIMIT = 100_000


def _conflicting_rows_lp():
    # x1 + x2 >= 2 and x1 + x2 <= 1 cannot both hold: primal infeasible.
    # Farkas ray y = (1, -1): A'y = 0, ray objective 2*1 + 1*(-1) = 1 > 0.
    A = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    lc = jnp.array([2.0, -jnp.inf])
    uc = jnp.array([jnp.inf, 1.0])
    return create_lp(
        jnp.array([1.0, 1.0]),
        A,
        lc,
        uc,
        jnp.zeros(2),
        jnp.full(2, jnp.inf),
        use_sparse_matrix=False,
    )


def test_rapdhg_detects_primal_infeasible():
    result = raPDHG(iteration_limit=_LIMIT).optimize(_conflicting_rows_lp())
    assert result.termination_status == TerminationStatus.PRIMAL_INFEASIBLE


def test_rapdhg_detects_primal_infeasible_leq_row():
    # x1 + x2 <= -1 with x >= 0: infeasible purely through the new <=-row
    # path (dual ray y = -1 has ray objective (-1)*(-1) = 1 > 0).
    A = jnp.array([[1.0, 1.0]])
    lp = create_lp(
        jnp.array([1.0, 1.0]),
        A,
        jnp.array([-jnp.inf]),
        jnp.array([-1.0]),
        jnp.zeros(2),
        jnp.full(2, jnp.inf),
        use_sparse_matrix=False,
    )
    result = raPDHG(iteration_limit=_LIMIT).optimize(lp)
    assert result.termination_status == TerminationStatus.PRIMAL_INFEASIBLE


def test_rapdhg_detects_dual_infeasible():
    # min -x1 - x2  s.t. x1 - x2 = 0, x >= 0: unbounded along the primal
    # ray (1, 1) (A @ ray = 0, objective -2 < 0) => dual infeasible.
    A = jnp.array([[1.0, -1.0]])
    lp = create_lp(
        jnp.array([-1.0, -1.0]),
        A,
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.zeros(2),
        jnp.full(2, jnp.inf),
        use_sparse_matrix=False,
    )
    result = raPDHG(iteration_limit=_LIMIT).optimize(lp)
    assert result.termination_status == TerminationStatus.DUAL_INFEASIBLE


def test_r2hpdhg_never_declares_infeasible():
    # r2HPDHG matches cuPDLP-x, which computes no infeasibility
    # certificates: an infeasible LP runs to the iteration limit instead
    # of terminating PRIMAL_INFEASIBLE.
    result = r2HPDHG(iteration_limit=1000).optimize(_conflicting_rows_lp())
    assert result.termination_status == TerminationStatus.ITERATION_LIMIT


def test_r2hpdhg_rejects_infeasibility_detection():
    with pytest.raises(ValueError, match="infeasibility_detection"):
        r2HPDHG(infeasibility_detection=True).optimize(_conflicting_rows_lp())
