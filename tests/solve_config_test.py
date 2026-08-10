import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp, create_qp
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus


def _tiny_qp():
    # min 1/2 x'Ix - x1 - x2  s.t. x1 + x2 >= 1, 0 <= x <= 1
    Q = jnp.eye(2)
    c = jnp.array([-1.0, -1.0])
    G = jnp.ones((1, 2))
    lc = jnp.ones(1)
    uc = jnp.array([jnp.inf])
    return create_qp(Q, c, G, lc, uc, jnp.zeros(2), jnp.ones(2),
                     use_sparse_matrix=False)


def _tiny_lp():
    # min x1 + 2 x2  s.t. x1 + x2 >= 1, 0 <= x <= 1  (optimum 1.0 at (1, 0))
    c = jnp.array([1.0, 2.0])
    G = jnp.ones((1, 2))
    lc = jnp.ones(1)
    uc = jnp.array([jnp.inf])
    return create_lp(c, G, lc, uc, jnp.zeros(2), jnp.ones(2),
                     use_sparse_matrix=False)


def test_qp_solve_does_not_mutate_config():
    solver = raPDHG(eps_abs=1e-4, eps_rel=1e-4)
    solver.optimize(_tiny_qp())
    # check_config used to flip these permanently on QP input.
    assert solver.adaptive_step_size is True
    assert solver.infeasibility_detection is True
    assert solver.primal_weight_update_smoothing == 0.5


def test_lp_after_qp_still_solves_with_lp_config():
    solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
    solver.optimize(_tiny_qp())
    result = solver.optimize(_tiny_lp())
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), rel=1e-2) == 1.0
