import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp, create_qp
from mpax.r2hpdhg import r2HPDHG
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus


def _tiny_lp():
    # min x1 + 2 x2  s.t. x1 + x2 >= 1, 0 <= x <= 1  (optimum 1.0)
    G = jnp.ones((1, 2))
    return create_lp(
        jnp.array([1.0, 2.0]),
        G,
        jnp.ones(1),
        jnp.array([jnp.inf]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )


def _tiny_qp():
    # min 1/2 x'x - x1 - x2  s.t. x1 + x2 <= 1  (optimum -0.75 at (0.5, 0.5))
    G = jnp.ones((1, 2))
    return create_qp(
        jnp.eye(2),
        jnp.array([-1.0, -1.0]),
        G,
        jnp.array([-jnp.inf]),
        jnp.ones(1),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )


def test_smoothing_stays_default():
    assert raPDHG().primal_weight_update == "smoothing"
    assert r2HPDHG().primal_weight_update == "smoothing"


def test_invalid_primal_weight_update_raises():
    solver = raPDHG(primal_weight_update="nonsense")
    with pytest.raises(ValueError, match="primal_weight_update"):
        solver.optimize(_tiny_lp())


def test_pid_mode_solves_lp_rapdhg():
    solver = raPDHG(primal_weight_update="pid", eps_abs=1e-6, eps_rel=1e-6)
    result = solver.optimize(_tiny_lp())
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), rel=1e-2) == 1.0


def test_pid_mode_solves_lp_r2hpdhg():
    solver = r2HPDHG(primal_weight_update="pid", eps_abs=1e-6, eps_rel=1e-6)
    result = solver.optimize(_tiny_lp())
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), rel=1e-2) == 1.0


def test_pid_mode_solves_qp():
    solver = raPDHG(primal_weight_update="pid", eps_abs=1e-6, eps_rel=1e-6)
    result = solver.optimize(_tiny_qp())
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), rel=1e-2) == -0.75
