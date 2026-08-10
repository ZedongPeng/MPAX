import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from jax.experimental.sparse import BCOO

from mpax.mp_io import create_lp, create_qp
from mpax.r2hpdhg import r2HPDHG
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus


def test_box_qp_no_constraints_sparse():
    # Issue #27 reproducer: box-constrained QP, zero constraint rows.
    # min 1/2 x'x - x1 - x2 on [0, 1]^2  =>  x = (1, 1), objective -1.
    Q = BCOO.fromdense(jnp.eye(2))
    A = BCOO.fromdense(jnp.zeros((0, 2)))
    qp = create_qp(
        Q,
        jnp.array([-1.0, -1.0]),
        A,
        jnp.zeros(0),
        jnp.zeros(0),
        jnp.zeros(2),
        jnp.ones(2),
    )
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(qp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert jnp.allclose(result.primal_solution, jnp.ones(2), atol=1e-4)
    assert pytest.approx(float(result.primal_objective), abs=1e-4) == -1.0


def test_box_lp_no_constraints_dense():
    # min x1 - 2 x2 on [0, 1]^2  =>  x = (0, 1), objective -2.
    lp = create_lp(
        jnp.array([1.0, -2.0]),
        jnp.zeros((0, 2)),
        jnp.zeros(0),
        jnp.zeros(0),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), abs=1e-4) == -2.0


def test_box_lp_no_constraints_r2hpdhg():
    lp = create_lp(
        jnp.array([1.0, -2.0]),
        jnp.zeros((0, 2)),
        jnp.zeros(0),
        jnp.zeros(0),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )
    result = r2HPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), abs=1e-4) == -2.0


def test_create_lp_rejects_wrong_objective_length():
    A = jnp.ones((1, 2))
    with pytest.raises(ValueError, match="columns"):
        create_lp(
            jnp.ones(3),
            A,
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(2),
            jnp.ones(2),
            use_sparse_matrix=False,
        )


def test_create_lp_rejects_wrong_constraint_bound_lengths():
    A = jnp.ones((1, 2))
    with pytest.raises(ValueError, match="rows"):
        create_lp(
            jnp.ones(2),
            A,
            jnp.zeros(2),
            jnp.ones(2),
            jnp.zeros(2),
            jnp.ones(2),
            use_sparse_matrix=False,
        )


def test_create_lp_rejects_wrong_variable_bound_lengths():
    A = jnp.ones((1, 2))
    with pytest.raises(ValueError, match="variable bounds"):
        create_lp(
            jnp.ones(2),
            A,
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(3),
            jnp.ones(3),
            use_sparse_matrix=False,
        )


def test_create_qp_rejects_wrong_q_shape():
    A = jnp.ones((1, 2))
    with pytest.raises(ValueError, match="objective matrix"):
        create_qp(
            jnp.eye(3),
            jnp.ones(2),
            A,
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(2),
            jnp.ones(2),
            use_sparse_matrix=False,
        )
