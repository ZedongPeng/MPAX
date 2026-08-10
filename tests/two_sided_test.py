import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp, create_lp_standard_form
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus

# min c'x  s.t.  x1+x2+x3 = 2 ; x1-x2 >= 0 ; x2+x3 >= 1 ; 0 <= x <= 2
_C = jnp.array([1.0, 2.0, 3.0])
_A_EQ = jnp.array([[1.0, 1.0, 1.0]])
_B = jnp.array([2.0])
_G = jnp.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
_H = jnp.array([0.0, 1.0])
_L = jnp.zeros(3)
_U = 2.0 * jnp.ones(3)


def test_standard_form_wrapper_matches_two_sided():
    with pytest.warns(DeprecationWarning, match="create_lp_standard_form"):
        lp_old = create_lp_standard_form(
            _C, _A_EQ, _B, _G, _H, _L, _U, use_sparse_matrix=False
        )
    A = jnp.concatenate([_A_EQ, _G], axis=0)
    lc = jnp.array([2.0, 0.0, 1.0])
    uc = jnp.array([2.0, jnp.inf, jnp.inf])
    lp_new = create_lp(_C, A, lc, uc, _L, _U, use_sparse_matrix=False)

    assert jnp.array_equal(lp_old.constraint_lower_bound, lp_new.constraint_lower_bound)
    assert jnp.array_equal(lp_old.constraint_upper_bound, lp_new.constraint_upper_bound)
    assert jnp.array_equal(lp_old.constraint_matrix, lp_new.constraint_matrix)

    r_old = raPDHG(eps_abs=1e-6, eps_rel=1e-6).optimize(lp_old)
    r_new = raPDHG(eps_abs=1e-6, eps_rel=1e-6).optimize(lp_new)
    # The wrapper must produce bit-identical problem data, hence an
    # identical trajectory. (The pre- vs post-migration behavior gate is
    # the benchmark compare against the stage-1 baseline, not this test.)
    assert r_old.iteration_count == r_new.iteration_count
    assert jnp.allclose(r_old.primal_solution, r_new.primal_solution)
    assert r_old.termination_status == TerminationStatus.OPTIMAL


def test_genuinely_two_sided_row():
    # min x1 + 2 x2  s.t.  1 <= x1 + x2 <= 2,  0 <= x <= 3   => x = (1, 0)
    A = jnp.array([[1.0, 1.0]])
    lp = create_lp(
        jnp.array([1.0, 2.0]),
        A,
        jnp.array([1.0]),
        jnp.array([2.0]),
        jnp.zeros(2),
        3.0 * jnp.ones(2),
        use_sparse_matrix=False,
    )
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), abs=1e-4) == 1.0


def test_leq_row_negative_dual():
    # max x1 + x2 (= min -x1 - x2)  s.t. x1 + x2 <= 1,  0 <= x <= 1
    A = jnp.array([[1.0, 1.0]])
    lp = create_lp(
        jnp.array([-1.0, -1.0]),
        A,
        jnp.array([-jnp.inf]),
        jnp.array([1.0]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), abs=1e-4) == -1.0
    # The <= row is binding, so its dual is strictly negative in the
    # two-sided convention (it was positive pre-migration because rows
    # were negated) — this pins the documented breaking change.
    assert float(result.dual_solution[0]) < -0.5


def test_free_row_gets_zero_dual():
    # A free row (both bounds infinite) must not affect the solution and
    # must carry a zero dual.
    A = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    lc = jnp.array([1.0, -jnp.inf])
    uc = jnp.array([jnp.inf, jnp.inf])
    lp = create_lp(
        jnp.array([1.0, 2.0]),
        A,
        lc,
        uc,
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), abs=1e-4) == 1.0
    assert abs(float(result.dual_solution[1])) < 1e-6
