import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax import solve
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
    # min 1/2 x'x - x1 - x2  s.t. x1 + x2 <= 1, 0 <= x <= 1
    # (optimum -0.75 at x = (0.5, 0.5); the <= row is active, so this also
    # exercises the two-sided <= path through solve())
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


def test_solve_dispatches_lp_to_r2hpdhg(monkeypatch):
    calls = []
    original = r2HPDHG.optimize

    def spy(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(r2HPDHG, "optimize", spy)
    result = solve(_tiny_lp(), eps_abs=1e-6, eps_rel=1e-6)
    assert calls == ["r2HPDHG"]
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), rel=1e-2) == 1.0


def test_solve_dispatches_qp_to_rapdhg(monkeypatch):
    calls = []
    original = raPDHG.optimize

    def spy(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    # r2HPDHG inherits from raPDHG, so patching raPDHG.optimize also sees
    # r2HPDHG calls — type(self).__name__ disambiguates.
    monkeypatch.setattr(raPDHG, "optimize", spy)
    result = solve(_tiny_qp(), eps_abs=1e-6, eps_rel=1e-6)
    assert calls == ["raPDHG"]
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(result.primal_objective), rel=1e-2) == -0.75


def test_solve_rejects_algorithm_specific_options():
    with pytest.raises(TypeError, match="restart_scheme"):
        solve(_tiny_lp(), restart_scheme=1)


def test_solve_warm_start_requires_initial_solutions():
    with pytest.raises(ValueError, match="warm_start"):
        solve(_tiny_lp(), warm_start=True)
