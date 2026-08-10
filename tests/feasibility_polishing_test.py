from pathlib import Path

import gurobipy as gp
import jax.numpy as jnp
import pytest
from jax import config

from mpax.mp_io import create_qp_from_gurobi
from mpax.rapdhg import raPDHG
from mpax.r2hpdhg import r2HPDHG
from mpax.utils import TerminationStatus

config.update("jax_enable_x64", True)
pytest_cache_dir = str(Path(__file__).parent.parent / ".pytest_cache")

EXPECTED_OBJ = 6.765209043e03  # gen-ip054 LP relaxation optimum
EXPECTED_QP_OBJ = 9.9336215e02  # AUG3DCQP known optimum
EPS = 1e-4


def _solve_gen_ip054(solver_cls, polish):
    model = gp.read(pytest_cache_dir + "/gen-ip054.mps")
    lp = create_qp_from_gurobi(model)
    solver = solver_cls(eps_abs=EPS, eps_rel=EPS, feasibility_polishing=polish)
    return solver.optimize(lp)


def _solve_aug3dcqp(polish):
    model = gp.read(pytest_cache_dir + "/maros_meszaros_dataset1/AUG3DCQP.QPS")
    qp = create_qp_from_gurobi(model)
    # r2HPDHG rejects QPs (objective matrix guard), so raPDHG only.
    solver = raPDHG(eps_abs=EPS, eps_rel=EPS, feasibility_polishing=polish)
    return solver.optimize(qp)


def _polish_was_accepted(plain, polished):
    """Polishing is gated: on rejection optimize() returns the plain iterate."""
    return not bool(jnp.allclose(polished.primal_solution, plain.primal_solution))


@pytest.mark.parametrize("solver_cls", [raPDHG, r2HPDHG])
def test_feasibility_polishing_runs_and_tightens_residual(solver_cls):
    plain = _solve_gen_ip054(solver_cls, polish=False)
    polished = _solve_gen_ip054(solver_cls, polish=True)

    assert polished.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(polished.primal_objective, rel=1e-2) == EXPECTED_OBJ
    if _polish_was_accepted(plain, polished):
        # Polishing solves the feasibility subproblems to eps_feas_polish=1e-6,
        # so an accepted polished residual must beat the plain 1e-4 solve.
        assert (
            polished.relative_primal_residual_norm
            < plain.relative_primal_residual_norm
        )
    # Whether the polished pair is accepted or rejected, the returned point
    # must honour the requested tolerances it reports OPTIMAL for.
    assert polished.relative_optimality_gap < EPS * 1.05


def test_feasibility_polishing_qp():
    plain = _solve_aug3dcqp(polish=False)
    polished = _solve_aug3dcqp(polish=True)

    assert polished.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(polished.primal_objective, rel=1e-2) == EXPECTED_QP_OBJ
    if _polish_was_accepted(plain, polished):
        assert (
            polished.relative_primal_residual_norm
            < plain.relative_primal_residual_norm
        )
    assert polished.relative_optimality_gap < EPS * 1.05
