from pathlib import Path

import gurobipy as gp
import pytest
from jax import config

from mpax.mp_io import create_qp_from_gurobi
from mpax.rapdhg import raPDHG
from mpax.r2hpdhg import r2HPDHG
from mpax.utils import TerminationStatus

config.update("jax_enable_x64", True)
pytest_cache_dir = str(Path(__file__).parent.parent / ".pytest_cache")

EXPECTED_OBJ = 6.765209043e03  # gen-ip054 LP relaxation optimum


def _solve_gen_ip054(solver_cls, polish):
    model = gp.read(pytest_cache_dir + "/gen-ip054.mps")
    lp = create_qp_from_gurobi(model)
    solver = solver_cls(
        eps_abs=1e-4, eps_rel=1e-4, feasibility_polishing=polish
    )
    return solver.optimize(lp)


@pytest.mark.parametrize("solver_cls", [raPDHG, r2HPDHG])
def test_feasibility_polishing_runs_and_tightens_residual(solver_cls):
    plain = _solve_gen_ip054(solver_cls, polish=False)
    polished = _solve_gen_ip054(solver_cls, polish=True)

    assert polished.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(polished.primal_objective, rel=1e-2) == EXPECTED_OBJ
    # Polishing solves the feasibility subproblems to eps_feas_polish=1e-6,
    # so the polished residual must beat the plain 1e-4-tolerance solve.
    assert (
        polished.relative_primal_residual_norm
        < plain.relative_primal_residual_norm
    )
