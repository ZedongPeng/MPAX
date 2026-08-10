from pathlib import Path

import gurobipy as gp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_qp_from_gurobi
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus

pytest_cache_dir = str(Path(__file__).parent.parent / ".pytest_cache")


def test_warm_start_reduces_iterations():
    model = gp.read(pytest_cache_dir + "/gen-ip054.mps")
    lp = create_qp_from_gurobi(model)

    cold = raPDHG(eps_abs=1e-6, eps_rel=1e-6).optimize(lp)
    assert cold.termination_status == TerminationStatus.OPTIMAL

    warm = raPDHG(eps_abs=1e-6, eps_rel=1e-6, warm_start=True).optimize(
        lp,
        initial_primal_solution=cold.primal_solution,
        initial_dual_solution=cold.dual_solution,
    )
    assert warm.termination_status == TerminationStatus.OPTIMAL
    assert warm.iteration_count < cold.iteration_count
