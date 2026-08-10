from pathlib import Path

import gurobipy as gp
import jax
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_qp_from_gurobi
from mpax.rapdhg import raPDHG

pytest_cache_dir = str(Path(__file__).parent.parent / ".pytest_cache")


def _stats_lines(text):
    return [l for l in text.splitlines() if "largest=" in l]


def test_problem_details_logging_has_values(capsys):
    # setup_logger (invoked by raPDHG.optimize) clears root handlers and
    # attaches its own StreamHandler, so caplog never observes the records;
    # capture stderr directly instead.
    model = gp.read(pytest_cache_dir + "/flugpl.mps")
    lp = create_qp_from_gurobi(model)
    solver = raPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=True,
                    iteration_limit=10)
    solver.optimize(lp)
    jax.effects_barrier()
    captured = capsys.readouterr()
    lines = _stats_lines(captured.err)
    # constraint matrix, objective vector, rhs, bound-gap stat lines exist
    assert len(lines) >= 3
    for line in lines:
        # every stats line carries formatted numbers, none of them the
        # impossible "smallest=0.000000" for a nonzero matrix
        assert "largest=" in line and "smallest=" in line
    cm_line = lines[0]  # constraint-matrix line
    assert "smallest=0.000000" not in cm_line
