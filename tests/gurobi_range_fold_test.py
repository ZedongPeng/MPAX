"""create_qp_from_gurobi folds Gurobi's range-constraint slack columns
(MPS_Rg* from the MPS reader, Rg* from Model.addRange) back into two-sided
row bounds, so the LP handed to the solver is the one the file describes."""

import os

import gurobipy as gp
import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_qp_from_gurobi

MPS_WITH_RANGES = """NAME          rangetest
ROWS
 N  obj
 L  rL
 G  rG
 E  rEp
 E  rEn
 E  plain
COLUMNS
    x         obj       1.0          rL        1.0
    x         rG        1.0          rEp       2.0
    x         rEn       1.0          plain     1.0
    y         obj       2.0          rL        1.0
    y         rG        -1.0         rEp       1.0
    y         rEn       -1.0         plain     1.0
RHS
    RHS       rL        10.0         rG        1.0
    RHS       rEp       4.0          rEn       3.0
    RHS       plain     2.0
RANGES
    RNG       rL        4.0          rG        6.0
    RNG       rEp       2.0          rEn       -1.5
BOUNDS
 UP BND       x         8.0
 UP BND       y         8.0
ENDATA
"""


def _dense(qp):
    A = np.zeros((qp.num_constraints, qp.num_variables))
    A[np.array(qp.constraint_matrix.indices[:, 0]), np.array(qp.constraint_matrix.indices[:, 1])] = np.array(
        qp.constraint_matrix.data
    )
    return A


def test_mps_ranges_fold_to_two_sided_rows(tmp_path):
    path = tmp_path / "rangetest.mps"
    path.write_text(MPS_WITH_RANGES)
    model = gp.read(str(path))
    assert model.NumVars == 6  # x, y + four MPS_Rg* slacks
    qp = create_qp_from_gurobi(model)
    assert qp.num_variables == 2 and qp.num_constraints == 5
    rows = {c.ConstrName: i for i, c in enumerate(model.getConstrs())}
    lc = np.array(qp.constraint_lower_bound)
    uc = np.array(qp.constraint_upper_bound)
    # MPS RANGES semantics: L: [rhs-|R|, rhs]; G: [rhs, rhs+|R|];
    # E, R>0: [rhs, rhs+R]; E, R<0: [rhs+R, rhs].
    assert (lc[rows["rL"]], uc[rows["rL"]]) == (6.0, 10.0)
    assert (lc[rows["rG"]], uc[rows["rG"]]) == (1.0, 7.0)
    assert (lc[rows["rEp"]], uc[rows["rEp"]]) == (4.0, 6.0)
    assert (lc[rows["rEn"]], uc[rows["rEn"]]) == (1.5, 3.0)
    assert (lc[rows["plain"]], uc[rows["plain"]]) == (2.0, 2.0)
    A = _dense(qp)
    np.testing.assert_array_equal(A[rows["rG"]], [1.0, -1.0])
    np.testing.assert_array_equal(np.array(qp.objective_vector), [1.0, 2.0])
    # Opt-out keeps Gurobi's layout.
    raw = create_qp_from_gurobi(model, fold_range_slacks=False)
    assert raw.num_variables == 6
    assert np.all(np.array(raw.constraint_lower_bound) == np.array(raw.constraint_upper_bound))


def test_addrange_slack_is_folded_and_optimum_preserved():
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVar(lb=0, ub=10, obj=1.0, name="x")
    y = m.addVar(lb=0, ub=10, obj=-1.0, name="y")
    m.addRange(x + y, 2.0, 5.0, name="r")  # Rg r slack
    m.addConstr(x - y >= -3.0, name="c")
    m.update()
    assert m.NumVars == 3
    qp = create_qp_from_gurobi(m)
    assert qp.num_variables == 2
    lc = np.array(qp.constraint_lower_bound)
    uc = np.array(qp.constraint_upper_bound)
    r = [c.ConstrName for c in m.getConstrs()].index("r")
    assert (lc[r], uc[r]) == (2.0, 5.0)
    # Same optimum when the folded problem is solved by Gurobi.
    m.optimize()
    m2 = gp.Model()
    m2.Params.OutputFlag = 0
    v = m2.addMVar(2, lb=np.array(qp.variable_lower_bound), ub=np.array(qp.variable_upper_bound), obj=np.array(qp.objective_vector))
    A = _dense(qp)
    m2.addConstr(A @ v <= uc)
    m2.addConstr(A @ v >= lc)
    m2.optimize()
    assert m2.ObjVal == pytest.approx(m.ObjVal)


def test_user_variable_named_rg_is_not_folded():
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVar(lb=0, ub=1, obj=1.0, name="Rgx")  # user name; obj != 0 -> not a slack
    y = m.addVar(lb=0, ub=1, obj=1.0, name="y")
    m.addConstr(x + y == 1.0)
    m.update()
    qp = create_qp_from_gurobi(m)
    assert qp.num_variables == 2
