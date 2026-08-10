import timeit
from pathlib import Path

import gurobipy as gp
import jax
import jax.numpy as jnp
import pytest
from jax import config, jit, vmap
from jax.experimental.sparse import BCOO
from jax.sharding import PartitionSpec as P
from pulp import LpProblem
from pulp2mat import convert_all

from mpax.mp_io import create_lp, create_qp_from_gurobi
from mpax.r2hpdhg import r2HPDHG

config.update("jax_enable_x64", True)
pytest_cache_dir = str(Path(__file__).parent.parent / ".pytest_cache")


def test_r2hpdhg():
    """Test the r2HPDHG solver on a sample LP problem."""
    gurobi_model = gp.read(pytest_cache_dir + "/gen-ip054.mps")
    lp = create_qp_from_gurobi(gurobi_model)
    solver = r2HPDHG(eps_abs=1e-6, eps_rel=1e-6)
    result = solver.optimize(lp)
    objective_value = (
        jnp.dot(lp.objective_vector, result.primal_solution) + lp.objective_constant
    )

    # Expected optimal objective value
    expected_obj = 6.765209043e03
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj


def test_r2hpdhg_with_jit():
    """Test the r2HPDHG solver on a sample LP problem."""
    gurobi_model = gp.read(pytest_cache_dir + "/gen-ip054.mps")
    lp = create_qp_from_gurobi(gurobi_model)
    solver = r2HPDHG(eps_abs=1e-6, eps_rel=1e-6)
    jit_optimize = jit(solver.optimize)
    result = jit_optimize(lp)
    objective_value = (
        jnp.dot(lp.objective_vector, result.primal_solution) + lp.objective_constant
    )

    # Expected optimal objective value
    expected_obj = 6.765209043e03
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj


def test_r2hpdhg_with_vmap():
    """Test the r2HPDHG solver on a batch of LP problems."""
    var, prob = LpProblem.fromMPS(pytest_cache_dir + "/gen-ip054.mps")
    c, integrality, constraints, bounds = convert_all(prob)
    c = jnp.array(c)
    constraint_lb = jnp.array(constraints.lb)
    constraint_ub = jnp.array(constraints.ub)

    A = BCOO.fromdense(jnp.array(constraints.A))
    var_lb = jnp.array(bounds.lb)
    var_ub = jnp.array(bounds.ub)

    solver = r2HPDHG(eps_abs=1e-6, eps_rel=1e-6)

    def solve_single(c):
        boxed_lp = create_lp(c, A, constraint_lb, constraint_ub, var_lb, var_ub)
        result = solver.optimize(boxed_lp)
        return (
            jnp.dot(boxed_lp.objective_vector, result.primal_solution)
            + boxed_lp.objective_constant
        )

    jit_solve_single = jit(solve_single)

    start_time = timeit.default_timer()
    objective_value = jit_solve_single(c)
    print("1st single run time = ", timeit.default_timer() - start_time)

    # Expected optimal objective value
    expected_obj = 6.765209043e03
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj

    start_time = timeit.default_timer()
    objective_value = jit_solve_single(c)
    print("2nd single run time = ", timeit.default_timer() - start_time)
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj

    batch_size = 100
    batch_c = jnp.tile(c, (batch_size, 1))
    batch_optimize = vmap(jit(solve_single))

    start_time = timeit.default_timer()
    batch_result = batch_optimize(batch_c)
    print("1st batch run time = ", timeit.default_timer() - start_time)

    for obj in batch_result:
        assert pytest.approx(obj, rel=1e-2) == expected_obj

    start_time = timeit.default_timer()
    batch_result = batch_optimize(batch_c)
    print("2nd batch run time = ", timeit.default_timer() - start_time)

    for obj in batch_result:
        assert pytest.approx(obj, rel=1e-2) == expected_obj


@pytest.mark.skipif(
    jax.device_count() <= 1,
    reason="Only one device available, skipping test_rapdhg_with_sharding",
)
def test_r2hpdhg_with_sharding():
    """Test the r2HPDHG solver on a batch of LP problems."""
    mesh = jax.make_mesh((2,), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, P("x"))

    gurobi_model = gp.read(pytest_cache_dir + "/flugpl.mps")
    lp_sharded = create_qp_from_gurobi(gurobi_model, sharding=sharding)
    jax.debug.visualize_array_sharding(lp_sharded.constraint_matrix)

    solver = r2HPDHG(eps_abs=1e-8, eps_rel=1e-8)
    jit_optimize = jax.jit(solver.optimize)
    result = jit_optimize(lp_sharded)

    objective_value = (
        jnp.dot(lp_sharded.objective_vector, result.primal_solution)
        + lp_sharded.objective_constant
    )

    # Expected optimal objective value
    expected_obj = 1.167185726e06
    assert pytest.approx(objective_value, rel=1e-2) == expected_obj
