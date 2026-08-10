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
from mpax.rapdhg import raPDHG

config.update("jax_enable_x64", True)
pytest_cache_dir = str(Path(__file__).parent.parent / ".pytest_cache")

lp_model_objs = {"gen-ip054.mps": 6.765209043e03, "flugpl.mps": 1.167185726e06}
qp_model_objs = {
    "maros_meszaros_dataset1/AUG2DC.QPS": 1.8183681e06,
    "maros_meszaros_dataset1/AUG3DCQP.QPS": 9.9336215e02,
}


def test_rapdhg_lp():
    """Test the raPDHG solver on a sample LP problem."""
    for model_filename, expected_obj in lp_model_objs.items():
        gurobi_model = gp.read(pytest_cache_dir + "/" + model_filename)
        qp = create_qp_from_gurobi(gurobi_model)
        solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
        result = solver.optimize(qp)

        assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj


def test_rapdhg_lp_constant_stepsize():
    """Test the raPDHG solver on a sample LP problem."""
    for model_filename, expected_obj in lp_model_objs.items():
        gurobi_model = gp.read(pytest_cache_dir + "/" + model_filename)
        qp = create_qp_from_gurobi(gurobi_model)
        solver = raPDHG(adaptive_step_size=False, eps_abs=1e-6, eps_rel=1e-6)
        result = solver.optimize(qp)

        assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj


def test_rapdhg_qp():
    """Test the raPDHG solver on a sample LP problem."""
    for model_filename, expected_obj in qp_model_objs.items():
        gurobi_model = gp.read(pytest_cache_dir + "/" + model_filename)
        qp = create_qp_from_gurobi(gurobi_model)
        solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
        result = solver.optimize(qp)

        assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj


def test_rapdhg_lp_with_jit():
    """Test the raPDHG solver on a sample LP problem."""
    for model_filename, expected_obj in lp_model_objs.items():
        gurobi_model = gp.read(pytest_cache_dir + "/" + model_filename)
        qp = create_qp_from_gurobi(gurobi_model)
        solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
        jit_optimize = jit(solver.optimize)
        result = jit_optimize(qp)

        assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj


def test_rapdhg_lp_with_jit_dense_matrix():
    """Test the raPDHG solver on a sample LP problem."""
    for model_filename, expected_obj in lp_model_objs.items():
        gurobi_model = gp.read(pytest_cache_dir + "/" + model_filename)
        qp = create_qp_from_gurobi(gurobi_model, use_sparse_matrix=False)
        solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6, verbose=True)
        result = solver.optimize(qp)

        assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj


def test_rapdhg_qp_with_jit():
    """Test the raPDHG solver on a sample LP problem."""
    for model_filename, expected_obj in qp_model_objs.items():
        gurobi_model = gp.read(pytest_cache_dir + "/" + model_filename)
        qp = create_qp_from_gurobi(gurobi_model)
        solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)
        jit_optimize = jit(solver.optimize)
        result = jit_optimize(qp)

        assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj


def test_rapdhg_lp_with_vmap():
    """Test the raPDHG solver on a batch of LP problems."""
    var, prob = LpProblem.fromMPS(pytest_cache_dir + "/gen-ip054.mps")
    c, integrality, constraints, bounds = convert_all(prob)
    c = jnp.array(c)
    constraint_lb = jnp.array(constraints.lb)
    constraint_ub = jnp.array(constraints.ub)

    A = BCOO.fromdense(jnp.array(constraints.A))
    var_lb = jnp.array(bounds.lb)
    var_ub = jnp.array(bounds.ub)

    solver = raPDHG(eps_abs=1e-6, eps_rel=1e-6)

    def solve_single(c):
        boxed_lp = create_lp(c, A, constraint_lb, constraint_ub, var_lb, var_ub)
        result = solver.optimize(boxed_lp)
        return result.primal_objective

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
def test_rapdhg_with_sharding():
    """Test the raPDHG solver on a batch of LP problems."""
    mesh = jax.make_mesh((2,), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, P("x"))

    gurobi_model = gp.read(pytest_cache_dir + "/flugpl.mps")
    lp_sharded = create_qp_from_gurobi(gurobi_model, sharding=sharding)
    jax.debug.visualize_array_sharding(lp_sharded.constraint_matrix)

    solver = raPDHG(eps_abs=1e-8, eps_rel=1e-8)
    jit_optimize = jax.jit(solver.optimize)
    result = jit_optimize(lp_sharded)

    # Expected optimal objective value
    expected_obj = 1.167185726e06
    assert pytest.approx(result.primal_objective, rel=1e-2) == expected_obj
