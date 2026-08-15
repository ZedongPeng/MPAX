"""Pins r2HPDHG's knobs and restart machinery to cuPDLPx (commit 9a3c258).

Each test names the reference behavior it pins; reference locations are
cuPDLPx src/utils.cu (defaults) and src/solver.cu (perform_restart).
"""

import inspect

import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp
from mpax.r2hpdhg import r2HPDHG, power_method_sigma_max
from mpax.restart import (
    compute_new_primal_weight_cupdlpx,
    update_best_primal_weight,
)
from mpax.utils import TerminationStatus, cupdlpx_constraint_bound_norm


def test_r2hpdhg_defaults_match_cupdlpx():
    # utils.cu set_default_parameters: eval frequency 200, artificial 0.36,
    # sufficient 0.2, necessary 0.5, bound_objective_rescaling true.
    solver = r2HPDHG()
    assert solver.termination_evaluation_frequency == 200
    assert solver.artificial_restart_threshold == 0.36
    assert solver.sufficient_reduction_for_restart == 0.2
    assert solver.necessary_reduction_for_restart == 0.5
    assert solver.bound_objective_rescaling is True


def test_power_iteration_default_cap_is_5000():
    # utils.cu: sv_max_iter = 5000, sv_tol = 1e-4.
    sig = inspect.signature(power_method_sigma_max)
    assert sig.parameters["max_iterations"].default == 5000
    assert sig.parameters["tolerance"].default == 1e-4


def test_pid_primal_weight_update():
    # solver.cu perform_restart: e = log(dd) - log(pd) - log(w);
    # S = 0.3*S + e; w *= exp(0.99*e + 0.01*S); last_error = e.
    w, error_sum, last_error, best_w = 2.0, 0.1, 0.05, 7.0
    pd, dd, ratio = 1.0, 4.0, 1.0
    new_w, new_sum, new_last = compute_new_primal_weight_cupdlpx(
        pd, dd, ratio, w, error_sum, last_error, best_w
    )
    e = jnp.log(4.0) - jnp.log(1.0) - jnp.log(2.0)
    expected_sum = 0.3 * 0.1 + e
    expected_w = 2.0 * jnp.exp(0.99 * e + 0.01 * expected_sum)
    assert new_w == pytest.approx(float(expected_w), rel=1e-12)
    assert new_sum == pytest.approx(float(expected_sum), rel=1e-12)
    assert new_last == pytest.approx(float(e), rel=1e-12)


@pytest.mark.parametrize(
    "pd,dd,ratio",
    [
        (0.0, 4.0, 1.0),  # primal distance below 1e-16
        (1.0, 0.0, 1.0),  # dual distance below 1e-16
        (1e13, 4.0, 1.0),  # primal distance above 1e12
        (1.0, 4.0, 1e9),  # residual ratio above 1e8
        (1.0, 4.0, 1e-9),  # residual ratio below 1e-8
    ],
)
def test_pid_guard_falls_back_to_best_weight(pd, dd, ratio):
    # solver.cu: on guard failure w = best_primal_weight and the PID
    # accumulators reset to zero.
    new_w, new_sum, new_last = compute_new_primal_weight_cupdlpx(
        pd, dd, ratio, 2.0, 0.1, 0.05, 7.0
    )
    assert new_w == pytest.approx(7.0)
    assert new_sum == 0.0
    assert new_last == 0.0


def test_best_primal_weight_tracking():
    # solver.cu: gap = |log10(rel_dual/rel_primal)|; a strictly smaller gap
    # records the just-updated weight as best, otherwise best is unchanged.
    best_w, best_gap = update_best_primal_weight(
        residual_ratio=10.0, new_weight=3.0, best_weight=7.0, best_gap=2.0
    )
    assert best_w == pytest.approx(3.0)
    assert best_gap == pytest.approx(1.0)

    best_w, best_gap = update_best_primal_weight(
        residual_ratio=1e4, new_weight=3.0, best_weight=7.0, best_gap=2.0
    )
    assert best_w == pytest.approx(7.0)
    assert best_gap == pytest.approx(2.0)


def test_constraint_bound_norm_counts_both_range_bounds():
    # solver.cu compute_constraint_bound_norm (L2): lower^2 enters when
    # finite and lower != upper, upper^2 enters when finite; a range row
    # therefore contributes both bounds, an equality row only one.
    lp = create_lp(
        jnp.array([1.0, 1.0]),
        jnp.ones((3, 2)),
        jnp.array([1.0, 2.0, -jnp.inf]),  # range / equality / <= rows
        jnp.array([3.0, 2.0, 5.0]),
        jnp.zeros(2),
        jnp.full(2, jnp.inf),
        use_sparse_matrix=False,
    )
    expected = jnp.sqrt(1.0 + 9.0 + 4.0 + 25.0)
    assert cupdlpx_constraint_bound_norm(lp, 2) == pytest.approx(float(expected))
    # L-inf: max over the same finite entries.
    assert cupdlpx_constraint_bound_norm(lp, jnp.inf) == pytest.approx(5.0)


def test_r2hpdhg_still_solves_tiny_lp():
    # End-to-end guard: the rewired restart path still reaches OPTIMAL.
    lp = create_lp(
        jnp.array([1.0, 2.0]),
        jnp.ones((1, 2)),
        jnp.ones(1),
        jnp.array([jnp.inf]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )
    result = r2HPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert float(result.primal_objective) == pytest.approx(1.0, rel=1e-4)
