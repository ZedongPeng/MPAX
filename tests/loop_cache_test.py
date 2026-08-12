"""The cached iteration loops must not serve stale configurations.

optimize() reuses jitted loop closures across calls (solver fields are
baked into them as compile-time constants), invalidating on any field
change. These tests pin the two hazards: a second solve must still be
correct, and a field mutated between solves must actually take effect.
"""

import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from jax.experimental.sparse import BCOO

from mpax.mp_io import create_lp
from mpax.r2hpdhg import r2HPDHG
from mpax.utils import TerminationStatus


def _lp():
    # min x1 + 2 x2  s.t. 1 <= x1 + x2 <= 4, 0.5 <= x1 - x2, 0 <= x <= 3
    c = jnp.array([1.0, 2.0])
    A = BCOO.fromdense(jnp.array([[1.0, 1.0], [1.0, -1.0]]))
    lc = jnp.array([1.0, 0.5])
    uc = jnp.array([4.0, jnp.inf])
    return create_lp(c, A, lc, uc, jnp.zeros(2), jnp.full(2, 3.0))


def test_second_solve_reuses_loop_and_stays_correct():
    solver = r2HPDHG(eps_abs=1e-6, eps_rel=1e-6)
    problem = _lp()
    first = solver.optimize(problem)
    second = solver.optimize(problem)
    assert first.termination_status == TerminationStatus.OPTIMAL
    assert second.termination_status == TerminationStatus.OPTIMAL
    assert pytest.approx(float(second.primal_objective), rel=1e-6) == float(
        first.primal_objective
    )


def test_field_mutation_between_solves_takes_effect():
    solver = r2HPDHG(eps_abs=1e-2, eps_rel=1e-2)
    problem = _lp()
    loose = solver.optimize(problem)
    solver.eps_abs = 1e-9
    solver.eps_rel = 1e-9
    tight = solver.optimize(problem)
    # A stale cached loop would keep terminating at the loose tolerance.
    assert tight.termination_status == TerminationStatus.OPTIMAL
    assert int(tight.iteration_count) > int(loose.iteration_count)
