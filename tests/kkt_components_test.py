import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp
from mpax.iteration_stats_utils import compute_kkt_components


def test_kkt_components_hand_computed():
    # min x1 + 2 x2   s.t. x1 + x2 = 1 ;  x1 - x2 >= 0 ;  0 <= x <= 10
    c = jnp.array([1.0, 2.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    G = jnp.array([[1.0, -1.0]])
    h = jnp.array([0.0])
    lp = create_lp(
        c, A, b, G, h, jnp.zeros(2), 10.0 * jnp.ones(2), use_sparse_matrix=False
    )

    x = jnp.array([0.5, 0.25])  # infeasible: x1+x2 = 0.75 != 1
    y = jnp.array([0.0, 1.0])  # inequality dual, nonnegative
    Ax = jnp.array([0.75, 0.25])  # [A;G] @ x
    ATy = jnp.array([1.0, -1.0])  # [A;G]' @ y
    qx = jnp.zeros(2)  # LP: Q @ x = 0

    prim, dual, pobj, dobj = compute_kkt_components(lp, x, y, Ax, ATy, qx, jnp.inf)

    # primal: only the equality row is violated, by 0.25; bounds hold
    assert jnp.isclose(prim, 0.25)
    # dual: gradient c - A'y + Qx = [0, 3]; both vars have finite bounds
    # => full gradient becomes reduced cost, violation = 0
    assert jnp.isclose(dual, 0.0)
    assert jnp.isclose(pobj, 1.0)  # c'x = 0.5 + 0.5

    # now a dual point whose gradient is NEGATIVE on an upper-bounded var:
    # y2 = 3 => gradient = c - G'y = [1-3, 2+3] = [-2, 5] -> reduced costs
    # absorb both (finite bounds), violation still 0; but with the bounds
    # made infinite the violation must appear:
    lp_free = create_lp(
        c,
        A,
        b,
        G,
        h,
        -jnp.inf * jnp.ones(2),
        jnp.inf * jnp.ones(2),
        use_sparse_matrix=False,
    )
    y2 = jnp.array([0.0, 3.0])
    ATy2 = jnp.array([3.0, -3.0])
    prim2, dual2, _, _ = compute_kkt_components(lp_free, x, y2, Ax, ATy2, qx, jnp.inf)
    # gradient = c - A'y + Qx = [-2, 5]; no finite bounds to absorb it
    # => reduced_costs_violation = |[-2, 5]|, inf-norm = 5
    assert jnp.isclose(dual2, 5.0)
