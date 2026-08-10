import jax.numpy as jnp

from mpax.mp_io import create_lp, create_qp
from mpax.utils import compute_objective_product


def test_objective_product_lp_is_zero_without_matmul():
    c = jnp.array([1.0, 2.0])
    G = jnp.array([[1.0, 1.0]])
    lc = jnp.array([1.0])
    uc = jnp.array([jnp.inf])
    lp = create_lp(c, G, lc, uc, jnp.zeros(2), jnp.ones(2), use_sparse_matrix=False)
    x = jnp.array([0.3, 0.7])
    assert jnp.all(compute_objective_product(lp, x) == 0.0)


def test_objective_product_qp_matches_matmul():
    Q = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    c = jnp.array([1.0, 2.0])
    G = jnp.array([[1.0, 1.0]])
    lc = jnp.array([1.0])
    uc = jnp.array([jnp.inf])
    qp = create_qp(Q, c, G, lc, uc, jnp.zeros(2), jnp.ones(2), use_sparse_matrix=False)
    x = jnp.array([0.3, 0.7])
    assert jnp.allclose(compute_objective_product(qp, x), Q @ x)
