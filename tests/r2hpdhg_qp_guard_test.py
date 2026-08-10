import jax.numpy as jnp
import pytest

from mpax.mp_io import create_qp
from mpax.r2hpdhg import r2HPDHG


def _tiny_qp():
    # min 1/2 x'Ix + 0'x  s.t. x1 + x2 >= 1, 0 <= x <= 1
    Q = jnp.eye(2)
    c = jnp.zeros(2)
    A = jnp.zeros((0, 2))
    b = jnp.zeros(0)
    G = jnp.ones((1, 2))
    h = jnp.ones(1)
    l = jnp.zeros(2)
    u = jnp.ones(2)
    return create_qp(Q, c, A, b, G, h, l, u, use_sparse_matrix=False)


def test_r2hpdhg_rejects_qp():
    qp = _tiny_qp()
    with pytest.raises(ValueError, match="LP only"):
        r2HPDHG().optimize(qp)
