import jax.numpy as jnp
import pytest

from mpax.mp_io import create_qp
from mpax.r2hpdhg import r2HPDHG


def _tiny_qp():
    # min 1/2 x'Ix + 0'x  s.t. x1 + x2 >= 1, 0 <= x <= 1
    Q = jnp.eye(2)
    c = jnp.zeros(2)
    G = jnp.ones((1, 2))
    lc = jnp.ones(1)
    uc = jnp.array([jnp.inf])
    return create_qp(Q, c, G, lc, uc, jnp.zeros(2), jnp.ones(2),
                     use_sparse_matrix=False)


def test_r2hpdhg_rejects_qp():
    qp = _tiny_qp()
    with pytest.raises(ValueError, match="LP only"):
        r2HPDHG().optimize(qp)
