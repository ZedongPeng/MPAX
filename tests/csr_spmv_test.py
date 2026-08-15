"""Pins the cuSparse CSR fast path for r2HPDHG's matvecs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from jax import config
from jax.experimental import sparse
from jax.experimental.sparse import BCOO

config.update("jax_enable_x64", True)

from mpax.mp_io import create_lp, create_qp
from mpax.preprocess import attach_csr_matrices
from mpax.r2hpdhg import r2HPDHG
from mpax.rapdhg import raPDHG
from mpax.utils import TerminationStatus


def _random_bcoo(m, n, nnz, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, m, nnz)
    cols = rng.integers(0, n, nnz)
    data = rng.standard_normal(nnz)
    data[0] = 0.0  # explicit zero survives the pipeline
    order = rng.permutation(nnz) if shuffle else np.arange(nnz)
    return BCOO(
        (jnp.asarray(data[order]), jnp.asarray(np.stack([rows, cols], 1)[order])),
        shape=(m, n),
    )


def test_attach_csr_matrices_matches_bcoo():
    m, n = 37, 53
    A = _random_bcoo(m, n, 400)  # unsorted entries, duplicates possible
    lp = create_lp(
        jnp.ones(n),
        A,
        -jnp.ones(m),
        jnp.ones(m),
        jnp.zeros(n),
        jnp.ones(n),
        use_sparse_matrix=True,
    )
    lp_csr = attach_csr_matrices(lp)
    assert lp_csr.constraint_matrix_csr is not None
    assert lp_csr.constraint_matrix_t_csr is not None

    x = jnp.asarray(np.random.default_rng(1).standard_normal(n))
    y = jnp.asarray(np.random.default_rng(2).standard_normal(m))
    ref_Ax = lp.constraint_matrix @ x
    ref_Aty = lp.constraint_matrix_t @ y
    got_Ax = sparse.csr_matvec(lp_csr.constraint_matrix_csr, x)
    got_Aty = sparse.csr_matvec(lp_csr.constraint_matrix_t_csr, y)
    assert float(jnp.max(jnp.abs(got_Ax - ref_Ax))) < 1e-10
    assert float(jnp.max(jnp.abs(got_Aty - ref_Aty))) < 1e-10
    # cusparse wants 32-bit index arrays
    assert lp_csr.constraint_matrix_csr.indices.dtype == jnp.int32
    assert lp_csr.constraint_matrix_csr.indptr.dtype == jnp.int32


def test_attach_csr_matrices_dense_noop():
    lp = create_lp(
        jnp.array([1.0, 2.0]),
        jnp.ones((1, 2)),
        jnp.ones(1),
        jnp.array([jnp.inf]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=False,
    )
    assert attach_csr_matrices(lp).constraint_matrix_csr is None


def _tiny_sparse_qp():
    # min 1/2 x'Ix - [1,1]x  s.t. x1 + x2 <= 2, 0 <= x <= 1
    # (unconstrained optimum x = (1,1) is feasible; objective = -1)
    Q = BCOO.fromdense(jnp.eye(2))
    A = BCOO.fromdense(jnp.ones((1, 2)))
    return create_qp(
        Q,
        jnp.array([-1.0, -1.0]),
        A,
        jnp.array([-jnp.inf]),
        jnp.array([2.0]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=True,
    )


def test_attach_csr_matrices_covers_objective_matrix():
    qp = _tiny_sparse_qp()
    qp_csr = attach_csr_matrices(qp)
    assert qp_csr.objective_matrix_csr is not None
    x = jnp.array([0.3, -0.7])
    ref = qp.objective_matrix @ x
    got = sparse.csr_matvec(qp_csr.objective_matrix_csr, x)
    assert float(jnp.max(jnp.abs(got - ref))) < 1e-12


def test_rapdhg_qp_solves_via_csr_path():
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(_tiny_sparse_qp())
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert float(result.primal_objective) == pytest.approx(-1.0, abs=1e-4)


def test_rapdhg_sparse_lp_solves_via_csr_path():
    # Exercises the adaptive-step-size / line-search path through cusparse.
    lp = create_lp(
        jnp.array([1.0, 2.0]),
        BCOO.fromdense(jnp.ones((1, 2))),
        jnp.ones(1),
        jnp.array([jnp.inf]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=True,
    )
    result = raPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert float(result.primal_objective) == pytest.approx(1.0, rel=1e-4)


def test_r2hpdhg_sparse_lp_solves_via_csr_path():
    # min x1 + 2 x2  s.t. x1 + x2 >= 1, 0 <= x <= 1 (optimum 1.0), sparse A.
    lp = create_lp(
        jnp.array([1.0, 2.0]),
        BCOO.fromdense(jnp.ones((1, 2))),
        jnp.ones(1),
        jnp.array([jnp.inf]),
        jnp.zeros(2),
        jnp.ones(2),
        use_sparse_matrix=True,
    )
    # cusparse is the default; the whole jitted loop (lean steps, window
    # end, restarts, stats) runs through it.
    result = r2HPDHG(eps_abs=1e-8, eps_rel=1e-8).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert float(result.primal_objective) == pytest.approx(1.0, rel=1e-4)
    # Opting out falls back to the fused XLA scatter path.
    result = r2HPDHG(eps_abs=1e-8, eps_rel=1e-8, use_cusparse=False).optimize(lp)
    assert result.termination_status == TerminationStatus.OPTIMAL
    assert float(result.primal_objective) == pytest.approx(1.0, rel=1e-4)
