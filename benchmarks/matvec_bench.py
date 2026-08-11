"""Micro-benchmark sparse matvec variants on the benchmark instances (P6).

Usage:
    $PY -m benchmarks.matvec_bench [--reps 200]

Times y = A @ x and z = At @ y for each registry instance's constraint
matrix in four formats: BCSR (the solver's format before the 2026-08-10 adoption of BCOO — the ratio baseline), BCOO (the solver's current format, adopted per the recorded verdict), sorted
BCOO, and CSR via jax.experimental.sparse.csr_matvec. The transpose is
stored explicitly per format, mirroring the solver (constraint_matrix_t
is a materialized transpose, never a transposed view).

Adoption rule (Stage-4 plan, Task 1): a variant graduates iff it is >=15%
faster than BCSR on BOTH directions for >=3 of the 5 instances.
"""

import argparse
import timeit

import gurobipy as gp
import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.experimental import sparse

from benchmarks.instances import INSTANCES, fetch_all

SPEEDUP_THRESHOLD = 0.85  # variant_time / bcsr_time below this counts as a win
REQUIRED_WINS = 3


def _best_time(fn, arg, reps):
    jax.block_until_ready(fn(arg))  # compile + warm cache
    times = timeit.repeat(lambda: jax.block_until_ready(fn(arg)), number=1, repeat=reps)
    return min(times)


def build_variants(a_csr):
    """Return {name: (matvec_matrix, matvec_t_matrix)} for one instance."""
    at_csr = a_csr.T.tocsr()
    variants = {
        "bcsr": (
            sparse.BCSR.from_scipy_sparse(a_csr),
            sparse.BCSR.from_scipy_sparse(at_csr),
        ),
        "bcoo": (
            sparse.BCOO.from_scipy_sparse(a_csr),
            sparse.BCOO.from_scipy_sparse(at_csr),
        ),
        "bcoo_sorted": (
            sparse.BCOO.from_scipy_sparse(a_csr).sort_indices(),
            sparse.BCOO.from_scipy_sparse(at_csr).sort_indices(),
        ),
        "csr": (
            sparse.CSR(
                (
                    jnp.array(a_csr.data),
                    jnp.array(a_csr.indices),
                    jnp.array(a_csr.indptr),
                ),
                shape=a_csr.shape,
            ),
            sparse.CSR(
                (
                    jnp.array(at_csr.data),
                    jnp.array(at_csr.indices),
                    jnp.array(at_csr.indptr),
                ),
                shape=at_csr.shape,
            ),
        ),
    }
    return variants


def matvec_fns(name, mat, mat_t):
    if name == "csr":
        f = jax.jit(lambda v: sparse.csr_matvec(mat, v))
        ft = jax.jit(lambda v: sparse.csr_matvec(mat_t, v))
    else:
        f = jax.jit(lambda v: mat @ v)
        ft = jax.jit(lambda v: mat_t @ v)
    return f, ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    paths = fetch_all()
    wins = {}  # variant -> count of instances where it beats bcsr on both dirs
    print(
        f"{'instance':15s} {'variant':12s} {'A@x (us)':>10s} "
        f"{'At@y (us)':>10s} {'ratio_fwd':>9s} {'ratio_t':>9s}"
    )
    for name in INSTANCES:
        a_csr = gp.read(str(paths[name])).getA().tocsr()
        m, n = a_csr.shape
        x = jnp.ones(n)
        y = jnp.ones(m)
        variants = build_variants(a_csr)
        base_fwd = base_t = None
        for vname, (mat, mat_t) in variants.items():
            f, ft = matvec_fns(vname, mat, mat_t)
            t_fwd = _best_time(f, x, args.reps)
            t_t = _best_time(ft, y, args.reps)
            if vname == "bcsr":
                base_fwd, base_t = t_fwd, t_t
            r_fwd = t_fwd / base_fwd
            r_t = t_t / base_t
            print(
                f"{name:15s} {vname:12s} {t_fwd * 1e6:10.1f} "
                f"{t_t * 1e6:10.1f} {r_fwd:9.3f} {r_t:9.3f}"
            )
            if (
                vname != "bcsr"
                and r_fwd < SPEEDUP_THRESHOLD
                and r_t < SPEEDUP_THRESHOLD
            ):
                wins[vname] = wins.get(vname, 0) + 1

    adopted = [v for v, c in wins.items() if c >= REQUIRED_WINS]
    if adopted:
        # Pick the variant with the most wins (ties: first alphabetically).
        best = sorted(adopted, key=lambda v: (-wins[v], v))[0]
        print(
            f"VERDICT: ADOPT {best} (beats bcsr on both directions on {wins[best]}/5 instances)"
        )
    else:
        print(f"VERDICT: NO ADOPTION (win counts: {wins or 'none'})")


if __name__ == "__main__":
    main()
