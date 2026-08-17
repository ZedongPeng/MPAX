"""Bit-exact reproduction of cuPDLP-x's power-iteration start vector.

cuPDLP-x (src/utils.cu) seeds its singular-value power iteration with

    std::mt19937 gen(1);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (i = 0; i < m; ++i) eigenvector_h[i] = dist(gen);

Both are process-wide globals, so a solve sees the first ``m`` draws of that
stream. r2HPDHG's ``power_method_sigma_max`` starts from the same vector so
its step size 0.998 / sigma_max matches the reference to roundoff. The
estimate is only ~1e-8 sensitive to the start vector, but on
restart-sensitive instances (physiciansched6-2 at 1e-8) that seed difference
is enough to land the two solvers on different trajectories.

The draws are reproduced with libstdc++ (GCC) semantics:

* ``std::mt19937(seed)`` is Matsumoto's ``init_genrand`` -- numpy's legacy
  MT19937 seeding -- so ``MT19937._legacy_seeding`` yields the same 32-bit
  stream (verified against the raw C++ outputs).
* ``std::normal_distribution`` uses the Marsaglia polar method; each attempt
  consumes two ``generate_canonical<double, 53>`` values, each of which is
  two 32-bit words: ``(r0 + r1 * 2**32) / 2**64`` clamped below 1. An
  accepted attempt returns ``y * mult`` first and caches ``x * mult`` for
  the next call.
* ``log`` goes through ``math.log`` (the C library's ``log``, i.e. the same
  glibc the reference binary links) rather than ``np.log``, whose SIMD
  implementation differs in the last ulp for ~0.3% of arguments.

Verified bit-for-bit against a g++ 12 reference on 300k draws; the first
values are pinned in tests/cupdlpx_parity_test.py.
"""

import math

import numpy as np

_TWO32 = 4294967296.0
_TWO64 = 18446744073709551616.0


def _generate_canonical(hi_lo):
    """libstdc++ generate_canonical<double, 53> over an (k, 2) uint64 array."""
    u = (hi_lo[:, 0].astype(np.float64) + hi_lo[:, 1].astype(np.float64) * _TWO32) / _TWO64
    return np.where(u >= 1.0, np.nextafter(1.0, 0.0), u)


def libstdcxx_normal_draws(n: int, seed: int = 1) -> np.ndarray:
    """First ``n`` values of ``std::normal_distribution<double>(0, 1)`` driven by
    ``std::mt19937(seed)`` under libstdc++, as float64."""
    n = int(n)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    bitgen = np.random.MT19937()
    bitgen._legacy_seeding(seed)  # init_genrand(seed) == std::mt19937(seed)
    filled = 0
    while filled < n:
        # Each polar attempt burns 4 words and is accepted with prob pi/4;
        # over-provision a little so most sizes finish in one round.
        attempts = max(1024, int((n - filled) / 2 / 0.78) + 64)
        words = bitgen.random_raw(4 * attempts).astype(np.uint64).reshape(attempts, 4)
        x = 2.0 * _generate_canonical(words[:, 0:2]) - 1.0
        y = 2.0 * _generate_canonical(words[:, 2:4]) - 1.0
        r2 = x * x + y * y
        keep = ~((r2 > 1.0) | (r2 == 0.0))
        x, y, r2 = x[keep], y[keep], r2[keep]
        log_r2 = np.fromiter(map(math.log, r2.tolist()), dtype=np.float64, count=r2.size)
        mult = np.sqrt(-2.0 * log_r2 / r2)
        pair = np.empty((r2.size, 2), dtype=np.float64)
        pair[:, 0] = y * mult  # returned by this call
        pair[:, 1] = x * mult  # _M_saved, returned by the next call
        flat = pair.ravel()
        take = min(flat.size, n - filled)
        out[filled : filled + take] = flat[:take]
        filled += take
    return out
