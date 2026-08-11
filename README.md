<p align="center">
  <img src="https://github.com/MIT-Lu-Lab/mpax/blob/main/docs/mpax.png" alt="MPAX" width="360">
</p>

# MPAX: Mathematical Programming in JAX

[![pypi](https://img.shields.io/pypi/v/mpax.svg?color=brightgreen)](https://pypi.org/pypi/mpax/)
![CI status](https://github.com/MIT-Lu-Lab/MPAX/actions/workflows/test.yml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MIT-Lu-Lab/MPAX/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09734-B31B1B.svg)](https://arxiv.org/abs/2412.09734)

MPAX is a hardware-accelerated, differentiable, batchable, and distributable solver for mathematical programming in JAX, designed to integrate with modern computational and deep learning workflows:

- **Hardware accelerated**: executes on multiple architectures including CPUs, GPUs and TPUs.
- **Differentiable**: computes derivatives of solutions with respect to inputs through unrolled differentiation.
- **Batchable**: solves multiple problem instances of the same shape simultaneously.
- **Distributed**: executes in parallel across multiple devices, such as several GPUs.

MPAX's primary motivation is to integrate mathematical programming with deep learning pipelines. To achieve this, MPAX aligns its algorithms and implementations with the requirements of deep learning hardware, ensuring compatibility with GPUs and TPUs. By being differentiable, MPAX can integrate directly into the backpropagation process of neural network training. Its batchability and distributability further enable scalable deployment in large-scale applications.

Currently, MPAX supports **linear programming (LP)** and **quadratic programming (QP)**, the foundational problems in mathematical programming. Future releases will expand support to include other problem classes of mathematical programming.

## Installation

You can install the latest released version of MPAX from PyPI via:
```
pip install mpax
```
or you can install the latest development version from GitHub:
```
pip install git+https://github.com/MIT-Lu-Lab/mpax.git
```

## Quickstart

Currently, MPAX focuses on solving linear programming (LP) and quadratic programming (QP) problems of the following form:

```math
\begin{equation}
\tag{LP}
\begin{aligned}
\min_{l \leq x \leq u}\ & c^\top x \\
\text{s.t.}\ & \ell_c \leq A x \leq u_c
\end{aligned}
\end{equation}
```

```math
\begin{equation}
\tag{QP}
\begin{aligned}
\min_{l \leq x \leq u}\ & \frac{1}{2} x^\top Q x + c^\top x \\
\text{s.t.}\ & \ell_c \leq A x \leq u_c
\end{aligned}
\end{equation}
```

Constraint rows are two-sided: equality rows have `lc == uc`, `>=` rows have `uc = +inf`, `<=` rows have `lc = -inf`, and rows with both bounds infinite are free.

MPAX implements two state-of-the-art first-order methods:
* $\boldsymbol{\mathrm{ra}}$**PDHG**: **restarted average Primal-Dual Hybrid Gradient**, supporting both LP ([paper](https://arxiv.org/abs/2311.12180)) and QP ([paper](https://arxiv.org/abs/2311.07710)). 
* $\boldsymbol{\mathrm{r^2}}$**HPDHG**: **reflected restarted Halpern Primal-Dual Hybrid Gradient**, supporting LP only ([paper](https://arxiv.org/abs/2407.16144)).

`mpax.solve()` picks the recommended solver automatically: LPs go to r2HPDHG, QPs to raPDHG. The classes remain available for algorithm-specific tuning.

### Solving a Single LP/QP Problem
MPAX supports both dense and sparse formats for the constraint matrix, controlled by the `use_sparse_matrix` parameter.
```python
from mpax import create_lp, create_qp, solve

# min c'x  s.t.  lc <= Ax <= uc,  l <= x <= u
lp = create_lp(c, A, lc, uc, l, u)          # sparse constraint matrix (default)
lp = create_lp(c, A, lc, uc, l, u, use_sparse_matrix=False)  # dense
result = solve(lp, eps_abs=1e-4, eps_rel=1e-4, verbose=True)

# min 1/2 x'Qx + c'x  s.t.  lc <= Ax <= uc,  l <= x <= u
qp = create_qp(Q, c, A, lc, uc, l, u)
result = solve(qp, eps_abs=1e-4, eps_rel=1e-4, verbose=True)
```

**Advanced: direct solver classes.** `solve()` only exposes the algorithm-agnostic options listed under [Solver Options](#solver-options). To tune algorithm-specific knobs (restart thresholds, step-size exponents, ...), instantiate `raPDHG` or `r2HPDHG` directly and call `.optimize()`:
```python
from mpax import create_lp, r2HPDHG

lp = create_lp(c, A, lc, uc, l, u)
solver = r2HPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=True)
result = solver.optimize(lp)
```

### Migrating from v0.2

* `create_lp(c, A, b, G, h, l, u)` / `create_qp(Q, c, A, b, G, h, l, u)` (pre-v0.3) became `create_lp(c, A, lc, uc, l, u)` / `create_qp(Q, c, A, lc, uc, l, u)` — constraints are now expressed as two-sided `lc <= Ax <= uc` bounds instead of separate equality (`Ax = b`) and inequality (`Gx >= h`) blocks. One-release compatibility wrappers `create_lp_standard_form(c, A, b, G, h, l, u)` / `create_qp_standard_form(Q, c, A, b, G, h, l, u)` reproduce the old semantics and emit a `DeprecationWarning`; they will be removed in v0.4.
* Duals of `<=` rows changed sign: rows are no longer negated internally, so the dual of a binding `<=` row is now nonpositive (the standard convention).
* For LPs, `objective_matrix` is `None` (no dense zero allocation).

### Batch solving
Batch solving allows you to solve multiple LP problems of the same shape simultaneously by using `jax.vmap`. `solve()` is vmap-compatible since solver dispatch is resolved at trace time:
```python
import jax
import jax.numpy as jnp
from mpax import create_lp, solve

def single_optimize(c_vector):
    lp = create_lp(c_vector, A, lc, uc, l, u)
    result = solve(lp, eps_abs=1e-4, eps_rel=1e-4)
    obj = jnp.dot(c_vector, result.primal_solution)
    return result.primal_solution, obj

batch_size = 100
batch_c = jnp.tile(c, (batch_size, 1))
batch_optimize = jax.vmap(single_optimize)

result = batch_optimize(batch_c)
```

### Device parallelism
Distribute computations across devices using JAX’s sharding capabilities. `solve()` works here too, but this example keeps the explicit `r2HPDHG` class to show `jax.jit` applied directly to `.optimize`:

```python
import jax
from jax.sharding import PartitionSpec as P
from mpax import create_lp, r2HPDHG

# Data sharding
mesh = jax.make_mesh((2,), ('x',))
sharding = jax.sharding.NamedSharding(mesh, P('x',))

A_sharded = jax.device_put(A, sharding)
lp_sharded = create_lp(c, A_sharded, lc, uc, l, u)

solver = r2HPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=True)
jit_optimize = jax.jit(solver.optimize)
result = jit_optimize(lp_sharded)
```

### Differentiation
An Example of computing the forward and backward passes of the "Smart Predict-then-Optimize+" loss using MPAX and `jax.custom_jvp()`.
```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def pso_fun(pred_cost, true_cost, true_sol, true_obj):
    sol, obj = batch_optimize(2*pred_cost - true_cost)
    loss = -obj + 2 * jnp.sum(pred_cost * true_sol, axis=1) - true_obj
    loss = jnp.mean(loss)
    return loss, sol

def spo_fwd(pred_cost, true_cost, true_sol, true_obj):
    loss, sol = pso_fun(pred_cost, true_cost, true_sol, true_obj)
    return loss, (sol, true_sol)

def spo_bwd(res, g):
    sol, true_sol = res
    grad = 2 * (true_sol - sol)
    # No gradients needed for true_cost, true_sol, or true_obj
    return grad * g, None, None, None

pso_fun.defvjp(spo_fwd, spo_bwd)
```

### Solver Options

**General options**
| Parameter                     | Type   | Default   | Description                                                             |
|:-------------------------------:|:--------:|:-----------:|-------------------------------------------------------------------------|
| `verbose`                    | bool   | `False`   | Enables detailed logging of the solver's progress.                     |
| `debug`                      | bool   | `False`   | Activates additional debugging information.                            |
| `display_frequency`          | int    | `10`      | Frequency (in every termination check) for displaying solver statistics.            |
| `jit`                        | bool   | `True`    | Enables JIT (Just-In-Time) compilation for faster execution.            |
| `unroll`                     | bool   | `False`   | Unrolls iteration loops  |
| `warm_start`                 | bool   | `False`   | Whether to perform warm starting  |
| `feasibility_polishing`      | bool   | `False`   | Whether to perform feasibility polishing  |

**Termination**
| Parameter                        | Type   | Default     | Description                                                           |
|:----------------------------------:|:--------:|:-------------:|-----------------------------------------------------------------------|
| `eps_abs`                       | float  | `1e-4`      | Absolute tolerance for convergence.                                   |
| `eps_rel`                       | float  | `1e-4`      | Relative tolerance for convergence.                                   |
| `eps_primal_infeasible`         | float  | `1e-8`      | Tolerance for detecting primal infeasibility.                         |
| `eps_dual_infeasible`           | float  | `1e-8`      | Tolerance for detecting dual infeasibility                           |
| `eps_feas_polish`               | float  | `1e-6`      | Tolerance for feasibility polishing |
| `iteration_limit`               | int    | `max_int`   | Maximum number of iterations allowed (interpreted as unlimited by default) |

**Precision**

By default, MPAX uses single-precision (32-bit). To enable double-precision (64-bit), add the following at the start of your script:

```python
jax.config.update("jax_enable_x64", True)
```

**Determinism**

Floating-point computations on GPUs in JAX may produce non-deterministic results. To ensure deterministic results, set:
```python
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
```
**Important**: If you are using batch solving, do not enable `--xla_gpu_deterministic_ops=true`, as it can significantly degrade performance.

## Citation
If MPAX is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```bibtex
@article{lu2024mpax,
  title={MPAX: Mathematical Programming in JAX},
  author={Lu, Haihao and Peng, Zedong and Yang, Jinwen},
  journal={arXiv preprint arXiv:2412.09734},
  year={2024}
}
```