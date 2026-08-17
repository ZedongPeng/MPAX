# Changelog

## Unreleased

### Breaking
- r2HPDHG is constant-step-size only: Halpern PDHG's convergence
  guarantee (and the cuPDLP-x reference) assume a fixed step, so the
  adaptive line-search path was removed from its `take_step` and
  `resolve_config` now rejects `adaptive_step_size=True` with a
  readable error. raPDHG is unchanged (adaptive remains its LP
  default). Iteration counts for r2HPDHG change accordingly: on the
  benchmark sweep, fewer iterations on flugpl/gen-ip054 (-14% to
  -50%), more on the degenerate knapsack-2000 (1 x 2000; +171% at
  1e-8) — re-baseline before gating.

### Changed
- `create_qp_from_gurobi` folds Gurobi's range-constraint slack columns
  back into two-sided row bounds (`fold_range_slacks=True`, new
  keyword). Gurobi encodes a ranged row -- an MPS RANGES entry or
  `Model.addRange` -- as an equality plus an auxiliary variable
  (`MPS_Rg<row>` / `Rg<row>`, bounded, zero cost, one nonzero); the
  problem handed to the solver used to inherit that column and lose
  the two-sided row. cont1/cont11/ns930473/ns1644855 are the affected
  LP-benchmark instances (ns930473: 33748 -> 11328 columns), and they
  now match what native MPS readers such as cuPDLPx solve. Only
  unmistakable slacks are folded (name prefix + singleton column + zero
  objective + equality row); when any are, the returned problem has
  fewer variables than `model.NumVars` -- pass `fold_range_slacks=False`
  to keep Gurobi's variable layout.
- r2HPDHG's power iteration starts from cuPDLP-x's exact start vector:
  the first m draws of `std::normal_distribution<double>(0,1)` on
  `std::mt19937(1)` under libstdc++, reproduced bit-for-bit on the host
  (`mpax/cupdlpx_random.py`, delivered through `pure_callback` so jit
  does not bake an m-element literal). Structure, cap and tolerance
  already matched; the start vector was the last difference, and it
  moved the 0.998/sigma_max step size by ~1e-8 relative -- enough on
  restart-sensitive instances (physiciansched6-2 at 1e-8: cuPDLPx
  1.9M iterations, MPAX >8M) to put the two solvers on different
  trajectories. With the reference start vector the step size agrees
  to 1 ulp and physiciansched6-2 reproduces cuPDLPx's 1e-4 count
  (951600) exactly.
- `optimize()` now reuses its jitted iteration loops across calls:
  the loop closures are cached on the solver instance (invalidated
  when any solver field changes) instead of re-traced and recompiled
  on every call, and the power method is jitted at module level. Warm
  same-shape solves drop ~60% wall time on GPU (a 3000-iteration
  budget on 30n20b8: 1.04s -> 0.41s), including solves of *different*
  problems with the same shapes — the batch-solving case. Numerics
  are unchanged; `tests/loop_cache_test.py` guards the invalidation.
- r2HPDHG evaluation windows run lean: the first N-1 steps of a
  window no longer compute or carry `pdhg_*`, `dual_slack` and
  `delta_*` (only the window's last step does, and nothing reads them
  earlier), and for sparse problems the lean step scatter-adds A dx
  straight onto A x instead of materializing the matvec separately.
  Same iterate arithmetic up to fp reassociation of that sum;
  iteration counts across the CPU sweep are unchanged. Measured
  per-iteration cost drops 22-45% on mid-size MIPLIB LP relaxations
  and 2-9% on large ones on an H100. (Measured and rejected along the
  way, all equal or slower on GPU: sorted-BCOO cusparse lowering,
  BCSR/CSR formats, fori_loop unrolling, WHILE command buffers.)

## v0.3.0 — 2026-08-10

### Breaking
- `create_lp` / `create_qp` re-signed for two-sided constraints:
  `create_lp(c, A, lc, uc, l, u)`, `create_qp(Q, c, A, lc, uc, l, u)`.
  Row classes are encoded by bounds (equality: `lc == uc`; `>=`:
  `uc = +inf`; `<=`: `lc = -inf`; free: both infinite).
- Duals of `<=` rows changed sign: rows are no longer negated
  internally, so the dual of a binding `<=` row is nonpositive (the
  standard convention). Affects `create_qp_from_gurobi` models with `<`
  constraints.
- For LPs, `QuadraticProgrammingProblem.objective_matrix` is `None`.

### Added
- `mpax.solve(problem, **options)`: unified entry point dispatching LPs
  to r2HPDHG and QPs to raPDHG at trace time (works under `jit`/`vmap`).
- Support for m = 0 (box-only) problems (#27), with shape validation and
  readable errors in the constructors.
- Infeasibility certificates for the two-sided form, including `<=`-row
  primal-infeasibility detection, with dedicated tests.
- Benchmark harness (`benchmarks/`): instance registry, sweep runner,
  CSV comparator, matvec micro-benchmark (#12).

### Changed
- Constraint matrices stay in BCOO format end-to-end (previously
  converted to BCSR after rescaling): 20-40% faster matvecs on the
  benchmark set, ~27% lower median solve time, identical iteration
  counts (`benchmarks/results/2026-08-10-matvec.txt`).
- Solver configuration is resolved per-solve into an immutable
  `SolveConfig`; solver configuration is no longer mutated by `optimize`
  (a QP solve no longer degrades subsequent LP solves on the same
  instance).

### Deprecated
- `create_lp_standard_form` / `create_qp_standard_form` reproduce the
  pre-v0.3 constructor semantics, emit `DeprecationWarning`, and will be
  removed in v0.4.

### Fixed
- Feasibility polishing runs end-to-end (#28).
- Verbose logging emits a single ordered summary block (#29).
- Restart KKT residual includes the dual-feasibility term and the QP
  gradient contribution.
- r2HPDHG rejects QP input with a clear error instead of silently
  solving the LP relaxation; its termination checks now honor
  `infeasibility_detection` and `is_lp`.
- `__version__` now reflects the installed package version.

### Removed
- Dead presolve subsystem (~430 lines: `presolve`, `remove_empty_*`,
  `undo_presolve`, bound-to-constraint transforms) — unreachable from
  any public entry point.

### Evaluated and not adopted
- PID primal-weight controller (from the mpax-dev reference): benchmark
  showed the smoothing rule dominates on both LPs and QPs; the port was
  removed per the design's no-dead-switch rule (implementation preserved
  in this branch's pre-merge history at 43d3d89).
