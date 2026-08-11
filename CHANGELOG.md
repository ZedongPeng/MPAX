# Changelog

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
