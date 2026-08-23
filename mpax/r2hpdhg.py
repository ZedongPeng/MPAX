import logging
import timeit
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax.experimental.sparse import BCOO, CSR
from jax.lax import cond

from mpax.cupdlpx_random import libstdcxx_normal_draws
from mpax.loop_utils import while_loop
from mpax.preprocess import (
    attach_csr_matrices,
    filter_small_matrix_entries,
    rescale_problem,
)
from mpax.rapdhg import (
    compute_next_solution,
    raPDHG,
)
from mpax.restart import (
    compute_cupdlpx_fixed_point_error,
    compute_new_primal_weight_cupdlpx,
    restart_criteria_met_fixed_point,
    should_do_adaptive_restart_cupdlpx,
    unscaled_saddle_point_output,
    update_best_primal_weight,
)
from mpax.solver_log import (
    display_iteration_stats_heading,
    pdhg_final_log,
    setup_logger,
)
from mpax.termination import (
    check_termination_criteria,
    check_termination_criteria_cupdlpx,
    check_primal_feasibility,
    check_dual_feasibility,
    optimality_criteria_met,
)
from mpax.utils import (
    CachedQuadraticProgramInfo,
    PdhgSolverState,
    QuadraticProgrammingProblem,
    RestartInfo,
    RestartScheme,
    RestartToCurrentMetric,
    SaddlePointOutput,
    ScaledQpProblem,
    SolveConfig,
    TerminationStatus,
    ConvergenceInformation,
    compute_objective_product,
    cupdlpx_constraint_bound_norm,
    safe_norm,
)
from mpax.feasibility_polishing import (
    set_dual_solution_to_zero,
    set_primal_solution_to_zero,
    init_primal_feasibility_polishing,
    init_dual_feasibility_polishing,
)
from mpax.iteration_stats_utils import compute_convergence_information

logger = logging.getLogger(__name__)


def cupdlpx_power_iteration_start(m, dtype):
    """cuPDLP-x's power-iteration start vector: the first m draws of
    std::normal_distribution<double>(0,1) on std::mt19937(1) (libstdc++).

    Produced on the host through pure_callback so that under jit the vector
    is materialized at run time instead of being baked into the executable
    as an m-element literal (m reaches 3e7 on the Mittelmann set).
    """
    return jax.pure_callback(
        lambda: libstdcxx_normal_draws(m).astype(dtype),
        jax.ShapeDtypeStruct((m,), dtype),
    )


@jax.jit
def power_method_sigma_max(matrix, matrix_t, tolerance=1e-4, max_iterations=5000):
    """sigma_max of A by power iteration on A A', stopped on the eigenpair residual.

    jitted at module level so repeated solves of same-shaped problems reuse
    the compilation (an eagerly-invoked lax.while_loop recompiles per call).

    Start vector, cap and tolerance mirror cuPDLP-x (utils.cu
    estimate_maximum_singular_value: mt19937(1) normal draws, sv_max_iter=5000,
    sv_tol=1e-4), so the estimate -- and the 0.998/sigma_max step size --
    agrees with the reference to roundoff. The
    cap matters on ill-separated spectra: power iteration approaches
    sigma_max from below, and an estimate cut short can push
    step * sigma_true past 1 -- a 400-iteration cap left qnet1_o 1.24% low
    and the run diverged. The residual test still exits early (sometimes in
    ~15 iterations) on well-separated problems.
    """
    z0 = cupdlpx_power_iteration_start(matrix.shape[0], matrix.dtype)

    def mv(M, v):
        # CSR operands take the cusparse kernel; BCOO/dense use @.
        if isinstance(M, CSR):
            return jsparse.csr_matvec(M, v)
        return M @ v

    def cond_fun(state):
        i, _, _, residual = state
        return (i < max_iterations) & (residual >= tolerance)

    def body_fun(state):
        i, z, _, _ = state
        q = z / jnp.linalg.norm(z)
        z_new = mv(matrix, mv(matrix_t, q))
        eigenvalue = jnp.dot(q, z_new)
        return i + 1, z_new, eigenvalue, jnp.linalg.norm(z_new - eigenvalue * q)

    _, _, eigenvalue, _ = jax.lax.while_loop(
        cond_fun, body_fun, (0, z0, jnp.array(1.0), jnp.array(tolerance))
    )
    return jnp.sqrt(eigenvalue)


@dataclass(eq=False)
class r2HPDHG(raPDHG):
    """
    The r2HPDHG solver class.
    """

    verbose: bool = False
    debug: bool = False
    display_frequency: int = 10
    jit: bool = True
    unroll: bool = False
    termination_evaluation_frequency: int = 200
    # cuPDLP-x measures residuals in the 2-norm; the inf-norm default this
    # replaced made r2HPDHG stop at a different point for the same tolerance.
    optimality_norm: float = 2
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    eps_primal_infeasible: float = 1e-8
    eps_dual_infeasible: float = 1e-8
    # time_sec_limit: float = float("inf")
    iteration_limit: int = jnp.iinfo(jnp.int32).max
    l_inf_ruiz_iterations: int = 10
    l2_norm_rescaling: bool = False
    pock_chambolle_alpha: float = 1.0
    primal_importance: float = 1.0
    scale_invariant_initial_primal_weight: bool = True
    restart_scheme: int = RestartScheme.ADAPTIVE_KKT
    restart_to_current_metric: int = RestartToCurrentMetric.KKT_GREEDY
    restart_frequency_if_fixed: int = 1000
    # Restart cadence constants are cuPDLP-x's defaults. The primal weight
    # itself is driven by the reference's PID controller (see
    # compute_new_primal_weight_cupdlpx), not by the parent's
    # primal_weight_update_smoothing rule.
    artificial_restart_threshold: float = 0.36
    sufficient_reduction_for_restart: float = 0.2
    necessary_reduction_for_restart: float = 0.5
    # Halpern PDHG's convergence theory and the cuPDLP-x reference both
    # assume a constant step size; take_step has no line-search path.
    # The field stays (inherited) only so resolve_config can reject an
    # explicit True with a readable error instead of silently ignoring it.
    adaptive_step_size: bool = False
    warm_start: bool = False
    feasibility_polishing: bool = False
    eps_feas_polish: float = 1e-06
    # cuPDLP-x carries no infeasibility certificates, so neither does the
    # r2HPDHG path. Like adaptive_step_size above, the field stays only so
    # resolve_config can reject an explicit True with a readable error.
    infeasibility_detection: bool = False
    # cuPDLP-x's post-Ruiz scalar pass on bounds and objective. It also fixes
    # the initial primal weight at 1.0, since both sides are order 1 after it.
    # On by default, as in the reference.
    bound_objective_rescaling: bool = True
    # use_cusparse is inherited from raPDHG (default True). On the LP
    # benchmark it pays off across the board: square41 (heavy rows,
    # 7.2ms -> 78us per SpMV) and cont1 (whose unsorted lazy-transpose
    # matvec cost 152us per call on the BCOO path); tiny instances pay a
    # bounded fixed overhead (2club: ~11us -> ~45us per iteration).

    def resolve_config(self, is_lp: bool) -> SolveConfig:
        if not is_lp:
            raise ValueError(
                "r2HPDHG supports LP only (its Qx terms are hardwired to "
                "zero); use raPDHG for QP."
            )
        if self.adaptive_step_size:
            raise ValueError(
                "r2HPDHG uses a constant step size (Halpern PDHG's "
                "convergence guarantee requires it); adaptive_step_size is "
                "not supported. Use raPDHG for an adaptive line search."
            )
        if self.infeasibility_detection:
            raise ValueError(
                "r2HPDHG matches cuPDLP-x, which computes no infeasibility "
                "certificates; infeasibility_detection is not supported. "
                "Use raPDHG to detect primal/dual infeasibility."
            )
        return super().resolve_config(is_lp)

    def initialize_solver_status(
        self,
        scaled_problem: ScaledQpProblem,
        initial_primal_solution: jnp.array,
        initial_dual_solution: jnp.array,
        is_lp: bool,
        cfg: SolveConfig,
    ) -> Tuple[PdhgSolverState, RestartInfo, float]:
        """Wire the constant step size to 0.998 / sigma_max(scaled A).

        raPDHG leaves step_size at a placeholder 1.0 on the non-adaptive
        path because its own take_step recomputes the step size every
        iteration via calculate_constant_step_size. r2HPDHG overrides
        take_step and reuses solver_state.step_size verbatim, so without
        this the constant-step-size path runs at 1.0 forever and the
        sigma_max the parent computed is dead.
        """
        solver_state, last_restart_info, initial_primal_weight = (
            super().initialize_solver_status(
                scaled_problem,
                initial_primal_solution,
                initial_dual_solution,
                is_lp,
                cfg,
            )
        )
        # _norm_A comes from this class's compute_constant_step_size_norms
        # override. It matters that that estimate is tight: power
        # iteration approaches sigma_max from below, 0.998 leaves only
        # 0.2% of margin, and the parent's 10%-relative-error default
        # under-estimated qnet1_o by 1.24%, putting step*sigma_max at
        # 1.010 -- past the stability threshold, and the run diverged.
        # The m = 0 test is on a static shape, so it stays a Python
        # branch; _norm_A itself is a tracer under jit and must not be.
        if scaled_problem.scaled_qp.constraint_matrix.shape[0] == 0:
            step_size = 1.0
        else:
            step_size = 0.998 / self._norm_A
        # Seed initial_step_size too, so take_step's Halpern weight needs
        # no per-iteration guard against the 0 sentinel.
        solver_state = solver_state.replace(
            step_size=step_size, initial_step_size=step_size
        )
        # These stay None on the shared state class (raPDHG never sets them),
        # but the scan carry needs a concrete shape from the first iteration.
        solver_state = solver_state.replace(
            pdhg_primal_solution=solver_state.current_primal_solution,
            pdhg_dual_solution=solver_state.current_dual_solution,
            dual_slack=jnp.zeros_like(solver_state.current_primal_solution),
        )
        if self.bound_objective_rescaling:
            # Bounds and objective are both order 1 after that pass, so the
            # reference starts the weight at 1 rather than at ||c||/||b||.
            initial_primal_weight = 1.0
        else:
            # cuPDLP-x's fallback: (||c|| + 1) / (||b|| + 1) on the unscaled
            # problem, with range rows contributing both bounds to ||b||.
            # The parent's ||c||/||b|| on the scaled problem is a different
            # starting weight on essentially every instance.
            initial_primal_weight = (
                safe_norm(
                    scaled_problem.original_qp.objective_vector,
                    ord=self.optimality_norm,
                )
                + 1.0
            ) / (
                cupdlpx_constraint_bound_norm(
                    scaled_problem.original_qp, self.optimality_norm
                )
                + 1.0
            )
        solver_state = solver_state.replace(primal_weight=initial_primal_weight)
        # The PID weight controller falls back to the best weight seen so
        # far when its update guard fails; before any restart that is the
        # initial weight.
        last_restart_info = last_restart_info.replace(
            best_primal_weight=initial_primal_weight
        )
        return solver_state, last_restart_info, initial_primal_weight

    def compute_constant_step_size_norms(self, scaled_qp, is_lp):
        """Residual-stopped power iteration in place of the parent's estimator.

        Also avoids the parent's `BCSR.from_bcoo(matrix.T)` conversion: the
        scaled problem already carries the transpose, so the operator can be
        applied straight out of the existing BCOO pair.
        """
        self._norm_Q = 0.0
        if scaled_qp.constraint_matrix.shape[0] == 0:
            self._norm_A = 0.0
        elif scaled_qp.constraint_matrix_csr is not None:
            self._norm_A = power_method_sigma_max(
                scaled_qp.constraint_matrix_csr, scaled_qp.constraint_matrix_t_csr
            )
        else:
            self._norm_A = power_method_sigma_max(
                scaled_qp.constraint_matrix, scaled_qp.constraint_matrix_t
            )

    @staticmethod
    def _lp_reflected_step(problem, solver_state, step_size, window_end=True):
        """One reflected PDHG step, specialized for LP.

        Same arithmetic as `compute_next_solution` with extrapolation 1.0,
        fused with the bookkeeping that used to follow it. Three redundant
        passes go away: the projected primal already *is* the pure PDHG
        iterate, `dual_slack` falls out of that same projection instead of
        re-forming c - A'y, and the Qx momentum terms (identically zero for
        an LP, since resolve_config rejects QP input) are dropped rather
        than multiplied by zero on every iteration.
        """
        primal_step = step_size / solver_state.primal_weight
        dual_step = step_size * solver_state.primal_weight

        gradient = problem.objective_vector - solver_state.current_dual_product
        unprojected = solver_state.current_primal_solution - primal_step * gradient
        pdhg_primal = jnp.clip(
            unprojected, problem.variable_lower_bound, problem.variable_upper_bound
        )
        # (proj(x~) - x~) / primal_step: the bound multiplier the projection
        # implies, needed by the cuPDLP-x dual residual. Only the last step
        # of an evaluation window has a consumer for it.
        dual_slack = (
            (pdhg_primal - unprojected) / primal_step if window_end else None
        )
        delta_primal = pdhg_primal - solver_state.current_primal_solution
        if window_end or problem.constraint_matrix_csr is not None:
            # The restart test and perform_restart read delta_primal_product,
            # so the window's last step forms it explicitly. With the CSR
            # fast path attached, every step forms it: one cusparse SpMV
            # beats the old scatter-add lean trick by ~90x on heavy-row
            # matrices, and the explicit A dx costs nothing extra there.
            delta_primal_product = problem.matvec(delta_primal)
            # A @ (x + 2 dx), i.e. the product of the reflected primal. The
            # dual update and the Halpern average of the product both need
            # it, and they sit in different scopes, so form it once here.
            reflected_primal_product = (
                solver_state.current_primal_product + 2 * delta_primal_product
            )
            if not window_end:
                delta_primal_product = None
        elif isinstance(problem.constraint_matrix, BCOO):
            # BCOO fallback (no CSR attached, e.g. CPU): scatter-add the
            # contributions straight onto A x, which drops the matvec's
            # zero-init and the separate A x + 2 dx pass. The default
            # scatter mode drops out-of-bounds rows, which is exactly what
            # padded BCOO entries need. Reassociates the sum relative to
            # the window-end formula (same values up to fp rounding).
            delta_primal_product = None
            indices = problem.constraint_matrix.indices
            reflected_primal_product = solver_state.current_primal_product.at[
                indices[:, 0]
            ].add(2 * problem.constraint_matrix.data * delta_primal[indices[:, 1]])
        else:
            delta_primal_product = None
            reflected_primal_product = solver_state.current_primal_product + 2 * (
                problem.constraint_matrix @ delta_primal
            )

        candidate = (
            solver_state.current_dual_solution - dual_step * reflected_primal_product
        )
        pdhg_dual = candidate - jnp.clip(
            candidate,
            -dual_step * problem.constraint_upper_bound,
            -dual_step * problem.constraint_lower_bound,
        )
        delta_dual = pdhg_dual - solver_state.current_dual_solution
        return (
            delta_primal,
            delta_primal_product,
            reflected_primal_product,
            delta_dual,
            pdhg_primal,
            pdhg_dual,
            dual_slack,
        )

    def take_step(
        self,
        solver_state: PdhgSolverState,
        problem: QuadraticProgrammingProblem,
        cfg: SolveConfig,
        window_end: bool = True,
    ) -> PdhgSolverState:
        """
        Take one reflected (Halpern) PDHG step at the constant step size.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        problem : QuadraticProgrammingProblem
            The problem being solved.
        cfg : SolveConfig
            The per-solve configuration derived by `resolve_config`.
        window_end : bool
            Whether this step's pdhg_*, dual_slack and delta_* fields will
            be read (the termination and restart checks consume them, and
            they only see the last step of an evaluation window). Inside a
            window they are neither computed nor written: the fields pass
            through unchanged, which XLA hoists out of the loop carry. The
            iterate update itself is identical either way.
        """
        # The pure PDHG iterate (before Halpern averaging) and the
        # projection's bound multiplier come out of the fused step;
        # cuPDLP-x terminates on that iterate, not on the Halpern
        # average. Both of its products are left to the termination
        # check, which runs 100x less often than the step does.
        (
            delta_primal,
            delta_primal_product,
            reflected_primal_product,
            delta_dual,
            pdhg_primal_solution,
            pdhg_dual_solution,
            dual_slack,
        ) = self._lp_reflected_step(
            problem, solver_state, solver_state.step_size, window_end
        )
        step_size = solver_state.step_size

        # Compute the weight according to the stepsize.
        new_solutions_count = solver_state.solutions_count + 1
        new_weights_sum = solver_state.weights_sum + solver_state.step_size
        # Both initialize_solver_status and perform_restart seed this, and
        # the step size never changes, so a zero-sentinel guard can never
        # fire -- and a lax.cond on every iteration is not free at this
        # problem size.
        initial_step_size = solver_state.initial_step_size
        weight = (new_weights_sum) / (new_weights_sum + initial_step_size)
        next_primal_solution = (
            weight * (solver_state.current_primal_solution + 2 * delta_primal)
            + (1 - weight) * solver_state.initial_primal_solution
        )
        next_primal_product = (
            weight * reflected_primal_product
            + (1 - weight) * solver_state.initial_primal_product
        )
        next_dual_solution = (
            weight * (solver_state.current_dual_solution + 2 * delta_dual)
            + (1 - weight) * solver_state.initial_dual_solution
        )
        next_dual_product = problem.matvec_t(next_dual_solution)

        if not window_end:
            # Pass-throughs: loop-invariant in the window's fori_loop, so no
            # per-iteration carry writes. The window's last step (a
            # window_end=True call) overwrites them all before any consumer
            # reads them.
            delta_primal = solver_state.delta_primal
            delta_dual = solver_state.delta_dual
            delta_primal_product = solver_state.delta_primal_product
            pdhg_primal_solution = solver_state.pdhg_primal_solution
            pdhg_dual_solution = solver_state.pdhg_dual_solution
            dual_slack = solver_state.dual_slack

        return PdhgSolverState(
            current_primal_solution=next_primal_solution,
            current_dual_solution=next_dual_solution,
            current_primal_product=next_primal_product,
            current_dual_product=next_dual_product,
            # Qx == 0 for every iterate (r2HPDHG is LP-only, enforced in
            # resolve_config) and the avg_* fields belong to raPDHG's
            # averaging, which this solver never reads. Carrying the arrays
            # through costs nothing; rebuilding them with zeros_like cost six
            # allocations and six kernel launches on every single iteration,
            # which dominated the step on small problems.
            current_primal_obj_product=solver_state.current_primal_obj_product,
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            avg_primal_solution=solver_state.avg_primal_solution,
            avg_dual_solution=solver_state.avg_dual_solution,
            avg_primal_product=solver_state.avg_primal_product,
            avg_dual_product=solver_state.avg_dual_product,
            avg_primal_obj_product=solver_state.avg_primal_obj_product,
            solutions_count=new_solutions_count,
            weights_sum=new_weights_sum,
            step_size=step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried + 1,
            num_iterations=solver_state.num_iterations + 1,
            termination_status=TerminationStatus.UNSPECIFIED,
            delta_primal=delta_primal,
            delta_dual=delta_dual,
            delta_primal_product=delta_primal_product,
            pdhg_primal_solution=pdhg_primal_solution,
            pdhg_dual_solution=pdhg_dual_solution,
            dual_slack=dual_slack,
            initial_step_size=initial_step_size,
        )

    def take_multiple_steps(
        self,
        solver_state: PdhgSolverState,
        problem: QuadraticProgrammingProblem,
        cfg: SolveConfig,
    ) -> PdhgSolverState:
        """One evaluation window: N-1 lean steps, then one full step.

        Same iterate arithmetic as the parent's N x take_step, up to fp
        reassociation of the reflected product in the lean steps (see
        _lp_reflected_step); only the last step computes and stores the
        pdhg_*, dual_slack and delta_* fields, which nothing reads until
        the window ends. Together this cuts the measured per-iteration
        cost by 25-45% on mid-size MIPLIB LP relaxations (H100, fp64).
        """
        if self.termination_evaluation_frequency < 1:
            return solver_state
        inner_state = jax.lax.fori_loop(
            lower=0,
            upper=self.termination_evaluation_frequency - 1,
            body_fun=lambda i, s: self.take_step(s, problem, cfg, window_end=False),
            init_val=solver_state,
        )
        return self.take_step(inner_state, problem, cfg)

    def perform_restart(
        self,
        solver_state,
        last_restart_info,
        kkt_reduction_ratio,
        problem,
        cfg,
        residual_ratio,
    ):
        # Take a pure PDHG step to get the new solution and set it as the initial solution for the outer iteration.
        # Use the pure PDHG step solution, instead of the Halpen PDHG step solution, as the initial solution for the restart.
        # Solver state has been updated to Halpen PDHG step solution, therefore, we need to retrieve the pure PDHG step solution.
        restart_length = solver_state.solutions_count
        # weight = 1 / solver_state.solutions_count
        weight = solver_state.initial_step_size / solver_state.weights_sum
        # Retrieve the last iteration solution and product.
        last_iteration_primal_solution = (
            (1 + weight) * solver_state.current_primal_solution
            - weight * solver_state.initial_primal_solution
            - solver_state.delta_primal
        )
        last_iteration_dual_solution = (
            (1 + weight) * solver_state.current_dual_solution
            - weight * solver_state.initial_dual_solution
            - solver_state.delta_dual
        )
        # Recompute both products fresh at the reconstructed point. The
        # reference forms A x by SpMV every iteration, while the windows here
        # accumulate it incrementally; one SpMV per restart resets that fp
        # drift so it cannot compound across restart periods.
        last_iteration_primal_product = problem.matvec(last_iteration_primal_solution)
        last_iteration_dual_product = problem.matvec_t(last_iteration_dual_solution)
        last_iteration_solver_state = PdhgSolverState(
            current_primal_solution=last_iteration_primal_solution,
            current_dual_solution=last_iteration_dual_solution,
            current_primal_product=last_iteration_primal_product,
            current_dual_product=last_iteration_dual_product,
            current_primal_obj_product=jnp.zeros_like(last_iteration_primal_solution),
            avg_primal_solution=solver_state.avg_primal_solution,
            avg_dual_solution=solver_state.avg_dual_solution,
            avg_primal_product=solver_state.avg_primal_product,
            avg_dual_product=solver_state.avg_dual_product,
            avg_primal_obj_product=jnp.zeros_like(solver_state.avg_primal_solution),
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            solutions_count=solver_state.solutions_count,
            weights_sum=solver_state.weights_sum,
            step_size=solver_state.step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried,
            num_iterations=solver_state.num_iterations,
            termination_status=TerminationStatus.UNSPECIFIED,
            # Restarting anchors current at the pure PDHG point, so the
            # pdhg_* view coincides with it here.
            pdhg_primal_solution=last_iteration_primal_solution,
            pdhg_dual_solution=last_iteration_dual_solution,
            dual_slack=solver_state.dual_slack,
        )
        # cuPDLP-x restarts *at* the pure-PDHG point: current and initial both
        # become that anchor and no step is taken. Advancing one step here
        # while resetting the counters to zero (as this did) put every restart
        # period one step out of phase with the reference.
        restarted_solver_state = last_iteration_solver_state
        restarted_solver_state.initial_step_size = restarted_solver_state.step_size
        restarted_solver_state.initial_primal_solution = last_iteration_primal_solution
        restarted_solver_state.initial_dual_solution = last_iteration_dual_solution
        restarted_solver_state.initial_primal_product = last_iteration_primal_product
        restarted_solver_state.initial_dual_product = last_iteration_dual_product

        # Movement over the restart period, as the reference measures it:
        # plain L2 norms of (new anchor - previous anchor). The PID guard
        # thresholds below are calibrated to these raw distances.
        primal_distance_moved_last_restart_period = safe_norm(
            restarted_solver_state.initial_primal_solution
            - last_restart_info.primal_solution
        )
        dual_distance_moved_last_restart_period = safe_norm(
            restarted_solver_state.initial_dual_solution
            - last_restart_info.dual_solution
        )

        new_primal_weight, new_error_sum, new_last_error = (
            compute_new_primal_weight_cupdlpx(
                primal_distance_moved_last_restart_period,
                dual_distance_moved_last_restart_period,
                residual_ratio,
                solver_state.primal_weight,
                last_restart_info.primal_weight_error_sum,
                last_restart_info.primal_weight_last_error,
                last_restart_info.best_primal_weight,
            )
        )
        new_best_primal_weight, new_best_gap = update_best_primal_weight(
            residual_ratio,
            new_primal_weight,
            last_restart_info.best_primal_weight,
            last_restart_info.best_primal_dual_residual_gap,
        )
        restarted_solver_state.primal_weight = new_primal_weight
        restarted_solver_state.solutions_count = 0
        restarted_solver_state.weights_sum = 0.0

        # The reference seeds its restart baseline from the first real step
        # of the next window, which runs with the new weight already synced.
        # Probing that step here (without advancing the iterate) reproduces
        # it exactly: the window's own first step recomputes the same deltas
        # from the same anchor with the same weight.
        probe_delta_primal, probe_delta_primal_product, probe_delta_dual, _ = (
            compute_next_solution(
                problem, restarted_solver_state, restarted_solver_state.step_size, 1.0
            )
        )
        restarted_solver_state.delta_primal = probe_delta_primal
        restarted_solver_state.delta_dual = probe_delta_dual
        restarted_solver_state.delta_primal_product = probe_delta_primal_product

        new_last_restart_info = RestartInfo(
            primal_solution=restarted_solver_state.initial_primal_solution,
            dual_solution=restarted_solver_state.initial_dual_solution,
            primal_diff=restarted_solver_state.delta_primal,
            dual_diff=restarted_solver_state.delta_dual,
            primal_diff_product=restarted_solver_state.delta_primal_product,
            primal_product=restarted_solver_state.initial_primal_product,
            dual_product=restarted_solver_state.initial_dual_product,
            primal_obj_product=restarted_solver_state.current_primal_obj_product,
            last_restart_length=restart_length,
            primal_distance_moved_last_restart_period=primal_distance_moved_last_restart_period,
            dual_distance_moved_last_restart_period=dual_distance_moved_last_restart_period,
            reduction_ratio_last_trial=kkt_reduction_ratio,
            initial_fixed_point_error=compute_cupdlpx_fixed_point_error(
                restarted_solver_state.delta_primal,
                restarted_solver_state.delta_dual,
                restarted_solver_state.delta_primal_product,
                restarted_solver_state.step_size,
                restarted_solver_state.primal_weight,
            ),
            last_trial_fixed_point_error=jnp.inf,
            primal_weight_error_sum=new_error_sum,
            primal_weight_last_error=new_last_error,
            best_primal_weight=new_best_primal_weight,
            best_primal_dual_residual_gap=new_best_gap,
        )
        return restarted_solver_state, new_last_restart_info

    def run_restart_scheme(
        self,
        problem: QuadraticProgrammingProblem,
        solver_state: PdhgSolverState,
        last_restart_info: RestartInfo,
        cfg: SolveConfig,
        convergence_information: ConvergenceInformation,
    ):
        """
        Check restart criteria based on current and average KKT residuals.

        Parameters
        ----------
        problem : QuadraticProgrammingProblem
            The quadratic programming problem instance.
        solver_state : PdhgSolverState
            The current solver state.
        last_restart_info : RestartInfo
            Information from the last restart.
        cfg : SolveConfig
            The per-solve configuration derived by `resolve_config`.
        convergence_information : ConvergenceInformation
            Residuals of the window just evaluated; the PID weight
            controller's guard reads their dual/primal ratio.

        Returns
        -------
        tuple
            The new solver state, and the new last restart info.
        """
        do_restart, fixed_point_error = should_do_adaptive_restart_cupdlpx(
            cfg.restart_params,
            solver_state,
            last_restart_info,
            self.termination_evaluation_frequency,
        )
        residual_ratio = (
            convergence_information.relative_dual_residual_norm
            / convergence_information.relative_primal_residual_norm
        )
        # A restart clears the trial error (the reference resets it to inf, so
        # `error_increased` cannot fire on the first check of a new restart
        # period); otherwise the check records what it just measured.
        return cond(
            do_restart,
            lambda: self.perform_restart(
                solver_state,
                last_restart_info,
                fixed_point_error,
                problem,
                cfg,
                residual_ratio,
            ),
            lambda: (
                solver_state,
                last_restart_info.replace(
                    last_trial_fixed_point_error=fixed_point_error
                ),
            ),
        )

    def run_restart_scheme_feasibility_polishing(
        self,
        problem: QuadraticProgrammingProblem,
        current_solver_state: PdhgSolverState,
        restart_solver_state: PdhgSolverState,
        last_restart_info: RestartInfo,
        cfg: SolveConfig,
    ):
        """
        Check restart criteria based on current and average KKT residuals.

        Parameters
        ----------
        problem : QuadraticProgrammingProblem
            The quadratic programming problem instance.
        solver_state : PdhgSolverState
            The current solver state.
        last_restart_info : RestartInfo
            Information from the last restart.
        cfg : SolveConfig
            The per-solve configuration derived by `resolve_config`.

        Returns
        -------
        tuple
            The new solver state, and the new last restart info.
        """
        do_restart, kkt_reduction_ratio = cond(
            restart_solver_state.solutions_count == 0,
            lambda: (False, last_restart_info.reduction_ratio_last_trial),
            lambda: restart_criteria_met_fixed_point(
                cfg.restart_params, restart_solver_state, last_restart_info
            ),
        )
        return cond(
            do_restart,
            lambda: self.perform_restart(
                restart_solver_state,
                last_restart_info,
                kkt_reduction_ratio,
                problem,
                cfg,
                # Polishing has no cuPDLP-x residual pair at hand; a neutral
                # ratio passes the PID guard and leaves the log10 gap at 0.
                jnp.asarray(1.0),
            ),
            lambda: (current_solver_state, last_restart_info),
        )

    def main_iteration_update(
        self,
        cfg,
        solver_state,
        last_restart_info,
        should_terminate,
        scaled_problem,
        qp_cache,
        ci,
    ):
        # cuPDLP-x ordering: advance a full evaluation window first, then
        # test termination and restart on the iterate that window produced.
        # (Previously the restart test ran on the previous window's iterate,
        # before any step had been taken.)
        stepped_solver_state = self.take_multiple_steps(
            solver_state, scaled_problem.scaled_qp, cfg
        )

        should_terminate, termination_status, convergence_information = (
            check_termination_criteria_cupdlpx(
                scaled_problem,
                stepped_solver_state,
                cfg.termination_criteria,
                qp_cache,
                stepped_solver_state.numerical_error,
                self.optimality_norm,
            )
        )

        new_solver_state, new_last_restart_info = self.run_restart_scheme(
            scaled_problem.scaled_qp,
            stepped_solver_state,
            last_restart_info,
            cfg,
            convergence_information,
        )

        new_solver_state.termination_status = termination_status
        return (
            new_solver_state,
            new_last_restart_info,
            should_terminate,
            scaled_problem,
            qp_cache,
            convergence_information,
        )

    def primal_feasibility_polishing(
        self, solver_state, scaled_problem, qp_cache, cfg, initial_primal_weight
    ):
        """Perform primal feasibility polishing.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        scaled_problem : ScaledQpProblem
            The original problem and scaled problem data.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.
        cfg : SolveConfig
            The per-solve configuration derived by `resolve_config`.
        initial_primal_weight : float
            The initial primal weight computed in `initialize_solver_status`.

        Returns
        -------
        jnp.array
            The primal solution.
        bool
            Whether primal feasibility succeeds.
        """
        (
            primal_feasibility_problem,
            primal_feasibility_solver_state,
            last_restart_info,
        ) = init_primal_feasibility_polishing(
            scaled_problem, solver_state, initial_primal_weight, average=False
        )
        polish_loop = self._cached_loop(
            "primal_polish",
            True,
            lambda: lambda init_val: while_loop(
                cond_fun=lambda state: state[2] == False,
                body_fun=lambda state: self.primal_feasibility_polishing_iterate(
                    cfg, *state
                ),
                init_val=init_val,
                maxiter=self.iteration_limit,
                unroll=self.unroll,
                jit=self.jit,
            ),
        )
        (new_solver_state, last_restart_info, should_terminate, _, _) = polish_loop(
            (
                primal_feasibility_solver_state,
                last_restart_info,
                False,
                primal_feasibility_problem,
                qp_cache,
            )
        )
        return new_solver_state.current_primal_solution, should_terminate

    def primal_feasibility_polishing_iterate(
        self,
        cfg,
        primal_polishing_solver_state,
        last_restart_info,
        should_terminate,
        primal_feasibility_problem,
        qp_cache,
    ):
        zeroed_dual_solver_state = set_dual_solution_to_zero(
            primal_polishing_solver_state
        )
        restarted_primal_polishing_solver_state, new_last_restart_info = (
            self.run_restart_scheme_feasibility_polishing(
                primal_feasibility_problem.scaled_qp,
                primal_polishing_solver_state,
                zeroed_dual_solver_state,
                last_restart_info,
                cfg,
            )
        )
        new_primal_polishing_solver_state = self.take_multiple_steps(
            restarted_primal_polishing_solver_state,
            primal_feasibility_problem.scaled_qp,
            cfg,
        )
        new_should_terminate = check_primal_feasibility(
            primal_feasibility_problem,
            new_primal_polishing_solver_state,
            cfg.polishing_termination_criteria,
            qp_cache,
            1.0,
            self.optimality_norm,
            average=False,
        )
        return (
            new_primal_polishing_solver_state,
            new_last_restart_info,
            new_should_terminate,
            primal_feasibility_problem,
            qp_cache,
        )

    def dual_feasibility_polishing(
        self, solver_state, scaled_problem, qp_cache, cfg, initial_primal_weight
    ):
        """Perform dual feasibility polishing.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        scaled_problem : ScaledQpProblem
            The original problem and scaled problem data.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.
        cfg : SolveConfig
            The per-solve configuration derived by `resolve_config`.
        initial_primal_weight : float
            The initial primal weight computed in `initialize_solver_status`.

        Returns
        -------
        jnp.array
            The dual solution.
        bool
            Whether dual feasibility succeeds.
        """
        dual_feasibility_problem, dual_feasibility_solver_state, last_restart_info = (
            init_dual_feasibility_polishing(
                scaled_problem, solver_state, initial_primal_weight, average=False
            )
        )

        polish_loop = self._cached_loop(
            "dual_polish",
            True,
            lambda: lambda init_val: while_loop(
                cond_fun=lambda state: state[2] == False,
                body_fun=lambda state: self.dual_feasibility_polishing_iterate(
                    cfg, *state
                ),
                init_val=init_val,
                maxiter=self.iteration_limit,
                unroll=self.unroll,
                jit=self.jit,
            ),
        )
        (new_solver_state, last_restart_info, should_terminate, _, _) = polish_loop(
            (
                dual_feasibility_solver_state,
                last_restart_info,
                False,
                dual_feasibility_problem,
                qp_cache,
            )
        )
        return new_solver_state.current_dual_solution, should_terminate

    def dual_feasibility_polishing_iterate(
        self,
        cfg,
        dual_polishing_solver_state,
        last_restart_info,
        should_terminate,
        dual_feasibility_problem,
        qp_cache,
    ):
        zeroed_primal_solver_state = set_primal_solution_to_zero(
            dual_polishing_solver_state
        )
        restarted_dual_polishing_solver_state, new_last_restart_info = (
            self.run_restart_scheme_feasibility_polishing(
                dual_feasibility_problem.scaled_qp,
                dual_polishing_solver_state,
                zeroed_primal_solver_state,
                last_restart_info,
                cfg,
            )
        )

        new_dual_polishing_solver_state = self.take_multiple_steps(
            restarted_dual_polishing_solver_state,
            dual_feasibility_problem.scaled_qp,
            cfg,
        )

        new_should_terminate = check_dual_feasibility(
            dual_feasibility_problem,
            new_dual_polishing_solver_state,
            cfg.polishing_termination_criteria,
            qp_cache,
            1.0,
            self.optimality_norm,
            average=False,
        )

        return (
            new_dual_polishing_solver_state,
            new_last_restart_info,
            new_should_terminate,
            dual_feasibility_problem,
            qp_cache,
        )

    def optimize(
        self,
        original_problem: QuadraticProgrammingProblem,
        initial_primal_solution=None,
        initial_dual_solution=None,
    ) -> SaddlePointOutput:
        """
        Main algorithm: given parameters and LP problem, return solutions.

        Parameters
        ----------
        original_problem : QuadraticProgrammingProblem
            The quadratic programming problem to be solved.

        Returns
        -------
        SaddlePointOutput
            The solution to the optimization problem.
        """
        setup_logger(self.verbose, self.debug)
        # validate(original_problem)
        # config_check(params)
        cfg = self.resolve_config(original_problem.is_lp)
        # cuPDLP-x drops |a_ij| <= 1e-9 before anything reads the matrix, so
        # the filtered matrix is what scaling, sigma_max, and residuals see.
        original_problem = filter_small_matrix_entries(original_problem)
        # ||b|| the reference's way: a range row contributes both of its
        # bounds, where cached_quadratic_program_info keeps only the larger.
        qp_cache = CachedQuadraticProgramInfo(
            safe_norm(original_problem.objective_vector, ord=self.optimality_norm),
            cupdlpx_constraint_bound_norm(original_problem, self.optimality_norm),
        )

        precondition_start_time = timeit.default_timer()
        scaled_problem = rescale_problem(
            self.l_inf_ruiz_iterations,
            self.l2_norm_rescaling,
            self.pock_chambolle_alpha,
            original_problem,
            self.bound_objective_rescaling,
        )
        # Route every per-iteration product through cusparse: the CSR copies
        # ride on the scaled problem so the jitted loops can reach them.
        if self.use_cusparse:
            scaled_problem = scaled_problem._replace(
                scaled_qp=attach_csr_matrices(scaled_problem.scaled_qp)
            )
        precondition_time = timeit.default_timer() - precondition_start_time
        logger.info("Preconditioning Time (seconds): %.2e", precondition_time)

        solver_state, last_restart_info, initial_primal_weight = (
            self.initialize_solver_status(
                scaled_problem,
                initial_primal_solution,
                initial_dual_solution,
                original_problem.is_lp,
                cfg,
            )
        )

        # Iteration loop
        display_iteration_stats_heading()

        iteration_start_time = timeit.default_timer()
        # No warm-up phase: cuPDLP-x goes straight into full evaluation
        # windows, so its first restart check lands at exactly
        # termination_evaluation_frequency iterations. The 10 single-step
        # iterations that used to run here shifted every later check by 10
        # and made iteration counts incomparable with the reference.
        main_loop = self._cached_loop(
            "main",
            True,
            lambda: lambda init_val: while_loop(
                cond_fun=lambda state: state[2] == False,
                body_fun=lambda state: self.main_iteration_update(cfg, *state),
                init_val=init_val,
                maxiter=self.iteration_limit,
                unroll=self.unroll,
                jit=self.jit,
            ),
        )
        (solver_state, last_restart_info, should_terminate, _, _, ci) = main_loop(
            (
                solver_state,
                last_restart_info,
                False,
                scaled_problem,
                qp_cache,
                ConvergenceInformation(),
            )
        )
        iteration_time = timeit.default_timer() - iteration_start_time

        if self.feasibility_polishing:
            feasibility_polishing_start_time = timeit.default_timer()
            polished_primal_solution, primal_feasibility = (
                self.primal_feasibility_polishing(
                    solver_state, scaled_problem, qp_cache, cfg, initial_primal_weight
                )
            )
            polished_dual_solution, dual_feasibility = self.dual_feasibility_polishing(
                solver_state, scaled_problem, qp_cache, cfg, initial_primal_weight
            )
            feasibility_polishing_time = (
                timeit.default_timer() - feasibility_polishing_start_time
            )
            polished_primal_product = (
                scaled_problem.scaled_qp.constraint_matrix @ polished_primal_solution
            )
            polished_dual_product = (
                scaled_problem.scaled_qp.constraint_matrix_t @ polished_dual_solution
            )
            polished_primal_obj_product = compute_objective_product(
                scaled_problem.scaled_qp, polished_primal_solution
            )
            # Convergence information of the polished candidate, computed before
            # acceptance so that the polished pair is only taken when it actually
            # satisfies the requested optimality tolerances.
            polished_ci = compute_convergence_information(
                scaled_problem.original_qp,
                qp_cache,
                polished_primal_solution / scaled_problem.variable_rescaling,
                polished_dual_solution / scaled_problem.constraint_rescaling,
                self.eps_abs / self.eps_rel,
                polished_primal_product * scaled_problem.constraint_rescaling,
                polished_dual_product * scaled_problem.variable_rescaling,
                polished_primal_obj_product * scaled_problem.variable_rescaling,
                self.optimality_norm,
            )
            accept_polished = (
                primal_feasibility
                & dual_feasibility
                & optimality_criteria_met(cfg.termination_criteria.eps_rel, polished_ci)
            )
            (
                solver_state.current_primal_solution,
                solver_state.current_primal_product,
                solver_state.current_dual_solution,
                solver_state.current_dual_product,
                solver_state.current_primal_obj_product,
                ci,
            ) = cond(
                accept_polished,
                lambda: (
                    polished_primal_solution,
                    polished_primal_product,
                    polished_dual_solution,
                    polished_dual_product,
                    polished_primal_obj_product,
                    polished_ci,
                ),
                lambda: (
                    solver_state.current_primal_solution,
                    solver_state.current_primal_product,
                    solver_state.current_dual_solution,
                    solver_state.current_dual_product,
                    solver_state.current_primal_obj_product,
                    ci,
                ),
            )
        else:
            feasibility_polishing_time = 0

        timing = {
            "Preconditioning": precondition_time,
            "Iteration loop": iteration_time,
            "Feasibility polishing": feasibility_polishing_time,
        }

        # Log the stats of the final iteration.
        pdhg_final_log(
            scaled_problem.scaled_qp,
            solver_state.current_primal_solution,
            solver_state.current_dual_solution,
            solver_state.num_iterations,
            solver_state.termination_status,
            timing,
            ci,
        )
        return unscaled_saddle_point_output(
            scaled_problem,
            # The reference reports the pure PDHG iterate, which is also the
            # point its residuals were measured at; returning the Halpern
            # average would hand back a point with different residuals than
            # the ones that satisfied the tolerance.
            solver_state.pdhg_primal_solution,
            solver_state.pdhg_dual_solution,
            solver_state.termination_status,
            # The -1 here compensated for the warm-up loop that used to run
            # before the main loop; without it the count is exactly the
            # reference's total_count.
            solver_state.num_iterations,
            ci,
            timing,
        )
