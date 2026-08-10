"""Run the benchmark sweep and write a CSV.

Usage:
    $PY -m benchmarks.run --out benchmarks/baseline/$(date +%F)-stage1.csv
"""
import argparse
import csv
import timeit

import gurobipy as gp
from jax import config

config.update("jax_enable_x64", True)

from mpax.mp_io import create_qp_from_gurobi
from mpax.rapdhg import raPDHG
from mpax.r2hpdhg import r2HPDHG
from mpax.utils import TerminationStatus

from benchmarks.instances import INSTANCES, fetch_all


def solvers_for(is_qp):
    if is_qp:
        return [("raPDHG", raPDHG)]
    return [("raPDHG", raPDHG), ("r2HPDHG", r2HPDHG)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", nargs="*", type=float, default=[1e-4, 1e-8])
    ap.add_argument("--instances", nargs="*", default=list(INSTANCES))
    args = ap.parse_args()

    paths = fetch_all()
    rows = []
    for name in args.instances:
        problem = create_qp_from_gurobi(gp.read(str(paths[name])))
        for solver_name, cls in solvers_for(INSTANCES[name]["is_qp"]):
            for tol in args.tol:
                solver = cls(eps_abs=tol, eps_rel=tol)
                t0 = timeit.default_timer()
                result = solver.optimize(problem)
                elapsed = timeit.default_timer() - t0
                rows.append({
                    "instance": name,
                    "solver": solver_name,
                    "tol": tol,
                    "status": int(result.termination_status),
                    "iterations": int(result.iteration_count),
                    "primal_objective": float(result.primal_objective),
                    "solve_time_sec": round(elapsed, 3),
                })
                print(rows[-1])

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
