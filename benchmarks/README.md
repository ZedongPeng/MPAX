# MPAX benchmarks

Not part of CI (a full sweep takes tens of minutes on CPU).

    PY=~/opt/anaconda3/envs/py310/bin/python
    $PY -m benchmarks.run --out benchmarks/baseline/<date>-<label>.csv
    $PY -m benchmarks.compare benchmarks/baseline/<old>.csv <new>.csv

`iterations` is the gate metric (deterministic on CPU); `solve_time_sec`
includes jit compile time and is indicative only. compare.py exits nonzero
if any (instance, solver, tol) shifts iterations by more than 5% or
changes termination status.
