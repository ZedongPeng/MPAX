"""Compare two benchmark CSVs: python -m benchmarks.compare OLD NEW"""
import csv
import sys


def load(path):
    with open(path) as f:
        return {
            (r["instance"], r["solver"], r["tol"]): r
            for r in csv.DictReader(f)
        }


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    flagged = False
    for key in sorted(old.keys() & new.keys()):
        o, n = old[key], new[key]
        it_o, it_n = int(o["iterations"]), int(n["iterations"])
        delta = (it_n - it_o) / max(it_o, 1) * 100
        line = (
            f"{'/'.join(key):40s} iters {it_o:>8d} -> {it_n:>8d} "
            f"({delta:+6.1f}%)  time {o['solve_time_sec']}s -> "
            f"{n['solve_time_sec']}s  status {o['status']}->{n['status']}"
        )
        if abs(delta) > 5 or o["status"] != n["status"]:
            line += "   <-- FLAG"
            flagged = True
        print(line)
    for key in sorted(new.keys() - old.keys()):
        print(f"{'/'.join(key):40s} NEW (no baseline)")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
