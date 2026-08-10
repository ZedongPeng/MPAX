"""Benchmark instance registry.

MPS/QPS instances are downloaded on first use into benchmarks/cache/
(same sources as tests/conftest.py). The synthetic knapsack LP is
generated deterministically.
"""

import gzip
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests

CACHE = Path(__file__).parent / "cache"

URLS = {
    "gen-ip054.mps.gz": "https://miplib.zib.de/WebData/instances/gen-ip054.mps.gz",
    "flugpl.mps.gz": "https://miplib.zib.de/WebData/instances/flugpl.mps.gz",
    "QPDATA1.ZIP": "http://www.doc.ic.ac.uk/~im/QPDATA1.ZIP",
}

# name -> (source archive member or None for synthetic, is_qp)
INSTANCES = {
    "gen-ip054": {"file": "gen-ip054.mps", "is_qp": False},
    "flugpl": {"file": "flugpl.mps", "is_qp": False},
    "AUG2DC": {"file": "qpdata/AUG2DC.QPS", "is_qp": True},
    "AUG3DCQP": {"file": "qpdata/AUG3DCQP.QPS", "is_qp": True},
    "knapsack-2000": {"file": None, "is_qp": False},
}


def _download(name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    target = CACHE / name
    if not target.exists():
        with requests.get(URLS[name], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return target


def _ensure_extracted():
    for gz in ("gen-ip054.mps.gz", "flugpl.mps.gz"):
        out = CACHE / gz.removesuffix(".gz")
        if not out.exists():
            with gzip.open(_download(gz), "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    qpdir = CACHE / "qpdata"
    if not qpdir.exists():
        with zipfile.ZipFile(_download("QPDATA1.ZIP")) as z:
            z.extractall(qpdir)


def write_knapsack_mps(path: Path, n: int = 2000, seed: int = 0) -> None:
    """LP relaxation of a knapsack: max v'x s.t. w'x <= W, 0 <= x <= 1."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(1.0, 100.0, n)
    w = rng.uniform(1.0, 100.0, n)
    W = 0.25 * w.sum()
    with open(path, "w") as f:
        f.write("NAME knapsack\nROWS\n N obj\n L cap\nCOLUMNS\n")
        for j in range(n):
            f.write(f" x{j} obj {-v[j]:.10f} cap {w[j]:.10f}\n")
        f.write(f"RHS\n rhs cap {W:.10f}\nBOUNDS\n")
        for j in range(n):
            f.write(f" UP bnd x{j} 1.0\n")
        f.write("ENDATA\n")


def fetch_all() -> dict:
    _ensure_extracted()
    paths = {}
    for name, meta in INSTANCES.items():
        if meta["file"] is None:
            p = CACHE / f"{name}.mps"
            if not p.exists():
                write_knapsack_mps(p)
            paths[name] = p
        else:
            paths[name] = CACHE / meta["file"]
    return paths
