"""MPAX - A Python package for Mathematical Programming in JAX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mpax")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "0.0.0.dev0"

from .r2hpdhg import r2HPDHG
from .rapdhg import raPDHG
from .mp_io import (
    create_lp,
    create_qp,
    create_lp_standard_form,
    create_qp_standard_form,
    create_qp_from_gurobi,
)
from .solve import solve

__all__ = [
    "solve",
    "r2HPDHG",
    "raPDHG",
    "create_lp",
    "create_qp",
    "create_lp_standard_form",
    "create_qp_standard_form",
    "create_qp_from_gurobi",
]
