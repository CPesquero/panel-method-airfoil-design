"""Panel method for NACA 4-digit airfoils: validation against XFOIL and inverse design."""

from .geometry import ALPHA_DEG, BASELINE, OPTIMIZED, Naca4, naca4, wedge
from .solver import Panels, Solution, influence, solve, system
from .studies import Candidate, Convergence, Design, inverse_design, mesh_convergence
from .xfoil import (
    POLAR_INVISCID,
    POLAR_VISCOUS,
    REF_NACA3217,
    CpDistribution,
    Polar,
    load_cp,
    load_polar,
)

__version__ = "1.0.0"

__all__ = [
    "ALPHA_DEG",
    "BASELINE",
    "OPTIMIZED",
    "POLAR_INVISCID",
    "POLAR_VISCOUS",
    "REF_NACA3217",
    "Candidate",
    "Convergence",
    "CpDistribution",
    "Design",
    "Naca4",
    "Panels",
    "Polar",
    "Solution",
    "influence",
    "inverse_design",
    "load_cp",
    "load_polar",
    "mesh_convergence",
    "naca4",
    "solve",
    "system",
    "wedge",
    "__version__",
]
