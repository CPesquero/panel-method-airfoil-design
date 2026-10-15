"""Mesh-convergence study (section 2.4) and inverse design (section 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ALPHA_DEG, BASELINE, Naca4
from .solver import solve


@dataclass(frozen=True)
class Convergence:
    n_panels: np.ndarray
    cl: np.ndarray
    rel_change: np.ndarray  # |Cl_k - Cl_{k-1}| / Cl_{k-1}, aligned with n_panels[1:]
    tol: float

    @property
    def n_opt(self) -> int:
        """First mesh whose change from the previous one is below ``tol``."""
        ok = np.flatnonzero(self.rel_change < self.tol)
        if ok.size == 0:
            raise RuntimeError("no mesh met the tolerance")
        return int(self.n_panels[1:][ok[0]])


def mesh_convergence(
    airfoil: Naca4 = BASELINE,
    alpha_deg: float = ALPHA_DEG,
    n_start: int = 40,
    factor: float = 1.2,
    n_steps: int = 8,
    tol: float = 0.01,
) -> Convergence:
    """Grow the mesh by ``factor`` each step, as in the report (40, 48, 56, 68, ...)."""
    ns, cls = [], []
    n = float(n_start)
    for _ in range(n_steps):
        n_even = int(n // 2 * 2)
        X, Y = airfoil.coordinates(n_even)
        ns.append(n_even)
        cls.append(solve(X, Y, alpha_deg).cl)
        n *= factor
    ns_arr, cl_arr = np.array(ns), np.array(cls)
    rel = np.abs(np.diff(cl_arr) / cl_arr[:-1])
    return Convergence(ns_arr, cl_arr, rel, tol)


@dataclass(frozen=True)
class Candidate:
    airfoil: Naca4
    cl: float
    cm_c4: float


@dataclass(frozen=True)
class Design:
    base: Candidate
    target_cl: float
    best: Candidate
    feasible: list[Candidate]

    @property
    def cl_gain(self) -> float:
        return self.best.cl / self.base.cl - 1.0

    @property
    def cm_change(self) -> float:
        return abs(self.best.cm_c4 - self.base.cm_c4) / abs(self.base.cm_c4)


def inverse_design(
    base: Naca4 = BASELINE,
    n_panels: int = 56,
    alpha_deg: float = 0.0,
    gain: float = 0.20,
    cl_tol: float = 0.015,
    f_digits=range(1, 10),
    p_digits=range(1, 7),
    t_digits=range(5, 26),
) -> Design:
    """Search NACA 4-digit sections whose Cl is within ``cl_tol`` of (1 + gain) * Cl_base.

    Among those, keep the one whose pitching moment changes least. Same grid and criterion as the
    original notebook: f = 1-9 %, p = 10-60 %, t = 5-25 %.
    """
    X, Y = base.coordinates(n_panels)
    s = solve(X, Y, alpha_deg)
    base_c = Candidate(base, s.cl, s.cm_c4)
    target = base_c.cl * (1 + gain)
    feasible = []
    for fd in f_digits:
        for pd in p_digits:
            for td in t_digits:
                af = Naca4(fd / 100, pd / 10, td / 100)
                X, Y = af.coordinates(n_panels)
                try:
                    s = solve(X, Y, alpha_deg)
                except np.linalg.LinAlgError:
                    continue
                if abs(s.cl - target) < cl_tol:
                    feasible.append(Candidate(af, s.cl, s.cm_c4))
    if not feasible:
        raise RuntimeError("no section met the target")
    feasible.sort(key=lambda c: abs(c.cm_c4 - base_c.cm_c4))
    return Design(base_c, target, feasible[0], feasible)
