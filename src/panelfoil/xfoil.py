"""Readers for the XFOIL outputs used as reference (shipped in ``panelfoil/data``)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

CP_NACA3217 = "xfoil_cp_naca3217_a3.1.txt"
POLAR_INVISCID = "xfoil_polar_naca4114_inviscid.txt"
POLAR_VISCOUS = "xfoil_polar_naca4114_re3e6.txt"

#: Inviscid XFOIL reference for NACA 3217 at alpha = 3.1 deg (report, tables 1 and 2).
#: The moment is nose-down, hence negative.
REF_NACA3217 = {"cl": 0.7373, "cm_c4": -0.0644}


def _path(name: str | Path) -> Path:
    p = Path(name)
    if p.exists():
        return p
    return Path(str(resources.files("panelfoil.data").joinpath(str(name))))


@dataclass(frozen=True)
class CpDistribution:
    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray


def load_cp(name: str | Path = CP_NACA3217) -> CpDistribution:
    """XFOIL ``CPWR`` file: two header lines, a ``#`` column line, then x, y, Cp."""
    d = np.loadtxt(_path(name), skiprows=3)
    return CpDistribution(d[:, 0], d[:, 1], d[:, 2])


@dataclass(frozen=True)
class Polar:
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray

    @property
    def cl_max(self) -> tuple[float, float]:
        i = int(np.argmax(self.cl))
        return float(self.alpha[i]), float(self.cl[i])

    def at(self, alpha: float) -> tuple[float, float]:
        return float(np.interp(alpha, self.alpha, self.cl)), float(np.interp(alpha, self.alpha, self.cm))


def load_polar(name: str | Path) -> Polar:
    """XFOIL polar save file: 12 header lines, then alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr."""
    d = np.loadtxt(_path(name), skiprows=12)
    return Polar(d[:, 0], d[:, 1], d[:, 2], d[:, 4])
