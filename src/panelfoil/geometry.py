"""Airfoil geometry: NACA 4-digit sections and the three-panel wedge of the hand calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Naca4:
    """NACA 4-digit section, all values as fractions of the chord.

    f : maximum camber, p : chordwise position of maximum camber, t : maximum thickness.
    """

    f: float
    p: float
    t: float

    @classmethod
    def from_code(cls, code: str) -> Naca4:
        code = code.strip().upper().removeprefix("NACA").strip()
        if len(code) != 4 or not code.isdigit():
            raise ValueError(f"not a NACA 4-digit code: {code!r}")
        return cls(int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100)

    @property
    def code(self) -> str:
        return f"{round(self.f * 100):d}{round(self.p * 10):d}{round(self.t * 100):02d}"

    def __str__(self) -> str:
        return f"NACA {self.code}"

    def coordinates(self, n_panels: int, chord: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Panel nodes, clockwise from the trailing edge: lower surface first, then upper.

        Cosine spacing; ``n_panels`` is rounded down to an even number. The contour is closed:
        the first and last node are both the trailing edge, so there are ``n_panels + 1`` nodes.
        """
        return naca4(self.f, self.p, self.t, n_panels, chord)


def naca4(f: float, p: float, t: float, n_panels: int, c: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Same construction as the original notebook (``generar_naca``)."""
    n_panels = int(n_panels // 2 * 2)
    if n_panels < 4:
        raise ValueError("need at least 4 panels")
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    x = c * 0.5 * (1 - np.cos(beta))

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if f > 0 and p > 0:
        fwd = x <= p * c
        yc[fwd] = (f / p**2) * (2 * p * (x[fwd] / c) - (x[fwd] / c) ** 2) * c
        dyc_dx[fwd] = (2 * f / p**2) * (p - x[fwd] / c)
        aft = ~fwd
        yc[aft] = (f / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * (x[aft] / c) - (x[aft] / c) ** 2) * c
        dyc_dx[aft] = (2 * f / (1 - p) ** 2) * (p - x[aft] / c)

    theta = np.arctan(dyc_dx)
    xc_ = x / c
    poly = 0.2969 * np.sqrt(xc_) - 0.1260 * xc_ - 0.3516 * xc_**2 + 0.2843 * xc_**3 - 0.1015 * xc_**4
    yt = 5 * t * c * poly
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    return np.concatenate([xl[::-1], xu[1:]]), np.concatenate([yl[::-1], yu[1:]])


#: Baseline section of the assignment and the redesigned one.
BASELINE = Naca4.from_code("3217")
OPTIMIZED = Naca4.from_code("4114")

#: Study angle of attack, from the ID digits: alpha = 1 + 0.3 X with X = 7.
ALPHA_DEG = 3.1


def wedge(x_v: float = 0.4, z_v: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    """Three-panel wedge of section 1 (x_v = 0.25 + 0.05 Y, z_v = 0.1 + 0.01 Z with Y = 3, Z = 8).

    Nodes clockwise from the trailing edge: TE -> LE (lower panel), LE -> vertex, vertex -> TE.
    """
    return np.array([1.0, 0.0, x_v, 1.0]), np.array([0.0, 0.0, z_v, 0.0])
