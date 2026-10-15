"""Constant-strength source panels plus one uniform vortex distribution (Hess-Smith type).

Unknowns: one source strength per panel and a single vortex strength gamma shared by all panels.
Equations: zero normal velocity at every control point (panel midpoint) and the Kutta condition,
equal and opposite tangential velocity on the first and last panel at the trailing edge.

Free stream of unit speed, chord 1. Pitching moment positive nose-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Panels:
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray

    @classmethod
    def from_nodes(cls, X: np.ndarray, Y: np.ndarray) -> Panels:
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        return cls(X[:-1], Y[:-1], X[1:], Y[1:])

    @property
    def n(self) -> int:
        return len(self.x1)

    @property
    def xc(self) -> np.ndarray:
        return 0.5 * (self.x1 + self.x2)

    @property
    def yc(self) -> np.ndarray:
        return 0.5 * (self.y1 + self.y2)

    @property
    def length(self) -> np.ndarray:
        return np.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle(self) -> np.ndarray:
        return np.arctan2(self.y2 - self.y1, self.x2 - self.x1)

    @property
    def normal(self) -> tuple[np.ndarray, np.ndarray]:
        """Outward normal for a clockwise contour."""
        return -np.sin(self.angle), np.cos(self.angle)

    @property
    def tangent(self) -> tuple[np.ndarray, np.ndarray]:
        return np.cos(self.angle), np.sin(self.angle)


@dataclass(frozen=True)
class Solution:
    panels: Panels
    alpha_deg: float
    sigma: np.ndarray  # source strength per panel
    gamma: float  # vortex strength (per unit length, same on every panel)
    vt: np.ndarray  # tangential velocity at the control points (V_inf = 1)

    @property
    def cp(self) -> np.ndarray:
        return 1.0 - self.vt**2

    @property
    def xc(self) -> np.ndarray:
        return self.panels.xc

    @property
    def lower(self) -> np.ndarray:
        """Boolean mask of the lower-surface panels (first half of the clockwise contour)."""
        return np.arange(self.panels.n) < self.panels.n // 2

    @property
    def cl(self) -> float:
        """Kutta-Joukowski: Cl = 2 Gamma / (V_inf c), Gamma = gamma * perimeter."""
        return float(2.0 * self.gamma * self.panels.length.sum())

    @property
    def cl_pressure(self) -> float:
        """Lift from integrating the surface pressure (cross-check of ``cl``)."""
        nx, ny = self.panels.normal
        fx = np.sum(-self.cp * self.panels.length * nx)
        fy = np.sum(-self.cp * self.panels.length * ny)
        a = np.radians(self.alpha_deg)
        return float(fy * np.cos(a) - fx * np.sin(a))

    def cm(self, x_ref: float = 0.25, y_ref: float = 0.0) -> float:
        """Pitching moment coefficient about (x_ref, y_ref), positive nose-up.

        The force on a panel is -Cp L n with n the outward normal. A nose-up moment is clockwise
        in x-right, y-up axes, i.e. minus the z component of r x F.
        """
        nx, ny = self.panels.normal
        dx, dy = self.panels.xc - x_ref, self.panels.yc - y_ref
        L = self.panels.length
        return float(np.sum(self.cp * L * (dx * ny - dy * nx)))

    @property
    def cm_c4(self) -> float:
        return self.cm(0.25, 0.0)


def influence(p: Panels) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Velocity induced at every control point by every panel, per unit strength (global axes).

    Returns (us, vs, uv, vv), each (N, N): [i, j] is the velocity at control point i due to a unit
    source (us, vs) or unit vortex (uv, vv) distribution on panel j.
    """
    phi = p.angle
    c, s = np.cos(phi)[None, :], np.sin(phi)[None, :]
    dx = p.xc[:, None] - p.x1[None, :]
    dy = p.yc[:, None] - p.y1[None, :]
    xp = dx * c + dy * s
    yp = -dx * s + dy * c
    L = p.length[None, :]
    r1 = np.hypot(xp, yp)
    r2 = np.hypot(xp - L, yp)
    th1 = np.arctan2(yp, xp)
    th2 = np.arctan2(yp, xp - L)
    diag = np.eye(p.n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ups = np.where(diag, 0.0, np.log(r1 / r2) / (2 * np.pi))
    vps = np.where(diag, 0.5, (th2 - th1) / (2 * np.pi))
    upv, vpv = vps, -ups
    us, vs = ups * c - vps * s, ups * s + vps * c
    uv, vv = upv * c - vpv * s, upv * s + vpv * c
    return us, vs, uv, vv


def solve(X: np.ndarray, Y: np.ndarray, alpha_deg: float) -> Solution:
    """Solve the panel method for the closed clockwise contour (X, Y) at ``alpha_deg``."""
    p = Panels.from_nodes(X, Y)
    n = p.n
    a = np.radians(alpha_deg)
    nx, ny = p.normal
    tx, ty = p.tangent
    us, vs, uv, vv = influence(p)

    st = us * tx[:, None] + vs * ty[:, None]  # tangential, sources
    vt = uv * tx[:, None] + vv * ty[:, None]  # tangential, vortices
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = us * nx[:, None] + vs * ny[:, None]
    A[:n, n] = np.sum(uv * nx[:, None] + vv * ny[:, None], axis=1)
    A[n, :n] = st[0] + st[n - 1]
    A[n, n] = np.sum(vt[0] + vt[n - 1])
    rhs = np.empty(n + 1)
    rhs[:n] = -(np.cos(a) * nx + np.sin(a) * ny)
    rhs[n] = -(np.cos(a) * (tx[0] + tx[n - 1]) + np.sin(a) * (ty[0] + ty[n - 1]))

    sol = np.linalg.solve(A, rhs)
    sigma, gamma = sol[:n], float(sol[n])
    v_t = (np.cos(a) * tx + np.sin(a) * ty) + st @ sigma + gamma * vt.sum(axis=1)
    return Solution(p, float(alpha_deg), sigma, gamma, v_t)


def system(X: np.ndarray, Y: np.ndarray, alpha_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The (N+1)x(N+1) matrix and right-hand side, as written out by hand in section 1."""
    p = Panels.from_nodes(X, Y)
    n = p.n
    a = np.radians(alpha_deg)
    nx, ny = p.normal
    tx, ty = p.tangent
    us, vs, uv, vv = influence(p)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = us * nx[:, None] + vs * ny[:, None]
    A[:n, n] = np.sum(uv * nx[:, None] + vv * ny[:, None], axis=1)
    A[n, :n] = (us * tx[:, None] + vs * ty[:, None])[[0, n - 1]].sum(axis=0)
    A[n, n] = np.sum((uv * tx[:, None] + vv * ty[:, None])[[0, n - 1]])
    rhs = np.empty(n + 1)
    rhs[:n] = -(np.cos(a) * nx + np.sin(a) * ny)
    rhs[n] = -(np.cos(a) * (tx[0] + tx[n - 1]) + np.sin(a) * (ty[0] + ty[n - 1]))
    return A, rhs
