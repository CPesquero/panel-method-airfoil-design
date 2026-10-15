"""Every figure in the README and the notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import Naca4
from .solver import Panels, Solution, solve
from .studies import Convergence, Design
from .xfoil import CpDistribution, Polar

BLUE, NAVY, RED, GREY = "#2f6db5", "#0b2a5b", "#c8332f", "#8a8f98"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save(fig, path: str | Path | None):
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def airfoil(af: Naca4, n_panels: int = 100, path=None):
    X, Y = af.coordinates(n_panels)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(X, Y, "-", color=NAVY, lw=1.4)
    ax.plot(X, Y, ".", color=BLUE, ms=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"{af}, {n_panels} panels (cosine spacing)")
    return _save(fig, path)


def wedge(X: np.ndarray, Y: np.ndarray, path=None):
    p = Panels.from_nodes(X, Y)
    nx, ny = p.normal
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(X, Y, "-o", color=NAVY)
    ax.quiver(p.xc, p.yc, nx, ny, color=RED, scale=12, width=0.004)
    for k in range(p.n):
        ax.annotate(f"p{k + 1}", (p.xc[k], p.yc[k]), textcoords="offset points", xytext=(6, -12))
    for k, (x, y) in enumerate(zip(X[:-1], Y[:-1])):
        ax.annotate(f"N{k + 1}", (x, y), textcoords="offset points", xytext=(-4, 6))
    ax.set_aspect("equal")
    ax.set_ylim(-0.14, 0.27)
    ax.set_title("Three-panel wedge: clockwise contour, outward normals", pad=12)
    return _save(fig, path)


def cp_vs_xfoil(sol: Solution, ref: CpDistribution | None, title: str, path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = sol.lower
    ax.plot(sol.xc[lo], sol.cp[lo], color=BLUE, label="Panel method, lower")
    ax.plot(sol.xc[~lo], sol.cp[~lo], color=NAVY, label="Panel method, upper")
    if ref is not None:
        ax.plot(ref.x, ref.cp, "--", color=RED, lw=1.6, label="XFOIL (inviscid)")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def convergence(conv: Convergence, path=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(conv.n_panels, conv.cl, "o-", color=NAVY, label="$C_l(N)$")
    ax1.set_xlabel("Number of panels N")
    ax1.set_ylabel("$C_l$")
    ax2 = ax1.twinx()
    ax2.semilogy(conv.n_panels[1:], conv.rel_change, "s--", color=RED, label="relative change")
    ax2.axhline(conv.tol, color=GREY, ls=":", label=f"tolerance {conv.tol:.0%}")
    ax2.axvline(conv.n_opt, color=GREY, lw=0.8)
    ax2.set_ylabel("Relative change from previous mesh")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title(f"Mesh convergence, N_opt = {conv.n_opt}")
    return _save(fig, path)


def design_geometry(design: Design, n_panels: int = 100, path=None):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for af, style, color in [(design.base.airfoil, "--", RED), (design.best.airfoil, "-", NAVY)]:
        X, Y = af.coordinates(n_panels)
        ax.plot(X, Y, style, color=color, label=str(af))
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.legend(loc="upper right")
    ax.set_title("Baseline and redesigned sections")
    return _save(fig, path)


def design_cp(design: Design, n_panels: int, alpha_deg: float = 0.0, path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for af, style, c_lo, c_up in [
        (design.base.airfoil, "--", "#e8907a", RED),
        (design.best.airfoil, "-", BLUE, NAVY),
    ]:
        s = solve(*af.coordinates(n_panels), alpha_deg)
        lo = s.lower
        ax.plot(s.xc[lo], s.cp[lo], style, color=c_lo, label=f"{af}, lower")
        ax.plot(s.xc[~lo], s.cp[~lo], style, color=c_up, label=f"{af}, upper")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"Pressure distribution at α = {alpha_deg:g}°, N = {n_panels}")
    ax.legend()
    return _save(fig, path)


def polars(inviscid: Polar, viscous: Polar, panel: tuple | None = None, path=None):
    """``panel`` is an optional (alpha, cl, cm) tuple from this solver, drawn as markers."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(inviscid.alpha, inviscid.cl, "--", color=BLUE, label="XFOIL inviscid")
    a1.plot(viscous.alpha, viscous.cl, "-", color=NAVY, lw=2.2, label="XFOIL viscous, Re = 3·10⁶")
    a2.plot(inviscid.alpha, inviscid.cm, "--", color=BLUE, label="XFOIL inviscid")
    a2.plot(viscous.alpha, viscous.cm, "-", color=NAVY, lw=2.2, label="XFOIL viscous")
    if panel is not None:
        al, cl, cm = panel
        a1.plot(al, cl, "o", color=RED, ms=5, label="This panel method")
        a2.plot(al, cm, "o", color=RED, ms=5, label="This panel method")
    am, clm = viscous.cl_max
    label = f"$C_{{l,max}}$ = {clm:.2f} at {am:g}°"
    a1.annotate(label, (am, clm), xytext=(-110, 10), textcoords="offset points")
    a1.set_xlabel("α [°]")
    a1.set_ylabel("$C_l$")
    a1.set_title("Lift, NACA 4114")
    a1.legend(loc="lower right")
    a2.set_xlabel("α [°]")
    a2.set_ylabel("$C_{m,c/4}$")
    a2.set_title("Pitching moment, NACA 4114")
    a2.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)
