"""Reproduce every table and figure of the study.

Usage::

    panelfoil                     # tables, figures into ./figures
    panelfoil --no-figures        # tables only
    panelfoil --json out.json     # also write the numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__
from .geometry import ALPHA_DEG, BASELINE, wedge
from .solver import solve, system
from .studies import inverse_design, mesh_convergence
from .xfoil import POLAR_INVISCID, POLAR_VISCOUS, REF_NACA3217, load_cp, load_polar


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _err(value: float, ref: float) -> float:
    return (value - ref) / ref


def run(figures_dir: Path | None) -> dict:
    res: dict = {"version": __version__}
    if figures_dir is not None:
        from . import plots

        figures_dir.mkdir(parents=True, exist_ok=True)

        def fig(name: str) -> Path:
            return figures_dir / name

    # 1. hand calculation
    _rule("1. Three-panel wedge (hand calculation), alpha = 3.1 deg")
    Xw, Yw = wedge()
    A, rhs = system(Xw, Yw, ALPHA_DEG)
    sw = solve(Xw, Yw, ALPHA_DEG)
    with np.printoptions(precision=3, suppress=True):
        print("A =\n", A)
        print("I =", rhs)
    print(f"gamma = {sw.gamma:.4f}   Cl = {sw.cl:.4f}")
    res["wedge"] = {"A": A.round(4).tolist(), "I": rhs.round(4).tolist(), "gamma": sw.gamma, "cl": sw.cl}

    # 2. validation
    ref = REF_NACA3217
    cp_ref = load_cp()
    res["validation"] = {}
    for n in (100, None):
        if n is None:
            conv = mesh_convergence()
            n = conv.n_opt
            _rule("2.4 Mesh convergence")
            for k, (ni, cl) in enumerate(zip(conv.n_panels, conv.cl)):
                ch = f"{conv.rel_change[k - 1]:.2%}" if k else "-"
                print(f"N = {ni:4d}   Cl = {cl:.5f}   change = {ch}")
            print(f"N_opt = {n}")
            res["convergence"] = {
                "n": conv.n_panels.tolist(),
                "cl": conv.cl.tolist(),
                "rel_change": conv.rel_change.tolist(),
                "n_opt": n,
            }
            if figures_dir is not None:
                plots.convergence(conv, fig("convergence.png"))
        s = solve(*BASELINE.coordinates(n), ALPHA_DEG)
        _rule(f"2. {BASELINE}, alpha = {ALPHA_DEG} deg, N = {n} vs XFOIL (inviscid)")
        print(f"Cl     = {s.cl:.4f}   XFOIL {ref['cl']:.4f}   error {_err(s.cl, ref['cl']):+.2%}")
        print(f"Cm_c/4 = {s.cm_c4:.4f}   XFOIL {ref['cm_c4']:.4f}   error {_err(s.cm_c4, ref['cm_c4']):+.2%}")
        print(f"Cl from pressure integration = {s.cl_pressure:.4f}")
        res["validation"][str(n)] = {"cl": s.cl, "cm_c4": s.cm_c4, "cl_pressure": s.cl_pressure}
        if figures_dir is not None:
            plots.cp_vs_xfoil(s, cp_ref, f"{BASELINE}, α = {ALPHA_DEG}°, N = {n}", fig(f"cp_xfoil_n{n}.png"))

    # 3. inverse design
    n_opt = res["convergence"]["n_opt"]
    d = inverse_design(n_panels=n_opt)
    _rule(f"3. Inverse design (+20 % Cl at alpha = 0, N = {n_opt})")
    print(f"base   {d.base.airfoil}: Cl = {d.base.cl:.4f}   Cm_c/4 = {d.base.cm_c4:.4f}")
    print(f"target Cl = {d.target_cl:.4f} +/- 0.015   ({len(d.feasible)} feasible sections)")
    print(
        f"best   {d.best.airfoil}: Cl = {d.best.cl:.4f} ({d.cl_gain:+.1%})   Cm_c/4 = {d.best.cm_c4:.4f}"
        f" (change {d.cm_change:.1%})"
    )
    inv = load_polar(POLAR_INVISCID)
    cl_x, cm_x = inv.at(0.0)
    print(f"XFOIL inviscid {d.best.airfoil} at 0 deg: Cl = {cl_x:.4f}   Cm_c/4 = {cm_x:.4f}")
    res["design"] = {
        "base": str(d.base.airfoil),
        "best": str(d.best.airfoil),
        "cl_base": d.base.cl,
        "cm_base": d.base.cm_c4,
        "cl_best": d.best.cl,
        "cm_best": d.best.cm_c4,
        "cl_gain": d.cl_gain,
        "cm_change": d.cm_change,
        "xfoil_cl_best": cl_x,
        "xfoil_cm_best": cm_x,
    }

    # 4. viscous comparison (XFOIL)
    vis = load_polar(POLAR_VISCOUS)
    am, clm = vis.cl_max
    _rule(f"4. XFOIL, {d.best.airfoil}: inviscid vs Re = 3e6")
    print(f"viscous Cl_max = {clm:.3f} at alpha = {am:g} deg")
    al = np.arange(-4, 13, 2.0)
    pan = [solve(*d.best.airfoil.coordinates(n_opt), a) for a in al]
    pcl = np.array([p.cl for p in pan])
    pcm = np.array([p.cm_c4 for p in pan])
    xcl = np.interp(al, inv.alpha, inv.cl)
    print("alpha   Cl panel   Cl XFOIL inviscid")
    for a_, c1, c2 in zip(al, pcl, xcl):
        print(f"{a_:5.1f}   {c1:8.4f}   {c2:8.4f}")
    res["viscous"] = {"cl_max": clm, "alpha_cl_max": am}
    if figures_dir is not None:
        plots.airfoil(BASELINE, 100, fig("airfoil.png"))
        plots.wedge(Xw, Yw, fig("wedge.png"))
        plots.design_geometry(d, 100, fig("design_geometry.png"))
        plots.design_cp(d, 100, 0.0, fig("design_cp.png"))
        plots.polars(inv, vis, (al, pcl, pcm), fig("polars_naca4114.png"))
        print(f"\nfigures written to {figures_dir}/")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="panelfoil", description=__doc__.splitlines()[0])
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--json", type=Path, help="write the numbers to this file")
    ap.add_argument("--version", action="version", version=f"panelfoil {__version__}")
    args = ap.parse_args(argv)
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
    res = run(None if args.no_figures else args.figures_dir)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print(f"numbers written to {args.json}")


if __name__ == "__main__":
    main()
