import numpy as np
import pytest

import panelfoil as pf


def test_naca_codes():
    af = pf.Naca4.from_code("NACA 3217")
    assert (af.f, af.p, af.t) == (0.03, 0.2, 0.17)
    assert af.code == "3217"
    with pytest.raises(ValueError):
        pf.Naca4.from_code("23012")


def test_geometry_closed_and_clockwise():
    X, Y = pf.BASELINE.coordinates(100)
    assert len(X) == 101
    # the NACA 4-digit thickness law leaves a small trailing-edge gap (about 0.002 c per side here)
    assert X[0] == pytest.approx(X[-1], abs=1e-3)
    assert abs(Y[0] - Y[-1]) < 5e-3
    signed_area = 0.5 * np.sum(X[:-1] * Y[1:] - X[1:] * Y[:-1])
    assert signed_area < 0  # clockwise


def test_mesh_convergence_report():
    conv = pf.mesh_convergence()
    np.testing.assert_array_equal(conv.n_panels, [40, 48, 56, 68, 82, 98, 118, 142])
    assert conv.n_opt == 56


def test_inverse_design_report():
    d = pf.inverse_design(n_panels=56)
    assert d.base.cl == pytest.approx(0.3542, abs=1e-4)
    assert d.base.cm_c4 == pytest.approx(-0.0520, abs=1e-4)
    assert d.best.airfoil.code == "4114"
    assert d.best.cl == pytest.approx(0.4355, abs=1e-4)
    assert d.best.cm_c4 == pytest.approx(-0.0518, abs=1e-4)
    assert d.cl_gain == pytest.approx(0.23, abs=0.005)
    assert d.cm_change < 0.005


def test_optimized_section_against_xfoil():
    inv = pf.load_polar(pf.POLAR_INVISCID)
    cl_x, _ = inv.at(0.0)
    s = pf.solve(*pf.OPTIMIZED.coordinates(56), 0.0)
    assert s.cl == pytest.approx(cl_x, rel=0.01)


def test_xfoil_files():
    cp = pf.load_cp()
    assert cp.x.size == 160 and cp.cp.min() == pytest.approx(-1.5748)
    vis = pf.load_polar(pf.POLAR_VISCOUS)
    assert vis.cl_max == pytest.approx((17.0, 2.0006))
    inv = pf.load_polar(pf.POLAR_INVISCID)
    assert np.all(inv.cm < 0)
