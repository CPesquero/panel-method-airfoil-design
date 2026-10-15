import numpy as np
import pytest

import panelfoil as pf


def _solve_reference(X, Y, alfa_deg):
    """Element-by-element solver of the original notebook (``resolver_perfil``), moment sign fixed."""
    alfa = np.radians(alfa_deg)
    N = len(X) - 1
    x1, y1, x2, y2 = X[:-1], Y[:-1], X[1:], Y[1:]
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    phi = np.arctan2(y2 - y1, x2 - x1)
    nx, ny = -np.sin(phi), np.cos(phi)
    tx, ty = np.cos(phi), np.sin(phi)
    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)
    st, vt = np.zeros((N, N)), np.zeros((N, N))
    rhs[:N] = -(np.cos(alfa) * nx + np.sin(alfa) * ny)
    for i in range(N):
        for j in range(N):
            dx, dy = xc[i] - x1[j], yc[i] - y1[j]
            xp = dx * np.cos(phi[j]) + dy * np.sin(phi[j])
            yp = -dx * np.sin(phi[j]) + dy * np.cos(phi[j])
            r1, r2 = np.sqrt(xp**2 + yp**2), np.sqrt((xp - L[j]) ** 2 + yp**2)
            th1, th2 = np.arctan2(yp, xp), np.arctan2(yp, xp - L[j])
            if i == j:
                ups, vps, upv, vpv = 0, 0.5, 0.5, 0
            else:
                ups, vps = (0.5 / np.pi) * np.log(r1 / r2), (th2 - th1) / (2 * np.pi)
                upv, vpv = vps, -ups
            us, vs = ups * np.cos(phi[j]) - vps * np.sin(phi[j]), ups * np.sin(phi[j]) + vps * np.cos(phi[j])
            uv, vv = upv * np.cos(phi[j]) - vpv * np.sin(phi[j]), upv * np.sin(phi[j]) + vpv * np.cos(phi[j])
            A[i, j] = us * nx[i] + vs * ny[i]
            A[i, N] += uv * nx[i] + vv * ny[i]
            st[i, j], vt[i, j] = us * tx[i] + vs * ty[i], uv * tx[i] + vv * ty[i]
    rhs[N] = -(np.cos(alfa) * (tx[0] + tx[N - 1]) + np.sin(alfa) * (ty[0] + ty[N - 1]))
    A[N, :-1] = st[0, :] + st[N - 1, :]
    A[N, N] = np.sum(vt[0, :] + vt[N - 1, :])
    sol = np.linalg.solve(A, rhs)
    gamma = sol[-1]
    Vt = (np.cos(alfa) * tx + np.sin(alfa) * ty) + st @ sol[:-1] + gamma * vt.sum(axis=1)
    Cp = 1 - Vt**2
    return 2 * gamma * L.sum(), np.sum(Cp * L * ((xc - 0.25) * ny - yc * nx)), Cp


@pytest.mark.parametrize("n", [20, 56, 100])
def test_vectorized_matches_original_loop(n):
    X, Y = pf.BASELINE.coordinates(n)
    s = pf.solve(X, Y, 3.1)
    cl, cm, cp = _solve_reference(X, Y, 3.1)
    assert s.cl == pytest.approx(cl, rel=1e-12)
    assert s.cm_c4 == pytest.approx(cm, rel=1e-10)
    np.testing.assert_allclose(s.cp, cp, rtol=1e-10, atol=1e-12)


def test_report_values_n100():
    s = pf.solve(*pf.BASELINE.coordinates(100), pf.ALPHA_DEG)
    assert s.cl == pytest.approx(0.7442, abs=1e-4)
    assert s.cm_c4 == pytest.approx(-0.0598, abs=1e-4)


def test_report_values_nopt():
    s = pf.solve(*pf.BASELINE.coordinates(56), pf.ALPHA_DEG)
    assert s.cl == pytest.approx(0.7384, abs=1e-4)
    assert s.cm_c4 == pytest.approx(-0.0584, abs=1e-4)


def test_agreement_with_xfoil():
    s = pf.solve(*pf.BASELINE.coordinates(100), pf.ALPHA_DEG)
    assert abs(s.cl / pf.REF_NACA3217["cl"] - 1) < 0.01
    assert abs(s.cm_c4 / pf.REF_NACA3217["cm_c4"] - 1) < 0.10


def test_positive_camber_gives_nose_down_moment():
    for code in ("2412", "3217", "4114"):
        s = pf.solve(*pf.Naca4.from_code(code).coordinates(80), 0.0)
        assert s.cm_c4 < 0


def test_symmetric_airfoil_zero_lift_and_moment():
    X, Y = pf.Naca4.from_code("0012").coordinates(100)
    s = pf.solve(X, Y, 0.0)
    assert abs(s.cl) < 1e-10
    assert abs(s.cm_c4) < 1e-3


def test_lift_slope_close_to_thin_airfoil_theory():
    # thickness raises the inviscid slope slightly above 2 pi
    X, Y = pf.Naca4.from_code("0006").coordinates(160)
    slope = pf.solve(X, Y, 4.0).cl / np.radians(4.0)
    assert 2 * np.pi < slope < 2 * np.pi * 1.10


def test_circulation_and_pressure_lift_agree():
    s = pf.solve(*pf.BASELINE.coordinates(140), pf.ALPHA_DEG)
    assert s.cl_pressure == pytest.approx(s.cl, rel=0.02)


def test_kutta_condition_holds():
    s = pf.solve(*pf.BASELINE.coordinates(100), pf.ALPHA_DEG)
    assert s.vt[0] + s.vt[-1] == pytest.approx(0.0, abs=1e-12)


def test_normals_point_outward():
    p = pf.Panels.from_nodes(*pf.BASELINE.coordinates(60))
    nx, ny = p.normal
    # sum of n * L only misses the small open trailing-edge gap of the NACA thickness law
    assert np.allclose([np.sum(nx * p.length), np.sum(ny * p.length)], 0, atol=5e-3)
    cx, cy = np.mean(p.xc), np.mean(p.yc)
    assert np.mean(nx * (p.xc - cx) + ny * (p.yc - cy)) > 0


def test_wedge_hand_calculation():
    X, Y = pf.wedge()
    A, rhs = pf.system(X, Y, 3.1)
    np.testing.assert_allclose(rhs, [0.054, 0.360, -0.339, 0.058], atol=1e-3)
    assert A[3, 3] == pytest.approx(1.859, abs=1e-3)
    s = pf.solve(X, Y, 3.1)
    assert s.gamma == pytest.approx(0.211, abs=1e-3)
    assert s.cl == pytest.approx(0.873, abs=1e-3)
