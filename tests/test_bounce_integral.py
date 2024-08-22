"""Test bounce integral methods."""

from functools import partial

import numpy as np
import pytest
from jax import grad
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import chebgauss, chebweight
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.interpolate import CubicHermiteSpline
from scipy.special import ellipe, ellipkm1, roots_chebyu
from tests.test_plotting import tol_1d

from desc.backend import jnp
from desc.compute.utils import dot
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import get_rtz_grid
from desc.examples import get
from desc.grid import LinearGrid
from desc.integrals.bounce_integral import (
    _get_extrema,
    _interp_to_argmin_B_hard,
    _interp_to_argmin_B_soft,
    bounce_integral,
    bounce_points,
    filter_bounce_points,
    get_pitch,
    plot_field_line,
    required_names,
)
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    leggausslob,
    tanh_sinh,
)


class TestBouncePoints:
    """Test that bounce points are computed correctly."""

    @staticmethod
    @pytest.mark.unit
    def test_bp1_first():
        """Test that bounce points are computed correctly."""
        start = np.pi / 3
        end = 6 * np.pi
        knots = np.linspace(start, end, 5)
        B = CubicHermiteSpline(knots, np.cos(knots), -np.sin(knots))
        pitch = 2.0
        intersect = B.solve(1 / pitch, extrapolate=False)
        bp1, bp2 = bounce_points(pitch, knots, B.c, B.derivative().c, check=True)
        bp1, bp2 = filter_bounce_points(bp1, bp2)
        assert bp1.size and bp2.size
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    @staticmethod
    @pytest.mark.unit
    def test_bp2_first():
        """Test that bounce points are computed correctly."""
        start = -3 * np.pi
        end = -start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch = 2.0
        intersect = B.solve(1 / pitch, extrapolate=False)
        bp1, bp2 = bounce_points(pitch, k, B.c, B.derivative().c, check=True)
        bp1, bp2 = filter_bounce_points(bp1, bp2)
        assert bp1.size and bp2.size
        np.testing.assert_allclose(bp1, intersect[1:-1:2])
        np.testing.assert_allclose(bp2, intersect[0::2][1:])

    @staticmethod
    @pytest.mark.unit
    def test_bp1_before_extrema():
        """Test that bounce points are computed correctly."""
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[3] + 1e-13
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = filter_bounce_points(bp1, bp2)
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[1], 1.982767, rtol=1e-6)
        np.testing.assert_allclose(bp1, intersect[[1, 2]], rtol=1e-6)
        # intersect array could not resolve double root as single at index 2,3
        np.testing.assert_allclose(intersect[2], intersect[3], rtol=1e-6)
        np.testing.assert_allclose(bp2, intersect[[3, 4]], rtol=1e-6)

    @staticmethod
    @pytest.mark.unit
    def test_bp2_before_extrema():
        """Test that bounce points are computed correctly."""
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 4,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 4,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2]
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = filter_bounce_points(bp1, bp2)
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[[0, -2]])
        np.testing.assert_allclose(bp2, intersect[[1, -1]])

    @staticmethod
    @pytest.mark.unit
    def test_extrema_first_and_before_bp1():
        """Test that bounce points are computed correctly."""
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 20,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 20,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2] - 1e-13
        bp1, bp2 = bounce_points(
            pitch, k[2:], B.c[:, 2:], B_z_ra.c[:, 2:], check=True, plot=False
        )
        plot_field_line(B, pitch, bp1, bp2, start=k[2])
        bp1, bp2 = filter_bounce_points(bp1, bp2)
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], 0.835319, rtol=1e-6)
        intersect = intersect[intersect >= k[2]]
        np.testing.assert_allclose(bp1, intersect[[0, 2, 4]], rtol=1e-6)
        np.testing.assert_allclose(bp2, intersect[[0, 3, 5]], rtol=1e-6)

    @staticmethod
    @pytest.mark.unit
    def test_extrema_first_and_before_bp2():
        """Test that bounce points are computed correctly."""
        start = -1.2 * np.pi
        end = -2 * start + 1
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 10,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 10,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[1] + 1e-13
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = filter_bounce_points(bp1, bp2)
        assert bp1.size and bp2.size
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], -0.671904, rtol=1e-6)
        np.testing.assert_allclose(bp1, intersect[[0, 3, 5]], rtol=1e-5)
        # intersect array could not resolve double root as single at index 0,1
        np.testing.assert_allclose(intersect[0], intersect[1], rtol=1e-5)
        np.testing.assert_allclose(bp2, intersect[[2, 4, 6]], rtol=1e-5)


class TestBounceQuadrature:
    """Test bounce quadrature accuracy."""

    @staticmethod
    def _mod_cheb_gauss(deg):
        x, w = chebgauss(deg)
        w /= chebweight(x)
        return x, w

    @staticmethod
    def _mod_chebu_gauss(deg):
        x, w = roots_chebyu(deg)
        w *= chebweight(x)
        return x, w

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "is_strong, quad, automorphism",
        [
            (True, tanh_sinh(40), None),
            (True, leggauss(25), "default"),
            (False, tanh_sinh(20), None),
            (False, leggausslob(10), "default"),
            # sin automorphism still helps out chebyshev quadrature
            (True, _mod_cheb_gauss(30), "default"),
            (False, _mod_chebu_gauss(10), "default"),
        ],
    )
    def test_bounce_quadrature(self, is_strong, quad, automorphism):
        """Test bounce integral matches elliptic integrals."""
        p = 1e-4
        m = 1 - p
        # Some prime number that doesn't appear anywhere in calculation.
        # Ensures no lucky cancellation occurs from this test case since otherwise
        # (bp2 - bp1) / pi = pi / (bp2 - bp1) which could mask errors since pi
        # appears often in transformations.
        v = 7
        bp1 = -np.pi / 2 * v
        bp2 = -bp1
        knots = np.linspace(bp1, bp2, 50)
        pitch = 1 + 50 * jnp.finfo(jnp.array(1.0).dtype).eps
        b = np.clip(np.sin(knots / v) ** 2, 1e-7, 1)
        db = np.sin(2 * knots / v) / v
        data = {"B^zeta": b, "B^zeta_z|r,a": db, "|B|": b, "|B|_z|r,a": db}

        if is_strong:
            integrand = lambda B, pitch: 1 / jnp.sqrt(1 - pitch * m * B)
            truth = v * 2 * ellipkm1(p)
        else:
            integrand = lambda B, pitch: jnp.sqrt(1 - pitch * m * B)
            truth = v * 2 * ellipe(m)
        kwargs = {}
        if automorphism != "default":
            kwargs["automorphism"] = automorphism
        bounce_integrate, _ = bounce_integral(data, knots, quad, check=True, **kwargs)
        result = bounce_integrate(integrand, [], pitch)
        assert np.count_nonzero(result) == 1
        np.testing.assert_allclose(np.sum(result), truth, rtol=1e-4)


@pytest.mark.unit
def test_bounce_integral_checks():
    """Test that all the internal correctness checks pass for real example."""

    def numerator(g_zz, B, pitch):
        f = (1 - pitch * B / 2) * g_zz
        return f / jnp.sqrt(1 - pitch * B)

    def denominator(B, pitch):
        return 1 / jnp.sqrt(1 - pitch * B)

    # Suppose we want to compute a bounce average of the function
    # f(ℓ) = (1 − λ|B|/2) * g_zz, where g_zz is the squared norm of the
    # toroidal basis vector on some set of field lines specified by (ρ, α)
    # coordinates. This is defined as
    # [∫ f(ℓ) / √(1 − λ|B|) dℓ] / [∫ 1 / √(1 − λ|B|) dℓ]
    eq = get("HELIOTRON")
    # Clebsch-Type field-line coordinates ρ, α, ζ.
    rho = np.linspace(0.1, 1, 6)
    alpha = np.array([0])
    knots = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    grid = get_rtz_grid(
        eq, rho, alpha, knots, coordinates="raz", period=(np.inf, 2 * np.pi, np.inf)
    )
    data = eq.compute(
        required_names() + ["min_tz |B|", "max_tz |B|", "g_zz"], grid=grid
    )
    bounce_integrate, spline = bounce_integral(
        data, knots, check=True, plot=False, quad=leggauss(3)
    )
    pitch = get_pitch(
        grid.compress(data["min_tz |B|"]), grid.compress(data["max_tz |B|"]), 10
    )
    # You can also plot the field line by uncommenting the following line.
    # Useful to see if the knot density was sufficient to reconstruct the field line.
    # _, _ = bounce_points(pitch, **spline, check=True, num=50000) # noqa: E800
    num = bounce_integrate(numerator, data["g_zz"], pitch)
    den = bounce_integrate(denominator, [], pitch)
    avg = num / den

    # Sum all bounce integrals across each particular field line.
    avg = np.nansum(avg, axis=-1)
    assert np.count_nonzero(avg)
    # Split the resulting data by field line.
    avg = avg.reshape(pitch.shape[0], rho.size, alpha.size)
    # The sum stored at index i, j
    i, j = 0, 0
    print(avg[:, i, j])
    # is the summed bounce average among wells along the field line with nodes
    # given in Clebsch-Type field-line coordinates ρ, α, ζ
    raz_grid = grid.source_grid
    nodes = raz_grid.nodes.reshape(rho.size, alpha.size, -1, 3)
    print(nodes[i, j])
    # for the pitch values stored in
    pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
    print(pitch[:, i, j])


@pytest.mark.unit
def test_get_extrema():
    """Test computation of extrema of |B|."""
    start = -np.pi
    end = -2 * start
    k = np.linspace(start, end, 5)
    B = CubicHermiteSpline(
        k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
    )
    B_z_ra = B.derivative()
    extrema, B_extrema = _get_extrema(k, B.c, B_z_ra.c)
    mask = ~np.isnan(extrema)
    extrema, B_extrema = extrema[mask], B_extrema[mask]
    idx = np.argsort(extrema)

    extrema_scipy = np.sort(B_z_ra.roots(extrapolate=False))
    B_extrema_scipy = B(extrema_scipy)
    assert extrema.size == extrema_scipy.size
    np.testing.assert_allclose(extrema[idx], extrema_scipy)
    np.testing.assert_allclose(B_extrema[idx], B_extrema_scipy)


@pytest.mark.unit
@pytest.mark.parametrize("func", [_interp_to_argmin_B_soft, _interp_to_argmin_B_hard])
def test_interp_to_argmin_B(func):
    """Test argmin interpolation."""  # noqa: D202

    # Test functions chosen with purpose; don't change unless plotted and compared.
    def f(z):
        return np.cos(3 * z) * np.sin(2 * np.cos(z)) + np.cos(1.2 * z)

    def B(z):
        return np.sin(3 * z) * np.cos(1 / (1 + z)) * np.cos(z**2) * z

    def dB_dz(z):
        return (
            3 * z * np.cos(3 * z) * np.cos(z**2) * np.cos(1 / (1 + z))
            - 2 * z**2 * np.sin(3 * z) * np.sin(z**2) * np.cos(1 / (1 + z))
            + z * np.sin(3 * z) * np.sin(1 / (1 + z)) * np.cos(z**2) / (1 + z) ** 2
            + np.sin(3 * z) * np.cos(z**2) * np.cos(1 / (1 + z))
        )

    zeta = np.linspace(0, 3 * np.pi, 175)
    _, spline = bounce_integral(
        {
            "B^zeta": np.ones_like(zeta),
            "B^zeta_z|r,a": np.ones_like(zeta),
            "|B|": B(zeta),
            "|B|_z|r,a": dB_dz(zeta),
        },
        zeta,
    )
    argmin = 5.61719
    np.testing.assert_allclose(
        f(argmin),
        func(
            f(zeta),
            bp1=np.array(0, ndmin=3),
            bp2=np.array(2 * np.pi, ndmin=3),
            **spline,
            method="cubic",
        ),
        rtol=1e-3,
    )


@partial(np.vectorize, excluded={0})
def _adaptive_elliptic(integrand, k):
    a = 0
    b = 2 * np.arcsin(k)
    return integrate.quad(integrand, a, b, args=(k,), points=b)[0]


def _fixed_elliptic(integrand, k, deg):
    k = np.atleast_1d(k)
    a = np.zeros_like(k)
    b = 2 * np.arcsin(k)
    x, w = leggauss(deg)
    w = w * grad_automorphism_sin(x)
    x = automorphism_sin(x)
    Z = bijection_from_disc(x, a[..., np.newaxis], b[..., np.newaxis])
    k = k[..., np.newaxis]
    quad = np.dot(integrand(Z, k), w) * grad_bijection_from_disc(a, b)
    return quad


def _elliptic_incomplete(k2):
    K_integrand = lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * (k / 4)
    E_integrand = lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) / (k * 4)
    # Scipy's elliptic integrals are broken.
    # https://github.com/scipy/scipy/issues/20525.
    k = np.sqrt(k2)
    K = _adaptive_elliptic(K_integrand, k)
    E = _adaptive_elliptic(E_integrand, k)
    # Make sure scipy's adaptive quadrature is not broken.
    np.testing.assert_allclose(K, _fixed_elliptic(K_integrand, k, 10))
    np.testing.assert_allclose(E, _fixed_elliptic(E_integrand, k, 10))

    I_0 = 4 / k * K
    I_1 = 4 * k * E
    I_2 = 16 * k * E
    I_3 = 16 * k / 9 * (2 * (-1 + 2 * k2) * E - (-1 + k2) * K)
    I_4 = 16 * k / 3 * ((-1 + 2 * k2) * E - 2 * (-1 + k2) * K)
    I_5 = 32 * k / 30 * (2 * (1 - k2 + k2**2) * E - (1 - 3 * k2 + 2 * k2**2) * K)
    I_6 = 4 / k * (2 * k2 * E + (1 - 2 * k2) * K)
    I_7 = 2 * k / 3 * ((-2 + 4 * k2) * E - 4 * (-1 + k2) * K)
    # Check for math mistakes.
    np.testing.assert_allclose(
        I_2,
        _adaptive_elliptic(
            lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * Z * np.sin(Z), k
        ),
    )
    np.testing.assert_allclose(
        I_3,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * Z * np.sin(Z), k
        ),
    )
    np.testing.assert_allclose(
        I_4,
        _adaptive_elliptic(
            lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.sin(Z) ** 2, k
        ),
    )
    np.testing.assert_allclose(
        I_5,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.sin(Z) ** 2, k
        ),
    )
    # scipy fails
    np.testing.assert_allclose(
        I_6,
        _fixed_elliptic(
            lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z),
            k,
            deg=10,
        ),
    )
    np.testing.assert_allclose(
        I_7,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z), k
        ),
    )
    return I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7


def _drift_analytic(data):
    """Compute analytic approximation for bounce-averaged binormal drift."""
    B = data["|B|"] / data["B ref"]
    B0 = np.mean(B)
    # epsilon should be changed to dimensionless, and computed in a way that
    # is independent of normalization length scales, like "effective r/R0".
    epsilon = data["a"] * data["rho"]  # Aspect ratio of the flux surface.
    np.testing.assert_allclose(epsilon, 0.05)
    theta_PEST = data["alpha"] + data["iota"] * data["zeta"]
    # same as 1 / (1 + epsilon cos(theta)) assuming epsilon << 1
    B_analytic = B0 * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(B, B_analytic, atol=3e-3)

    gradpar = data["a"] * data["B^zeta"] / data["|B|"]
    # This method of computing G0 suggests a fixed point iteration.
    G0 = data["a"]
    gradpar_analytic = G0 * (1 - epsilon * np.cos(theta_PEST))
    gradpar_theta_analytic = data["iota"] * gradpar_analytic
    G0 = np.mean(gradpar_theta_analytic)
    np.testing.assert_allclose(gradpar, gradpar_analytic, atol=5e-3)

    # Comparing coefficient calculation here with coefficients from compute/_metric
    normalization = -np.sign(data["psi"]) * data["B ref"] * data["a"] ** 2
    cvdrift = data["cvdrift"] * normalization
    gbdrift = data["gbdrift"] * normalization
    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * data["|B|"] ** 2)
    alpha_MHD = -0.5 * dPdrho / data["iota"] ** 2
    gds21 = (
        -np.sign(data["iota"])
        * data["shear"]
        * dot(data["grad(psi)"], data["grad(alpha)"])
        / data["B ref"]
    )
    gds21_analytic = -data["shear"] * (
        data["shear"] * theta_PEST - alpha_MHD / B**4 * np.sin(theta_PEST)
    )
    gds21_analytic_low_order = -data["shear"] * (
        data["shear"] * theta_PEST - alpha_MHD / B0**4 * np.sin(theta_PEST)
    )
    np.testing.assert_allclose(gds21, gds21_analytic, atol=2e-2)
    np.testing.assert_allclose(gds21, gds21_analytic_low_order, atol=2.7e-2)

    fudge_1 = 0.19
    gbdrift_analytic = fudge_1 * (
        -data["shear"]
        + np.cos(theta_PEST)
        - gds21_analytic / data["shear"] * np.sin(theta_PEST)
    )
    gbdrift_analytic_low_order = fudge_1 * (
        -data["shear"]
        + np.cos(theta_PEST)
        - gds21_analytic_low_order / data["shear"] * np.sin(theta_PEST)
    )
    fudge_2 = 0.07
    cvdrift_analytic = gbdrift_analytic + fudge_2 * alpha_MHD / B**2
    cvdrift_analytic_low_order = (
        gbdrift_analytic_low_order + fudge_2 * alpha_MHD / B0**2
    )
    np.testing.assert_allclose(gbdrift, gbdrift_analytic, atol=1e-2)
    np.testing.assert_allclose(cvdrift, cvdrift_analytic, atol=2e-2)
    np.testing.assert_allclose(gbdrift, gbdrift_analytic_low_order, atol=1e-2)
    np.testing.assert_allclose(cvdrift, cvdrift_analytic_low_order, atol=2e-2)

    pitch = get_pitch(np.min(B), np.max(B), 100)[1:]
    k2 = 0.5 * ((1 - pitch * B0) / (epsilon * pitch * B0) + 1)
    I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7 = _elliptic_incomplete(k2)
    y = np.sqrt(2 * epsilon * pitch * B0)
    I_0, I_2, I_4, I_6 = map(lambda I: I / y, (I_0, I_2, I_4, I_6))
    I_1, I_3, I_5, I_7 = map(lambda I: I * y, (I_1, I_3, I_5, I_7))

    drift_analytic_num = (
        fudge_2 * alpha_MHD / B0**2 * I_1
        - 0.5
        * fudge_1
        * (
            data["shear"] * (I_0 + I_1 - I_2 - I_3)
            + alpha_MHD / B0**4 * (I_4 + I_5)
            - (I_6 + I_7)
        )
    ) / G0
    drift_analytic_den = I_0 / G0
    drift_analytic = drift_analytic_num / drift_analytic_den
    return drift_analytic, cvdrift, gbdrift, pitch


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_drift():
    """Test bounce-averaged drift with analytical expressions."""
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
    psi_boundary = eq.Psi / (2 * np.pi)
    psi = 0.25 * psi_boundary
    rho = np.sqrt(psi / psi_boundary)
    np.testing.assert_allclose(rho, 0.5)

    # Make a set of nodes along a single fieldline.
    grid_fsa = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, sym=eq.sym, NFP=eq.NFP)
    data = eq.compute(["iota"], grid=grid_fsa)
    iota = grid_fsa.compress(data["iota"]).item()
    alpha = 0
    zeta = np.linspace(-np.pi / iota, np.pi / iota, (2 * eq.M_grid) * 4 + 1)
    grid = get_rtz_grid(
        eq,
        rho,
        alpha,
        zeta,
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
        iota=np.array([iota]),
    )
    data = eq.compute(
        required_names()
        + [
            "cvdrift",
            "gbdrift",
            "grad(psi)",
            "grad(alpha)",
            "shear",
            "iota",
            "psi",
            "a",
        ],
        grid=grid,
    )
    np.testing.assert_allclose(data["psi"], psi)
    np.testing.assert_allclose(data["iota"], iota)
    assert np.all(data["B^zeta"] > 0)
    B_ref = 2 * np.abs(psi_boundary) / data["a"] ** 2
    data["B ref"] = B_ref
    data["rho"] = rho
    data["alpha"] = alpha
    data["zeta"] = zeta
    data["psi"] = grid.compress(data["psi"])
    data["iota"] = grid.compress(data["iota"])
    data["shear"] = grid.compress(data["shear"])

    # Compute analytic approximation.
    drift_analytic, cvdrift, gbdrift, pitch = _drift_analytic(data)
    # Compute numerical result.
    bounce_integrate, _ = bounce_integral(
        data,
        knots=zeta,
        B_ref=B_ref,
        L_ref=data["a"],
        quad=leggauss(28),  # converges to absolute and relative tolerance of 1e-7
        check=True,
    )

    def integrand_num(cvdrift, gbdrift, B, pitch):
        g = jnp.sqrt(1 - pitch * B)
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    def integrand_den(B, pitch):
        return 1 / jnp.sqrt(1 - pitch * B)

    drift_numerical_num = bounce_integrate(
        integrand=integrand_num,
        f=[cvdrift, gbdrift],
        pitch=pitch[:, np.newaxis],
        num_well=1,
    )
    drift_numerical_den = bounce_integrate(
        integrand=integrand_den,
        f=[],
        pitch=pitch[:, np.newaxis],
        num_well=1,
        weight=np.ones(zeta.size),
    )
    drift_numerical = np.squeeze(drift_numerical_num / drift_numerical_den)
    msg = "There should be one bounce integral per pitch in this example."
    assert drift_numerical.size == drift_analytic.size, msg
    np.testing.assert_allclose(drift_numerical, drift_analytic, atol=5e-3, rtol=5e-2)

    _test_bounce_autodiff(
        bounce_integrate,
        integrand_num,
        f=[cvdrift, gbdrift],
        weight=np.ones(zeta.size),
    )

    fig, ax = plt.subplots()
    ax.plot(1 / pitch, drift_analytic)
    ax.plot(1 / pitch, drift_numerical)
    return fig


def _test_bounce_autodiff(bounce_integrate, integrand, **kwargs):
    """Make sure reverse mode AD works correctly on this algorithm."""

    def fun1(pitch):
        return jnp.sum(bounce_integrate(integrand, pitch=pitch, **kwargs))

    def fun2(pitch):
        return jnp.sum(bounce_integrate(integrand_grad, pitch=pitch, **kwargs))

    def integrand_grad(*args, **kwargs2):
        fun = jnp.vectorize(
            grad(integrand, -1), signature="()," * len(kwargs["f"]) + "(),()->()"
        )
        return fun(*args, *kwargs2.values())

    pitch = 1.0
    truth = 650  # Extrapolated from plot.
    assert np.isclose(grad(fun1)(pitch), truth, rtol=1e-3)
    # Make sure bounce points get differentiated too.
    result = fun2(pitch)
    assert np.isfinite(result) and not np.isclose(result, truth, rtol=1e-3)
