"""Test bounce integral methods."""

import inspect
from functools import partial

import numpy as np
import pytest
from interpax import Akima1DInterpolator
from matplotlib import pyplot as plt

# TODO: can use the one from interpax once .solve() is implemented
from scipy.interpolate import CubicHermiteSpline
from scipy.special import ellipe, ellipk

from desc.backend import complex_sqrt, flatnonzero, fori_loop, put, root_scalar
from desc.compute.bounce_integral import (
    bounce_integral_map,
    bounce_points,
    pitch_of_extrema,
    poly_der,
    poly_int,
    poly_root,
    poly_val,
    take_mask,
)
from desc.compute.utils import dot, safediv
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import desc_grid_from_field_line_coords
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.objectives import (
    ObjectiveFromUser,
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile


@partial(np.vectorize, signature="(m)->()")
def _last_value(a):
    """Return the last non-nan value in ``a``."""
    a = np.ravel(a)[::-1]
    idx = np.squeeze(flatnonzero(~np.isnan(a), size=1, fill_value=0))
    return a[idx]


def _filter_not_nan(a):
    """Filter out nan from ``a`` while asserting nan is padded at right."""
    is_nan = np.isnan(a)
    assert np.array_equal(is_nan, np.sort(is_nan, axis=-1))
    return a[~is_nan]


@pytest.mark.unit
def test_mask_operations():
    """Test custom masked array operation."""
    rows = 5
    cols = 7
    a = np.random.rand(rows, cols)
    nan_idx = np.random.choice(rows * cols, size=(rows * cols) // 2, replace=False)
    a.ravel()[nan_idx] = np.nan
    taken = take_mask(a, ~np.isnan(a))
    last = _last_value(taken)
    for i in range(rows):
        desired = a[i, ~np.isnan(a[i])]
        assert np.array_equal(
            taken[i],
            np.pad(desired, (0, cols - desired.size), constant_values=np.nan),
            equal_nan=True,
        ), "take_mask has bugs."
        assert np.array_equal(last[i], desired[-1]), "flatnonzero has bugs."


@pytest.mark.unit
def test_reshape_convention():
    """Test the reshaping convention separates data across field lines."""
    rho = np.linspace(0, 1, 3)
    alpha = np.linspace(0, 2 * np.pi, 4)
    zeta = np.linspace(0, 6 * np.pi, 5)
    r, a, z = map(np.ravel, np.meshgrid(rho, alpha, zeta, indexing="ij"))
    # functions of zeta should separate along first two axes
    # since those are contiguous, this should work
    f = z.reshape(-1, zeta.size)
    for i in range(1, f.shape[0]):
        np.testing.assert_allclose(f[i - 1], f[i])
    # likewise for rho
    f = r.reshape(rho.size, -1)
    for i in range(1, f.shape[-1]):
        np.testing.assert_allclose(f[:, i - 1], f[:, i])
    # test final reshape of bounce integral result won't mix data
    f = (a**2 + z).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[0]):
        np.testing.assert_allclose(f[i - 1], f[i])
    f = (r**2 + z).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[1]):
        np.testing.assert_allclose(f[:, i - 1], f[:, i])
    f = (r**2 + a).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[-1]):
        np.testing.assert_allclose(f[..., i - 1], f[..., i])

    err_msg = "The ordering conventions are required for correctness."
    assert "P, S, N" in inspect.getsource(bounce_points), err_msg
    src = inspect.getsource(bounce_integral_map)
    assert "S, zeta.size" in src, err_msg
    assert "pitch.shape[0], rho.size, alpha.size" in src, err_msg
    src = inspect.getsource(desc_grid_from_field_line_coords)
    assert 'indexing="ij"' in src, err_msg
    assert 'meshgrid(rho, alpha, zeta, indexing="ij")' in src, err_msg


@pytest.mark.unit
def test_poly_root():
    """Test vectorized computation of cubic polynomial exact roots."""
    cubic = 4
    c = np.arange(-24, 24).reshape(cubic, 6, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    constant = np.broadcast_to(np.arange(c.shape[-1]), c.shape[1:])
    constant = np.stack([constant, constant])
    root = poly_root(c, constant, sort=True)

    for i in range(constant.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                d = c[-1, j, k] - constant[i, j, k]
                np.testing.assert_allclose(
                    actual=root[i, j, k],
                    desired=np.sort(np.roots([*c[:-1, j, k], d])),
                )

    c = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, -8, 12],
            [1, -6, 11, -6],
            [0, -6, 11, -2],
        ]
    )
    root = poly_root(c.T, sort=True, distinct=True)
    for j in range(c.shape[0]):
        unique_roots = np.unique(np.roots(c[j]))
        if j == 4:
            # There are only two distinct roots.
            unique_roots = unique_roots[[0, 1]]
        np.testing.assert_allclose(
            actual=_filter_not_nan(root[j]),
            desired=unique_roots,
            err_msg=str(j),
        )
    c = np.array([0, 1, -1, -8, 12])
    np.testing.assert_allclose(
        actual=_filter_not_nan(poly_root(c, sort=True, distinct=True)),
        desired=np.unique(np.roots(c)),
    )


@pytest.mark.unit
def test_poly_int():
    """Test vectorized computation of polynomial primitive."""
    quintic = 6
    c = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    constant = np.broadcast_to(np.arange(c.shape[-1]), c.shape[1:])
    primitive = poly_int(c, k=constant)
    for j in range(c.shape[1]):
        for k in range(c.shape[2]):
            np.testing.assert_allclose(
                actual=primitive[:, j, k],
                desired=np.polyint(c[:, j, k], k=constant[j, k]),
            )
    assert poly_int(c).shape == primitive.shape, "Failed broadcasting default k."


@pytest.mark.unit
def test_poly_der():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    c = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    derivative = poly_der(c)
    for j in range(c.shape[1]):
        for k in range(c.shape[2]):
            np.testing.assert_allclose(
                actual=derivative[:, j, k], desired=np.polyder(c[:, j, k])
            )


@pytest.mark.unit
def test_poly_val():
    """Test vectorized computation of polynomial evaluation."""

    def test(x, c):
        val = poly_val(x=x, c=c)
        for index in np.ndindex(c.shape[1:]):
            idx = (..., *index)
            np.testing.assert_allclose(
                actual=val[idx],
                desired=np.poly1d(c[idx])(x[idx]),
                err_msg=f"Failed with shapes {x.shape} and {c.shape}.",
            )

    quartic = 5
    c = np.arange(-60, 60).reshape(quartic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    x = np.linspace(0, 20, c.shape[1] * c.shape[2]).reshape(c.shape[1], c.shape[2])
    test(x, c)

    x = np.stack([x, x * 2], axis=0)
    x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
    # make sure broadcasting won't hide error in implementation
    assert np.unique(x.shape).size == x.ndim
    assert c.shape[1:] == x.shape[x.ndim - (c.ndim - 1) :]
    assert np.unique((c.shape[0],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
    test(x, c)

    # integrate piecewise polynomial and set constants to preserve continuity
    y = np.arange(2, 8)
    y = np.arange(y.prod()).reshape(*y)
    x = np.arange(y.shape[-1])
    a1d = Akima1DInterpolator(x, y, axis=-1)
    primitive = poly_int(a1d.c)
    # choose evaluation points at d just to match choice made in a1d.antiderivative()
    d = np.diff(x)
    # evaluate every spline at d
    k = poly_val(x=d, c=primitive)
    # don't want to use jax.ndarray.at[].add() in case jax is not installed
    primitive = np.array(primitive)
    primitive[-1, 1:] += np.cumsum(k, axis=-1)[:-1]
    np.testing.assert_allclose(primitive, a1d.antiderivative().c)


@pytest.mark.unit
def test_pitch_of_extrema():
    """Test that these pitch intersect extrema of |B|."""
    start = -np.pi
    end = -2 * start
    k = np.linspace(start, end, 5)
    B = CubicHermiteSpline(
        k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
    )
    B_z_ra = B.derivative()
    pitch_scipy = 1 / B(B_z_ra.roots(extrapolate=False))
    pitch = _filter_not_nan(pitch_of_extrema(k, B.c, B_z_ra.c))
    np.testing.assert_allclose(pitch, pitch_scipy)


@pytest.mark.unit
def test_bounce_points():
    """Test that bounce points are computed correctly."""

    def plot_field_line(B, pitch, start, end):
        # Can observe correctness of bounce points through this plot.
        fig, ax = plt.subplots()
        for knot in B.x:
            ax.axvline(x=knot, color="red", linestyle="--")
        z = np.linspace(start, end, 100)
        ax.plot(z, B(z), label=r"$\vert B \vert (\zeta)$")
        ax.plot(z, np.full(z.size, 1 / pitch), label=r"$1 / \lambda$")
        ax.set_xlabel(r"Field line $\zeta$")
        ax.set_ylabel("Tesla")
        ax.legend()
        plt.show()
        plt.close()

    def test_bp1_first(plot=False):
        start = np.pi / 3
        end = 6 * np.pi
        knots = np.linspace(start, end, 5)
        B = CubicHermiteSpline(knots, np.cos(knots), -np.sin(knots))
        pitch = 2
        if plot:
            plot_field_line(B, pitch, start, end)
        bp1, bp2 = bounce_points(knots, B.c, B.derivative().c, pitch, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    def test_bp2_first(plot=False):
        start = -3 * np.pi
        end = -start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch = 2
        if plot:
            plot_field_line(B, pitch, start, end)
        bp1, bp2 = bounce_points(k, B.c, B.derivative().c, pitch, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[1::2])
        np.testing.assert_allclose(bp2, intersect[0::2][1:])

    def test_bp1_before_extrema(plot=False):
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[3]
        if plot:
            plot_field_line(B, pitch, start, end)

        bp1, bp2 = bounce_points(k, B.c, B_z_ra.c, pitch, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[1], 1.9827671337414938)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[1]), bp1[1])
        np.testing.assert_allclose(bp1, intersect[[1, 2]])
        np.testing.assert_allclose(bp2, intersect[[2, 3]])

    def test_bp2_before_extrema(plot=False):
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
        if plot:
            plot_field_line(B, pitch, start, end)

        bp1, bp2 = bounce_points(k, B.c, B_z_ra.c, pitch, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[[0, -2]])
        np.testing.assert_allclose(bp2, intersect[[1, -1]])

    def test_extrema_first_and_before_bp1(plot=False):
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 20,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 20,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2]
        if plot:
            plot_field_line(B, pitch, k[2], end)

        bp1, bp2 = bounce_points(k[2:], B.c[:, 2:], B_z_ra.c[:, 2:], pitch, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], 0.8353192766102349)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[0]), bp1[0])
        intersect = intersect[intersect >= k[2]]
        np.testing.assert_allclose(bp1, intersect[[0, 1, 3]])
        np.testing.assert_allclose(bp2, intersect[[0, 2, 4]])

    def test_extrema_first_and_before_bp2(plot=False):
        start = -1.2 * np.pi
        end = -2 * start + 1
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 10,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 10,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[1]
        if plot:
            plot_field_line(B, pitch, start, end)

        bp1, bp2 = bounce_points(k, B.c, B_z_ra.c, pitch, check=True)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], -0.6719044147510538)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[0]), bp1[0])
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    # These are all the unique cases, if all tests pass then the bounce_points
    # should work correctly for all inputs. Pass in True to see plots.
    test_bp1_first()
    test_bp2_first()
    test_bp1_before_extrema()
    test_bp2_before_extrema()
    # In theory, this test should only pass if distinct=True when computing the
    # intersections in bounce points. However, we can get lucky due to floating
    # point errors, and it may also pass when distinct=False.
    test_extrema_first_and_before_bp1()
    test_extrema_first_and_before_bp2()


@pytest.mark.unit
def test_example_code_and_hairy_ball():
    """Test example code in bounce_integral docstring and ensure B does not vanish."""

    def integrand_num(g_zz, B, pitch):
        """Integrand in integral in numerator of bounce average."""
        f = (1 - pitch * B) * g_zz  # something arbitrary
        # When 1 - pitch * B is negative, we want g to evaluate as nan.
        # jnp.sqrt() will do this as desired, but np.sqrt() will give a runtime error.
        g = complex_sqrt(1 - pitch * B)  # typical to have this in denominator
        g = np.where(np.isclose(np.imag(g), 0), np.real(g), np.nan)
        return safediv(f, g, fill=np.nan)

    def integrand_den(B, pitch):
        """Integrand in integral in denominator of bounce average."""
        # When 1 - pitch * B is negative, we want g to evaluate as nan.
        # jnp.sqrt() will do this as desired, but np.sqrt() will give a runtime error.
        g = complex_sqrt(1 - pitch * B)  # typical to have this in denominator
        g = np.where(np.isclose(np.imag(g), 0), np.real(g), np.nan)
        return safediv(1, g, fill=np.nan)

    eq = get("HELIOTRON")
    rho = np.linspace(1e-12, 1, 6)
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 5)
    knots = np.linspace(0, 6 * np.pi, 20)

    bounce_integral, items = bounce_integral_map(eq, rho, alpha, knots)

    # start hairy ball test
    B = eq.compute("B", grid=items["grid_desc"], data=items["data"])["B"]
    assert not np.isclose(B, 0, atol=1e-19).any(), "B should never vanish."
    B = items["data"]["B"]
    assert not np.isclose(B, 0, atol=1e-19).any(), "B should never vanish."
    # end hairy ball test

    g_zz = eq.compute("g_zz", grid=items["grid_desc"], data=items["data"])["g_zz"]
    pitch = pitch_of_extrema(knots, items["B.c"], items["B_z_ra.c"])
    num = bounce_integral(integrand_num, g_zz, pitch)
    den = bounce_integral(integrand_den, [], pitch)
    average = num / den
    assert np.isfinite(average).any()

    # Now we can group the data by field line.
    average = average.reshape(pitch.shape[0], rho.size, alpha.size, -1)
    # The bounce averages stored at index i, j
    i, j = 0, 0
    print(average[:, i, j])
    # are the bounce averages along the field line with nodes
    nodes = items["grid_fl"].nodes.reshape(rho.size, alpha.size, -1, 3)
    print(nodes[i, j])
    # for the pitch values stored in
    pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
    print(pitch[:, i, j])
    # Some of these bounce averages will evaluate as nan.
    # You should filter out these nan values when computing stuff.
    average_sum_over_field_line = np.nansum(average, axis=-1)
    print(average_sum_over_field_line)
    assert not np.allclose(average_sum_over_field_line, 0)


# @pytest.mark.unit
def test_elliptic_integral_limit():
    """Test bounce integral matches elliptic integrals.

    In the limit of a low beta, large aspect ratio tokamak the bounce integral
    should converge to the elliptic integrals of the first kind.
    todo: would be nice to understand physics for why these are supposed
        to be proportional to bounce integral. Is this discussed in any book?
        Also, looking at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html
        Are we saying that in this limit, we expect that |B| ~ sin(t)^2, with m as the
        pitch angle? I assume that we want to add g_zz to the integrand in the
        definition of the function in the scipy documentation above,
        and after a change of variables the bounce points will be the endpoints of
        the integration.
        So this test will test whether the quadrature is accurate
        (and not whether the bounce points were accurate).

    """
    assert False, "Test not finished yet."
    L, M, N, NFP, sym = 6, 6, 6, 1, True
    surface = FourierRZToroidalSurface(
        R_lmn=[1.0, 0.1],
        Z_lmn=[0.0, -0.1],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
        sym=sym,
        NFP=NFP,
    )
    eq = Equilibrium(
        L=L,
        M=M,
        N=N,
        NFP=NFP,
        surface=surface,
        pressure=PowerSeriesProfile([1e2, 0, -1e2]),
        iota=PowerSeriesProfile([1, 0, 2]),
        Psi=1.0,
    )
    eq = solve_continuation_automatic(eq)[-1]

    def beta(grid, data):
        return data["<beta>_vol"]

    low_beta = 0.01
    # todo: error that objective function has no linear attribute?
    objective = ObjectiveFunction(
        (ObjectiveFromUser(fun=beta, eq=eq, target=low_beta),)
    )

    constraints = (*get_fixed_boundary_constraints(eq), get_equilibrium_objective(eq))
    opt = Optimizer("proximal-lsq-exact")
    eq, result = eq.optimize(
        objective=objective, constraints=constraints, optimizer=opt
    )
    print(result)

    rho = np.array([0.5])
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 10)
    knots = np.linspace(0, 6 * np.pi, 20)
    # TODO now compare result to elliptic integral
    bounce_integral, items = bounce_integral_map(eq, rho, alpha, knots, check=True)
    pitch = pitch_of_extrema(knots, items["B.c"], items["B_z_ra.c"])
    bp1, bp2 = bounce_points(knots, items["B.c"], items["B_z_ra.c"], pitch)


@pytest.mark.unit
def test_bounce_averaged_drifts():
    """Test bounce-averaged drift with analytical expressions.

    Calculate bounce-averaged drifts using the bounce-average routine and
    compare it with the analytical expression
    # Note 1: This test can be merged with the elliptic integral test as
    we do calculate elliptic integrals here
    # Note 2: Remove tests/test_equilibrium :: test_shifted_circle_geometry
    # once all the epsilons and Gammas have been implemented and tested
    """
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")

    psi = 0.25  # normalized psi
    rho = np.sqrt(psi)
    alpha = np.array([0])
    data_eq = eq.compute(["iota", "iota_r", "a", "rho", "psi"])

    iotas = np.interp(rho, data_eq["rho"], data_eq["iota"])
    shears = np.interp(rho, data_eq["rho"], data_eq["iota_r"])

    N = int((2 * eq.M_grid) * 4 + 1)
    zeta = np.linspace(-1.0 * np.pi / iotas, 1.0 * np.pi / iotas, N)
    theta_PEST = alpha * np.ones(N, dtype=int) + iotas * zeta

    # Creating a grid along a field line
    coords1 = np.zeros((N, 3))
    coords1[:, 0] = np.sqrt(psi) * np.ones(N, dtype=int)
    coords1[:, 1] = theta_PEST
    coords1[:, 2] = zeta
    # c1 = eq.compute_theta_coords(coords1)  # noqa: E800
    # grid = Grid(c1, sort=False)  # noqa: E800

    # TODO: Request: The bounce integral operator should be able to take a grid.
    #       Response: Currently the API is such that the method does all the
    #                 above preprocessing for you. Let's test it for correctness
    #                 first then do this later.
    bounce_integral, items = bounce_integral_map(eq, rho, alpha, knots=zeta, check=True)
    grid = items["grid_desc"]
    data_keys = [
        "|grad(psi)|^2",
        "grad(psi)",
        "B",
        "iota",
        "|B|",
        "B^zeta",
        "cvdrift0",
        "cvdrift",
        "gbdrift",
    ]
    # FIXME (outside scope of the bounce integral pull request):
    #  override_grid should not be required for the test to pass.
    data = eq.compute(data_keys, grid=grid, override_grid=False)
    # If this is the toroidal flux at the boundary, the [-1] retrieval
    # assumes that the grid which computed the data_eq["psi"] happens to have
    # its last node at the last closed flux surface.
    psib = data_eq["psi"][-1]

    # signs
    sign_psi = psib / np.abs(psib)
    sign_iota = iotas / np.abs(iotas)

    # normalizations
    Lref = data_eq["a"]
    Bref = 2 * np.abs(psib) / Lref**2

    modB = data["|B|"]
    bmag = modB / Bref

    x = Lref * np.sqrt(psi)
    s_hat = -x / iotas * shears / Lref

    iota = data["iota"]
    gradpar = Lref * data["B^zeta"] / modB

    ## Comparing coefficient calculation here with coefficients from compute/_mtric
    cvdrift = -2 * sign_psi * Bref * Lref**2 * np.sqrt(psi) * data["cvdrift"]
    gbdrift = -2 * sign_psi * Bref * Lref**2 * np.sqrt(psi) * data["gbdrift"]

    epsilon = Lref * np.sqrt(psi)
    B0 = np.mean(bmag)

    bmag_an = B0 * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(bmag, bmag_an, atol=5e-3, rtol=5e-3)

    gradpar_an = 2 * Lref * iota * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(gradpar, gradpar_an, atol=9e-3, rtol=5e-3)

    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * modB**2)
    alpha_MHD = -dPdrho * 1 / iota**2 * 0.5

    grad_psi = data["grad(psi)"]
    grad_alpha = data["grad(alpha)"]

    gds21 = -sign_iota * np.array(dot(grad_psi, grad_alpha)) * s_hat / Bref

    fudge_factor2 = 0.19
    gbdrift_an = fudge_factor2 * (
        -1 * s_hat + (np.cos(theta_PEST) - 1.0 * gds21 / s_hat * np.sin(theta_PEST))
    )

    fudge_factor3 = 0.07
    cvdrift_an = gbdrift_an + fudge_factor3 * alpha_MHD / bmag**2

    # Comparing coefficients with their analytical expressions
    np.testing.assert_allclose(gbdrift, gbdrift_an, atol=1.5e-2, rtol=5e-3)
    np.testing.assert_allclose(cvdrift, cvdrift_an, atol=1.8e-2, rtol=5e-3)

    # Values of pitch angle for which to evaluate the bounce averages
    pitch = np.linspace(1 / np.max(bmag), 1 / np.min(bmag), 11)
    pitch = pitch.reshape(pitch.shape[0], -1)

    k2 = 0.5 * ((1 - pitch * B0) / epsilon + 1)

    bavg_drift_an = (
        0.5 * cvdrift_an * ellipe(k2)
        + gbdrift_an * ellipk(k2)
        + dPdrho / bmag**2 * ellipe(k2)
    )

    def integrand(cvdrift, gbdrift, B, pitch):
        # The arguments to this function will be interpolated
        # onto the quadrature points before these quantities are evaluated.
        bmag = B / Bref
        # When 1 - pitch * B is negative, we want g to evaluate as nan.
        # jnp.sqrt() will do this as desired, but np.sqrt() will give a runtime error.
        g = complex_sqrt(1 - pitch * bmag)
        g = np.where(np.isclose(np.imag(g), 0), np.real(g), np.nan)
        return (g * 0.5 * cvdrift) + (gbdrift / g) + (dPdrho / bmag**2 * g)

    bavg_drift_num = bounce_integral(
        integrand=integrand,
        # additional things to interpolate onto quadrature points besides B and pitch
        f=[cvdrift, gbdrift],
        pitch=pitch,
    )
    assert np.isfinite(bavg_drift_num).any(), "Quadrature failed."
    # there's only one field line on the grid, so squeeze out that axis
    bavg_drift_num = np.squeeze(bavg_drift_num, axis=1)
    for i in range(pitch.shape[0]):
        np.testing.assert_allclose(
            _filter_not_nan(bavg_drift_num[i]),
            bavg_drift_an[i],
            atol=2e-2,
            rtol=1e-2,
            err_msg=f"Failed on index {i} for pitch {pitch[i]}",
        )


# TODO: if deemed useful finish details using methods in desc.compute.bounce_integral
def _compute_bounce_points_with_root_finding(
    eq, pitch, rho, alpha, resolution=20, zeta_max=6 * np.pi
):
    # TODO: avoid separate root finding routines in residual and jac
    #       and use previous desc coords as initial guess for next iteration
    def residual(zeta, i):
        grid, data = desc_grid_from_field_line_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|"], grid=grid, data=data)
        return data["|B|"] - pitch[i]

    def jac(zeta):
        grid, data = desc_grid_from_field_line_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|_z|r,a"], grid=grid, data=data)
        return data["|B|_z|r,a"]

    # Compute |B| - 1/pitch on a dense grid.
    # For every field line, find the roots of this linear spline.
    # These estimates for the true roots will serve as an initial guess, and
    # let us form a boundary mesh around root estimates to limit search domain
    # of the root finding algorithms.
    zeta = np.linspace(0, zeta_max, 3 * resolution)
    grid, data = desc_grid_from_field_line_coords(rho, alpha, zeta, eq)
    data = eq.compute(["|B|"], grid=grid, data=data)
    B_norm = data["|B|"].reshape(alpha.size, rho.size, -1)  # constant field line chunks

    boundary_lt = np.zeros((pitch.size, resolution, alpha.size, rho.size))
    boundary_rt = np.zeros((pitch.size, resolution, alpha.size, rho.size))
    guess = np.zeros((pitch.size, resolution, alpha.size, rho.size))
    # todo: scan over this
    for i in range(pitch.size):
        for j in range(alpha.size):
            for k in range(rho.size):
                # indices of zeta values observed prior to sign change
                idx = np.nonzero(np.diff(np.sign(B_norm[j, k] - pitch[i])))[0]
                guess[i, :, j, k] = grid.nodes[idx, 2]
                boundary_lt[i, :, j, k] = np.append(zeta[0], guess[:-1])
                boundary_rt[i, :, j, k] = np.append(guess[1:], zeta[-1])
    guess = guess.reshape(pitch.size, resolution, alpha.size * rho.size)
    boundary_lt = boundary_lt.reshape(pitch.size, resolution, alpha.size * rho.size)
    boundary_rt = boundary_rt.reshape(pitch.size, resolution, alpha.size * rho.size)

    def body_pitch(i, out):
        def body_roots(j, out_i):
            def fixup(z):
                return np.clip(z, boundary_lt[i, j], boundary_rt[i, j])

            # todo: call vmap to vectorize on guess[i, j] so that we solve
            #  guess[i, j].size independent root finding problems
            root = root_scalar(residual, guess[i, j], jac=jac, args=i, fixup=fixup)
            out_i = put(out_i, j, root)
            return out_i

        out = put(out, i, fori_loop(0, resolution, body_roots, out[i]))
        return out

    bounce_points = np.zeros(shape=(pitch.size, alpha.size, rho.size, resolution))
    bounce_points = fori_loop(0, pitch.size, body_pitch, bounce_points)
    return bounce_points
