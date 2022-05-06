import numpy as np
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    RadialForceBalance,
    HelicalForceBalance,
    AspectRatio,
    get_fixed_boundary_constraints,
)
from desc.vmec import VMECIO
from desc.vmec_utils import vmec_boundary_subspace


def test_SOLOVEV_results(SOLOVEV):
    """Tests that the SOLOVEV example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    rho_err, theta_err = VMECIO.area_difference_vmec(eq, SOLOVEV["vmec_nc_path"])

    np.testing.assert_allclose(rho_err, 0, atol=1e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    rho_err, theta_err = VMECIO.area_difference_vmec(eq, DSHAPE["vmec_nc_path"])

    np.testing.assert_allclose(rho_err, 0, atol=2e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


def test_HELIOTRON_results(HELIOTRON):
    """Tests that the HELIOTRON example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(HELIOTRON["desc_h5_path"]))[-1]
    rho_err, theta_err = VMECIO.area_difference_vmec(eq, HELIOTRON["vmec_nc_path"])

    np.testing.assert_allclose(rho_err.mean(), 0, atol=1e-2)
    np.testing.assert_allclose(theta_err.mean(), 0, atol=2e-2)


def test_force_balance_grids():
    """Compares radial & helical force balance on same vs different grids."""

    res = 3

    eq1 = Equilibrium(sym=True)
    eq1.change_resolution(L=res, M=res)
    eq1.L_grid = res
    eq1.M_grid = res

    eq2 = Equilibrium(sym=True)
    eq2.change_resolution(L=res, M=res)
    eq2.L_grid = res
    eq2.M_grid = res

    # force balances on the same grids
    obj1 = ObjectiveFunction(ForceBalance())
    eq1.solve(objective=obj1)

    # force balances on different grids
    obj2 = ObjectiveFunction((RadialForceBalance(), HelicalForceBalance()))
    eq2.solve(objective=obj2)

    np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn, atol=5e-4)
    np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn, atol=5e-4)
    np.testing.assert_allclose(eq1.L_lmn, eq2.L_lmn, atol=2e-3)


def test_1d_optimization(SOLOVEV):
    """Tests 1D optimization for target aspect ratio."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    objective = ObjectiveFunction(AspectRatio(target=3))
    constraints = get_fixed_boundary_constraints()
    perturb_options = {"dZb": True, "subspace": vmec_boundary_subspace(eq, ZBS=[0, 1])}
    eq = eq.optimize(objective, constraints, perturb_options=perturb_options)

    np.testing.assert_allclose(eq.compute("V")["R0/a"], 3)
