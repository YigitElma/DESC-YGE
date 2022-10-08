"""Classes defining objectives for equilibrium and optimization."""

from .objective_funs import ObjectiveFunction
from .linear_objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixLambdaGauge,
    FixPressure,
    FixIota,
    FixCurrent,
    FixPsi,
)
from ._generic import GenericObjective, ToroidalCurrent
from ._equilibrium import (
    Energy,
    ForceBalance,
    RadialForceBalance,
    HelicalForceBalance,
    CurrentDensity,
)
from ._geometry import Volume, AspectRatio
from ._qs import (
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
)
from ._stability import MercierStability, MagneticWell
from .utils import (
    get_fixed_boundary_constraints,
    get_equilibrium_objective,
)
from ._wrappers import WrappedEquilibriumObjective
