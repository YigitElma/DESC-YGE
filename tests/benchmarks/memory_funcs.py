#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import desc

if sys.argv[3] in ["GPU", "gpu"]:
    # Set the environment variable to use the GPU
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    from desc import set_device

    set_device("gpu")
import warnings

from desc.backend import jax
import desc.examples
from desc.magnetic_fields import ToroidalMagneticField
from desc.objectives import (
    BoundaryError,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.optimize import LinearConstraintProjection, ProximalProjection


def test_proximal_freeb_compute(res):
    """Benchmark computing free boundary objective with proximal constraint."""
    jax.clear_caches()
    eq = desc.examples.get("ESTELL")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    objective = ObjectiveFunction(BoundaryError(eq, field=field))
    constraint = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=1)
    prox = ProximalProjection(objective, constraint, eq)
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build()
    x = obj.x(eq)
    obj.compute_scaled_error(x, obj.constants).block_until_ready()


def test_proximal_freeb_jac(res):
    """Benchmark computing free boundary jacobian with proximal constraint."""
    jax.clear_caches()
    eq = desc.examples.get("ESTELL")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    objective = ObjectiveFunction(BoundaryError(eq, field=field))
    constraint = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=1)
    prox = ProximalProjection(objective, constraint, eq)
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build()
    x = obj.x(eq)
    obj.jac_scaled_error(x, prox.constants).block_until_ready()


if __name__ == "__main__":
    func = str(sys.argv[1])
    res = int(sys.argv[2])
    if func == "proximal_freeb_compute":
        test_proximal_freeb_compute(res)
    elif func == "proximal_freeb_jac":
        test_proximal_freeb_jac(res)
    else:
        print(
            f"Invalid function name {func}. Use 'proximal_freeb_compute' or 'proximal_freeb_jac'."
        )
        sys.exit(1)
