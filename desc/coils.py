"""Classes for magnetic field coils."""

from abc import ABC
from collections.abc import MutableSequence

import numpy as np

from desc.backend import jnp
from desc.geometry import FourierPlanarCurve, FourierRZCurve, FourierXYZCurve, XYZCurve
from desc.geometry.utils import rpz2xyz, xyz2rpz_vec
from desc.grid import Grid
from desc.magnetic_fields import MagneticField, biot_savart


class Coil(MagneticField, ABC):
    """Base class representing a magnetic field coil.

    Represents coils as a combination of a Curve and current

    Subclasses for a particular parameterization of a coil should inherit
    from Coil and the appropriate Curve type, eg MyCoil(Coil, MyCurve)
    - note that Coil must be the first parent for correct inheritance.

    Subclasses based on curves that follow the Curve API should only have
    to implement a new __init__ method, all others will be handled by default

    Parameters
    ----------
    current : float
        current passing through the coil, in Amperes
    """

    _io_attrs_ = MagneticField._io_attrs_ + ["_current"]

    def __init__(self, current, *args, **kwargs):
        self._current = current
        super().__init__(*args, **kwargs)

    @property
    def current(self):
        """float: Current passing through the coil, in Amperes."""
        return self._current

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._current = new

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
        """Compute magnetic field at a set of points.

        The coil is discretized into a series of straight line segments, using
        the coil ``grid`` attribute. To override this, include 'grid' as a key
        in the `params` dictionary with the desired grid resolution.

        Similarly, the coil current may be overridden by including `current`
        in the `params` dictionary.

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict, optional
            parameters to pass to curve
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        if params is None:
            params = {}
        assert basis.lower() in ["rpz", "xyz"]
        if isinstance(coords, Grid):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "rpz":
            coords = rpz2xyz(coords)
        current = params.pop("current", self.current)
        coil_coords = self.compute_coordinates(**params, basis="xyz")
        B = biot_savart(coords, coil_coords, current)
        if basis == "rpz":
            B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        return B

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, current={})".format(self.name, self.current)
        )


class FourierRZCoil(Coil, FourierRZCurve):
    """Coil parameterized by fourier series for R,Z in terms of toroidal angle phi.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
    R_n, Z_n: array-like
        fourier coefficients for R, Z
    modes_R : array-like
        mode numbers associated with R_n. If not given defaults to [-n:n]
    modes_Z : array-like
        mode numbers associated with Z_n, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry
    grid : Grid
        default grid for computation
    name : str
        name for this coil
    """

    _io_attrs_ = Coil._io_attrs_ + FourierRZCurve._io_attrs_

    def __init__(
        self,
        current=1,
        R_n=10,
        Z_n=0,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        grid=None,
        name="",
    ):
        super().__init__(current, R_n, Z_n, modes_R, modes_Z, NFP, sym, grid, name)


class FourierXYZCoil(Coil, FourierXYZCurve):
    """Coil parameterized by fourier series for X,Y,Z in terms of arbitrary angle phi.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
    X_n, Y_n, Z_n: array-like
        fourier coefficients for X, Y, Z
    modes : array-like
        mode numbers associated with X_n etc.
    grid : Grid
        default grid or computation
    name : str
        name for this coil

    """

    _io_attrs_ = Coil._io_attrs_ + FourierXYZCurve._io_attrs_

    def __init__(
        self,
        current=1,
        X_n=[0, 10, 2],
        Y_n=[0, 0, 0],
        Z_n=[-2, 0, 0],
        modes=None,
        grid=None,
        name="",
    ):
        super().__init__(current, X_n, Y_n, Z_n, modes, grid, name)


# TODO: add a from_XYZ?


class FourierPlanarCoil(Coil, FourierPlanarCurve):
    """Coil that lines in a plane.

    Parameterized by a point (the center of the coil), a vector (normal to the plane),
    and a fourier series defining the radius from the center as a function of a polar
    angle theta.

    Parameters
    ----------
    current : float
        current through the coil, in Amperes
    center : array-like, shape(3,)
        x,y,z coordinates of center of coil
    normal : array-like, shape(3,)
        x,y,z components of normal vector to planar surface
    r_n : array-like
        fourier coefficients for radius from center as function of polar angle
    modes : array-like
        mode numbers associated with r_n
    grid : Grid
        default grid for computation
    name : str
        name for this coil

    """

    _io_attrs_ = Coil._io_attrs_ + FourierPlanarCurve._io_attrs_

    def __init__(
        self,
        current=1,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        modes=None,
        grid=None,
        name="",
    ):
        super().__init__(current, center, normal, r_n, modes, grid, name)


class XYZCoil(Coil, XYZCurve):
    """Coil parameterized by points in X,Y,Z.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
    X, Y, Z: array-like
        points for X, Y, Z descriving a closed curve
    name : str
        name for this coil

    """

    _io_attrs_ = Coil._io_attrs_

    def __init__(
        self,
        current,
        X,
        Y,
        Z,
        grid=None,
        name="",
    ):
        super().__init__(current, X, Y, Z, grid, name)
        self.X = X
        self.Y = Y
        self.Z = Z
        self.coords = np.vstack((X, Y, Z)).T


class CoilSet(Coil, MutableSequence):
    """Set of coils of different geometry.

    Parameters
    ----------
    coils : Coil or array-like of Coils
        collection of coils
    currents : float or array-like of float
        currents in each coil, or a single current shared by all coils in the set
    """

    _io_attrs_ = Coil._io_attrs_ + ["_coils"]

    def __init__(
        self, *coils, name=""
    ):  # FIXME: if a list of of Coils is passed, this fails...
        assert all([isinstance(coil, (Coil)) for coil in coils])
        self._coils = list(coils)
        self._name = str(name)

    @property
    def name(self):
        """str: Name of the curve."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def coils(self):
        """list: coils in the coilset."""
        return self._coils

    @property
    def current(self):
        """list: currents in each coil."""
        return [coil.current for coil in self.coils]

    @current.setter
    def current(self, new):
        if jnp.isscalar(new):
            new = [new] * len(self)
        for coil, cur in zip(self.coils, new):
            coil.current = cur

    @property
    def grid(self):
        """Grid: nodes for computation."""
        return self.coils[0].grid

    @grid.setter
    def grid(self, new):
        for coil in self.coils:
            coil.grid = new

    def compute_coordinates(self, *args, **kwargs):
        """Compute real space coordinates using underlying curve method."""
        return [coil.compute_coordinates(*args, **kwargs) for coil in self.coils]

    def compute_frenet_frame(self, *args, **kwargs):
        """Compute Frenet frame using underlying curve method."""
        return [coil.compute_frenet_frame(*args, **kwargs) for coil in self.coils]

    def compute_curvature(self, *args, **kwargs):
        """Compute curvature using underlying curve method."""
        return [coil.compute_curvature(*args, **kwargs) for coil in self.coils]

    def compute_torsion(self, *args, **kwargs):
        """Compute torsion using underlying curve method."""
        return [coil.compute_torsion(*args, **kwargs) for coil in self.coils]

    def compute_length(self, *args, **kwargs):
        """Compute the length of the curve using underlying curve method."""
        return [coil.compute_length(*args, **kwargs) for coil in self.coils]

    def translate(self, *args, **kwargs):
        """Translate the coils along an axis."""
        [coil.translate(*args, **kwargs) for coil in self.coils]

    def rotate(self, *args, **kwargs):
        """Rotate the coils about an axis."""
        [coil.rotate(*args, **kwargs) for coil in self.coils]

    def flip(self, *args, **kwargs):
        """Flip the coils across a plane."""
        [coil.flip(*args, **kwargs) for coil in self.coils]

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict or array-like of dict, optional
            parameters to pass to curves, either the same for all curves,
            or one for each member
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        if isinstance(params, dict) or params is None:
            params = [params] * len(self)
        assert len(params) == len(self)
        B = 0
        for coil, par in zip(self.coils, params):
            B += coil.compute_magnetic_field(coords, par, basis)

        return B

    @classmethod
    def linspaced_angular(
        cls, coil, current=None, axis=[0, 0, 1], angle=2 * np.pi, n=10, endpoint=False
    ):
        """Create a coil set by repeating a coil n times rotationally.

        Parameters
        ----------
        coil : Coil
            base coil to repeat
        current : float or array-like, shape(n,)
            current in (each) coil, overrides coil.current
        axis : array-like, shape(3,)
            axis to rotate about
        angle : float
            total rotational extend of coil set.
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final angle
        """
        assert isinstance(coil, Coil)
        if current is None:
            current = coil.current
        currents = jnp.broadcast_to(current, (n,))
        coils = []
        phis = jnp.linspace(0, angle, n, endpoint=endpoint)
        for i in range(n):
            coili = coil.copy()
            coili.rotate(axis, angle=phis[i])
            coili.current = currents[i]
            coils.append(coili)
        return cls(*coils)

    @classmethod
    def linspaced_linear(
        cls, coil, current=None, displacement=[2, 0, 0], n=4, endpoint=False
    ):
        """Create a coil group by repeating a coil n times in a straight line.

        Parameters
        ----------
        coil : Coil
            base coil to repeat
        current : float or array-like, shape(n,)
            current in (each) coil
        displacement : array-like, shape(3,)
            total displacement of the final coil
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final point
        """
        assert isinstance(coil, Coil)
        if current is None:
            current = coil.current
        currents = jnp.broadcast_to(current, (n,))
        displacement = jnp.asarray(displacement)
        coils = []
        a = jnp.linspace(0, 1, n, endpoint=endpoint)
        for i in range(n):
            coili = coil.copy()
            coili.translate(a[i] * displacement)
            coili.current = currents[i]
            coils.append(coili)
        return cls(*coils)

    @classmethod
    def from_symmetry(cls, coils, NFP, sym=False):
        """Create a coil group by reflection and symmetry.

        Given coils over one field period, repeat coils NFP times between
        0 and 2pi to form full coil set.

        Or, give coils over 1/2 of a field period, repeat coils 2*NFP times
        between 0 and 2pi to form full stellarator symmetric coil set.

        Parameters
        ----------
        coils : Coil, CoilGroup, Coilset
            base coil or collection of coils to repeat
        NFP : int
            number of field periods
        sym : bool
            whether coils should be stellarator symmetric
        """
        if not isinstance(coils, CoilSet):
            coils = CoilSet(coils)
        coilset = []
        if sym:
            # first reflect/flip original coilset
            # ie, given coils [1,2,3] at angles [0, pi/6, 2pi/6]
            # we want a new set like [1,2,3,flip(3),flip(2),flip(1)]
            # at [0, pi/6, 2pi/6, 3pi/6, 4pi/6, 5pi/6]
            flipped_coils = []
            normal = jnp.array([-jnp.sin(jnp.pi / NFP), jnp.cos(jnp.pi / NFP), 0])
            for coil in coils[::-1]:
                fcoil = coil.copy()
                fcoil.flip(normal)
                fcoil.flip([0, 0, 1])
                fcoil.current = -1 * coil.current
                flipped_coils.append(fcoil)
            coils = coils + flipped_coils
        for k in range(0, NFP):
            coil = coils.copy()
            coil.rotate(axis=[0, 0, 1], angle=2 * jnp.pi * k / NFP)
            coilset.append(coil)

        return cls(*coilset)

    @classmethod
    def from_makegrid_coilfile(cls, coil_file, grid=None):
        """Create a CoilSet of XYZCoils from a MAKEGRID-formatted coil txtfile.

        Parameters
        ----------
        coil_file : str or path-like
            path to coil file in txt format
        """
        coils = []  # list of XYZCoils
        coilinds = []

        # read in the coils file
        with open(coil_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.find("Modular") != -1:
                    coilinds.append(i)
                if line.find("mirror") != -1:
                    coilinds.append(i)
        for i, (start, end) in enumerate(zip(coilinds[0:-1], coilinds[1:])):
            coords = np.genfromtxt(lines[start + 1 : end])
            if i % 20 == 0:
                print("reading coil " + f"{i}")

            tempx = np.append(coords[:, 0], np.array([coords[0, 0]]))
            tempy = np.append(coords[:, 1], np.array([coords[0, 1]]))
            tempz = np.append(coords[:, 2], np.array([coords[0, 2]]))

            coils.append(XYZCoil(coords[:, -1][0], tempx, tempy, tempz, grid=grid))
            # FIXME: # sign of current may need to be negative, need to check
            # makegrid convention, but tests show
            # the resulting field agrees better with DESC eq if negative curr

        return CoilSet(*coils)

    def save_in_MAKEGRID_format(self, coilsFilename, params={}):
        """Save CoilSet of as a MAKEGRID-formatted coil txtfile.

        Parameters
        ----------
        filename : str or path-like
            path save CoilSet as a file in MAKEGRID txt format
        params : dict or array-like of dict, optional
            parameters to pass to curves, either the same for all curves,
            or one for each member
        """
        if isinstance(params, dict):
            params = [params] * len(self)
        assert len(params) == len(self)

        with open(coilsFilename, "w") as f:
            f.write("periods " + str(1) + "\n")
            f.write("begin filament\n")
            f.write("mirror NIL\n")

            # FIXME: proper way to pass in params?
            # or just do grid as a param, since we want to use the current coils?
            # TODO: use numpy to make this faster? instead of line by line
            for coil, par in zip(self.coils, params):
                if isinstance(coil, XYZCoil):
                    contour_X = coil.X[0:-1]
                    contour_Y = coil.Y[0:-1]
                    contour_Z = coil.Z[0:-1]
                else:
                    coords = coil.compute_coordinates(basis="xyz")
                    contour_X = coords[0:-1, 0]
                    contour_Y = coords[0:-1, 1]
                    contour_Z = coords[0:-1, 2]

                for k in range(contour_X.size):
                    f.write(
                        "{:14.22e} {:14.22e} {:14.22e} {:14.22e}\n".format(
                            contour_X[k], contour_Y[k], contour_Z[k], coil.current
                        )
                    )
                # Close the loop
                k = 0
                f.write(
                    "{:14.22e} {:14.22e} {:14.22e} {:14.22e} 1 Modular\n".format(
                        contour_X[k], contour_Y[k], contour_Z[k], 0
                    )
                )

            f.write("end\n")
        print(f"Saved coils file at : {coilsFilename}")

    # FIXME: implement this properly for all coiltypes

    def __add__(self, other):
        if isinstance(other, (CoilSet)):
            return CoilSet(*self.coils, *other.coils)
        if isinstance(other, (list, tuple)):
            return CoilSet(*self.coils, *other)
        raise TypeError

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.coils[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils[i] = new_item

    def __delitem__(self, i):
        del self._coils[i]

    def __len__(self):
        return len(self._coils)

    def insert(self, i, new_item):
        """Insert a new coil into the coilset at position i."""
        if not isinstance(new_item, Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils.insert(i, new_item)

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, with {} submembers)".format(self.name, len(self))
        )
