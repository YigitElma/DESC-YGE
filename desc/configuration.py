import numpy as np
import copy
import warnings
from termcolor import colored
from abc import ABC
from shapely.geometry import LineString, MultiLineString
from desc.io import IOAble
from desc.utils import unpack_state, copy_coeffs
from desc.grid import Grid, LinearGrid
from desc.transform import Transform
from desc.grid import QuadratureGrid
from desc.objective_funs import get_objective_function
from desc.boundary_conditions import BoundaryCondition
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    ZernikePolynomial,
    FourierZernikeBasis,
)

from desc.compute_funs import (
    compute_profiles,
    compute_toroidal_coords,
    compute_cartesian_coords,
    compute_covariant_basis,
    compute_jacobian,
    compute_contravariant_basis,
    compute_magnetic_field_magnitude_axis,
    compute_current_density,
    compute_magnetic_pressure_gradient,
    compute_magnetic_tension,
    compute_force_error_magnitude,
    compute_energy,
)


class _Configuration(IOAble, ABC):
    """Configuration is an abstract base class for equilibrium information.

    It contains information about a plasma state, including the
    shapes of flux surfaces and profile inputs. It can compute additional
    information, such as the magnetic field and plasma currents.
    """

    _io_attrs_ = [
        "_sym",
        "_Psi",
        "_NFP",
        "_L",
        "_M",
        "_N",
        "_x",
        "_R_lmn",
        "_Z_lmn",
        "_L_lmn",
        "_Rb_lmn",
        "_Zb_lmn",
        "_p_l",
        "_i_l",
        "_R_basis",
        "_Z_basis",
        "_L_basis",
        "_Rb_basis",
        "_Zb_basis",
        "_p_basis",
        "_i_basis",
        "_spectral_indexing",
        "_bdry_mode",
        "_zeta_ratio",
    ]

    _object_lib_ = {
        "PowerSeries": PowerSeries,
        "FourierSeries": FourierSeries,
        "DoubleFourierSeries": DoubleFourierSeries,
        "FourierZernikeBasis": FourierZernikeBasis,
    }

    def __init__(self, inputs):
        """Initializes a Configuration

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs with the following required keys:
                Psi : float, total toroidal flux (in Webers) within LCFS
                NFP : int, number of field periods
                L : int, radial resolution
                M : int, poloidal resolution
                N : int, toroidal resolution
                profiles : ndarray, array of profile coeffs [l, p_l, i_l]
                boundary : ndarray, array of boundary coeffs [m, n, Rb_lmn, Zb_lmn]
            And the following optional keys:
                sym : bool, is the problem stellarator symmetric or not, default is False
                spectral_indexing : str, type of Zernike indexing scheme to use, default is 'ansi'
                bdry_mode : {'lcfs', 'poincare'}, where the BC are enforced
                zeta_ratio : float, Multiplier on the toroidal derivatives. Default = 1.0.
                axis : ndarray, array of magnetic axis coeffs [n, R0_n, Z0_n]
                x : ndarray, state vector [R_lmn, Z_lmn, L_lmn]
                R_lmn : ndarray, spectral coefficients of R
                Z_lmn : ndarray, spectral coefficients of Z
                L_lmn : ndarray, spectral coefficients of lambda

        """
        self.inputs = inputs
        try:
            self._Psi = float(inputs["Psi"])
            self._NFP = inputs["NFP"]
            self._L = inputs["L"]
            self._M = inputs["M"]
            self._N = inputs["N"]
            self._profiles = inputs["profiles"]
            self._boundary = inputs["boundary"]
        except:
            raise ValueError(colored("input dict does not contain proper keys", "red"))

        # optional inputs
        self._sym = inputs.get("sym", False)
        self._spectral_indexing = inputs.get("spectral_indexing", "fringe")
        self._bdry_mode = inputs.get("bdry_mode", "lcfs")
        self._zeta_ratio = float(inputs.get("zeta_ratio", 1.0))

        # keep track of where it came from
        self._parent = None
        self._children = []

        # stellarator symmetry for bases
        if self._sym:
            self._R_sym = "cos"
            self._Z_sym = "sin"
        else:
            self._R_sym = None
            self._Z_sym = None

        # create bases
        self._set_basis()

        # format profiles
        self._p_l, self._i_l = format_profiles(
            self._profiles, self.p_basis, self.i_basis
        )

        # format boundary
        self._Rb_lmn, self._Zb_lmn = format_boundary(
            self._boundary, self.Rb_basis, self.Zb_basis, self.bdry_mode
        )

        # check if state vector is provided
        try:
            self._x = inputs["x"]
            self._R_lmn, self._Z_lmn, self._L_lmn = unpack_state(
                self.x, self.R_basis.num_modes, self.Z_basis.num_modes
            )
        # default initial guess
        except:
            axis = inputs.get(
                "axis", self._boundary[np.where(self._boundary[:, 1] == 0)[0], 2:]
            )
            # check if R is provided
            try:
                self._R_lmn = inputs["R_lmn"]
            except:
                self._R_lmn = initial_guess(
                    self.R_basis, self.Rb_lmn, self.Rb_basis, axis[:, 0:-1]
                )
            # check if Z is provided
            try:
                self._Z_lmn = inputs["Z_lmn"]
            except:
                self._Z_lmn = initial_guess(
                    self.Z_basis, self.Zb_lmn, self.Zb_basis, axis[:, (0, -1)]
                )
            # check if lambda is provided
            try:
                self._L_lmn = inputs["L_lmn"]
            except:
                self._L_lmn = np.zeros((self.L_basis.num_modes,))
            self._x = np.concatenate([self.R_lmn, self.Z_lmn, self.L_lmn])

    def _set_basis(self):

        self._R_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._R_sym,
            spectral_indexing=self.spectral_indexing,
        )
        self._Z_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._Z_sym,
            spectral_indexing=self.spectral_indexing,
        )
        self._L_basis = FourierZernikeBasis(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            sym=self._Z_sym,
            spectral_indexing=self.spectral_indexing,
        )

        if np.all(self._boundary[:, 0] == 0):
            self._Rb_basis = DoubleFourierSeries(
                M=self.M, N=self.N, NFP=self.NFP, sym=self._R_sym
            )
            self._Zb_basis = DoubleFourierSeries(
                M=self.M, N=self.N, NFP=self.NFP, sym=self._Z_sym
            )
        elif np.all(self._boundary[:, 2] == 0):
            self._Rb_basis = ZernikePolynomial(
                L=self.L, M=self.M, sym=self._R_sym, index=self.spectral_indexing
            )
            self._Zb_basis = ZernikePolynomial(
                L=self.L, M=self.M, sym=self._Z_sym, index=self.spectral_indexing
            )
        else:
            raise ValueError("boundary should either have l=0 or n=0")

        nonzero_modes = self._boundary[
            np.argwhere(self._boundary[:, 3:] != np.array([0, 0]))[:, 0]
        ]
        if nonzero_modes.size and (
            self.L < np.max(abs(nonzero_modes[:, 0]))
            or self.M < np.max(abs(nonzero_modes[:, 1]))
            or self.N < np.max(abs(nonzero_modes[:, 2]))
        ):
            warnings.warn(
                colored(
                    "Configuration resolution does not fully resolve boundary inputs, "
                    + "Configuration L,M,N={},{},{}, "
                    + "boundary resolution L,M,N={},{},{}".format(
                        self.L,
                        self.M,
                        self.N,
                        int(np.max(abs(nonzero_modes[:, 0]))),
                        int(np.max(abs(nonzero_modes[:, 1]))),
                        int(np.max(abs(nonzero_modes[:, 2]))),
                    ),
                    "yellow",
                )
            )
        self._p_basis = PowerSeries(L=max(self.L, int(np.max(self._profiles[:, 0]))))
        self._i_basis = PowerSeries(L=max(self.L, int(np.max(self._profiles[:, 0]))))

    @property
    def parent(self):
        """Pointer to the equilibrium this was derived from."""
        return self._parent

    @property
    def children(self):
        """List of configurations that were derived from this one."""
        return self._children

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this equilibrium."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        new._parent = self
        self._children.append(new)
        return new

    def change_resolution(self, L=None, M=None, N=None, *args, **kwargs):
        """Set the spectral resolution.

        Parameters
        ----------
        L : int
            maximum radial zernike mode number
        M : int
            maximum poloidal fourier mode number
        N : int
            maximum toroidal fourier mode number

        """
        L_change = M_change = N_change = False
        if L is not None and L != self.L:
            L_change = True
            self._L = L
        if M is not None and M != self.M:
            M_change = True
            self._M = M
        if N is not None and N != self.N:
            N_change = True
            self._N = N

        if not np.any([L_change, M_change, N_change]):
            return

        old_modes_R = self.R_basis.modes
        old_modes_Z = self.Z_basis.modes
        old_modes_L = self.L_basis.modes
        old_modes_p = self.p_basis.modes
        old_modes_i = self.i_basis.modes
        old_modes_Rb = self.Rb_basis.modes
        old_modes_Zb = self.Zb_basis.modes

        self._set_basis()

        # previous resolution may have left off some coeffs, so we should add them back
        # in but need to check if "profiles" is still accurate, might have been
        # perturbed so we reuse the old coeffs up to the old resolution
        full_p_l, full_i_l = format_profiles(self._profiles, self.p_basis, self.i_basis)
        self._p_l = copy_coeffs(self.p_l, old_modes_p, self.p_basis.modes, full_p_l)
        self._i_l = copy_coeffs(self.i_l, old_modes_i, self.p_basis.modes, full_i_l)

        # format boundary
        full_Rb_lmn, full_Zb_lmn = format_boundary(
            self._boundary, self.Rb_basis, self.Zb_basis, self.bdry_mode
        )
        self._Rb_lmn = copy_coeffs(
            self.Rb_lmn, old_modes_Rb, self.Rb_basis.modes, full_Rb_lmn
        )
        self._Zb_lmn = copy_coeffs(
            self.Zb_lmn, old_modes_Zb, self.Zb_basis.modes, full_Zb_lmn
        )

        self._R_lmn = copy_coeffs(self.R_lmn, old_modes_R, self.R_basis.modes)
        self._Z_lmn = copy_coeffs(self.Z_lmn, old_modes_Z, self.Z_basis.modes)
        self._L_lmn = copy_coeffs(self.L_lmn, old_modes_L, self.L_basis.modes)

        # state vector
        self._x = np.concatenate([self.R_lmn, self.Z_lmn, self.L_lmn])
        self._make_labels()

    @property
    def spectral_indexing(self):
        """Type of indexing used for the spectral basis (str)."""
        return self._spectral_indexing

    @property
    def sym(self):
        """Whether this equilibrium is stellarator symmetric (bool)."""
        return self._sym

    @property
    def bdry_mode(self):
        """Mode for specifying plasma boundary (str)."""
        return self._bdry_mode

    @property
    def Psi(self):
        """Total toroidal flux within the last closed flux surface in Webers (float)."""
        return self._Psi

    @Psi.setter
    def Psi(self, Psi):
        self._Psi = float(Psi)

    @property
    def NFP(self):
        """Number of (toroidal) field periods (int)."""
        return self._NFP

    @NFP.setter
    def NFP(self, NFP):
        self._NFP = NFP

    @property
    def L(self):
        """Maximum radial mode number (int)."""
        return self._L

    @property
    def M(self):
        """Maximum poloidal fourier mode number (int)."""
        return self._M

    @property
    def N(self):
        """Maximum toroidal fourier mode number (int)."""
        return self._N

    @property
    def x(self):
        """Optimization state vector (ndarray)."""
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
        self._R_lmn, self._Z_lmn, self._L_lmn = unpack_state(
            self.x, self.R_basis.num_modes, self.Z_basis.num_modes
        )

    @property
    def R_lmn(self):
        """Spectral coefficients of R (ndarray)."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, R_lmn):
        self._R_lmn = R_lmn
        self._x = np.concatenate([self.R_lmn, self.Z_lmn, self.L_lmn])

    @property
    def Z_lmn(self):
        """Spectral coefficients of Z (ndarray)."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, Z_lmn):
        self._Z_lmn = Z_lmn
        self._x = np.concatenate([self.R_lmn, self.Z_lmn, self.L_lmn])

    @property
    def L_lmn(self):
        """Spectral coefficients of lambda (ndarray)."""
        return self._L_lmn

    @L_lmn.setter
    def L_lmn(self, L_lmn):
        self._L_lmn = L_lmn
        self._x = np.concatenate([self.R_lmn, self.Z_lmn, self.L_lmn])

    @property
    def Rb_lmn(self):
        """Spectral coefficients of R at the boundary (ndarray)."""
        return self._Rb_lmn

    @Rb_lmn.setter
    def Rb_lmn(self, Rb_lmn):
        self._Rb_lmn = Rb_lmn

    @property
    def Zb_lmn(self):
        """Spectral coefficients of Z at the boundary (ndarray)."""
        return self._Zb_lmn

    @Zb_lmn.setter
    def Zb_lmn(self, Zb_lmn):
        self._Zb_lmn = Zb_lmn

    @property
    def p_l(self):
        """Spectral coefficients of pressure profile (ndarray)."""
        return self._p_l

    @p_l.setter
    def p_l(self, p_l):
        self._p_l = p_l

    @property
    def i_l(self):
        """Spectral coefficients of iota profile (ndarray)."""
        return self._i_l

    @i_l.setter
    def i_l(self, i_l):
        self._i_l = i_l

    @property
    def R_basis(self):
        """Spectral basis for R (FourierZernikeBasis)."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z (FourierZernikeBasis)."""
        return self._Z_basis

    @property
    def L_basis(self):
        """Spectral basis for lambda (FourierZernikeBasis)."""
        return self._L_basis

    @property
    def Rb_basis(self):
        """Spectral basis for R at the boundary (Basis)."""
        return self._Rb_basis

    @property
    def Zb_basis(self):
        """Spectral basis for Z at the boundary (Basis)."""
        return self._Zb_basis

    @property
    def p_basis(self):
        """Spectral basis for pressure (PowerSeries)."""
        return self._p_basis

    @property
    def i_basis(self):
        """Spectral basis for rotational transform (PowerSeries)."""
        return self._i_basis

    @property
    def zeta_ratio(self):
        """Multiplier on toroidal derivatives (float)."""
        return self._zeta_ratio

    @zeta_ratio.setter
    def zeta_ratio(self, zeta_ratio):
        self._zeta_ratio = zeta_ratio

    def _make_labels(self):
        R_label = ["R_{},{},{}".format(l, m, n) for l, m, n in self.R_basis.modes]
        Z_label = ["Z_{},{},{}".format(l, m, n) for l, m, n in self.Z_basis.modes]
        L_label = ["L_{},{},{}".format(l, m, n) for l, m, n in self.L_basis.modes]

        x_label = R_label + Z_label + L_label

        self.xlabel = {i: val for i, val in enumerate(x_label)}
        self.rev_xlabel = {val: i for i, val in self.xlabel.items()}

    def get_xlabel_by_idx(self, idx):
        """Find which mode corresponds to a given entry in x.

        Parameters
        ----------
        idx : int or array-like of int
            index into optimization vector x

        Returns
        -------
        label : str or list of str
            label for the coefficient at index idx, eg R_0,1,3 or L_4,3,0

        """
        self._make_labels()
        idx = np.atleast_1d(idx)
        labels = [self.xlabel.get(i, None) for i in idx]
        return labels

    def get_idx_by_xlabel(self, labels):
        """Find which index of x corresponds to a given mode.

        Parameters
        ----------
        label : str or list of str
            label for the coefficient at index idx, eg R_0,1,3 or L_4,3,0

        Returns
        -------
        idx : int or array-like of int
            index into optimization vector x

        """
        self._make_labels()
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        idx = [self.rev_xlabel.get(label, None) for label in labels]
        return np.array(idx)

    def compute_profiles(self, grid):
        """Compute magnetic flux, pressure, and rotational transform profiles.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        profiles : dict
            dictionary of ndarray, shape(num_nodes,) of profiles.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=0, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=0, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=0, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        profiles = compute_profiles(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.Z_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return profiles

    def compute_toroidal_coords(self, grid):
        """Compute toroidal coordinates from polar coordinates.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        toroidal_coords : dict
            dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """

        # TODO: option to return intermediate variables for all these
        R_transform = Transform(grid, self.R_basis, derivs=0, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=0, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=0, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=0, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=0, method="fft")

        toroidal_coords = compute_toroidal_coords(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return toroidal_coords

    def compute_cartesian_coords(self, grid):
        """Compute cartesian coordinates from toroidal coordinates.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        toroidal_coords : dict
            dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
            Keys are of the form 'X_y' meaning the derivative of X wrt to y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=0, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=0, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=0, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=0, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=0, method="fft")

        (cartesian_coords, toroidal_coords) = compute_cartesian_coords(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return cartesian_coords

    def compute_covariant_basis(self, grid):
        """Compute covariant basis vectors.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        cov_basis : dict
            dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
            Keys are of the form 'e_x_y', meaning the covariant basis vector in
            the x direction, differentiated wrt to y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=1, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=1, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=0, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=0, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=0, method="fft")

        (cov_basis, toroidal_coords) = compute_covariant_basis(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return cov_basis

    def compute_jacobian(self, grid):
        """Compute coordinate system jacobian.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        jacobian : dict
            dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
            Keys are of the form 'g_x' meaning the x derivative of the coordinate
            system jacobian g.

        """
        R_transform = Transform(grid, self.R_basis, derivs=1, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=1, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=0, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=0, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=0, method="fft")

        (jacobian, cov_basis, toroidal_coords) = compute_jacobian(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return jacobian

    def compute_contravariant_basis(self, grid):
        """Compute contravariant basis vectors.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        con_basis : dict
            dictionary of ndarray, shape(3,num_nodes), of contravariant basis vectors.
            Keys are of the form 'e^x_y', meaning the contravariant basis vector
            in the x direction, differentiated wrt to y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=1, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=1, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=0, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=0, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=0, method="fft")

        (con_basis, jacobian, cov_basis, toroidal_coords) = compute_contravariant_basis(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return con_basis

    def compute_magnetic_field(self, grid):
        """Compute magnetic field components.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        magnetic_field: dict
            dictionary of ndarray, shape(num_nodes,) of magnetic field components.
            Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
            or contravariant (B^x) component of the magnetic field, with the
            derivative wrt to y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=2, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=1, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        (
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_magnetic_field_magnitude_axis(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return magnetic_field

    def compute_current_density(self, grid):
        """Compute current density field components.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        current_density : dict
            dictionary of ndarray, shape(num_nodes,), of current density components.
            Keys are of the form 'J^x_y' meaning the contravariant (J^x)
            component of the current, with the derivative wrt to y.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=2, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=2, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        (
            current_density,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_current_density(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return current_density

    def compute_magnetic_pressure_gradient(self, grid):
        """Compute magnetic pressure gradient components and its magnitude.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        magnetic_pressure : dict
            dictionary of ndarray, shape(num_nodes,), of magnetic pressure gradient components.
            Keys are of the form 'grad_B^x' meaning the contravariant (grad_B^x) component of the
            magnetic pressure gradient.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=2, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=2, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        (
            magnetic_pressure,
            current_density,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_magnetic_pressure_gradient(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return magnetic_pressure

    def compute_magnetic_tension(self, grid):
        """Compute magnetic tension vector and its magnitude.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        magnetic_tension : dict
            dictionary of ndarray, shape(num_nodes,), of magnetic tension vector.
            Keys are of the form 'gradB' for the vector form and '|gradB|' for its
            magnitude.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=2, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=2, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        (
            magnetic_tension,
            current_density,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_magnetic_tension(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return magnetic_tension

    def compute_force_error(self, grid):
        """Compute force errors and magnitude.

        Parameters
        ----------
        grid : Grid
            Collocation grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at.

        Returns
        -------
        force_error : dict
            dictionary of ndarray, shape(num_nodes,), of force error components.
            Keys are of the form 'F_x' meaning the covariant (F_x) component of the
            force error.

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=2, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=2, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        (
            force_error,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_force_error_magnitude(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return force_error

    def compute_energy(self, grid):
        """Compute total MHD energy.

        Also computes the individual components (magnetic and pressure)

        Parameters
        ----------
        grid : Grid
            Quadrature grid containing the (rho, theta, zeta) coordinates of
            the nodes to evaluate at

        Returns
        -------
        energy : dict
            Keys are 'W_B' for magnetic energy (B**2 / 2mu0 integrated over volume),
            'W_p' for pressure energy (-p integrated over volume), and 'W' for total
            MHD energy (W_B + W_p)

        """
        R_transform = Transform(grid, self.R_basis, derivs=2, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=2, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=2, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")

        (
            energy,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_energy(
            self.Psi,
            self.R_lmn,
            self.Z_lmn,
            self.L_lmn,
            self.p_l,
            self.i_l,
            R_transform,
            Z_transform,
            L_transform,
            p_transform,
            i_transform,
            self.zeta_ratio,
        )

        return energy

    def compute_axis_location(self, zeta=0):
        """Find the axis location on specified zeta plane(s).

        Parameters
        ----------
        zeta : float or array-like of float
            zeta planes to find axis on

        Returns
        -------
        R0 : ndarray
            R coordinate of axis on specified zeta planes
        Z0 : ndarray
            Z coordinate of axis on specified zeta planes

        """
        z = np.atleast_1d(zeta).flatten()
        r = np.zeros_like(z)
        t = np.zeros_like(z)
        nodes = np.array([r, t, z]).T
        R0 = np.dot(self.R_basis.evaluate(nodes), self.R_lmn)
        Z0 = np.dot(self.Z_basis.evaluate(nodes), self.Z_lmn)

        return R0, Z0

    def is_nested(self, nsurfs=10, ntheta=20, zeta=0, Nt=45, Nr=20):
        """Check that an equilibrium has properly nested flux surfaces in a plane.

        Parameters
        ----------
        nsurfs : int, optional
            number of radial surfaces to check (Default value = 10)
        ntheta : int, optional
            number of sfl poloidal contours to check (Default value = 20)
        zeta : float, optional
            toroidal plane to check (Default value = 0)
        Nt : int, optional
            number of theta points to use for the r contours (Default value = 45)
        Nr : int, optional
            number of r points to use for the theta contours (Default value = 20)

        Returns
        -------
        is_nested : bool
            whether or not the surfaces are nested

        """
        r_grid = LinearGrid(L=nsurfs, M=Nt, zeta=zeta, endpoint=True)
        t_grid = LinearGrid(L=Nr, M=ntheta, zeta=zeta, endpoint=False)

        r_coords = self.compute_toroidal_coords(r_grid)
        t_coords = self.compute_toroidal_coords(t_grid)

        v_nodes = t_grid.nodes
        v_nodes[:, 1] = t_grid.nodes[:, 1] - t_coords["lambda"]
        v_grid = Grid(v_nodes)
        v_coords = self.compute_toroidal_coords(v_grid)

        # rho contours
        Rr = r_coords["R"].reshape((r_grid.L, r_grid.M, r_grid.N))[:, :, 0]
        Zr = r_coords["Z"].reshape((r_grid.L, r_grid.M, r_grid.N))[:, :, 0]

        # theta contours
        Rv = v_coords["R"].reshape((t_grid.L, t_grid.M, t_grid.N))[:, :, 0]
        Zv = v_coords["Z"].reshape((t_grid.L, t_grid.M, t_grid.N))[:, :, 0]

        rline = MultiLineString(
            [LineString(np.array([R, Z]).T) for R, Z in zip(Rr, Zr)]
        )
        vline = MultiLineString(
            [LineString(np.array([R, Z]).T) for R, Z in zip(Rv.T, Zv.T)]
        )

        return rline.is_simple and vline.is_simple

    def compute_dW(self, free_bdry=True, grid=None):
        """Compute the dW ideal MHD stability matrix, ie the Hessian of the energy.

        Parameters
        ----------
        grid : Grid, optional
            grid to use for computation. If None, a QuadratureGrid is created

        Returns
        -------
        dW : ndarray
            symmetric matrix whose eigenvalues determine mhd stability and eigenvectors
            describe the shape of unstable perturbations

        """
        if grid is None:
            grid = QuadratureGrid(L=2 * self.L + 1, M=2 * self.M + 1, N=2 * self.N + 1)
        R_transform = Transform(grid, self.R_basis, derivs=1, method="fft")
        Z_transform = Transform(grid, self.Z_basis, derivs=1, method="fft")
        L_transform = Transform(grid, self.L_basis, derivs=1, method="fft")
        p_transform = Transform(grid, self.p_basis, derivs=1, method="fft")
        i_transform = Transform(grid, self.i_basis, derivs=1, method="fft")
        Rb_transform = Transform(grid, self.Rb_basis, derivs=1, method="fft")
        Zb_transform = Transform(grid, self.Zb_basis, derivs=1, method="fft")

        obj = get_objective_function(
            "energy",
            R_transform,
            Z_transform,
            L_transform,
            Rb_transform,
            Zb_transform,
            p_transform,
            i_transform,
            BC_constraint=None,
            use_jit=False,
        )
        x = self.x
        dW = obj.hess_x(
            x, self.Rb_lmn, self.Zb_lmn, self.p_l, self.i_l, self.Psi, self.zeta_ratio
        )
        return dW


def format_profiles(profiles, p_basis, i_basis):
    """Format profile input arrays.

    Parameters
    ----------
    profiles : ndarray, shape(Nprof,3)
        array of fourier coeffs [l, p, i]
    p_basis : PowerSeries
        spectral basis for p_l coefficients
    i_basis : PowerSeries
        spectral basis for i_l coefficients

    Returns
    -------
    p_l : ndarray
        spectral coefficients for pressure profile
    i_l : ndarray
        spectral coefficients for rotational transform profile

    """
    p_l = np.zeros((p_basis.num_modes,))
    i_l = np.zeros((i_basis.num_modes,))

    for l, p, i in profiles:
        idx_p = np.where(p_basis.modes[:, 0] == int(l))[0]
        idx_i = np.where(i_basis.modes[:, 0] == int(l))[0]
        p_l[idx_p] = p
        i_l[idx_i] = i

    return p_l.astype(float), i_l.astype(float)


def format_boundary(boundary, Rb_basis, Zb_basis, mode="lcfs"):
    """Format boundary arrays and converts between real and fourier representations.

    Parameters
    ----------
    boundary : ndarray, shape(Nbdry,5)
        array of fourier coeffs [l, m, n, Rb_lmn, Zb_lmn]
        or array of real space coordinates, [rho, theta, phi, R, Z]
    Rb_basis : DoubleFourierSeries
        spectral basis for Rb_lmn coefficients
    Zb_basis : DoubleFourierSeries
        spectral basis for Zb_lmn coefficients
    mode : str
        One of 'lcfs', 'poincare'.
        Whether the boundary condition is specified by the last closed flux surface
        (rho=1) or the Poincare section (zeta=0).

    Returns
    -------
    Rb_lmn : ndarray
        spectral coefficients for R boundary
    Zb_lmn : ndarray
        spectral coefficients for Z boundary

    """
    Rb_lmn = np.zeros((Rb_basis.num_modes,))
    Zb_lmn = np.zeros((Zb_basis.num_modes,))

    if mode == "lcfs":
        # boundary is on m,n LCFS
        for m, n, R1, Z1 in boundary[:, 1:]:
            idx_R = np.where((Rb_basis.modes[:, 1:] == [int(m), int(n)]).all(axis=1))[0]
            idx_Z = np.where((Zb_basis.modes[:, 1:] == [int(m), int(n)]).all(axis=1))[0]
            Rb_lmn[idx_R] = R1
            Zb_lmn[idx_Z] = Z1
    elif mode == "poincare":
        # boundary is on l,m poincare section
        for l, m, R1, Z1 in boundary[:, (0, 1, 3, 4)]:
            idx_R = np.where((Rb_basis.modes[:, :2] == [int(l), int(m)]).all(axis=1))[0]
            idx_Z = np.where((Zb_basis.modes[:, :2] == [int(l), int(m)]).all(axis=1))[0]
            Rb_lmn[idx_R] = R1
            Zb_lmn[idx_Z] = Z1
    else:
        raise ValueError("Boundary mode should be either 'lcfs' or 'poincare'.")

    return Rb_lmn.astype(float), Zb_lmn.astype(float)


def initial_guess(x_basis, b_lmn, b_basis, axis, mode="lcfs"):
    """Create an initial guess from the boundary coefficients and magnetic axis guess.

    Parameters
    ----------
    x_basis : FourierZernikeBais
        basis of the flux surfaces (for R, Z, or Lambda).
    b_lmn : ndarray, shape(b_basis.num_modes,)
        vector of boundary coefficients associated with b_basis.
    b_basis : Basis
        basis of the boundary surface (for Rb or Zb)
    axis : ndarray, shape(num_modes,2)
        coefficients of the magnetic axis. axis[i, :] = [n, x0].
        Only used for 'lcfs' boundary mode.
    mode : str
        One of 'lcfs', 'poincare'.
        Whether the boundary condition is specified by the last closed flux surface
        (rho=1) or the Poincare section (zeta=0).

    Returns
    -------
    x_lmn : ndarray
        vector of flux surface coefficients associated with x_basis.

    """
    x_lmn = np.zeros((x_basis.num_modes,))

    if mode == "lcfs":
        for k, (l, m, n) in enumerate(b_basis.modes):
            idx = np.where((x_basis.modes == [np.abs(m), m, n]).all(axis=1))[0]
            if m == 0:
                idx2 = np.where((x_basis.modes == [np.abs(m) + 2, m, n]).all(axis=1))[0]
                x0 = np.where(axis[:, 0] == n, axis[:, 1], b_lmn[k])[0]
                x_lmn[idx] = x0
                x_lmn[idx2] = b_lmn[k] - x0
            else:
                x_lmn[idx] = b_lmn[k]

    elif mode == "poincare":
        for k, (l, m, n) in enumerate(b_basis.modes):
            idx = np.where((x_basis.modes == [l, m, n]).all(axis=1))[0]
            x_lmn[idx] = b_lmn[k]

    else:
        raise ValueError("Boundary mode should be either 'lcfs' or 'poincare'.")

    return x_lmn
