# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Density field and cross-correlation calculations.
"""
from abc import ABC

import MAS_library as MASL
import numpy
import Pk_library as PKL
import smoothing_library as SL
from tqdm import trange

from .utils import force_single_precision


class BaseField(ABC):
    """Base class for density field calculations."""
    _box = None
    _MAS = None

    @property
    def boxsize(self):
        """
        Box size. Particle positions are always assumed to be in box units,
        therefore this is 1.

        Returns
        -------
        boxsize : float
        """
        return 1.

    @property
    def box(self):
        """
        Simulation box information and transformations.

        Returns
        -------
        box : :py:class:`csiborgtools.units.BoxUnits`
        """
        return self._box

    @box.setter
    def box(self, box):
        try:
            assert box._name == "box_units"
            self._box = box
        except AttributeError as err:
            raise TypeError from err

    @property
    def MAS(self):
        """
        Mass-assignment scheme.

        Returns
        -------
        MAS : str
        """
        if self._MAS is None:
            raise ValueError("`mas` is not set.")
        return self._MAS

    @MAS.setter
    def MAS(self, MAS):
        assert MAS in ["NGP", "CIC", "TSC", "PCS"]
        self._MAS = MAS

#     def evaluate_field(self, *field, pos):
#         """
#         Evaluate the field at Cartesian coordinates using CIC interpolation.
#
#         Parameters
#         ----------
#         field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
#             Density field that is to be interpolated.
#         pos : 2-dimensional array of shape `(n_samples, 3)`
#             Positions to evaluate the density field. The coordinates span range
#             of [0, boxsize].
#
#         Returns
#         -------
#         interp_field : (list of) 1-dimensional array of shape `(n_samples,).
#         """
#         self._force_f32(pos, "pos")
#
#         interp_field = [numpy.zeros(pos.shape[0], dtype=numpy.float32)
#                         for __ in range(len(field))]
#         for i, f in enumerate(field):
#             MASL.CIC_interp(f, self.boxsize, pos, interp_field[i])
#         return interp_field
#
#     def evaluate_sky(self, *field, pos, isdeg=True):
#         """
#         Evaluate the field at given distance, right ascension and declination.
#         Assumes that the observed is in the centre of the box and uses CIC
#         interpolation.
#
#         Parameters
#         ----------
#         field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
#             Density field that is to be interpolated. Assumed to be defined
#             on a Cartesian grid.
#         pos : 2-dimensional array of shape `(n_samples, 3)`
#             Spherical coordinates to evaluate the field. Should be distance,
#             right ascension, declination, respectively.
#         isdeg : bool, optional
#             Whether `ra` and `dec` are in degres. By default `True`.
#
#         Returns
#         -------
#         interp_field : (list of) 1-dimensional array of shape `(n_samples,).
#         """
#         # TODO: implement this
#         raise NotImplementedError("This method is not yet implemented.")
# #         self._force_f32(pos, "pos")
# #         X = numpy.vstack(
# #             radec_to_cartesian(*(pos[:, i] for i in range(3)), isdeg)).T
# #         X = X.astype(numpy.float32)
# #         # Place the observer at the center of the box
# #         X += 0.5 * self.boxsize
# #         return self.evaluate_field(*field, pos=X)
#
#     def make_sky_map(self, ra, dec, field, dist_marg, isdeg=True,
#                      verbose=True):
#         """
#         Make a sky map of a density field. Places the observed in the center of
#         the box and evaluates the field in directions `ra`, `dec`. At each such
#         position evaluates the field at distances `dist_marg` and sums these
#         interpolated values of the field.
#
#         NOTE: Supports only scalar fields.
#
#         Parameters
#         ----------
#         ra, dec : 1-dimensional arrays of shape `(n_pos, )`
#             Directions to evaluate the field. Assumes `dec` is in [-90, 90]
#             degrees (or equivalently in radians).
#         field : 3-dimensional array of shape `(grid, grid, grid)`
#             Density field that is to be interpolated. Assumed to be defined
#             on a Cartesian grid `[0, self.boxsize]^3`.
#         dist_marg : 1-dimensional array
#             Radial distances to evaluate the field.
#         isdeg : bool, optional
#             Whether `ra` and `dec` are in degres. By default `True`.
#         verbose : bool, optional
#             Verbosity flag.
#
#         Returns
#         -------
#         interp_field : 1-dimensional array of shape `(n_pos, )`.
#         """
#         # Angular positions at which to evaluate the field
#         Nang = ra.size
#         pos = numpy.vstack([ra, dec]).T
#
#         # Now loop over the angular positions, each time evaluating a vector
#         # of distances. Pre-allocate arrays for speed
#         ra_loop = numpy.ones_like(dist_marg)
#         dec_loop = numpy.ones_like(dist_marg)
#         pos_loop = numpy.ones((dist_marg.size, 3), dtype=numpy.float32)
#
#         out = numpy.zeros(Nang, dtype=numpy.float32)
#         for i in trange(Nang) if verbose else range(Nang):
#             # Get the position vector for this choice of theta, phi
#             ra_loop[:] = pos[i, 0]
#             dec_loop[:] = pos[i, 1]
#             pos_loop[:] = numpy.vstack([dist_marg, ra_loop, dec_loop]).T
#             # Evaluate and sum it up
#             out[i] = numpy.sum(self.evaluate_sky(field, pos_loop, isdeg)[0, :])
#
#         return out


###############################################################################
#                         Density field calculation                           #
###############################################################################


class DensityField(BaseField):
    r"""
    Density field calculations. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    pos : 2-dimensional array of shape `(N, 3)`
        Particle position array. Columns must be ordered as `['x', 'y', 'z']`.
        The positions are assumed to be in box units, i.e. :math:`\in [0, 1 ]`.
    mass : 1-dimensional array of shape `(N,)`
        Particle mass array. Assumed to be in box units.
    box : :py:class:`csiborgtools.read.BoxUnits`
        The simulation box information and transformations.
    MAS : str
        Mass assignment scheme. Options are Options are: 'NGP' (nearest grid
        point), 'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).

    References
    ----------
    [1] https://pylians3.readthedocs.io/
    """
    _pos = None
    _mass = None

    def __init__(self, pos, mass, box, MAS):
        self.pos = pos
        self.mass = mass
        self.box = box
        self.MAS = MAS

    @property
    def pos(self):
        """
        Particle position array.

        Returns
        -------
        particles : 2-dimensional array
        """
        return self._particles

    @pos.setter
    def pos(self, pos):
        assert pos.ndim == 2
        pos = force_single_precision(pos)
        self._pos = pos

    @property
    def mass(self):
        """
        Particle mass array.

        Returns
        -------
        mass : 1-dimensional array
        """
        return self._mass

    @mass.setter
    def mass(self, mass):
        assert mass.ndim == 1
        mass = force_single_precision(mass)
        self._mass = mass

    def smoothen(self, field, smooth_scale, threads=1):
        """
        Smooth a field with a Gaussian filter.

        Parameters
        ----------
        field : 3-dimensional array of shape `(grid, grid, grid)`
            Field to be smoothed.
        smooth_scale : float, optional
            Gaussian kernal scale to smoothen the density field, in box units.
        threads : int, optional
            Number of threads. By default 1.

        Returns
        -------
        smoothed_field : 3-dimensional array of shape `(grid, grid, grid)`
        """
        filter_kind = "Gaussian"
        grid = field.shape[0]
        # FFT of the filter
        W_k = SL.FT_filter(self.boxsize, smooth_scale, grid, filter_kind,
                           threads)
        return SL.field_smoothing(field, W_k, threads)

    def overdensity_field(self, delta):
        r"""
        Calculate the overdensity field from the density field.
        Defined as :math:`\rho/ <\rho> - 1`. Overwrites the input array.


        Parameters
        ----------
        delta : 3-dimensional array of shape `(grid, grid, grid)`
            The density field.

        Returns
        -------
        overdensity : 3-dimensional array of shape `(grid, grid, grid)`.
        """
        delta /= delta.mean()
        delta -= 1
        return delta

    def __call__(self, grid, smooth_scale=None, verbose=True):
        """
        Calculate the density field using a Pylians routine [1, 2].

        Parameters
        ----------
        grid : int
            Grid size.
        smooth_scale : float, optional
            Gaussian kernal scale to smoothen the density field, in box units.
        verbose : bool
            Verbosity flag.

        Returns
        -------
        rho : 3-dimensional array of shape `(grid, grid, grid)`.
            Density field.

        References
        ----------
        [1] https://pylians3.readthedocs.io/
        [2] https://github.com/franciscovillaescusa/Pylians3/blob/master
            /library/MAS_library/MAS_library.pyx
        """
        # Pre-allocate and do calculations
        rho = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        MASL.MA(self.pos, rho, self.boxsize, self.MAS, W=self.mass,
                verbose=verbose)
        if smooth_scale is not None:
            rho = self.smoothen(rho, smooth_scale)
        return rho


###############################################################################
#                         Potential field calculation                         #
###############################################################################


class PotentialField(BaseField):
    """
    Potential field calculation.

    Parameters
    ----------
    box : :py:class:`csiborgtools.read.BoxUnits`
        The simulation box information and transformations.
    MAS : str
        Mass assignment scheme. Options are Options are: 'NGP' (nearest grid
        point), 'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).
    """
    def __init__(self, box, MAS):
        self.box = box
        self.MAS = MAS

    def __call__(self, overdensity_field):
        """
        Calculate the potential field.

        Parameters
        ----------
        overdensity_field : 3-dimensional array of shape `(grid, grid, grid)`
            The overdensity field.

        Returns
        -------
        potential : 3-dimensional array of shape `(grid, grid, grid)`.
        """
        return MASL.potential(overdensity_field, self.box._omega_m,
                              self.box._aexp, self.MAS)


###############################################################################
#                        Tidal tensor field calculation                       #
###############################################################################


class TidalTensorField(BaseField):
    """
    Tidal tensor field calculation.

    Parameters
    ----------
    box : :py:class:`csiborgtools.read.BoxUnits`
        The simulation box information and transformations.
    MAS : str
        Mass assignment scheme. Options are Options are: 'NGP' (nearest grid
        point), 'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).
    """
    def __init__(self, box, MAS):
        self.box = box
        self.MAS = MAS

    @staticmethod
    def tensor_field_eigvals(tidal_tensor):
        """
        Calculate eigenvalues of the tidal tensor field, sorted in increasing
        order.

        Parameters
        ----------
        tidal_tensor : :py:class:`MAS_library.tidal_tensor`
            Tidal tensor object, whose attributes `tidal_tensor.Tij` contain
            the relevant tensor components.

        Returns
        -------
        eigvals : 3-dimensional array of shape `(grid, grid, grid)`
        """
        n_samples = tidal_tensor.T00.size
        # We create a array and then calculate the eigenvalues.
        Teval = numpy.full((n_samples, 3, 3), numpy.nan, dtype=numpy.float32)
        Teval[:, 0, 0] = tidal_tensor.T00
        Teval[:, 0, 1] = tidal_tensor.T01
        Teval[:, 0, 2] = tidal_tensor.T02
        Teval[:, 1, 1] = tidal_tensor.T11
        Teval[:, 1, 2] = tidal_tensor.T12
        Teval[:, 2, 2] = tidal_tensor.T22

        eigvals = numpy.full((n_samples, 3), numpy.nan, dtype=numpy.float32)
        for i in range(n_samples):
            eigvals[i, :] = numpy.linalg.eigvalsh(Teval[i, ...], 'U')
            eigvals[i, :] = numpy.sort(eigvals[i, :])

        return eigvals

    def __call__(self, overdensity_field):
        """
        Calculate the tidal tensor field.

        Parameters
        ----------
        overdensity_field : 3-dimensional array of shape `(grid, grid, grid)`
            The overdensity field.

        Returns
        -------
        tidal_tensor : :py:class:`MAS_library.tidal_tensor`
            Tidal tensor object, whose attributes `tidal_tensor.Tij` contain
            the relevant tensor components.
        """
        return MASL.tidal_tensor(overdensity_field, self.box._omega_m,
                                 self.box._aexp, self.MAS)
