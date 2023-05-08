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
import smoothing_library as SL
from tqdm import trange

from .utils import force_single_precision
from ..read.utils import radec_to_cartesian, real2redshift


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

    def evaluate_cartesian(self, *fields, pos):
        """
        Evaluate a scalar field at Cartesian coordinates using CIC
        interpolation.

        Parameters
        ----------
        field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
            Fields to be interpolated.
        pos : 2-dimensional array of shape `(n_samples, 3)`
            Positions to evaluate the density field. Assumed to be in box
            units.

        Returns
        -------
        interp_fields : (list of) 1-dimensional array of shape `(n_samples,).
        """
        pos = force_single_precision(pos, "pos")

        nsamples = pos.shape[0]
        interp_fields = [numpy.full(nsamples, numpy.nan, dtype=numpy.float32)
                         for __ in range(len(fields))]
        for i, field in enumerate(fields):
            MASL.CIC_interp(field, self.boxsize, pos, interp_fields[i])

        if len(fields) == 1:
            return interp_fields[0]
        return interp_fields

    def evaluate_sky(self, *fields, pos, isdeg=True):
        """
        Evaluate the scalar fields at given distance, right ascension and
        declination. Assumes an observed in the centre of the box, with
        distance being in :math:`Mpc`. Uses CIC interpolation.

        Parameters
        ----------
        fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
            Field to be interpolated.
        pos : 2-dimensional array of shape `(n_samples, 3)`
            Spherical coordinates to evaluate the field. Columns are distance,
            right ascension, declination, respectively.
        isdeg : bool, optional
            Whether `ra` and `dec` are in degres. By default `True`.

        Returns
        -------
        interp_fields : (list of) 1-dimensional array of shape `(n_samples,).
        """
        pos = force_single_precision(pos, "pos")
        # We first calculate convert the distance to box coordinates and then
        # convert to Cartesian coordinates.
        X = numpy.copy(pos)
        X[:, 0] = self.box.mpc2box(X[:, 0])
        X = radec_to_cartesian(pos, isdeg)
        # Then we move the origin to match the box coordinates
        X -= 0.5
        return self.evaluate_field(*fields, pos=X)

    def make_sky(self, field, angpos, dist, verbose=True):
        r"""
        Make a sky map of a scalar field. The observer is in the centre of the
        box the field is evaluated along directions `angpos`. Along each
        direction, the field is evaluated distances `dist_marg` and summed.
        Uses CIC interpolation.

        Parameters
        ----------
        field : 3-dimensional array of shape `(grid, grid, grid)`
            Field to be interpolated
        angpos : 2-dimensional arrays of shape `(ndir, 2)`
            Directions to evaluate the field. Assumed to be RA
            :math:`\in [0, 360]` and dec :math:`\in [-90, 90]` degrees,
            respectively.
        dist : 1-dimensional array
            Radial distances to evaluate the field.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        interp_field : 1-dimensional array of shape `(n_pos, )`.
        """
        assert angpos.ndim == 2 and dist.ndim == 1
        # We loop over the angular directions, at each step evaluating a vector
        # of distances. We pre-allocate arrays for speed.
        dir_loop = numpy.full((dist.size, 3), numpy.nan, dtype=numpy.float32)
        ndir = angpos.shape[0]
        out = numpy.zeros(ndir, numpy.nan, dtype=numpy.float32)
        for i in trange(ndir) if verbose else range(ndir):
            dir_loop[1, :] = angpos[i, 0]
            dir_loop[2, :] = angpos[i, 1]
            out[i] = numpy.sum(self.evaluate_sky(field, dir_loop, isdeg=True))
        return out


###############################################################################
#                         Density field calculation                           #
###############################################################################


class DensityField(BaseField):
    r"""
    Density field calculations. Based primarily on routines of Pylians [1].

    Parameters
    ----------
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

    def __init__(self, box, MAS):
        self.box = box
        self.MAS = MAS

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

    def __call__(self, parts, grid, in_rsp, flip_xz=True, nbatch=30,
                 verbose=True):
        """
        Calculate the density field using a Pylians routine [1, 2].
        Iteratively loads the particles into memory, flips their `x` and `z`
        coordinates. Particles are assumed to be in box units, with positions
        in [0, 1] and observer in the centre of the box if RSP is applied.

        Parameters
        ----------
        parts : 2-dimensional array of shape `(n_parts, 7)`
            Particle positions, velocities and masses.
            Columns are: `x`, `y`, `z`, `vx`, `vy`, `vz`, `M`.
        grid : int
            Grid size.
        in_rsp : bool
            Whether to calculate the density field in redshift space.
        flip_xz : bool, optional
            Whether to flip the `x` and `z` coordinates.
        nbatch : int, optional
            Number of batches to split the particle loading into.
        verbose : bool, optional
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
        rho = numpy.zeros((grid, grid, grid), dtype=numpy.float32)

        nparts = parts.shape[0]
        batch_size = nparts // nbatch
        start = 0
        for __ in trange(nbatch + 1) if verbose else range(nbatch + 1):
            end = min(start + batch_size, nparts)
            pos = parts[start:end]
            pos, vel, mass = pos[:, :3], pos[:, 3:6], pos[:, 6]

            pos = force_single_precision(pos, "particle_position")
            vel = force_single_precision(vel, "particle_velocity")
            mass = force_single_precision(mass, "particle_mass")
            if flip_xz:
                pos[:, [0, 2]] = pos[:, [2, 0]]
                vel[:, [0, 2]] = vel[:, [2, 0]]

            if in_rsp:
                pos = real2redshift(pos, vel, [0.5, 0.5, 0.5], self.box,
                                    in_box_units=True, periodic_wrap=True,
                                    make_copy=False)

            MASL.MA(pos, rho, self.boxsize, self.MAS, W=mass, verbose=False)
            if end == nparts:
                break
            start = end
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
