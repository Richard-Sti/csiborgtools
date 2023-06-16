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

import numpy

import MAS_library as MASL
from numba import jit
from tqdm import trange

from ..read.utils import real2redshift
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
        box : :py:class:`csiborgtools.units.CSiBORGBox`
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


###############################################################################
#                         Density field calculation                           #
###############################################################################


class DensityField(BaseField):
    r"""
    Density field calculation. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    box : :py:class:`csiborgtools.read.CSiBORGBox`
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
#                         Density field calculation                           #
###############################################################################


class VelocityField(BaseField):
    r"""
    Velocity field calculation. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    box : :py:class:`csiborgtools.read.CSiBORGBox`
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

    @staticmethod
    @jit(nopython=True)
    def radial_velocity(rho_vel):
        """
        Calculate the radial velocity field around the observer in the centre
        of the box.

        Parameters
        ----------
        rho_vel : 4-dimensional array of shape `(3, grid, grid, grid)`.
            Velocity field along each axis.

        Returns
        -------
        radvel : 3-dimensional array of shape `(grid, grid, grid)`.
            Radial velocity field.
        """
        grid = rho_vel.shape[1]
        radvel = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        for i in range(grid):
            px = i - 0.5 * (grid - 1)
            for j in range(grid):
                py = j - 0.5 * (grid - 1)
                for k in range(grid):
                    pz = k - 0.5 * (grid - 1)
                    vx, vy, vz = rho_vel[:, i, j, k]
                    radvel[i, j, k] = ((px * vx + py * vy + pz * vz)
                                       / numpy.sqrt(px**2 + py**2 + pz**2))
        return radvel

    def __call__(self, parts, grid, mpart, flip_xz=True, nbatch=30,
                 verbose=True):
        """
        Calculate the velocity field using a Pylians routine [1, 2].
        Iteratively loads the particles into memory, flips their `x` and `z`
        coordinates. Particles are assumed to be in box units.

        Parameters
        ----------
        parts : 2-dimensional array of shape `(n_parts, 7)`
            Particle positions, velocities and masses.
            Columns are: `x`, `y`, `z`, `vx`, `vy`, `vz`, `M`.
        grid : int
            Grid size.
        mpart : float
            Particle mass.
        flip_xz : bool, optional
            Whether to flip the `x` and `z` coordinates.
        nbatch : int, optional
            Number of batches to split the particle loading into.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        rho_vel : 4-dimensional array of shape `(3, grid, grid, grid)`.
            Velocity field along each axis.

        References
        ----------
        [1] https://pylians3.readthedocs.io/
        [2] https://github.com/franciscovillaescusa/Pylians3/blob/master
            /library/MAS_library/MAS_library.pyx
        """
        rho_velx = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        rho_vely = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        rho_velz = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        rho_vel = [rho_velx, rho_vely, rho_velz]

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
            vel *= mass.reshape(-1, 1) / mpart

            for i in range(3):
                MASL.MA(pos, rho_vel[i], self.boxsize, self.MAS, W=vel[:, i],
                        verbose=False)
            if end == nparts:
                break
            start = end

        return numpy.stack(rho_vel)


###############################################################################
#                         Potential field calculation                         #
###############################################################################


class PotentialField(BaseField):
    """
    Potential field calculation.

    Parameters
    ----------
    box : :py:class:`csiborgtools.read.CSiBORGBox`
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
    box : :py:class:`csiborgtools.read.CSiBORGBox`
        The simulation box information and transformations.
    MAS : str
        Mass assignment scheme used to calculate the density field. Options
        are: 'NGP' (nearest grid point), 'CIC' (cloud-in-cell), 'TSC'
        (triangular-shape cloud), 'PCS' (piecewise cubic spline).
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
        grid = tidal_tensor.T00.shape[0]
        eigvals = numpy.full((grid, grid, grid, 3), numpy.nan,
                             dtype=numpy.float32)
        dummy_vector = numpy.full(3, numpy.nan, dtype=numpy.float32)
        dummy_tensor = numpy.full((3, 3), numpy.nan, dtype=numpy.float32)

        tidal_tensor_to_eigenvalues(eigvals, dummy_vector, dummy_tensor,
                                    tidal_tensor.T00, tidal_tensor.T01,
                                    tidal_tensor.T02, tidal_tensor.T11,
                                    tidal_tensor.T12, tidal_tensor.T22)
        return eigvals

    @staticmethod
    def eigvals_to_environment(eigvals, threshold=0.0):
        """
        Calculate the environment of each grid cell based on the eigenvalues
        of the tidal tensor field.

        Parameters
        ----------
        eigvals : 4-dimensional array of shape `(grid, grid, grid, 3)`
            The eigenvalues of the tidal tensor field.

        Returns
        -------
        environment : 3-dimensional array of shape `(grid, grid, grid)`
            The environment of each grid cell. Possible values are 0 (void),
            1 (sheet), 2 (filament), 3 (knot).
        """
        environment = numpy.full(eigvals.shape[:-1], numpy.nan,
                                 dtype=numpy.float32)
        eigenvalues_to_environment(environment, eigvals, threshold)
        return environment

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


@jit(nopython=True)
def tidal_tensor_to_eigenvalues(eigvals, dummy_vector, dummy_tensor,
                                T00, T01, T02, T11, T12, T22):
    """
    Calculate eigenvalues of the tidal tensor field, sorted in decreasing
    absolute value order. JIT implementation to speed up the work.

    Parameters
    ----------
    eigvals : 3-dimensional array of shape `(grid, grid, grid)`
        Array to store the eigenvalues.
    dummy_vector : 1-dimensional array of shape `(3,)`
        Dummy vector to store the eigenvalues.
    dummy_tensor : 2-dimensional array of shape `(3, 3)`
        Dummy tensor to store the tidal tensor.
    T00 : 3-dimensional array of shape `(grid, grid, grid)`
        Tidal tensor component :math:`T_{00}`.
    T01 : 3-dimensional array of shape `(grid, grid, grid)`
        Tidal tensor component :math:`T_{01}`.
    T02 : 3-dimensional array of shape `(grid, grid, grid)`
        Tidal tensor component :math:`T_{02}`.
    T11 : 3-dimensional array of shape `(grid, grid, grid)`
        Tidal tensor component :math:`T_{11}`.
    T12 : 3-dimensional array of shape `(grid, grid, grid)`
        Tidal tensor component :math:`T_{12}`.
    T22 : 3-dimensional array of shape `(grid, grid, grid)`
        Tidal tensor component :math:`T_{22}`.

    Returns
    -------
    eigvals : 3-dimensional array of shape `(grid, grid, grid)`
    """
    grid = T00.shape[0]
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                dummy_tensor[0, 0] = T00[i, j, k]
                dummy_tensor[1, 1] = T11[i, j, k]
                dummy_tensor[2, 2] = T22[i, j, k]

                dummy_tensor[0, 1] = T01[i, j, k]
                dummy_tensor[1, 0] = T01[i, j, k]

                dummy_tensor[0, 2] = T02[i, j, k]
                dummy_tensor[2, 0] = T02[i, j, k]

                dummy_tensor[1, 2] = T12[i, j, k]
                dummy_tensor[2, 1] = T12[i, j, k]
                dummy_vector[:] = numpy.linalg.eigvalsh(dummy_tensor)

                eigvals[i, j, k, :] = dummy_vector[
                    numpy.argsort(numpy.abs(dummy_vector))][::-1]
    return eigvals


@jit(nopython=True)
def eigenvalues_to_environment(environment, eigvals, th):
    """
    Classify the environment of each grid cell based on the eigenvalues of the
    tidal tensor field.

    Parameters
    ----------
    environment : 3-dimensional array of shape `(grid, grid, grid)`
        Array to store the environment.
    eigvals : 4-dimensional array of shape `(grid, grid, grid, 3)`
        The eigenvalues of the tidal tensor field.
    th : float
        Threshold value to classify the environment.

    Returns
    -------
    environment : 3-dimensional array of shape `(grid, grid, grid)`
    """
    grid = eigvals.shape[0]
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                lmbda1, lmbda2, lmbda3 = eigvals[i, j, k, :]
                if lmbda1 < th and lmbda2 < th and lmbda3 < th:
                    environment[i, j, k] = 0
                elif lmbda1 < th and lmbda2 < th:
                    environment[i, j, k] = 1
                elif lmbda1 < th:
                    environment[i, j, k] = 2
                else:
                    environment[i, j, k] = 3
    return environment
