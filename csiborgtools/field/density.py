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
Density field, potential and tidal tensor field calculations. Most routines
here do not support SPH-calculated density fields because of the unknown
corrrections necessary when performing the fast Fourier transform.
"""
from abc import ABC

import MAS_library as MASL
import numpy
import Pk_library as PKL
from numba import jit
from tqdm import trange

from .utils import divide_nonzero, force_single_precision


class BaseField(ABC):
    """Base class for density field calculations."""
    _MAS = None
    _boxsize = None

    @property
    def boxsize(self):
        """Size of the box in units matching the particle coordinates."""
        return self._boxsize

    @boxsize.setter
    def boxsize(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("`boxsize` must be an integer.")

    @property
    def MAS(self):
        """Mass-assignment scheme."""
        if self._MAS is None:
            raise ValueError("`MAS` is not set.")
        return self._MAS

    @MAS.setter
    def MAS(self, MAS):
        if MAS == "SPH":
            raise ValueError("`SPH` is not supported. Use `cosmotool` scripts to calculate the density field with SPH.")  # noqa

        if MAS not in ["NGP", "CIC", "TSC", "PCS"]:
            raise ValueError(f"Invalid `MAS` value: {MAS}")

        self._MAS = MAS


###############################################################################
#                         Density field calculation                           #
###############################################################################


class DensityField(BaseField):
    r"""
    Density field calculation. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    boxsize : float
        Size of the periodic box.
    MAS : str
        Mass assignment scheme. Options are: 'NGP' (nearest grid point),
        'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).

    References
    ----------
    [1] https://pylians3.readthedocs.io/
    """

    def __init__(self, boxsize, MAS):
        self.boxsize = boxsize
        self.MAS = MAS

    def __call__(self, pos, mass, grid, nbatch=30, verbose=True):
        """
        Calculate the density field using a Pylians routine [1, 2]. Iteratively
        loads the particles into memory. Particle coordinates units should
        match that of `boxsize`.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(n_parts, 3)`
            Particle positions
        mass : 1-dimensional array of shape `(n_parts,)`
            Particle masses
        grid : int
            Grid size.
        nbatch : int, optional
            Number of batches to split the particle loading into.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        3-dimensional array of shape `(grid, grid, grid)`.

        References
        ----------
        [1] https://pylians3.readthedocs.io/
        [2] https://github.com/franciscovillaescusa/Pylians3/blob/master
            /library/MAS_library/MAS_library.pyx
        """
        rho = numpy.zeros((grid, grid, grid), dtype=numpy.float32)

        nparts = pos.shape[0]
        batch_size = nparts // nbatch
        start = 0

        for __ in trange(nbatch + 1, disable=not verbose,
                         desc="Loading particles for the density field"):
            end = min(start + batch_size, nparts)
            batch_pos = pos[start:end]
            batch_mass = mass[start:end]

            batch_pos = force_single_precision(batch_pos)
            batch_mass = force_single_precision(batch_mass)

            MASL.MA(batch_pos, rho, self.boxsize, self.MAS, W=batch_mass,
                    verbose=False)

            if end == nparts:
                break

            start = end

        # Divide by the cell volume in (kpc / h)^3
        rho /= (self.boxsize / grid * 1e3)**3

        return rho


def overdensity_field(delta, make_copy=True):
    r"""
    Get the overdensity field from the density field as `rho / <rho> - 1`.

    Parameters
    ----------
    delta : 3-dimensional array of shape `(grid, grid, grid)`
        The density field.
    make_copy : bool, optional
        Whether to make a copy of the input array.

    Returns
    -------
    3-dimensional array of shape `(grid, grid, grid)`.
    """
    if make_copy:
        delta = numpy.copy(delta)

    delta /= delta.mean()
    delta -= 1

    return delta


###############################################################################
#                         Velocity field calculation                          #
###############################################################################


class VelocityField(BaseField):
    r"""
    Velocity field calculation. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    boxsize : float
        Size of the periodic box.
    MAS : str
        Mass assignment scheme. Options are: 'NGP' (nearest grid point),
        'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).

    References
    ----------
    [1] https://pylians3.readthedocs.io/
    """

    def __init__(self, boxsize, MAS):
        self.boxsize = boxsize
        self.MAS = MAS

    def __call__(self, pos, vel, mass, grid, nbatch=30, verbose=True):
        """
        Calculate the velocity field using a Pylians routine [1, 2].
        Iteratively loads the particles into memory, flips their `x` and `z`
        coordinates. Particles are assumed to be in box units.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(n_parts, 3)`
            Particle positions.
        vel : 2-dimensional array of shape `(n_parts, 3)`
            Particle velocities.
        mass : 1-dimensional array of shape `(n_parts,)`
            Particle masses.
        grid : int
            Grid size.
        nbatch : int, optional
            Number of batches to split the particle loading into.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        4-dimensional array of shape `(3, grid, grid, grid)`.

        References
        ----------
        [1] https://pylians3.readthedocs.io/
        [2] https://github.com/franciscovillaescusa/Pylians3/blob/master
            /library/MAS_library/MAS_library.pyx
        """
        rho_vel = [numpy.zeros((grid, grid, grid), dtype=numpy.float32),
                   numpy.zeros((grid, grid, grid), dtype=numpy.float32),
                   numpy.zeros((grid, grid, grid), dtype=numpy.float32),
                   ]
        cellcounts = numpy.zeros((grid, grid, grid), dtype=numpy.float32)

        nparts = pos.shape[0]
        batch_size = nparts // nbatch
        start = 0
        for __ in trange(nbatch + 1) if verbose else range(nbatch + 1):
            end = min(start + batch_size, nparts)

            batch_pos = pos[start:end]
            batch_vel = vel[start:end]
            batch_mass = mass[start:end]

            batch_pos = force_single_precision(batch_pos)
            batch_vel = force_single_precision(batch_vel)
            batch_mass = force_single_precision(batch_mass)

            vel *= mass.reshape(-1, 1)

            for i in range(3):
                MASL.MA(pos, rho_vel[i], self.boxsize, self.MAS, W=vel[:, i],
                        verbose=False)

            MASL.MA(pos, cellcounts, self.boxsize, self.MAS, W=mass,
                    verbose=False)

            if end == nparts:
                break
            start = end

        for i in range(3):
            divide_nonzero(rho_vel[i], cellcounts)

        return numpy.stack(rho_vel)


@jit(nopython=True)
def radial_velocity(rho_vel, observer_velocity):
    """
    Calculate the radial velocity field around the observer in the centre
    of the box.

    Parameters
    ----------
    rho_vel : 4-dimensional array of shape `(3, grid, grid, grid)`.
        Velocity field along each axis.
    observer_velocity : 3-dimensional array of shape `(3,)`
        Observer velocity.

    Returns
    -------
    3-dimensional array of shape `(grid, grid, grid)`.
    """
    grid = rho_vel.shape[1]
    radvel = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
    vx0, vy0, vz0 = observer_velocity

    for i in range(grid):
        px = i - 0.5 * (grid - 1)
        for j in range(grid):
            py = j - 0.5 * (grid - 1)
            for k in range(grid):
                pz = k - 0.5 * (grid - 1)

                vx = rho_vel[0, i, j, k] - vx0
                vy = rho_vel[1, i, j, k] - vy0
                vz = rho_vel[2, i, j, k] - vz0

                radvel[i, j, k] = ((px * vx + py * vy + pz * vz)
                                   / numpy.sqrt(px**2 + py**2 + pz**2))
    return radvel


###############################################################################
#                         Potential field calculation                         #
###############################################################################


class PotentialField(BaseField):
    """
    Potential field calculation.

    Parameters
    ----------
    boxsize : float
        Size of the periodic box.
    MAS : str
        Mass assignment scheme. Options are: 'NGP' (nearest grid point),
        'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).
    """
    def __init__(self, boxsize, MAS):
        self.boxsize = boxsize
        self.MAS = MAS

    def __call__(self, overdensity_field, omega_m, aexp):
        """
        Calculate the potential field.

        Parameters
        ----------
        overdensity_field : 3-dimensional array of shape `(grid, grid, grid)`
            The overdensity field.
        omega_m : float
            TODO
        aexp : float
            TODO

        Returns
        -------
        3-dimensional array of shape `(grid, grid, grid)`.
        """
        return MASL.potential(overdensity_field, omega_m, aexp, self.MAS)


###############################################################################
#                        Tidal tensor field calculation                       #
###############################################################################


class TidalTensorField(BaseField):
    """
    Tidal tensor field calculation.

    Parameters
    ----------
    boxsize : float
        Size of the periodic box.
    MAS : str
        Mass assignment scheme. Options are Options are: 'NGP' (nearest grid
        point), 'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).
    """
    def __init__(self, boxsize, MAS):
        self.boxsize = boxsize
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
        3-dimensional array of shape `(grid, grid, grid)`
        """
        return tidal_tensor_to_eigenvalues(
            tidal_tensor.T00, tidal_tensor.T01, tidal_tensor.T02,
            tidal_tensor.T11, tidal_tensor.T12, tidal_tensor.T22)

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
        3-dimensional array of shape `(grid, grid, grid)`
            The environment of each grid cell. Possible values are 0 (void),
            1 (sheet), 2 (filament), 3 (knot).
        """
        return eigenvalues_to_environment(eigvals, threshold)

    def __call__(self, overdensity_field, omega_m, aexp):
        """
        Calculate the tidal tensor field.

        Parameters
        ----------
        overdensity_field : 3-dimensional array of shape `(grid, grid, grid)`
            The overdensity field.
        omega_m : float
            TODO
        aexp : float
            TODO

        Returns
        -------
        :py:class:`MAS_library.tidal_tensor`
            Tidal tensor object, whose attributes `tidal_tensor.Tij` contain
            the relevant tensor components.
        """
        return MASL.tidal_tensor(overdensity_field, omega_m, aexp, self.MAS)


@jit(nopython=True)
def tidal_tensor_to_eigenvalues(T00, T01, T02, T11, T12, T22):
    """
    Calculate eigenvalues of the tidal tensor field, sorted in decreasing
    absolute value order.

    Parameters
    ----------
    T00, T01, T02, T11, T12, T22 : 3-dimensional array `(grid, grid, grid)`
        Tidal tensor components.

    Returns
    -------
    3-dimensional array of shape `(grid, grid, grid)`
    """
    grid = T00.shape[0]
    eigvals = numpy.full((grid, grid, grid, 3), numpy.nan, dtype=numpy.float32)
    dummy_vector = numpy.full(3, numpy.nan, dtype=numpy.float32)
    dummy_tensor = numpy.full((3, 3), numpy.nan, dtype=numpy.float32)

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
def eigenvalues_to_environment(eigvals, th):
    """
    Classify the environment of each grid cell based on the eigenvalues of the
    tidal tensor field.

    Parameters
    ----------
    eigvals : 4-dimensional array of shape `(grid, grid, grid, 3)`
        The eigenvalues of the tidal tensor field.
    th : float
        Threshold value to classify the environment.

    Returns
    -------
    3-dimensional array of shape `(grid, grid, grid)`
    """
    env = numpy.full(eigvals.shape[:-1], numpy.nan, dtype=numpy.float32)

    grid = eigvals.shape[0]
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                lmbda1, lmbda2, lmbda3 = eigvals[i, j, k, :]
                if lmbda1 < th and lmbda2 < th and lmbda3 < th:
                    env[i, j, k] = 0
                elif lmbda1 < th and lmbda2 < th:
                    env[i, j, k] = 1
                elif lmbda1 < th:
                    env[i, j, k] = 2
                else:
                    env[i, j, k] = 3
    return env


###############################################################################
#                       Power spectrum calculation                            #
###############################################################################


def power_spectrum(delta, boxsize, MAS, threads=1, verbose=True):
    """
    Calculate the monopole power spectrum of the density field.

    Parameters
    ----------
    delta : 3-dimensional array of shape `(grid, grid, grid)`
        The over-density field.
    boxsize : float
        The simulation box size in `Mpc / h`.
    MAS : str
        Mass assignment scheme used to calculate the density field. Options
        are: 'NGP' (nearest grid point), 'CIC' (cloud-in-cell), 'TSC'
        (triangular-shape cloud), 'PCS' (piecewise cubic spline).
    threads : int, optional
        Number of threads to use.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    k, Pk : 1-dimensional arrays of shape `(grid,)`
        The wavenumbers and the power spectrum.
    """
    # Axis along which compute the quadrupole and hexadecapole, is not used
    # for the monopole that we calculat here.
    axis = 2
    Pk = PKL.Pk(delta, boxsize, axis, MAS, threads, verbose)
    return Pk.k3D, Pk.Pk[:, 0]
