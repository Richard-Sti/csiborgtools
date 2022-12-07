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
import numpy
import MAS_library as MASL
import Pk_library as PKL
import smoothing_library as SL
from warnings import warn
from tqdm import trange
from ..units import (BoxUnits, radec_to_cartesian)


class DensityField:
    r"""
    Density field calculations. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    particles : structured array
        Particle array. Must contain keys `['x', 'y', 'z', 'M']`. Particle
        coordinates are assumed to be :math:`\in [0, 1]` or in box units
        otherwise.
    boxsize : float
        Box length. Multiplies `particles` positions to fix the power spectum
        units.
    box : :py:class:`csiborgtools.units.BoxUnits`
        The simulation box information and transformations.
    MAS : str, optional
        Mass assignment scheme. Options are Options are: 'NGP' (nearest grid
        point), 'CIC' (cloud-in-cell), 'TSC' (triangular-shape cloud), 'PCS'
        (piecewise cubic spline).

    References
    ----------
    [1] https://pylians3.readthedocs.io/
    """
    _particles = None
    _boxsize = None
    _box = None
    _MAS = None

    def __init__(self, particles, boxsize, box, MAS="CIC"):
        self.particles = particles
        self.box = box
        self.boxsize = boxsize
        self.MAS = MAS

    @property
    def particles(self):
        """
        Particles structured array.

        Returns
        -------
        particles : structured array
        """
        return self._particles

    @particles.setter
    def particles(self, particles):
        """Set `particles`, checking it has the right columns."""
        if any(p not in particles.dtype.names for p in ('x', 'y', 'z', 'M')):
            raise ValueError("`particles` must be a structured array "
                             "containing `['x', 'y', 'z', 'M']`.")
        self._particles = particles

    @property
    def boxsize(self):
        """
        Box length. Determines the power spectrum units.

        Returns
        -------
        boxsize : float
        """
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        """Set `self.boxsize`."""
        self._boxsize = boxsize

    @property
    def box(self):
        """
        The simulation box information and transformations.

        Returns
        -------
        box : :py:class:`csiborgtools.units.BoxUnits`
        """
        return self._box

    @box.setter
    def box(self, box):
        """Set the simulation box."""
        if not isinstance(box, BoxUnits):
            raise TypeError("`box` must be `BoxUnits` instance.")
        self._box = box

    @property
    def MAS(self):
        """
        The mass-assignment scheme.

        Returns
        -------
        MAS : str
        """
        return self._MAS

    @MAS.setter
    def MAS(self, MAS):
        """Sets `self.MAS`."""
        opts = ["NGP", "CIC", "TSC", "PCS"]
        if MAS not in opts:
            raise ValueError("Invalid MAS `{}`. Options are: `{}`."
                             .format(MAS, opts))
        self._MAS = MAS

    @staticmethod
    def _force_f32(x, name):
        if x.dtype != numpy.float32:
            warn("Converting `{}` to float32.".format(name))
            x = x.astype(numpy.float32)
        return x

    def density_field(self, grid, verbose=True):
        """
        Calculate the density field using a Pylians routine [1, 2]. Enforces
        float32 precision.

        Parameters
        ----------
        grid : int
            The grid size.
        verbose : float, optional
            A verbosity flag. By default `True`.

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
        pos = numpy.vstack([self.particles[p] for p in ('x', 'y', 'z')]).T
        pos *= self.boxsize
        pos = self._force_f32(pos, "pos")
        weights = self._force_f32(self.particles['M'], 'M')

        # Pre-allocate and do calculations
        rho = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        MASL.MA(pos, rho, self.boxsize, self.MAS, W=weights, verbose=verbose)
        return rho

    def overdensity_field(self, grid, verbose=True):
        r"""
        Calculate the overdensity field using Pylians routines.
        Defined as :math:`\rho/ <\rho> - 1`.

        Parameters
        ----------
        grid : int
            The grid size.
        verbose : float, optional
            A verbosity flag. By default `True`.

        Returns
        -------
        overdensity : 3-dimensional array of shape `(grid, grid, grid)`.
            Overdensity field.
        """
        # Get the overdensity
        delta = self.density_field(grid, verbose)
        delta /= delta.mean()
        delta -= 1
        return delta

    def potential_field(self, grid, verbose=True):
        """
        Calculate the potential field using Pylians routines.

        Parameters
        ----------
        grid : int
            The grid size.
        verbose : float, optional
            A verbosity flag. By default `True`.

        Returns
        -------
        potential : 3-dimensional array of shape `(grid, grid, grid)`.
            Potential field.
        """
        delta = self.overdensity_field(grid, verbose)
        if verbose:
            print("Calculating potential from the overdensity..")
        return MASL.potential(
            delta, self.box._omega_m, self.box._aexp, self.MAS)

    def tensor_field(self, grid, verbose=True):
        """
        Calculate the tidal tensor field.

        Parameters
        ----------
        grid : int
            The grid size.
        verbose : float, optional
            A verbosity flag. By default `True`.

        Returns
        -------
        tidal_tensor : :py:class:`MAS_library.tidal_tensor`
            Tidal tensor object, whose attributes `tidal_tensor.Tij` contain
            the relevant tensor components.
        """
        delta = self.overdensity_field(grid, verbose)
        return MASL.tidal_tensor(
            delta, self.box._omega_m, self.box._aexp, self.MAS)

    def gravitational_field(self, grid, verbose=True):
        """
        Calculate the gravitational tensor field. Note that this method is
        only defined in fork of `Pylians`.

        Parameters
        ----------
        grid : int
            The grid size.
        verbose : float, optional
            A verbosity flag. By default `True`.

        Returns
        -------
        grav_field_tensor : :py:class:`MAS_library.grav_field_tensor`
            Tidal tensor object, whose attributes `grav_field_tensor.gi`
            contain the relevant tensor components.
        """
        delta = self.overdensity_field(grid, verbose)
        return MASL.grav_field_tensor(
            delta, self.box._omega_m, self.box._aexp, self.MAS)

    def auto_powerspectrum(self, grid, verbose=True):
        """
        Calculate the auto 1-dimensional power spectrum.

        Parameters
        ----------
        grid : int
            The grid size.
        verbose : float, optional
            A verbosity flag. By default `True`.

        Returns
        -------
        pk : py:class`Pk_library.Pk`
            Power spectrum object.
        """
        delta = self.overdensity_field(grid, verbose)
        return PKL.Pk(
            delta, self.boxsize, axis=1, MAS=self.MAS, threads=1,
            verbose=verbose)

    def smooth_field(self, field, scale, threads=1):
        """
        Smooth a field with a Gaussian filter.

        Parameters
        ----------
        field : 3-dimensional array of shape `(grid, grid, grid)`
            The field to be smoothed.
        scale : float
            The smoothing scale of the Gaussian filter. Units must match that
            of `self.boxsize`.
        threads : int, optional
            Number of threads. By default 1.

        Returns
        -------
        smoothed_field : 3-dimensional array of shape `(grid, grid, grid)`
        """
        Filter = "Gaussian"
        grid = field.shape[0]
        # FFT of the filter
        W_k = SL.FT_filter(self.boxsize, scale, grid, Filter, threads)
        return SL.field_smoothing(field, W_k, threads)

    def evaluate_field(self, pos, field):
        """
        Evaluate the field at Cartesian coordinates.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(n_samples, 3)`
            Positions to evaluate the density field. The coordinates span range
            of [0, boxsize].
        field : 3-dimensional array of shape `(grid, grid, grid)`
            The density field that is to be interpolated.

        Returns
        -------
        interp_field : 1-dimensional array of shape `(n_samples,).
            Interpolated field at `pos`.
        """
        self._force_f32(pos, "pos")
        density_interpolated = numpy.zeros(pos.shape[0], dtype=numpy.float32)
        MASL.CIC_interp(field, self.boxsize, pos, density_interpolated)
        return density_interpolated

    def evaluate_sky(self, pos, field, isdeg=True):
        """
        Evaluate the field at given distance, right ascension and declination.
        Assumes that the observed is in the centre of the box.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(n_samples, 3)`
            Spherical coordinates to evaluate the field. Should be distance,
            right ascension, declination, respectively.
        field : 3-dimensional array of shape `(grid, grid, grid)`
            The density field that is to be interpolated. Assumed to be defined
            on a Cartesian grid.
        isdeg : bool, optional
            Whether `ra` and `dec` are in degres. By default `True`.

        Returns
        -------
        interp_field : 1-dimensional array of shape `(n_samples,).
            Interpolated field at `pos`.
        """
        self._force_f32(pos, "pos")
        X = numpy.vstack(
            radec_to_cartesian(*(pos[:, i] for i in range(3)), isdeg)).T
        X = X.astype(numpy.float32)
        # Place the observer at the center of the box
        X += 0.5 * self.boxsize
        return self.evaluate_field(X, field)

    def make_sky_map(self, ra, dec, field, dist_marg, isdeg=True,
                     verbose=True):
        """
        Make a sky map of a density field. Places the observed in the center of
        the box and evaluates the field in directions `ra`, `dec`. At each such
        position evaluates the field at distances `dist_marg` and sums these
        interpolated values of the field.

        Parameters
        ----------
        ra, dec : 1-dimensional arrays of shape `(n_pos, )`
            Directions to evaluate the field. Assumes `dec` is in [-90, 90]
            degrees (or equivalently in radians).
        field : 3-dimensional array of shape `(grid, grid, grid)`
            The density field that is to be interpolated. Assumed to be defined
            on a Cartesian grid `[0, self.boxsize]^3`.
        dist_marg : 1-dimensional array
            Radial distances to evaluate the field.
        isdeg : bool, optional
            Whether `ra` and `dec` are in degres. By default `True`.
        verbose : bool, optional
            Verbosity flag. By default `True`.

        Returns
        -------
        interp_field : 1-dimensional array of shape `(n_pos, )`.
        """
        # Angular positions at which to evaluate the field
        Nang = ra.size
        pos = numpy.vstack([ra, dec]).T

        # Now loop over the angular positions, each time evaluating a vector
        # of distances. Pre-allocate arrays for speed
        ra_loop = numpy.ones_like(dist_marg)
        dec_loop = numpy.ones_like(dist_marg)
        pos_loop = numpy.ones((dist_marg.size, 3), dtype=numpy.float32)

        out = numpy.zeros(Nang, dtype=numpy.float32)
        for i in trange(Nang) if verbose else range(Nang):
            # Get the position vector for this choice of theta, phi
            ra_loop[:] = pos[i, 0]
            dec_loop[:] = pos[i, 1]
            pos_loop[:] = numpy.vstack([dist_marg, ra_loop, dec_loop]).T
            # Evaluate and sum it up
            out[i] = numpy.sum(self.evaluate_sky(pos_loop, field, isdeg))

        return out
