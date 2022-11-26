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

import numpy
import MAS_library as MASL
from warnings import warn


class DensityField:
    """
    Density field calculations. Based primarily on routines of Pylians [1].

    Parameters
    ----------
    particles : structured array
        Particle array. Must contain keys `['x', 'y', 'z', 'M']`.
    boxsiez : float, optional
        Size of the box. By default 1.

    References
    ----------
    [1] https://pylians3.readthedocs.io/
    """
    _particles = None
    _boxsize = None

    def __init__(self, particles, boxsize=1.):
        self.particles = particles
        self.boxsize = boxsize

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
        The boxsize.

        Returns
        -------
        boxsize : float
        """
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        """Sets boxsize, checking its a float."""
        boxsize = float(boxsize) if isinstance(boxsize, int) else boxsize
        if not isinstance(boxsize, float) and boxsize > 0:
            raise ValueError("Invalid `boxsize` of `{}`.".format(boxsize))
        self._boxsize = boxsize

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
        pos = pos.astype(numpy.float32)
        weights = self.particles['M'].astype(numpy.float32)
        MAS = "CIC"  # Cloud in cell

        # Pre-allocate and do calculations
        rho = numpy.zeros((grid, grid, grid), dtype=numpy.float32)
        MASL.MA(pos, rho, self.boxsize, MAS, W=weights, verbose=verbose)
        return rho

#       def smooth_field(self):
#            import smoothing_library as SL
#
#            BoxSize = 75.0 #Mpc/h
#            R       = 5.0  #Mpc.h
#            grid    = field.shape[0]
#            Filter  = 'Top-Hat'
#            threads = 28
#
#            # compute FFT of the filter
#            W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
#
#            # smooth the field
#            field_smoothed = SL.field_smoothing(field, W_k, threads)

    def evaluate_field(self, pos, field):
        """
        Evaluate the field at given positions.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(n_samples, 3)`
            Positions to evaluate the density field.
        field : 3-dimensional array of shape `(grid, grid, grid)`
            The density field that is to be interpolated.

        Returns
        -------
        interp_field : 1-dimensional array of shape `(n_samples,).
            Interpolated field at `pos`.
        """
        if pos.dtype != numpy.float32:
            warn("Converting `pos` to float32.")
            pos = pos.astype(numpy.float32)
        density_interpolated = numpy.zeros(pos.shape[0], dtype=numpy.float32)
        MASL.CIC_interp(field, self.boxsize, pos, density_interpolated)
        return density_interpolated
