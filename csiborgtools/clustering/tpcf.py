# Copyright (C) 2023 Richard Stiskalek
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
"""2PCF calculation."""
import numpy
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf
from .utils import BaseRVS


class Mock2PCF:
    """
    Tool to calculate the 2PCF of a catalogue.
    """
    def __call__(self, pos, rvs_gen, nrandom, bins, random_state=42):
        """
        Calculate the 2PCF from 3D pair counts.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(ndata, 3)`
            Positions of the data.
        rvs_gen : :py:class:`csiborgtools.clustering.BaseRVS`
            Uniform RVS generator.
        nrandom : int
            Number of random points to generate.
        bins : 1-dimensional array of shape `(nbins,)`
            Separation bins.
        random_state : int, optional
            Random state for the RVS generator.

        Returns
        -------
        rp : 1-dimensional array of shape `(nbins - 1,)`
            Projected separation where the auto-2PCF is evaluated.
        xi : 1-dimensional array of shape `(nbins - 1,)`
            The auto-2PCF.
        """
        assert isinstance(rvs_gen, BaseRVS)
        pos = pos.astype(numpy.float64)
        rand_pos = rvs_gen(nrandom, random_state=random_state,
                           dtype=numpy.float64)

        dd = DD(autocorr=1, nthreads=1, binfile=bins,
                X1=pos[:, 0], Y1=pos[:, 1], Z1=pos[:, 2], periodic=False)
        dr = DD(autocorr=0, nthreads=1, binfile=bins,
                X1=pos[:, 0], Y1=pos[:, 1], Z1=pos[:, 2],
                X2=rand_pos[:, 0], Y2=rand_pos[:, 1], Z2=rand_pos[:, 2],
                periodic=False)
        rr = DD(autocorr=1, nthreads=1, binfile=bins,
                X1=rand_pos[:, 0], Y1=rand_pos[:, 1], Z1=rand_pos[:, 2],
                periodic=False)

        ndata = pos.shape[0]
        xi = convert_3d_counts_to_cf(ndata, ndata, nrandom, nrandom,
                                     dd, dr, dr, rr)
        rp = 0.5 * (bins[1:] + bins[:-1])
        return rp, xi
