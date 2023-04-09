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
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from .utils import BaseRVS


class Mock2PCF:
    """
    Tool to calculate the 2PCF of a catalogue.
    """
    def __call__(self, pos, rvs_gen, nrandom, bins, pimax):
        """
        Projected auto-2PCF.

        Parameters
        ----------
        pos : 2-dimensional array of shape `(ndata, 3)`
            Positions of the data.
        rvs_gen : :py:class:`csiborgtools.clustering.BaseRVS`
            Uniform RVS generator.
        nrandom : int
            Number of random points to generate.
        bins : 1-dimensional array of shape `(nbins,)`
            Projected separation bins.
        pimax : float
            Maximum line-of-sight separation.

        Returns
        -------
        rp : 1-dimensional array of shape `(nbins - 1,)`
            Projected separation where the auto-2PCF is evaluated.
        wp : 1-dimensional array of shape `(nbins - 1,)`
            The auto-2PCF.
        """
        assert isinstance(rvs_gen, BaseRVS)
        rand_pos = rvs_gen(nrandom)

        dd = DDrppi(autocorr=1, nthreads=1, pimax=pimax, binfile=bins,
                    X1=pos[:, 0], Y1=pos[:, 1], Z1=pos[:, 2], periodic=False)
        dr = DDrppi(autocorr=0, nthreads=1, pimax=pimax, binfile=bins,
                    X1=pos[:, 0], Y1=pos[:, 1], Z1=pos[:, 2], periodic=False,
                    X2=rand_pos[:, 0], Y2=rand_pos[:, 1], Z2=rand_pos[:, 2])
        rr = DDrppi(autocorr=1, nthreads=1, pimax=pimax, binfile=bins,
                    X1=rand_pos[:, 0], Y1=rand_pos[:, 1], Z1=rand_pos[:, 2],
                    periodic=False)

        ndata = pos.shape[0]
        wp = convert_rp_pi_counts_to_wp(ndata, ndata, nrandom, nrandom,
                                        dd, dr, dr, rr, bins.size - 1, pimax)
        rp = 0.5 * (bins[1:] + bins[:-1])
        return rp, wp
