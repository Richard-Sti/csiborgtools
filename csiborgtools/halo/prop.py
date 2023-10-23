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

import numpy
from scipy.stats import binned_statistic

from ..utils import periodic_distance


def density_profile(pos, mass, center, nbins, boxsize):
    """
    Calculate a density profile.
    """
    raise NotImplementedError("Not implemented yet..")

    rdist = periodic_distance(pos, center, boxsize)
    rmin, rmax = numpy.min(rdist), numpy.max(rdist)

    bin_edges = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), nbins)


    rho, __, __ = binned_statistic(rdist, mass, statistic='sum',
                                   bins=bin_edges)

    rho /= 4. / 3 * numpy.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    print(bin_edges)

    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])

#    r = numpy.sqrt(bin_edges[:1] * bin_edges[:-1])

    return r, rho

