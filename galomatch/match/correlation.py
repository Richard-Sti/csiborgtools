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


def get_randoms_sphere(N, seed=42):
    """
    Generate random points on a sphere.

    Parameters
    ----------
    N : int
        Number of points.
    seed : int
        Random seed.

    Returns
    -------
    ra : 1-dimensional array
        Right ascension in :math:`[0, 360)` degrees.
    dec : 1-dimensional array
        Declination in :math:`[-90, 90]` degrees.
    """
    gen = numpy.random.default_rng(seed)
    ra = gen.random(N) * 360
    dec = numpy.rad2deg(numpy.arcsin(2 * (gen.random(N) - 0.5)))
    return ra, dec
