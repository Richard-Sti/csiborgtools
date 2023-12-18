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
from numba import jit

###############################################################################
#                Select particles in a box utility functions                  #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def pbc_distance(x1, x2, boxsize):
    """Calculate periodic distance between two points."""
    delta = abs(x1 - x2)
    return min(delta, boxsize - delta)


@jit(nopython=True, fastmath=True, boundscheck=False)
def find_next_particle(start_index, end_index, pos, x0, y0, z0,
                       half_width, boxsize):
    """
    Find the next particle in a box of size `half_width` centered on `x0`,
    `y0`, `z0`, where the periodic simulation box size is `boxsize`.
    """
    for i in range(start_index, end_index):
        x, y, z = pos[i]
        if ((pbc_distance(x, x0, boxsize) < half_width) and (pbc_distance(y, y0, boxsize) < half_width) and (pbc_distance(z, z0, boxsize) < half_width)):  # noqa
            return i

    return None


def find_boxed(pos, center, subbox_size, boxsize):
    """
    Find indices of `pos` in a box of size `subbox_size` centered on
    `center`, where the simulation box size is `boxsize`.

    Parameters
    ----------
    pos : 2-dimensional array of shape (nsamples, 3)
        Positions of all particles in the simulation.
    center : 1-dimensional array
        Center of the sub-box.
    subbox_size : float
        Size of the sub-box.
    boxsize : float
        Size of the simulation box.

    Returns
    -------
    indxs : 1-dimensional array of shape
    """
    if isinstance(center, list):
        center = numpy.asanyarray(center)

    half_width = subbox_size / 2.

    indxs, start_index, end_index = [], 0, len(pos)
    while True:
        i = find_next_particle(start_index, end_index, pos,
                               *center, half_width, boxsize)

        if i is None:
            break

        indxs.append(i)
        start_index = i + 1

    return indxs
