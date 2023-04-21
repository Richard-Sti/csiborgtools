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
"""Useful functions."""
import numpy


def concatenate_parts(list_parts, include_velocities=False):
    """
    Concatenate a list of particle arrays into a single array.

    Parameters
    ----------
    list_parts : list of structured arrays
        List of particle arrays.
    include_velocities : bool, optional
        Whether to include velocities in the output array.

    Returns
    -------
    parts_out : structured array
    """
    # Count how large array will be needed
    N = 0
    for part in list_parts:
        N += part.size
    # Infer dtype of positions
    if list_parts[0]["x"].dtype.char in numpy.typecodes["AllInteger"]:
        posdtype = numpy.int32
    else:
        posdtype = numpy.float32

    # We pre-allocate an empty array. By default, we include just particle
    # positions, which may be specified by cell IDs if integers, and masses.
    # Additionally also outputs velocities.
    if include_velocities:
        dtype = {
            "names": ["x", "y", "z", "vx", "vy", "vz", "M"],
            "formats": [posdtype] * 3 + [numpy.float32] * 4,
        }
    else:
        dtype = {
            "names": ["x", "y", "z", "M"],
            "formats": [posdtype] * 3 + [numpy.float32],
        }
    parts_out = numpy.full(N, numpy.nan, dtype)

    # Fill it one clump by another
    start = 0
    for parts in list_parts:
        end = start + parts.size
        for p in dtype["names"]:
            parts_out[p][start:end] = parts[p]
        start = end

    return parts_out
