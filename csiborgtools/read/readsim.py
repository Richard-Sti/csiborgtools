# Copyright (C) 2022 Richard Stiskalek, Harry Desmond
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
Functions to read in the particle and clump files.
"""
from warnings import warn

###############################################################################
#                         Supplementary functions                             #
###############################################################################


def make_halomap_dict(halomap):
    """
    Make a dictionary mapping halo IDs to their start and end indices in the
    snapshot particle array.
    """
    return {hid: (int(start), int(end)) for hid, start, end in halomap}


def load_halo_particles(hid, particles, hid2map):
    """
    Load a halo's particles from a particle array. If it is not there, i.e
    halo has no associated particles, return `None`.

    Parameters
    ----------
    hid : int
        Halo ID.
    particles : 2-dimensional array
        Array of particles.
    hid2map : dict
        Dictionary mapping halo IDs to `halo_map` array positions.

    Returns
    -------
    parts : 1- or 2-dimensional array
    """
    try:
        k0, kf = hid2map[hid]
        return particles[k0:kf + 1]
    except KeyError:
        return None


def convert_str_to_num(s):
    """
    Convert a string representation of a number to its appropriate numeric type
    (int or float).

    Parameters
    ----------
    s : str
        The string representation of the number.

    Returns
    -------
    num : int or float
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            warn(f"Cannot convert string '{s}' to number", UserWarning)
            return s
