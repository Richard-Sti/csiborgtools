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
I/O functions for analysing the CSiBORG realisations.
"""

from os.path import join
import numpy


def dump_split(arr, Nsplit, Nsim, Nsnap, outdir):
    """
    Dump an array from a split.

    Parameters
    ----------
    arr : n-dimensional or structured array
        Array to be saved.
    Nsplit : int
        The split index.
    Nsim : int
        The CSiBORG realisation index.
    Nsnap : int
        The index of a redshift snapshot.
    outdir : string
        Directory where to save the temporary files.

    Returns
    -------
    None
    """
    Nsim = str(Nsim).zfill(5)
    Nsnap = str(Nsnap).zfill(5)
    fname = join(outdir, "ramses_out_{}_{}_{}.npy".format(Nsim, Nsnap, Nsplit))
    numpy.save(fname, arr)


def combine_split(Nsplits, Nsim, Nsnap, srcdir):
    """
    Figure out how many splits.
    Get kind of array from the first split
    Initi an empty array
    """
    pass
