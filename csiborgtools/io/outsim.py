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


import numpy
from os.path import join
from os import remove
from tqdm import trange
from .readsim import (get_sim_path, read_clumps)

I64 = numpy.int64
F64 = numpy.float64


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


def combine_splits(Nsplits, Nsim, Nsnap, outdir, cols_add, remove_splits=False,
                   verbose=True):
    """
    Combine results of many splits saved from `dump_split`. Identifies to which
    clump the clumps in the split correspond to by matching their index.
    Returns an array that contains the original clump data along with the newly
    calculated quantities.

    Paramaters
    ----------
    Nsplits : int
        The total number of clump splits.
    Nsim : int
        The CSiBORG realisation index.
    Nsnap : int
        The index of a redshift snapshot.
    outdir : str
        Directory where to save the new array.
    cols_add : list of `(str, dtype)`
        Colums to add. Must be formatted as, for example,
        `[("npart", numpy.float64), ("totpartmass", numpy.float64)]`.
    remove_splits : bool, optional
        Whether to remove the splits files. By default `False`.
    verbose : bool, optional
        Verbosity flag. By default `True`.

    Returns
    -------
    out : structured array
        Clump array with appended results from the splits.
    """
    # Load clumps to see how many there are and will add to this array
    simpath = get_sim_path(Nsim)
    clumps = read_clumps(Nsnap, simpath, cols=None)
    # Get the old + new dtypes and create an empty array
    descr = clumps.dtype.descr + cols_add
    out = numpy.full(clumps.size, numpy.nan, dtype=descr)
    # Now put the old values into the array
    for par in clumps.dtype.names:
        out[par] = clumps[par]

    # Filename of splits data
    froot = "ramses_out_{}_{}".format(str(Nsim).zfill(5), str(Nsnap).zfill(5))
    fname = join(outdir, froot + "_{}.npy")

    # Iterate over splits and add to the output array
    cols_add_names = [col[0] for col in cols_add]
    iters = trange(Nsplits) if verbose else range(Nsplits)
    for n in iters:
        fnamesplit = fname.format(n)
        arr = numpy.load(fnamesplit)

        # Check that all halo indices from the split are in the clump file
        if not numpy.alltrue(numpy.isin(arr["index"], out["index"])):
            raise KeyError("....")
        # Mask of where to put the values from the split
        mask = numpy.isin(out["index"], arr["index"])
        for par in cols_add_names:
            out[par][mask] = arr[par]

        # Now remove this split
        if remove_splits:
            remove(fnamesplit)

    return out
